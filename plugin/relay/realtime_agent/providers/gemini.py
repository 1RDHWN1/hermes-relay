"""Gemini Live provider-native realtime-agent connection.

Talks to the Gemini Live API (BidiGenerateContent WebSocket) directly.
Gemini Live is available on the Gemini API free tier with an AI Studio
API key, which makes this the zero-cost speech-to-speech provider.

Protocol notes (v1beta):
- Input audio must be PCM16 at 16 kHz; output audio is PCM16 at 24 kHz,
  which matches the broker session rate, so only the uplink is resampled.
- The broker drives turns manually, so server VAD is disabled and turns
  are framed with realtimeInput activityStart / activityEnd.
- Gemini auto-continues after a tool response; there is no explicit
  response.create or response.cancel primitive.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import aiohttp

from ....voice_lab.auth import load_voice_lab_env_file
from ....voice_lab.providers.base import ProviderRunError, ProviderUnavailable
from ....voice_lab.transport import (
    merge_transport_headers,
    resolve_voice_transport_options,
)

from ..models import (
    ProviderEvent,
    ProviderEventKind,
    RealtimeAgentCapabilities,
    RealtimeAgentSessionConfig,
    ToolCallEvent,
)
from .base import RealtimeAgentProviderAdapter

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_VOICE = "Kore"
DEFAULT_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
DEFAULT_TIMEOUT_SECONDS = 60.0
SUPPORTED_SAMPLE_RATE = 24000
GEMINI_INPUT_SAMPLE_RATE = 16000


class GeminiProviderSocket(Protocol):
    async def send_json(self, payload: dict[str, Any]) -> None: ...

    async def receive_json(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...


SocketFactory = Callable[[str, dict[str, str], float], Any]


@dataclass(frozen=True, slots=True)
class AuthToken:
    value: str
    source: str


class GeminiLiveRealtimeAgentAdapter(RealtimeAgentProviderAdapter):
    """Render-only Gemini adapter kept for the non-native fallback path."""

    provider_id = "gemini_live"


class GeminiLiveRealtimeAgentProvider:
    """Connection-oriented Gemini Live provider for Hermes Realtime Agent."""

    provider_id = "gemini_live"
    capabilities = RealtimeAgentCapabilities(
        provider_id=provider_id,
        supports_function_calls=True,
        # Turns are broker-driven (manual activity framing), not server VAD.
        supports_server_vad=False,
    )

    def __init__(self, socket_factory: SocketFactory | None = None) -> None:
        self._socket_factory = socket_factory

    async def connect(
        self,
        config: RealtimeAgentSessionConfig,
    ) -> "GeminiLiveRealtimeAgentConnection":
        if config.sample_rate != SUPPORTED_SAMPLE_RATE:
            raise ProviderUnavailable(
                "Gemini Live currently supports sample_rate=24000 for "
                "provider-native Realtime Agent sessions."
            )
        auth = _resolve_auth_token(config.provider_options)
        if auth is None:
            raise ProviderUnavailable(
                "Gemini Live auth is not configured. Set relay-owned "
                "GEMINI_API_KEY or GOOGLE_API_KEY, or a gemini_live "
                "provider api_key before using Realtime Agent."
            )
        timeout = _float_option(
            config.provider_options,
            "timeout",
            DEFAULT_TIMEOUT_SECONDS,
        )
        url = (
            _option(config.provider_options, "url")
            or os.getenv("GEMINI_LIVE_URL")
            or DEFAULT_URL
        ).strip()
        transport = resolve_voice_transport_options(
            config.provider_options,
            base_url=url,
        )
        headers = {"x-goog-api-key": auth.value}
        headers = merge_transport_headers(headers, transport)

        try:
            if self._socket_factory is None:
                socket = await _create_aiohttp_websocket(
                    url,
                    headers,
                    timeout,
                    ssl=transport.aiohttp_ssl,
                )
            else:
                socket = await self._socket_factory(url, headers, timeout)
        except aiohttp.WSServerHandshakeError as exc:
            if exc.status in {401, 403}:
                raise ProviderUnavailable(
                    "Gemini Live rejected the relay auth "
                    f"({exc.status}; source: {auth.source}). Update the "
                    "relay-side Gemini API key."
                ) from exc
            raise ProviderUnavailable(
                f"Gemini Live websocket handshake failed with HTTP {exc.status}"
            ) from exc
        connection = GeminiLiveRealtimeAgentConnection(
            socket=socket,
            config=config,
            auth_source=auth.source,
        )
        await connection.configure()
        return connection


class GeminiLiveRealtimeAgentConnection:
    def __init__(
        self,
        *,
        socket: GeminiProviderSocket,
        config: RealtimeAgentSessionConfig,
        auth_source: str,
    ) -> None:
        self.socket = socket
        self.config = config
        self.auth_source = auth_source
        self.resumption_handle: str | None = None
        self._activity_open = False
        self._in_turn = False
        self._turn_index = 0
        self._input_transcript = ""
        self._output_transcript = ""
        self._tool_names: dict[str, str] = {}

    async def configure(self) -> None:
        await self.socket.send_json(_setup_message(self.config))

    async def send_audio(self, pcm: bytes, sample_rate: int) -> None:
        if not self._activity_open:
            await self.socket.send_json(
                {"realtimeInput": {"activityStart": {}}}
            )
            self._activity_open = True
        pcm = _resample_pcm16_mono(pcm, sample_rate, GEMINI_INPUT_SAMPLE_RATE)
        if not pcm:
            return
        await self.socket.send_json(
            {
                "realtimeInput": {
                    "audio": {
                        "data": base64.b64encode(pcm).decode("ascii"),
                        "mimeType": f"audio/pcm;rate={GEMINI_INPUT_SAMPLE_RATE}",
                    }
                }
            }
        )

    async def commit_audio(self) -> None:
        if self._activity_open:
            await self.socket.send_json({"realtimeInput": {"activityEnd": {}}})
            self._activity_open = False

    async def send_text(self, text: str) -> None:
        await self.socket.send_json(
            _client_content_turn(role="user", text=text, turn_complete=True)
        )

    async def clear_audio(self) -> None:
        # Audio already streamed cannot be retracted from Gemini. Closing an
        # open activity is the closest equivalent; the partial turn may
        # still produce a (short) response.
        await self.commit_audio()

    async def cancel_response(self) -> None:
        # Gemini Live has no response-cancel primitive. Playback drain is
        # handled broker/app side; server generation ends at turnComplete.
        logger.debug("gemini_live cancel_response: no native primitive; no-op")

    async def send_tool_result(self, call_id: str, output: dict[str, Any]) -> None:
        name = self._tool_names.get(call_id, "")
        await self.socket.send_json(
            {
                "toolResponse": {
                    "functionResponses": [
                        {
                            "id": call_id,
                            "name": name,
                            "response": output,
                        }
                    ]
                }
            }
        )

    async def append_context_item(self, *, role: str, text: str) -> None:
        # Seed history without triggering a response (turnComplete=false).
        gemini_role = "model" if role == "assistant" else "user"
        await self.socket.send_json(
            _client_content_turn(
                role=gemini_role,
                text=text,
                turn_complete=False,
            )
        )

    async def request_response(
        self,
        *,
        instructions: str | None = None,
        exact_text: str | None = None,
    ) -> None:
        # Gemini auto-continues after tool responses, so a bare
        # request_response (the post-tool-call path) is a no-op. Explicit
        # broker-driven speech is triggered with a client-content turn.
        if exact_text:
            await self.socket.send_json(
                _client_content_turn(
                    role="user",
                    text=(
                        "Say exactly the following, word for word, with no "
                        f"additions: {exact_text}"
                    ),
                    turn_complete=True,
                )
            )
        elif instructions:
            await self.socket.send_json(
                _client_content_turn(
                    role="user",
                    text=instructions,
                    turn_complete=True,
                )
            )

    async def close(self) -> None:
        await self.socket.close()

    async def events(self) -> AsyncIterator[ProviderEvent]:
        while True:
            try:
                raw = await self.socket.receive_json()
            except (EOFError, StopAsyncIteration):
                return
            except ProviderRunError as exc:
                yield ProviderEvent(
                    ProviderEventKind.ERROR,
                    payload={"message": str(exc)},
                )
                return
            for event in self._provider_events(raw):
                yield event

    # ── server message translation ─────────────────────────────────────

    def _response_id(self) -> str:
        return f"gemini-turn-{self._turn_index}"

    def _begin_turn(self) -> list[ProviderEvent]:
        if self._in_turn:
            return []
        self._in_turn = True
        self._turn_index += 1
        return [
            ProviderEvent(
                ProviderEventKind.RESPONSE_STARTED,
                response_id=self._response_id(),
                payload={"provider_event_type": "gemini.turn_started"},
            )
        ]

    def _end_turn(self, *, interrupted: bool) -> list[ProviderEvent]:
        events = [
            ProviderEvent(
                ProviderEventKind.AUDIO_DONE,
                response_id=self._response_id(),
                payload={"provider_event_type": "gemini.turn_complete"},
            ),
            ProviderEvent(
                ProviderEventKind.RESPONSE_DONE,
                response_id=self._response_id(),
                payload={
                    "provider_event_type": "gemini.turn_complete",
                    "interrupted": interrupted,
                },
            ),
        ]
        self._in_turn = False
        self._input_transcript = ""
        self._output_transcript = ""
        return events

    def _provider_events(self, raw: dict[str, Any]) -> list[ProviderEvent]:
        if "setupComplete" in raw:
            return [
                ProviderEvent(
                    ProviderEventKind.READY,
                    payload={
                        "provider_event_type": "setupComplete",
                        "resolved_model": self.config.model or DEFAULT_MODEL,
                    },
                )
            ]
        if "error" in raw:
            return [
                ProviderEvent(
                    ProviderEventKind.ERROR,
                    payload={"message": _format_gemini_error(raw)},
                )
            ]
        if "goAway" in raw:
            return [
                ProviderEvent(
                    ProviderEventKind.ERROR,
                    payload={
                        "message": (
                            "Gemini Live session is ending (goAway); "
                            "reconnect to continue the conversation."
                        )
                    },
                )
            ]
        resumption = raw.get("sessionResumptionUpdate")
        if isinstance(resumption, dict):
            handle = resumption.get("newHandle")
            if isinstance(handle, str) and handle:
                self.resumption_handle = handle
            return []
        tool_call = raw.get("toolCall")
        if isinstance(tool_call, dict):
            return self._tool_call_events(tool_call)
        server_content = raw.get("serverContent")
        if isinstance(server_content, dict):
            return self._server_content_events(server_content)
        return []

    def _tool_call_events(self, tool_call: dict[str, Any]) -> list[ProviderEvent]:
        events: list[ProviderEvent] = []
        calls = tool_call.get("functionCalls")
        if not isinstance(calls, list):
            return events
        for call in calls:
            if not isinstance(call, dict):
                continue
            events.extend(self._begin_turn())
            call_id = str(call.get("id") or "").strip()
            name = str(call.get("name") or "").strip()
            args = call.get("args")
            arguments = args if isinstance(args, dict) else {}
            if call_id and name:
                self._tool_names[call_id] = name
            tool_event = ToolCallEvent(
                call_id=call_id,
                name=name,
                arguments=arguments,
            )
            events.append(
                ProviderEvent(
                    ProviderEventKind.FUNCTION_CALL_COMPLETED,
                    response_id=self._response_id(),
                    payload={
                        "call": tool_event,
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                    },
                )
            )
        return events

    def _server_content_events(
        self,
        content: dict[str, Any],
    ) -> list[ProviderEvent]:
        events: list[ProviderEvent] = []
        input_transcription = content.get("inputTranscription")
        if isinstance(input_transcription, dict):
            events.extend(self._input_transcription_events(input_transcription))
        output_transcription = content.get("outputTranscription")
        if isinstance(output_transcription, dict):
            delta = self._transcript_delta(
                output_transcription,
                previous=self._output_transcript,
            )
            if delta:
                self._output_transcript += delta
                events.append(
                    ProviderEvent(
                        ProviderEventKind.OUTPUT_TEXT_DELTA,
                        response_id=self._response_id(),
                        payload={"delta": delta},
                    )
                )
        model_turn = content.get("modelTurn")
        if isinstance(model_turn, dict):
            parts = model_turn.get("parts")
            if isinstance(parts, list):
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    events.extend(self._model_part_events(part))
        if content.get("interrupted"):
            events.extend(self._end_turn(interrupted=True))
        elif content.get("turnComplete"):
            events.extend(self._end_turn(interrupted=False))
        return events

    def _input_transcription_events(
        self,
        transcription: dict[str, Any],
    ) -> list[ProviderEvent]:
        events: list[ProviderEvent] = []
        text = transcription.get("text")
        snapshot = text if isinstance(text, str) else ""
        if snapshot:
            if snapshot.startswith(self._input_transcript):
                delta = snapshot[len(self._input_transcript) :]
            else:
                delta = snapshot
            if delta:
                events.append(
                    ProviderEvent(
                        ProviderEventKind.INPUT_TRANSCRIPT_DELTA,
                        response_id=self._response_id(),
                        payload={"delta": delta},
                    )
                )
            self._input_transcript = snapshot
        if transcription.get("finished"):
            final_text = self._input_transcript
            self._input_transcript = ""
            if final_text:
                events.append(
                    ProviderEvent(
                        ProviderEventKind.INPUT_TRANSCRIPT_FINAL,
                        response_id=self._response_id(),
                        payload={"text": final_text},
                    )
                )
        return events

    def _transcript_delta(
        self,
        transcription: dict[str, Any],
        *,
        previous: str,
    ) -> str:
        text = transcription.get("text")
        if not isinstance(text, str) or not text:
            return ""
        if text.startswith(previous):
            return text[len(previous) :]
        return text

    def _model_part_events(self, part: dict[str, Any]) -> list[ProviderEvent]:
        events: list[ProviderEvent] = []
        inline_data = part.get("inlineData")
        if isinstance(inline_data, dict):
            audio64 = inline_data.get("data")
            if isinstance(audio64, str) and audio64:
                try:
                    chunk = base64.b64decode(audio64)
                except Exception as exc:
                    raise ProviderRunError(
                        "Gemini Live audio part was not valid base64"
                    ) from exc
                events.extend(self._begin_turn())
                events.append(
                    ProviderEvent(
                        ProviderEventKind.AUDIO_DELTA,
                        response_id=self._response_id(),
                        payload={
                            "audio": chunk,
                            "audio_base64": audio64,
                            "byte_count": len(chunk),
                        },
                        metadata={"provider_event_type": "modelTurn.audio"},
                    )
                )
            return events
        text = part.get("text")
        if isinstance(text, str) and text:
            events.extend(self._begin_turn())
            events.append(
                ProviderEvent(
                    ProviderEventKind.OUTPUT_TEXT_DELTA,
                    response_id=self._response_id(),
                    payload={"delta": text},
                )
            )
        return events


class _AioHttpGeminiProviderSocket:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        ws: aiohttp.ClientWebSocketResponse,
    ) -> None:
        self._session = session
        self._ws = ws

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self._ws.send_str(json.dumps(payload, separators=(",", ":")))

    async def receive_json(self) -> dict[str, Any]:
        msg = await self._ws.receive()
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError as exc:
                raise ProviderRunError("Gemini Live returned non-JSON text") from exc
            if not isinstance(payload, dict):
                raise ProviderRunError("Gemini Live returned a non-object event")
            return payload
        if msg.type == aiohttp.WSMsgType.BINARY:
            raise ProviderRunError("Gemini Live returned an unexpected binary frame")
        if msg.type == aiohttp.WSMsgType.ERROR:
            raise ProviderRunError(
                f"Gemini Live websocket error: {self._ws.exception()}"
            )
        raise EOFError

    async def close(self) -> None:
        try:
            await self._ws.close()
        finally:
            await self._session.close()


async def _create_aiohttp_websocket(
    url: str,
    headers: dict[str, str],
    timeout: float,
    *,
    ssl: Any = None,
) -> GeminiProviderSocket:
    # Liveness via WS heartbeat rather than an ambient total-timeout — realtime
    # sessions legitimately live for many minutes; `timeout` bounds the connect.
    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, connect=timeout, sock_connect=timeout)
    )
    try:
        kwargs: dict[str, Any] = {"headers": headers, "heartbeat": 20.0}
        if ssl is not None:
            kwargs["ssl"] = ssl
        ws = await session.ws_connect(url, **kwargs)
    except aiohttp.WSServerHandshakeError as exc:
        await session.close()
        if exc.status in {401, 403}:
            raise ProviderUnavailable(
                "Gemini Live rejected the relay auth. Update the relay-side "
                "Gemini API key."
            ) from exc
        raise ProviderUnavailable(
            f"Gemini Live websocket handshake failed with HTTP {exc.status}"
        ) from exc
    except Exception:
        await session.close()
        raise
    return _AioHttpGeminiProviderSocket(session, ws)


def _setup_message(config: RealtimeAgentSessionConfig) -> dict[str, Any]:
    speech_config: dict[str, Any] = {
        "voiceConfig": {
            "prebuiltVoiceConfig": {
                "voiceName": config.voice or DEFAULT_VOICE,
            }
        }
    }
    language = _option(config.provider_options, "language")
    if language:
        speech_config["languageCode"] = language
    setup: dict[str, Any] = {
        "model": f"models/{config.model or DEFAULT_MODEL}",
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": speech_config,
        },
        "systemInstruction": {
            "parts": [
                {"text": config.instructions or _default_instructions(config)}
            ]
        },
        "tools": [
            {
                "functionDeclarations": [
                    _gemini_function_declaration(tool) for tool in config.tools
                ]
            }
        ],
        # The broker owns turn-taking (floor control); disable server VAD
        # and frame turns with activityStart / activityEnd instead.
        "realtimeInputConfig": {
            "automaticActivityDetection": {"disabled": True},
        },
        "inputAudioTranscription": {},
        "outputAudioTranscription": {},
        "contextWindowCompression": {"slidingWindow": {}},
        "sessionResumption": {},
    }
    return {"setup": setup}


def _gemini_function_declaration(schema: dict[str, Any]) -> dict[str, Any]:
    declaration: dict[str, Any] = {
        "name": str(schema.get("name") or ""),
        "description": str(schema.get("description") or ""),
    }
    parameters = schema.get("parameters")
    if isinstance(parameters, dict) and parameters:
        declaration["parameters"] = dict(parameters)
    return declaration


def _client_content_turn(
    *,
    role: str,
    text: str,
    turn_complete: bool,
) -> dict[str, Any]:
    return {
        "clientContent": {
            "turns": [
                {
                    "role": role,
                    "parts": [{"text": text}],
                }
            ],
            "turnComplete": turn_complete,
        }
    }


def _format_gemini_error(raw: dict[str, Any]) -> str:
    error = raw.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("status") or error.get("code")
        if message:
            return f"Gemini Live error: {message}"
    return f"Gemini Live error event: {raw}"


def _default_instructions(config: RealtimeAgentSessionConfig) -> str:
    profile = config.profile or "active Hermes profile"
    return (
        "You are the voice front-end for Hermes Relay. Use the provided Hermes "
        "functions for user requests that need memory, tools, device or desktop "
        "actions, confirmations, persistent context, research, current facts, "
        "news, external data, live checks, latest/versioned info, personal or "
        "project context, side effects, precision-sensitive answers, or media and "
        "artifact handling. Hermes is the durable conversation memory; use any "
        "seeded recent chat context for follow-up references when it is enough, "
        "and route missing/stale/verification-sensitive references through "
        "Hermes before answering. Do not say you lack context before a Hermes "
        "call. You may speak one brief acknowledgement such as 'I'll check "
        "Hermes' or 'I'll check that' before the tool call, then call Hermes "
        "immediately. Do not give a substantive answer until Hermes returns. "
        "Android will provide restrained local status while Hermes runs. "
        "When speaking, summarize dense machine-readable "
        "values instead of reading raw IDs, URLs, paths, JSON, logs, or long "
        f"numbers character by character. Active profile: {profile}. "
        "Do not call web_search, x_search, MCP, or any non-Hermes tool."
    )


def _resolve_auth_token(options: dict[str, Any]) -> AuthToken | None:
    for name in ("api_key", "google_api_key", "gemini_api_key"):
        explicit = _option(options, name)
        if explicit:
            return AuthToken(value=explicit, source=f"provider-option:{name}")
    load_voice_lab_env_file()
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return AuthToken(value=value, source=f"env:{name}")
    return None


def _option(options: dict[str, Any], name: str) -> str | None:
    value = options.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_option(options: dict[str, Any], name: str, default: float) -> float:
    value = _option(options, name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ProviderUnavailable(f"{name} must be a number") from exc


def _resample_pcm16_mono(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    if not pcm or source_rate <= 0 or target_rate <= 0 or source_rate == target_rate:
        return pcm
    sample_count = len(pcm) // 2
    if sample_count <= 1:
        return pcm
    target_count = max(1, round(sample_count * target_rate / source_rate))
    out = bytearray(target_count * 2)
    for i in range(target_count):
        source_index = min(sample_count - 1, round(i * source_rate / target_rate))
        start = source_index * 2
        out[i * 2 : i * 2 + 2] = pcm[start : start + 2]
    return bytes(out)
