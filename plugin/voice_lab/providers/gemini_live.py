"""Gemini Live WebSocket provider for the standalone voice lab."""

from __future__ import annotations

import base64
import json
import os
import wave
from collections.abc import Callable
from typing import Any, Protocol

from ..auth import load_voice_lab_env_file
from ..metrics import MetricsRecorder
from ..transport import (
    header_lines,
    merge_transport_headers,
    resolve_voice_transport_options,
)
from .base import (
    ProviderInfo,
    ProviderRunError,
    ProviderUnavailable,
    VoiceProvider,
    VoiceRequest,
    VoiceResponse,
)

DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_VOICE = "Kore"
DEFAULT_OUTPUT_FORMAT = "audio/pcm"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

GEMINI_LIVE_MODELS = (
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-3.8-live",
)

GEMINI_LIVE_VOICES = (
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
)


class WebSocketLike(Protocol):
    def send(self, payload: str) -> Any: ...

    def recv(self) -> str | bytes: ...

    def close(self) -> Any: ...


SocketFactory = Callable[[str, list[str], float], WebSocketLike]


class GeminiLiveProvider(VoiceProvider):
    """Render text prompts through Gemini Live over a server-side socket."""

    info = ProviderInfo(
        id="gemini_live",
        name="Gemini Live",
        description=(
            "Server-side WebSocket adapter for Gemini Live "
            "(BidiGenerateContent), defaulting to "
            "gemini-3.1-flash-live-preview and WAV output for CLI testing. "
            "Runs on the Gemini API free tier with an AI Studio API key."
        ),
        models=GEMINI_LIVE_MODELS,
        voices=GEMINI_LIVE_VOICES,
        sample_rates=(24000,),
        supports_tts=True,
        supports_realtime=True,
        supports_speech_to_speech=True,
        supports_tool_use=True,
        supports_interruption=True,
        supports_expression=False,
    )

    def __init__(self, socket_factory: SocketFactory | None = None) -> None:
        self._socket_factory = socket_factory

    def run_text(
        self,
        request: VoiceRequest,
        recorder: MetricsRecorder,
    ) -> VoiceResponse:
        options = request.provider_options
        api_key = _resolve_api_key(options)
        if not api_key:
            raise ProviderUnavailable(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for "
                "provider gemini_live"
            )

        model = _option(options, "model") or DEFAULT_MODEL
        voice = _option(options, "voice") or DEFAULT_VOICE
        output_format = _option(options, "output_format") or DEFAULT_OUTPUT_FORMAT
        if output_format != DEFAULT_OUTPUT_FORMAT:
            raise ProviderUnavailable(
                "gemini_live currently supports output_format=audio/pcm only"
            )
        sample_rate = _int_option(options, "sample_rate", DEFAULT_SAMPLE_RATE)
        if sample_rate != DEFAULT_SAMPLE_RATE:
            raise ProviderUnavailable(
                "gemini_live currently supports sample_rate=24000 only"
            )
        timeout = _float_option(options, "timeout", DEFAULT_TIMEOUT_SECONDS)
        url = _option(options, "url") or os.getenv("GEMINI_LIVE_URL") or DEFAULT_URL

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        transport = resolve_voice_transport_options(options, base_url=url)
        headers = {"x-goog-api-key": api_key}
        headers = merge_transport_headers(headers, transport)
        header_values = header_lines(headers)

        recorder.event(
            "request_started",
            provider=self.info.id,
            model=model,
            voice=voice,
            text_chars=len(request.text),
            expression=request.expression.to_dict(),
        )

        ws: WebSocketLike | None = None
        transcript_parts: list[str] = []
        audio_bytes = 0
        events_seen: list[str] = []
        try:
            if self._socket_factory is None:
                ws = _create_websocket(
                    url,
                    header_values,
                    timeout,
                    sslopt=transport.websocket_sslopt,
                )
            else:
                ws = self._socket_factory(url, header_values, timeout)
            recorder.event("websocket_connected", provider=self.info.id, model=model)
            _send_json(ws, _setup_message(model=model, voice=voice, options=options))
            recorder.event("client_event_sent", client_event_type="setup")
            _send_json(ws, _client_content(request))
            recorder.event("client_event_sent", client_event_type="clientContent")

            with wave.open(str(request.output_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)

                while True:
                    event = _recv_json(ws)
                    event_type = _event_type(event)
                    events_seen.append(event_type)
                    recorder.event(
                        "server_event",
                        server_event_type=event_type,
                    )

                    if event_type == "error":
                        raise ProviderRunError(_format_gemini_error(event))

                    content = event.get("serverContent")
                    if not isinstance(content, dict):
                        continue
                    transcription = content.get("outputTranscription")
                    if isinstance(transcription, dict):
                        text = transcription.get("text")
                        if isinstance(text, str) and text:
                            transcript_parts.append(text)
                    model_turn = content.get("modelTurn")
                    if isinstance(model_turn, dict):
                        parts = model_turn.get("parts")
                        if isinstance(parts, list):
                            for part in parts:
                                if not isinstance(part, dict):
                                    continue
                                chunk = _decode_audio_part(part)
                                if chunk is None:
                                    continue
                                wav.writeframes(chunk)
                                audio_bytes += len(chunk)
                                recorder.audio_chunk(
                                    byte_count=len(chunk),
                                    label="modelTurn.audio",
                                    pcm=chunk,
                                )
                                if request.audio_sink is not None:
                                    request.audio_sink(
                                        chunk,
                                        {
                                            "label": "modelTurn.audio",
                                            "sample_rate": sample_rate,
                                            "channels": 1,
                                            "sample_width": 2,
                                        },
                                    )
                    if content.get("turnComplete") or content.get("interrupted"):
                        break
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

        if audio_bytes <= 0:
            _delete_empty_output(request.output_path)
            raise ProviderRunError(
                "Gemini Live completed without audio delta events"
            )

        metrics = recorder.finish(provider=self.info.id, model=model, voice=voice)
        return VoiceResponse(
            provider=self.info.id,
            model=model,
            voice=voice,
            audio_path=request.output_path,
            metrics=metrics,
            metadata={
                "audio_bytes": audio_bytes,
                "output_format": output_format,
                "sample_rate": sample_rate,
                "server_event_types": events_seen,
                "transcript": "".join(transcript_parts).strip() or None,
            },
        )


def _create_websocket(
    url: str,
    headers: list[str],
    timeout: float,
    *,
    sslopt: dict[str, Any] | None = None,
) -> WebSocketLike:
    try:
        import websocket  # type: ignore
    except ImportError as exc:
        raise ProviderUnavailable(
            "gemini_live requires websocket-client; install with "
            "`pip install websocket-client`"
        ) from exc
    kwargs: dict[str, Any] = {"header": headers, "timeout": timeout}
    if sslopt is not None:
        kwargs["sslopt"] = sslopt
    return websocket.create_connection(url, **kwargs)


def _setup_message(
    *,
    model: str,
    voice: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    speech_config: dict[str, Any] = {
        "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
    }
    language = _option(options, "language")
    if language:
        speech_config["languageCode"] = language
    return {
        "setup": {
            "model": f"models/{model}",
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": speech_config,
            },
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "You are a text-to-speech renderer. Speak the "
                            "user's message verbatim, exactly as written, "
                            "with no additions or commentary."
                        )
                    }
                ]
            },
            "outputAudioTranscription": {},
        }
    }


def _client_content(request: VoiceRequest) -> dict[str, Any]:
    return {
        "clientContent": {
            "turns": [
                {
                    "role": "user",
                    "parts": [{"text": request.text}],
                }
            ],
            "turnComplete": True,
        }
    }


def _send_json(ws: WebSocketLike, payload: dict[str, Any]) -> None:
    ws.send(json.dumps(payload, separators=(",", ":")))


def _recv_json(ws: WebSocketLike) -> dict[str, Any]:
    raw = ws.recv()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderRunError("Gemini Live returned non-JSON text") from exc
    if not isinstance(payload, dict):
        raise ProviderRunError("Gemini Live returned a non-object event")
    return payload


def _event_type(event: dict[str, Any]) -> str:
    for key in (
        "setupComplete",
        "serverContent",
        "toolCall",
        "toolCallCancellation",
        "goAway",
        "error",
        "usageMetadata",
        "sessionResumptionUpdate",
    ):
        if key in event:
            return key
    return "unknown"


def _decode_audio_part(part: dict[str, Any]) -> bytes | None:
    inline_data = part.get("inlineData")
    if not isinstance(inline_data, dict):
        return None
    data = inline_data.get("data")
    if not isinstance(data, str) or not data:
        return None
    try:
        return base64.b64decode(data)
    except Exception as exc:
        raise ProviderRunError("Gemini Live audio part was not valid base64") from exc


def _format_gemini_error(event: dict[str, Any]) -> str:
    error = event.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("status") or error.get("code")
        if message:
            return f"Gemini Live error: {message}"
    return f"Gemini Live error event: {event}"


def _delete_empty_output(path: Any) -> None:
    try:
        if path.exists() and path.stat().st_size <= 44:
            path.unlink()
    except OSError:
        pass


def _resolve_api_key(options: dict[str, Any]) -> str:
    for name in ("api_key", "google_api_key", "gemini_api_key"):
        explicit = _option(options, name)
        if explicit:
            return explicit
    load_voice_lab_env_file()
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _option(options: dict[str, Any], name: str) -> str | None:
    value = options.get(name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_option(options: dict[str, Any], name: str, default: int) -> int:
    value = _option(options, name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ProviderUnavailable(f"{name} must be a number") from exc


def _float_option(options: dict[str, Any], name: str, default: float) -> float:
    value = _option(options, name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ProviderUnavailable(f"{name} must be a number") from exc
