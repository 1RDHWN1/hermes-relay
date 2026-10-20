"""Tests for the Gemini Live provider-native realtime-agent adapter."""

from __future__ import annotations

import base64
import os
import unittest
from typing import Any
from unittest.mock import patch

from plugin.relay.realtime_agent.models import (
    HERMES_TOOL_SURFACE,
    ProviderEventKind,
    RealtimeAgentSessionConfig,
)
from plugin.relay.realtime_agent.providers.gemini import (
    GeminiLiveRealtimeAgentProvider,
    _setup_message,
)
from plugin.voice_lab.providers.base import ProviderUnavailable


class FakeGeminiSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.incoming: list[dict[str, Any]] = []
        self.closed = False

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def receive_json(self) -> dict[str, Any]:
        if not self.incoming:
            raise EOFError
        return self.incoming.pop(0)

    async def close(self) -> None:
        self.closed = True


def _config(**overrides: Any) -> RealtimeAgentSessionConfig:
    values: dict[str, Any] = {
        "provider": "gemini_live",
        "model": "gemini-3.1-flash-live-preview",
        "voice": "Kore",
        "sample_rate": 24000,
        "profile": "default",
        "hermes_session_id": "chat-123",
        "instructions": "Use Hermes for current data.",
        "provider_options": {"api_key": "gemini-test-key"},
    }
    values.update(overrides)
    return RealtimeAgentSessionConfig(**values)


async def _connect(fake_socket: FakeGeminiSocket, **overrides: Any):
    captured: dict[str, Any] = {}

    async def factory(url: str, headers: dict[str, str], timeout: float):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return fake_socket

    provider = GeminiLiveRealtimeAgentProvider(socket_factory=factory)
    with patch.dict(os.environ, {}, clear=True):
        connection = await provider.connect(_config(**overrides))
    return connection, captured


class GeminiLiveRealtimeAgentProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_connect_sends_setup_with_manual_turns_and_hermes_tools(
        self,
    ) -> None:
        fake_socket = FakeGeminiSocket()
        connection, captured = await _connect(fake_socket)

        self.assertIs(connection.socket, fake_socket)
        self.assertEqual(
            captured["url"],
            "wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1beta."
            "GenerativeService.BidiGenerateContent",
        )
        self.assertEqual(captured["headers"]["x-goog-api-key"], "gemini-test-key")

        self.assertEqual(len(fake_socket.sent), 1)
        setup = fake_socket.sent[0]["setup"]
        self.assertEqual(setup["model"], "models/gemini-3.1-flash-live-preview")
        generation = setup["generationConfig"]
        self.assertEqual(generation["responseModalities"], ["AUDIO"])
        voice_config = generation["speechConfig"]["voiceConfig"]
        self.assertEqual(
            voice_config["prebuiltVoiceConfig"]["voiceName"],
            "Kore",
        )
        self.assertEqual(
            setup["systemInstruction"]["parts"][0]["text"],
            "Use Hermes for current data.",
        )
        self.assertEqual(
            setup["realtimeInputConfig"]["automaticActivityDetection"],
            {"disabled": True},
        )
        self.assertEqual(setup["inputAudioTranscription"], {})
        self.assertEqual(setup["outputAudioTranscription"], {})
        declarations = setup["tools"][0]["functionDeclarations"]
        tool_names = [tool["name"] for tool in declarations]
        self.assertEqual(tool_names, list(HERMES_TOOL_SURFACE))
        run_tool = next(
            tool for tool in declarations if tool["name"] == "hermes_run_task"
        )
        self.assertIn("parameters", run_tool)
        self.assertIn("text", run_tool["parameters"]["properties"])

    async def test_missing_auth_fails_before_opening_socket(self) -> None:
        provider = GeminiLiveRealtimeAgentProvider(
            socket_factory=lambda *args: self.fail("socket must not open")
        )
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ProviderUnavailable):
                await provider.connect(_config(provider_options={}))

    async def test_unsupported_sample_rate_fails_before_opening_socket(
        self,
    ) -> None:
        provider = GeminiLiveRealtimeAgentProvider(
            socket_factory=lambda *args: self.fail("socket must not open")
        )
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ProviderUnavailable):
                await provider.connect(_config(sample_rate=16000))

    async def test_setup_message_defaults_model_and_voice(self) -> None:
        message = _setup_message(_config(model="", voice=""))
        setup = message["setup"]
        self.assertEqual(setup["model"], "models/gemini-3.1-flash-live-preview")
        voice_config = setup["generationConfig"]["speechConfig"]["voiceConfig"]
        self.assertEqual(voice_config["prebuiltVoiceConfig"]["voiceName"], "Kore")

    async def test_audio_activity_framing_and_resample(self) -> None:
        fake_socket = FakeGeminiSocket()
        connection, _ = await _connect(fake_socket)
        fake_socket.sent.clear()

        pcm = b"\x01\x00" * 24  # 24 samples @ 24 kHz
        await connection.send_audio(pcm, 24000)
        self.assertEqual(fake_socket.sent[0], {"realtimeInput": {"activityStart": {}}})
        audio_message = fake_socket.sent[1]["realtimeInput"]["audio"]
        self.assertEqual(audio_message["mimeType"], "audio/pcm;rate=16000")
        forwarded = base64.b64decode(audio_message["data"])
        self.assertEqual(len(forwarded), 32)  # 16 samples @ 16 kHz

        await connection.send_audio(pcm, 24000)
        starts = [
            item
            for item in fake_socket.sent
            if item.get("realtimeInput", {}).get("activityStart") is not None
        ]
        self.assertEqual(len(starts), 1)

        await connection.commit_audio()
        self.assertEqual(fake_socket.sent[-1], {"realtimeInput": {"activityEnd": {}}})

        await connection.send_audio(pcm, 24000)
        starts = [
            item
            for item in fake_socket.sent
            if item.get("realtimeInput", {}).get("activityStart") is not None
        ]
        self.assertEqual(len(starts), 2)

    async def test_events_normalize(self) -> None:
        fake_socket = FakeGeminiSocket()
        connection, _ = await _connect(fake_socket)

        pcm = b"\x02\x00" * 8
        fake_socket.incoming.extend(
            [
                {"setupComplete": {}},
                {
                    "serverContent": {
                        "inputTranscription": {"text": "halo", "finished": False}
                    }
                },
                {
                    "serverContent": {
                        "inputTranscription": {"text": "halo jarvis", "finished": True}
                    }
                },
                {
                    "serverContent": {
                        "modelTurn": {
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "audio/pcm;rate=24000",
                                        "data": base64.b64encode(pcm).decode("ascii"),
                                    }
                                }
                            ]
                        }
                    }
                },
                {
                    "toolCall": {
                        "functionCalls": [
                            {
                                "id": "call-1",
                                "name": "hermes_run_task",
                                "args": {"text": "cek status"},
                            }
                        ]
                    }
                },
                {"serverContent": {"turnComplete": True}},
                {"goAway": {"timeLeft": "30s"}},
            ]
        )
        events = []
        async for event in connection.events():
            events.append(event)

        kinds = [event.kind for event in events]
        self.assertEqual(
            kinds,
            [
                ProviderEventKind.READY,
                ProviderEventKind.INPUT_TRANSCRIPT_DELTA,
                ProviderEventKind.INPUT_TRANSCRIPT_DELTA,
                ProviderEventKind.INPUT_TRANSCRIPT_FINAL,
                ProviderEventKind.RESPONSE_STARTED,
                ProviderEventKind.AUDIO_DELTA,
                ProviderEventKind.FUNCTION_CALL_COMPLETED,
                ProviderEventKind.AUDIO_DONE,
                ProviderEventKind.RESPONSE_DONE,
                ProviderEventKind.ERROR,
            ],
        )
        self.assertEqual(events[1].payload["delta"], "halo")
        self.assertEqual(events[2].payload["delta"], " jarvis")
        self.assertEqual(events[3].payload["text"], "halo jarvis")
        self.assertEqual(events[5].payload["audio"], pcm)
        call = events[6].payload["call"]
        self.assertEqual(call.call_id, "call-1")
        self.assertEqual(call.name, "hermes_run_task")
        self.assertEqual(call.arguments, {"text": "cek status"})
        self.assertIn("goAway", events[9].payload["message"])

    async def test_tool_result_uses_remembered_name(self) -> None:
        fake_socket = FakeGeminiSocket()
        connection, _ = await _connect(fake_socket)
        fake_socket.incoming.append(
            {
                "toolCall": {
                    "functionCalls": [
                        {"id": "call-9", "name": "hermes_get_status", "args": {}}
                    ]
                }
            }
        )
        async for _event in connection.events():
            break
        fake_socket.sent.clear()
        await connection.send_tool_result("call-9", {"status": "idle"})
        response = fake_socket.sent[-1]["toolResponse"]["functionResponses"][0]
        self.assertEqual(response["id"], "call-9")
        self.assertEqual(response["name"], "hermes_get_status")
        self.assertEqual(response["response"], {"status": "idle"})

    async def test_text_context_and_response_triggers(self) -> None:
        fake_socket = FakeGeminiSocket()
        connection, _ = await _connect(fake_socket)
        fake_socket.sent.clear()

        await connection.send_text("halo")
        turn = fake_socket.sent[-1]["clientContent"]
        self.assertTrue(turn["turnComplete"])
        self.assertEqual(turn["turns"][0]["parts"][0]["text"], "halo")

        await connection.append_context_item(role="assistant", text="sudah dicek")
        turn = fake_socket.sent[-1]["clientContent"]
        self.assertFalse(turn["turnComplete"])
        self.assertEqual(turn["turns"][0]["role"], "model")

        sent_before = len(fake_socket.sent)
        await connection.request_response()
        self.assertEqual(len(fake_socket.sent), sent_before)

        await connection.request_response(exact_text="presensi dibuka")
        turn = fake_socket.sent[-1]["clientContent"]
        self.assertTrue(turn["turnComplete"])
        self.assertIn("presensi dibuka", turn["turns"][0]["parts"][0]["text"])

    async def test_error_event_normalizes(self) -> None:
        fake_socket = FakeGeminiSocket()
        connection, _ = await _connect(fake_socket)
        fake_socket.incoming.append(
            {"error": {"code": 500, "message": "internal boom", "status": "INTERNAL"}}
        )
        events = []
        async for event in connection.events():
            events.append(event)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, ProviderEventKind.ERROR)
        self.assertIn("internal boom", events[0].payload["message"])


if __name__ == "__main__":
    unittest.main()
