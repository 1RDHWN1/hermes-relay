"""Forced-summary response gate (`_should_forward_provider_response_event`).

Regression coverage for the live failure of 2026-10-22: after a promoted
Hermes run completed, an event carrying the already-finished
acknowledgement turn's response id reached the gate first and claimed the
forced-summary slot. The real summary response was then suppressed as a
"duplicate" 208 times, no summary audio ever played, and the voice turn
never closed. The gate must only ever pin the first *live* response:
stragglers of completed responses are dropped without pinning, and done
events never pin a response that was never started.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from plugin.relay.config import RelayConfig
from plugin.relay.realtime_agent.broker import (
    RealtimeAgentHandler,
    RealtimeAgentSession,
)
from plugin.relay.realtime_agent.models import ProviderEvent, ProviderEventKind


def _event(kind: ProviderEventKind, response_id: str) -> ProviderEvent:
    return ProviderEvent(kind, response_id=response_id, payload={})


class ForcedSummaryGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.handler = RealtimeAgentHandler(RelayConfig())
        self.session = RealtimeAgentSession(
            session_id="sess-summary-gate",
            provider="gemini_live",
            model="gemini-3.8-live",
            voice="Leda",
            sample_rate=24000,
            profile=None,
            chat_session_id="chat-1",
            config_scope="test",
            config_path=None,
            auth_kind="test",
            bearer_token=None,
            created_at=time.time(),
            event_log_path=Path(self._tmp.name) / "events.jsonl",
            resume_token_hash="hash",
        )
        self.session.native_forced_summary_active = True

    def gate(self, kind: ProviderEventKind, response_id: str) -> bool:
        return self.handler._should_forward_provider_response_event(
            self.session, _event(kind, response_id)
        )

    def test_stale_completed_response_cannot_pin_or_block_summary(self) -> None:
        session = self.session
        session.response_ids_done.add("gemini-turn-2")

        # Straggler events of the finished acknowledgement turn.
        self.assertFalse(self.gate(ProviderEventKind.AUDIO_DONE, "gemini-turn-2"))
        self.assertFalse(self.gate(ProviderEventKind.RESPONSE_DONE, "gemini-turn-2"))
        self.assertFalse(
            self.gate(ProviderEventKind.OUTPUT_TEXT_DELTA, "gemini-turn-2")
        )
        self.assertIsNone(session.native_forced_summary_response_id)

        # The live summary response pins the slot and flows.
        self.assertTrue(self.gate(ProviderEventKind.AUDIO_DELTA, "gemini-turn-3"))
        self.assertEqual("gemini-turn-3", session.native_forced_summary_response_id)
        self.assertTrue(self.gate(ProviderEventKind.RESPONSE_DONE, "gemini-turn-3"))

        # A genuinely different second response is still a duplicate.
        self.assertFalse(self.gate(ProviderEventKind.AUDIO_DELTA, "gemini-turn-4"))
        self.assertEqual("gemini-turn-3", session.native_forced_summary_response_id)

    def test_unstarted_done_event_does_not_pin(self) -> None:
        self.assertFalse(self.gate(ProviderEventKind.RESPONSE_DONE, "ghost"))
        self.assertFalse(self.gate(ProviderEventKind.AUDIO_DONE, "ghost"))
        self.assertIsNone(self.session.native_forced_summary_response_id)

    def test_done_event_for_started_live_response_may_pin(self) -> None:
        self.session.response_ids_started.add("resp-9")
        self.assertTrue(self.gate(ProviderEventKind.RESPONSE_DONE, "resp-9"))
        self.assertEqual("resp-9", self.session.native_forced_summary_response_id)

    def test_started_event_pins_and_later_ids_are_duplicates(self) -> None:
        self.assertTrue(self.gate(ProviderEventKind.RESPONSE_STARTED, "resp-1"))
        self.assertEqual("resp-1", self.session.native_forced_summary_response_id)
        self.assertFalse(self.gate(ProviderEventKind.RESPONSE_STARTED, "resp-2"))

    def _conclude_summary(self) -> None:
        self.session.native_forced_summary_active = False
        self.session.native_forced_summary_done = True
        self.session.native_forced_summary_done_at = 100.0
        self.session.response_ids_done.add("gemini-turn-4")

    def test_followup_response_with_fresh_input_lifts_mute(self) -> None:
        # Live failure 2026-10-22 (second session): the provider answered
        # a follow-up question before the transcript-final reset ran, and
        # the whole response was eaten by the post-summary mute.
        self._conclude_summary()
        self.session.native_last_input_audio_at = 150.0
        self.assertTrue(
            self.gate(ProviderEventKind.RESPONSE_STARTED, "gemini-turn-5")
        )
        self.assertFalse(self.session.native_forced_summary_done)
        self.assertTrue(self.gate(ProviderEventKind.AUDIO_DELTA, "gemini-turn-5"))

    def test_auto_continuation_without_fresh_input_stays_muted(self) -> None:
        self._conclude_summary()
        self.session.native_last_input_audio_at = 50.0
        self.assertFalse(
            self.gate(ProviderEventKind.RESPONSE_STARTED, "gemini-turn-5")
        )
        self.assertTrue(self.session.native_forced_summary_done)

    def test_completed_response_id_never_lifts_mute(self) -> None:
        self._conclude_summary()
        self.session.native_last_input_audio_at = 150.0
        self.assertFalse(
            self.gate(ProviderEventKind.RESPONSE_STARTED, "gemini-turn-4")
        )
        self.assertTrue(self.session.native_forced_summary_done)

    def test_mark_done_records_completion_without_active_summary(self) -> None:
        self.session.native_forced_summary_active = False
        self.handler._mark_provider_response_done(
            self.session, _event(ProviderEventKind.RESPONSE_DONE, "resp-7")
        )
        self.assertIn("resp-7", self.session.response_ids_done)
        self.assertFalse(self.session.native_forced_summary_done)


if __name__ == "__main__":
    unittest.main()
