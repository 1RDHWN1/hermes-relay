package com.hermesandroid.relay.voice

import android.app.Application
import android.util.Log
import androidx.lifecycle.viewModelScope
import com.hermesandroid.relay.audio.RealtimePcmPlayer
import com.hermesandroid.relay.data.ChatMessage
import com.hermesandroid.relay.data.MessageRole
import com.hermesandroid.relay.data.VoiceAudioRoute
import com.hermesandroid.relay.network.shared.VoiceAudioClient
import com.hermesandroid.relay.network.shared.VoiceSpeechStream
import com.hermesandroid.relay.network.shared.VoiceSpeechStreamCallbacks
import com.hermesandroid.relay.network.shared.VoiceSpeechStreamOutcome
import com.hermesandroid.relay.network.shared.VoiceSpeechStreamStatus
import com.hermesandroid.relay.viewmodel.ChatViewModel
import com.hermesandroid.relay.viewmodel.InteractionMode
import com.hermesandroid.relay.viewmodel.VoiceState
import com.hermesandroid.relay.viewmodel.VoiceUiState
import com.hermesandroid.relay.viewmodel.VoiceViewModel
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.unmockkAll
import io.mockk.verify
import io.mockk.verifyOrder
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import kotlin.coroutines.CoroutineContext

@OptIn(ExperimentalCoroutinesApi::class)
class StandardVoiceTurnPlaybackTest {
    private val dispatcher = StandardTestDispatcher()
    private lateinit var vm: VoiceViewModel
    private lateinit var pcm: RealtimePcmPlayer
    private val messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    private val running = MutableStateFlow(true)
    private val session = MutableStateFlow<String?>("voice-test")
    private val appended = mutableListOf<String>()
    private var opens = 0
    private var finishes = 0
    private var stops = 0
    private var remaining = 500L
    private lateinit var callbacks: VoiceSpeechStreamCallbacks
    private val outcome = CompletableDeferred<VoiceSpeechStreamOutcome>()
    private val stream = object : VoiceSpeechStream {
        override fun append(text: String) { appended += text }
        override fun finish() { finishes++ }
        override fun stop() { stops++ }
        override suspend fun awaitOutcome() = outcome.await()
    }

    @Before
    fun setup() {
        Dispatchers.setMain(dispatcher)
        mockkStatic(Log::class)
        every { Log.i(any(), any<String>()) } returns 0
        every { Log.d(any(), any<String>()) } returns 0
        every { Log.w(any(), any<String>()) } returns 0
        every { Log.e(any(), any<String>()) } returns 0
        pcm = mockk(relaxed = true)
        every { pcm.estimatedRemainingPlaybackMs(any()) } answers { remaining }
        vm = VoiceViewModel(mockk<Application>(relaxed = true))
        vm.standardSpeechWriteDispatcher = dispatcher
        field("realtimePcmPlayer", pcm)
        field("voiceAudioClient", object : VoiceAudioClient {
            override val route = VoiceAudioRoute.Standard
            override suspend fun transcribe(audioFile: File) = Result.success("")
            override suspend fun synthesize(text: String): Result<File> = error("Unexpected file fallback")
            override suspend fun openSpeechStream(callbacks: VoiceSpeechStreamCallbacks): Result<VoiceSpeechStream?> {
                this@StandardVoiceTurnPlaybackTest.callbacks = callbacks
                opens++
                return Result.success(stream)
            }
        })
        @Suppress("UNCHECKED_CAST")
        val ui = VoiceViewModel::class.java.getDeclaredField("_uiState").apply { isAccessible = true }
            .get(vm) as MutableStateFlow<VoiceUiState>
        ui.value = VoiceUiState(voiceMode = true, state = VoiceState.Thinking,
            interactionMode = InteractionMode.HoldToTalk)
    }

    @After
    fun cleanup() {
        vm.viewModelScope.cancel()
        Dispatchers.resetMain()
        unmockkAll()
    }

    private fun observe(finalOnly: Boolean = false) {
        field("finalAnswerOnly", finalOnly)
        if (!finalOnly) invoke("prepareStandardSpeechStream")
        val chat = mockk<ChatViewModel>(relaxed = true)
        every { chat.messages } returns messages
        every { chat.isStreaming } returns running
        every { chat.currentSessionId } returns session
        VoiceViewModel::class.java.getDeclaredMethod("startStreamObserver", ChatViewModel::class.java)
            .apply { isAccessible = true }.invoke(vm, chat)
    }

    @Test
    fun toolGapDoesNotFinishStreamAndFinalPcmDrainsBeforeIdle() = runTest(dispatcher) {
        observe()
        runCurrent()
        messages.value = listOf(message("first", "The opening sentence."))
        runCurrent()
        callbacks.onPcm(ByteArray(90_000), 24_000)
        runCurrent()
        remaining = 0
        advanceTimeBy(5_000)
        runCurrent()
        assertEquals(0, finishes)
        assertEquals(0, stops)
        assertTrue(vm.uiState.value.state != VoiceState.Idle)

        messages.value += message("last", "The final answer.")
        running.value = false
        runCurrent()
        assertEquals("The opening sentence.\n\nThe final answer.", appended.joinToString(""))
        assertEquals(1, finishes)
        remaining = 1_000
        callbacks.onPcm(ByteArray(48_000), 24_000)
        outcome.complete(VoiceSpeechStreamOutcome(VoiceSpeechStreamStatus.Completed, true))
        runCurrent()
        advanceTimeBy(500)
        runCurrent()
        assertEquals(VoiceState.Speaking, vm.uiState.value.state)
        verify(exactly = 2) { pcm.write(any(), any(), any()) }
        remaining = 0
        // The production echo tail guard uses wall time, independent of the test scheduler.
        field("lastRealtimeAudioDeltaAtMs", 1L)
        advanceTimeBy(3_000)
        runCurrent()
        assertEquals(VoiceState.Idle, vm.uiState.value.state)
    }

    @Test
    fun finalAnswerOnlyOpensSpeechForSettledFinalBubble() = runTest(dispatcher) {
        observe(finalOnly = true)
        runCurrent()
        messages.value = listOf(message("interim", "I will check."))
        runCurrent()
        advanceTimeBy(5_000)
        assertEquals(0, opens)
        messages.value += message("final", "The complete final answer.")
        running.value = false
        runCurrent()
        assertEquals(1, opens)
        assertEquals(listOf("The complete final answer."), appended)
        assertEquals(1, finishes)
    }

    @Test
    fun cancelledStreamCannotPlayItsLatePcm() = runTest(dispatcher) {
        observe()
        runCurrent()
        val old = callbacks
        VoiceViewModel::class.java.getDeclaredMethod("cancelStandardSpeechStream", String::class.java)
            .apply { isAccessible = true }.invoke(vm, "test cancellation")
        old.onPcm(ByteArray(90_000), 24_000)
        runCurrent()
        verify(exactly = 0) { pcm.write(any(), any(), any()) }
        assertTrue(stops > 0)
        assertFalse(outcome.isCompleted)
    }

    @Test
    fun socketEndWaitsForPendingIoAndPreservesFrameOrder() = runTest(dispatcher) {
        val io = PausedDispatcher()
        vm.standardSpeechWriteDispatcher = io
        observe()
        runCurrent()
        messages.value = listOf(message("answer", "A complete answer."))
        running.value = false
        runCurrent()
        val first = ByteArray(90_000) { 1 }
        val last = ByteArray(48_000) { 2 }
        callbacks.onPcm(first, 24_000)
        callbacks.onPcm(last, 24_000)
        outcome.complete(VoiceSpeechStreamOutcome(VoiceSpeechStreamStatus.Completed, true))
        runCurrent()
        advanceTimeBy(5_000)
        runCurrent()
        verify(exactly = 0) { pcm.flushBufferedPlayback(any()) }
        assertEquals(VoiceState.Speaking, vm.uiState.value.state)
        io.runNext()
        runCurrent()
        verify(exactly = 0) { pcm.flushBufferedPlayback(any()) }
        io.runNext()
        runCurrent()
        verifyOrder {
            pcm.write(first, 24_000, any())
            pcm.write(last, 24_000, any())
            pcm.flushBufferedPlayback(any())
        }
    }

    @Test
    fun replacementStreamRejectsOldSocketAndScheduledIo() = runTest(dispatcher) {
        val io = PausedDispatcher()
        vm.standardSpeechWriteDispatcher = io
        observe()
        runCurrent()
        val old = callbacks
        old.onPcm(ByteArray(90_000), 24_000)
        runCurrent()
        invoke("prepareStandardSpeechStream")
        runCurrent()
        old.onPcm(ByteArray(48_000), 24_000)
        io.runNext()
        runCurrent()
        verify(exactly = 0) { pcm.write(any(), any(), any()) }
        callbacks.onPcm(ByteArray(960), 24_000)
        runCurrent()
        io.runNext()
        runCurrent()
        verify(exactly = 1) { pcm.write(any(), any(), any()) }
    }

    private class PausedDispatcher : CoroutineDispatcher() {
        private val pending = ArrayDeque<Runnable>()
        override fun dispatch(context: CoroutineContext, block: Runnable) { pending.addLast(block) }
        fun runNext() { pending.removeAt(0).run() }
    }

    private fun message(id: String, text: String) = ChatMessage(
        id = id, role = MessageRole.ASSISTANT, content = text, timestamp = 1L, isStreaming = false,
    )
    private fun invoke(name: String) {
        VoiceViewModel::class.java.getDeclaredMethod(name).apply { isAccessible = true }.invoke(vm)
    }
    private fun field(name: String, value: Any) {
        VoiceViewModel::class.java.getDeclaredField(name).apply { isAccessible = true }.set(vm, value)
    }
}
