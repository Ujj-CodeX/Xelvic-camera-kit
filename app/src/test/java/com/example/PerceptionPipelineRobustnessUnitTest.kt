package com.example

import android.content.Context
import android.content.res.AssetManager
import android.graphics.ImageFormat
import android.graphics.Rect
import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import com.example.engine.capture.AnalysisFrame
import com.example.engine.capture.CameraSession
import com.example.engine.capture.ClockOffsetCalibrator
import com.example.engine.capture.FrameClock
import com.example.engine.debug.LatencyProbe
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.perception.MediaPipeFaceGeometryProvider
import com.example.engine.perception.ProviderConfig
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import com.example.engine.signal.SignalPipeline
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import org.junit.After
import org.junit.Assert.*
import org.junit.Test
import java.lang.reflect.Proxy
import java.nio.ByteBuffer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Pure-JVM unit test suite for Perception Pipeline Robustness + Throughput
 * ([MediaPipeFaceGeometryProvider], [CameraSession], [LatencyProbe], [SignalPipeline]):
 *
 * 1. `detectAsync` never runs on MediaPipe's `ResultListener` thread; `handleResult` only publishes
 *    and posts `pumpOnAnalysisThread()` to `analysisExecutor`.
 * 2. Overlapped conversion of Frame $N+1$ into the free ping-pong slot (`slots[inFlightSlotIndex xor 1]`)
 *    while Inference $N$ is in flight, with `framesDroppedCount`, `pendingReplacedCount`,
 *    `pendingWaitMs` propagation (`FaceGeometrySnapshot` -> `FaceState` -> `LatencyProbe` -> HUD & CSV),
 *    and `computeExpectedOverlappedThroughputHz(conv, inf) = 1000 / max(conv, inf)`.
 * 3. Safe Bitmap lifecycle: never calls `Bitmap.recycle()` on a slot bitmap when size or rotation changes.
 * 4. Watchdog grace (5 s first inference vs 500 ms steady state) scheduled via `watchdogScheduler`
 *    posting to `analysisExecutor`, plus 3-resets-in-30s permanent CPU fallback (`"GPU->CPU (watchdog)"`).
 * 5. Non-blocking reaper thread (`reaperExecutor`) for `close()` so a blocking `close()` never
 *    stalls `analysisExecutor`.
 * 6. `Throwable` / `Error` catching across landmarker creation, conversion, and `detectAsync`, plus
 *    `"ERROR: landmarker init failed"` HUD status when both GPU and CPU creation fail.
 * 7. Strictly monotonic `lastDetectTimestampMs` across delegate switches and camera rebinds.
 * 8. Governor tracker cadence limit (`targetTrackerCadenceHz`) enforced between `detectAsync` dispatches.
 * 9. Every `ImageProxy` is closed synchronously and exactly once inside the `submit()` / `analyze()`
 *    `try/finally` scope and never held across calls (`activeHeldImageProxyCount == 0`).
 * 10. Stale result/error callbacks from closed or watchdog-reset landmarkers are ignored by `dispatchId`
 *     and never release the slot or clear `isInferenceInFlight` of a newer dispatch.
 */
class PerceptionPipelineRobustnessUnitTest {

    @After
    fun tearDown() {
        FrameClock.customElapsedRealtimeSource = null
    }

    @Test
    fun testDetectAsyncNeverCalledOnCallbackThread() {
        var fakeClockNs = 1_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val provider = MediaPipeFaceGeometryProvider(createMockContext())
        provider.enforceCadenceLimit = false

        val detectAsyncThreadIds = ArrayList<Long>()
        val detectAsyncThreadNames = ArrayList<String>()
        var pendingCallback1: (() -> Unit)? = null
        var pendingCallback2: (() -> Unit)? = null
        var callIndex = 0

        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    synchronized(detectAsyncThreadIds) {
                        detectAsyncThreadIds.add(Thread.currentThread().id)
                        detectAsyncThreadNames.add(Thread.currentThread().name)
                    }
                    callIndex++
                    if (callIndex == 1) {
                        pendingCallback1 = {
                            resultListener(null, mpImage)
                        }
                    } else {
                        pendingCallback2 = {
                            resultListener(null, mpImage)
                        }
                    }
                }
                override fun close() {
                    isClosed = true
                }
            }
        }

        provider.start(ProviderConfig(analysisLongEdgePx = 320))
        assertTrue(provider.awaitAnalysisIdle())

        // 1. Submit Frame 1 -> dispatched on dispatchExecutor
        val (proxy1, closeCount1) = createCountedYuvProxy(320, 180, 90)
        provider.submit(
            AnalysisFrame(
                imageProxy = proxy1,
                timestampNs = fakeClockNs,
                rotationDegrees = 0,
                isFrontCamera = true
            )
        )
        assertEquals(1, closeCount1.get())
        assertTrue(provider.awaitAnalysisIdle())
        assertNotNull(pendingCallback1)

        // 2. While Frame 1 is in flight, submit Frame 2 -> converted immediately and staged in pendingConvertedFrameRef
        fakeClockNs += 33_000_000L
        val (proxy2, closeCount2) = createCountedYuvProxy(320, 180, 140)
        provider.submit(
            AnalysisFrame(
                imageProxy = proxy2,
                timestampNs = fakeClockNs,
                rotationDegrees = 0,
                isFrontCamera = true
            )
        )
        assertEquals(1, closeCount2.get())
        assertTrue(provider.awaitAnalysisIdle())
        assertEquals("Only Frame 1 should be dispatched so far", 1, detectAsyncThreadIds.size)

        // 3. Invoke Frame 1's resultListener on a simulated MediaPipe native callback thread
        var callbackThreadId = -1L
        val cbLatch = CountDownLatch(1)
        val callbackThread = Thread({
            callbackThreadId = Thread.currentThread().id
            pendingCallback1!!.invoke()
            cbLatch.countDown()
        }, "FakeMediaPipeCallbackThread")
        callbackThread.start()
        assertTrue(cbLatch.await(2, TimeUnit.SECONDS))

        // Wait for dispatchExecutor to pump and dispatch Frame 2
        assertTrue(provider.awaitAnalysisIdle())
        assertNotNull("Frame 2 must be dispatched on dispatchExecutor after Frame 1 finishes", pendingCallback2)

        synchronized(detectAsyncThreadIds) {
            assertEquals(2, detectAsyncThreadIds.size)
            for (i in detectAsyncThreadIds.indices) {
                assertNotEquals(
                    "detectAsync (call #$i) must NEVER run on MediaPipe's ResultListener thread",
                    callbackThreadId,
                    detectAsyncThreadIds[i]
                )
                assertEquals("XelvicDispatchThread", detectAsyncThreadNames[i])
            }
        }
        assertEquals("FakeMediaPipeCallbackThread", provider.lastResultCallbackThreadName)
        assertNotEquals(provider.lastResultCallbackThreadId, provider.lastDetectAsyncThreadId)

        pendingCallback2!!.invoke()
        provider.awaitAnalysisIdle()
        provider.stop()
    }

    @Test
    fun testOverlappedConversionOfFrameNPlus1StartsBeforeInferenceNEndsAndPendingWaitMsAndCounters() {
        var fakeClockNs = 5_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val provider = MediaPipeFaceGeometryProvider(createMockContext())
        provider.enforceCadenceLimit = false
        val pipeline = SignalPipeline()
        val probe = LatencyProbe(
            windowDurationNs = 5_000_000_000L,
            hudRefreshIntervalNs = 100_000_000L
        )

        var pendingCallback: (() -> Unit)? = null
        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    pendingCallback = {
                        resultListener(null, mpImage)
                    }
                }
                override fun close() {
                    isClosed = true
                }
            }
        }

        val snapshots = ArrayList<FaceGeometrySnapshot>()
        provider.setListener { snap ->
            snapshots.add(snap)
            probe.recordTrackerSnapshot(snap)
            pipeline.processSnapshot(snap)
        }

        provider.start(ProviderConfig(analysisLongEdgePx = 320))
        assertTrue(provider.awaitAnalysisIdle())

        // 1. Submit Frame 1 at t = 5.000s (occupies Slot 0)
        val (proxy1, close1) = createCountedYuvProxy(320, 180, yFill = 60)
        provider.submit(
            AnalysisFrame(
                imageProxy = proxy1,
                timestampNs = fakeClockNs,
                rotationDegrees = 0,
                isFrontCamera = true,
                submittedAtNs = fakeClockNs - 1_500_000L
            )
        )
        assertEquals(1, close1.get())
        assertTrue(provider.awaitAnalysisIdle())
        assertEquals(0, provider.inFlightSlotIndex)
        assertTrue(provider.slots[0].inUseByInference.get())
        val slot0PixelDuringFlight = provider.slots[0].lastUnpackedPixelArgb

        // 2. While Frame 1 is still in flight on Slot 0, submit Frame 2 at t = 5.010s and Frame 3 at t = 5.020s
        fakeClockNs += 10_000_000L
        val (proxy2, close2) = createCountedYuvProxy(320, 180, yFill = 150)
        provider.submit(
            AnalysisFrame(
                imageProxy = proxy2,
                timestampNs = fakeClockNs,
                rotationDegrees = 0,
                isFrontCamera = true,
                submittedAtNs = fakeClockNs - 1_000_000L
            )
        )
        assertEquals(1, close2.get())
        assertEquals(
            "Frame 2 must convert in overlap into Slot 1 while Slot 0 is in flight",
            1,
            provider.overlappedConversionCount
        )
        assertEquals(
            "In-flight Slot 0 unpacked pixels must NEVER be mutated while Frame 1 is in flight",
            slot0PixelDuringFlight,
            provider.slots[0].lastUnpackedPixelArgb
        )

        // Submit Frame 3 at t = 5.020s while Frame 1 is STILL in flight -> converts into FREE Slot 2 and replaces STAGED Slot 1
        fakeClockNs += 10_000_000L
        val frame3ConversionDoneNs = fakeClockNs
        val (proxy3, close3) = createCountedYuvProxy(320, 180, yFill = 220)
        provider.submit(
            AnalysisFrame(
                imageProxy = proxy3,
                timestampNs = fakeClockNs,
                rotationDegrees = 0,
                isFrontCamera = true,
                submittedAtNs = fakeClockNs - 2_000_000L
            )
        )
        assertEquals(1, close3.get())
        assertEquals(2, provider.overlappedConversionCount)
        assertEquals("Frame 3 must replace pending Frame 2", 1, provider.pendingReplacedCount)
        assertEquals("Dropped frame counter must increment when pending Frame 2 is replaced", 1, provider.framesDroppedCount)
        assertEquals(slot0PixelDuringFlight, provider.slots[0].lastUnpackedPixelArgb)

        // 3. Advance clock by 9.5ms (so Frame 3 waits 9.5ms in stagedPreparedFrame) and finish Frame 1
        fakeClockNs = frame3ConversionDoneNs + 9_500_000L
        pendingCallback!!.invoke()
        assertTrue(provider.awaitAnalysisIdle())

        // Frame 3 is now in flight on Slot 2 (since Slot 0 was IN_FLIGHT and Slot 1 was STAGED when Frame 3 converted into FREE Slot 2)!
        assertEquals(2, provider.inFlightSlotIndex)
        assertFalse(provider.slots[0].inUseByInference.get())
        assertFalse(provider.slots[1].inUseByInference.get())
        assertTrue(provider.slots[2].inUseByInference.get())

        // 4. Advance clock by 12.0ms for Frame 3's inference and complete Frame 3
        fakeClockNs += 12_000_000L
        pendingCallback!!.invoke()
        assertTrue(provider.awaitAnalysisIdle())

        assertEquals(2, snapshots.size)
        val snap3 = snapshots[1]
        assertEquals("pendingWaitMs must equal time from conversionDone to detectAsync dispatch", 9.5f, snap3.pendingWaitMs, 0.2f)
        assertEquals("inferenceMs must equal time from detectAsync dispatch to callback", 12.0f, snap3.inferenceMs, 0.2f)
        assertEquals(2.0f, snap3.queueMs, 0.1f)

        // Verify FaceState, LatencyProbe, HUD line, and CSV include pendingWaitMs
        val currentFaceState = pipeline.getLatestState()
        assertEquals(9.5f, currentFaceState.pendingWaitMs, 0.2f)
        assertEquals(9.5f, probe.lastPendingWaitMs, 0.2f)

        probe.recordRenderFrame(
            renderTimestampNs = fakeClockNs + 4_000_000L,
            faceState = FaceState(
                timestampNs = fakeClockNs,
                poseTimestampNs = snap3.timestampNs,
                lifecycle = FaceLifecycle.State.TRACKING,
                isVisible = true,
                queueMs = snap3.queueMs,
                conversionMs = snap3.conversionMs,
                pendingWaitMs = snap3.pendingWaitMs,
                inferenceMs = snap3.inferenceMs
            )
        )
        probe.refreshMetrics(fakeClockNs + 600_000_000L)
        assertTrue(
            "HUD cached line must include wait: X.Xms, was: ${probe.cachedHudLine}",
            probe.cachedHudLine.contains("wait:")
        )

        val csv = probe.dumpCsv()
        assertTrue(
            "CSV header must include pendingWaitMs",
            csv.startsWith("timestamp,ageMs,conversionMs,pendingWaitMs,inferenceMs,queueMs,")
        )
        assertTrue("CSV body must contain recorded row", probe.csvRowCount() >= 1)

        // Verify expected throughput formula: 1000 / max(conv, inf) instead of 1000 / (conv + inf)
        val overlappedHz = MediaPipeFaceGeometryProvider.computeExpectedOverlappedThroughputHz(8.0f, 16.0f)
        assertEquals(62.5f, overlappedHz, 0.01f)
        provider.stop()
    }

    @Test
    fun testNoBitmapRecycleDuringInFlightOnSizeOrRotationChange() {
        val provider = MediaPipeFaceGeometryProvider(createMockContext())
        provider.enforceCadenceLimit = false

        var pendingCallback: (() -> Unit)? = null
        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    pendingCallback = {
                        resultListener(null, mpImage)
                    }
                }
                override fun close() {
                    isClosed = true
                }
            }
        }
        provider.start(ProviderConfig(analysisLongEdgePx = 320))
        assertTrue(provider.awaitAnalysisIdle())

        // 1. Frame 1 at 640x360, rot 90, longEdge 320 -> target 180x320 in Slot 0
        val (p1, _) = createCountedYuvProxy(640, 360, 100)
        provider.submit(AnalysisFrame(imageProxy = p1, timestampNs = 1_000_000_000L, rotationDegrees = 90, isFrontCamera = true))
        assertTrue(provider.awaitAnalysisIdle())
        assertEquals(0, provider.inFlightSlotIndex)
        assertEquals(180, provider.slots[0].targetWidth)
        assertEquals(320, provider.slots[0].targetHeight)

        // 2. While Slot 0 is in flight, change analysis size to 256 and rotation to 270 -> converts into Slot 1
        provider.updateAnalysisSize(256)
        val (p2, _) = createCountedYuvProxy(640, 360, 120)
        provider.submit(AnalysisFrame(imageProxy = p2, timestampNs = 1_050_000_000L, rotationDegrees = 270, isFrontCamera = true))
        assertEquals(144, provider.slots[1].targetWidth)
        assertEquals(256, provider.slots[1].targetHeight)

        // Complete Frame 1 -> Frame 2 is now in flight on Slot 1
        pendingCallback!!.invoke()
        assertTrue(provider.awaitAnalysisIdle())
        assertEquals(1, provider.inFlightSlotIndex)

        // 3. While Slot 1 is in flight, change analysis size to 224 and source resolution to 384x216 -> converts into Slot 2
        provider.updateAnalysisSize(224)
        val (p3, _) = createCountedYuvProxy(384, 216, 140)
        provider.submit(AnalysisFrame(imageProxy = p3, timestampNs = 1_100_000_000L, rotationDegrees = 90, isFrontCamera = true))
        assertEquals(126, provider.slots[2].targetWidth)
        assertEquals(224, provider.slots[2].targetHeight)

        // Complete Frame 2 -> Frame 3 is now in flight on Slot 2; submit Frame 4 at longEdge 160 to reallocate Slot 0 without recycling
        pendingCallback!!.invoke()
        assertTrue(provider.awaitAnalysisIdle())
        assertEquals(2, provider.inFlightSlotIndex)

        provider.updateAnalysisSize(160)
        val (p4, _) = createCountedYuvProxy(384, 216, 160)
        provider.submit(AnalysisFrame(imageProxy = p4, timestampNs = 1_150_000_000L, rotationDegrees = 90, isFrontCamera = true))
        assertEquals(90, provider.slots[0].targetWidth)
        assertEquals(160, provider.slots[0].targetHeight)

        assertEquals("Bitmap.recycle() must NEVER be called on Slot 0", 0, provider.slots[0].recycleCallCount)
        assertEquals("Bitmap.recycle() must NEVER be called on Slot 1", 0, provider.slots[1].recycleCallCount)
        assertEquals("Bitmap.recycle() must NEVER be called on Slot 2", 0, provider.slots[2].recycleCallCount)
        provider.slots[0].sourceBitmap?.let { assertFalse(it.isRecycled) }
        provider.slots[0].targetBitmap?.let { assertFalse(it.isRecycled) }
        provider.slots[1].sourceBitmap?.let { assertFalse(it.isRecycled) }
        provider.slots[1].targetBitmap?.let { assertFalse(it.isRecycled) }
        provider.slots[2].sourceBitmap?.let { assertFalse(it.isRecycled) }
        provider.slots[2].targetBitmap?.let { assertFalse(it.isRecycled) }

        pendingCallback!!.invoke()
        provider.awaitAnalysisIdle()
        provider.stop()
    }

    @Test
    fun testWatchdogFirstInference5sGraceAndSteadyState500msRecoveryAnd3StrikeCpuFallback() {
        var fakeClockNs = 10_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val provider = MediaPipeFaceGeometryProvider(createMockContext())
        provider.enforceCadenceLimit = false
        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        val pipeline = SignalPipeline()
        val calibrator = ClockOffsetCalibrator(warmupFrames = 5)
        val probe = LatencyProbe()
        val session = CameraSession(provider, pipeline, gov, calibrator, probe)

        val createdHandles = ArrayList<MediaPipeFaceGeometryProvider.LandmarkerHandle>()
        val createdDelegates = ArrayList<Delegate>()
        var latestCallback: (() -> Unit)? = null

        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            createdDelegates.add(delegate)
            val handle = object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    latestCallback = {
                        resultListener(null, mpImage)
                    }
                }
                override fun close() {
                    isClosed = true
                }
            }
            createdHandles.add(handle)
            handle
        }

        // Start with GPU delegate
        session.startProvider(ProviderConfig(useGpu = true))
        assertTrue(provider.awaitAnalysisIdle())
        assertEquals("GPU", provider.activeDelegateLabel)
        assertEquals(1, createdHandles.size)
        val gpuHandle1 = createdHandles[0]

        // 1. Submit first GPU inference at t = 10.0s
        val (p1, _) = createCountedYuvProxy(320, 180, 100)
        provider.submit(AnalysisFrame(imageProxy = p1, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        assertTrue(provider.awaitAnalysisIdle())
        assertEquals(0, provider.inFlightSlotIndex)

        // Advance clock by 2,000 ms (2.0 s): exceeds 500 ms steady-state timeout, but is within 5,000 ms first-inference warm-up grace!
        fakeClockNs += 2_000_000_000L
        provider.triggerWatchdogCheck(fakeClockNs)
        assertEquals(
            "Slow 2s first GPU inference (shader warm-up) must NOT trigger a watchdog reset (5s grace)",
            0,
            provider.watchdogResetCount
        )
        assertFalse("gpuHandle1 must remain open during 5s first-inference grace", gpuHandle1.isClosed)
        assertEquals(0, provider.inFlightSlotIndex)

        // Complete the first inference at t = 12.0s -> gpuHandle1 now transitions to 500ms steady-state timeout
        latestCallback!!.invoke()
        assertTrue(provider.awaitAnalysisIdle())
        assertEquals(-1, provider.inFlightSlotIndex)

        // 2. Submit second GPU inference on gpuHandle1 at t = 12.1s and wedge it for 600 ms (> 500 ms steady-state timeout)
        fakeClockNs += 100_000_000L
        val (p2, _) = createCountedYuvProxy(320, 180, 110)
        provider.submit(AnalysisFrame(imageProxy = p2, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        assertTrue(provider.awaitAnalysisIdle())
        assertEquals(1, provider.inFlightSlotIndex)

        fakeClockNs += 600_000_000L // 600 ms > 500 ms steady-state timeout -> Reset #1
        provider.triggerWatchdogCheck(fakeClockNs)
        provider.awaitReaperIdle()

        assertEquals("Steady-state inference > 500ms must trigger watchdog reset #1", 1, provider.watchdogResetCount)
        assertTrue("Wedged gpuHandle1 must be closed on reaperExecutor", gpuHandle1.isClosed)
        assertEquals(MediaPipeFaceGeometryProvider.STATUS_TRACKER_RESET, provider.statusMessage)
        assertEquals(MediaPipeFaceGeometryProvider.STATUS_TRACKER_RESET, probe.trackerStatusLabel)
        assertEquals("GPU", provider.activeDelegateLabel)
        assertEquals(2, createdHandles.size)
        val gpuHandle2 = createdHandles[1]

        // 3. Submit inference on gpuHandle2 (newly recreated -> 5s first-inference timeout) and wedge it for 5.1s -> Reset #2
        fakeClockNs += 100_000_000L
        val (p3, _) = createCountedYuvProxy(320, 180, 120)
        provider.submit(AnalysisFrame(imageProxy = p3, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        assertTrue(provider.awaitAnalysisIdle())
        fakeClockNs += 5_100_000_000L // 5.1s -> Reset #2 within 30s window
        provider.triggerWatchdogCheck(fakeClockNs)
        provider.awaitReaperIdle()

        assertEquals(2, provider.watchdogResetCount)
        assertFalse("2 resets in 30s should not yet force permanent CPU fallback", provider.isWatchdogForcedCpuForSession)
        assertTrue(gpuHandle2.isClosed)
        assertEquals(3, createdHandles.size)
        val gpuHandle3 = createdHandles[2]

        // 4. Submit inference on gpuHandle3 and wedge it for 5.1s -> Reset #3 within 30s -> permanent CPU fallback!
        fakeClockNs += 100_000_000L
        val (p4, _) = createCountedYuvProxy(320, 180, 130)
        provider.submit(AnalysisFrame(imageProxy = p4, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        assertTrue(provider.awaitAnalysisIdle())
        fakeClockNs += 5_100_000_000L // Reset #3 within 30s window!
        provider.triggerWatchdogCheck(fakeClockNs)
        provider.awaitReaperIdle()

        assertEquals(3, provider.watchdogResetCount)
        assertTrue("3 watchdog resets within 30s must permanently switch session to CPU", provider.isWatchdogForcedCpuForSession)
        assertEquals(MediaPipeFaceGeometryProvider.STATUS_GPU_TO_CPU_WATCHDOG, provider.activeDelegateLabel)
        assertEquals(MediaPipeFaceGeometryProvider.STATUS_GPU_TO_CPU_WATCHDOG, provider.statusMessage)
        assertEquals(MediaPipeFaceGeometryProvider.STATUS_GPU_TO_CPU_WATCHDOG, probe.delegateLabel)
        assertEquals(4, createdHandles.size)
        assertEquals("4th landmarker after 3-strike watchdog must use Delegate.CPU", Delegate.CPU, createdDelegates[3])

        // Attempting to switch back to GPU during the same session must stay on CPU ("GPU->CPU (watchdog)")
        session.setUseGpuDelegate(true)
        provider.awaitAnalysisIdle()
        assertEquals(MediaPipeFaceGeometryProvider.STATUS_GPU_TO_CPU_WATCHDOG, provider.activeDelegateLabel)
        assertEquals(4, createdHandles.size)
        provider.stop()
    }

    @Test
    fun testReaperThreadClosesLandmarkerWithoutBlockingAnalysisThread() {
        var fakeClockNs = 20_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val provider = MediaPipeFaceGeometryProvider(createMockContext())
        provider.enforceCadenceLimit = false

        val closeEnteredLatch = CountDownLatch(1)
        val allowCloseToFinishLatch = CountDownLatch(1)
        val handle1Closed = AtomicBoolean(false)
        val handle2Dispatches = AtomicInteger(0)

        var handleCount = 0
        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            handleCount++
            val idx = handleCount
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    if (idx >= 2) {
                        handle2Dispatches.incrementAndGet()
                    }
                    resultListener(null, mpImage)
                }
                override fun close() {
                    if (idx == 1) {
                        closeEnteredLatch.countDown()
                        // Simulate a wedged GPU landmarker whose close() blocks for seconds
                        allowCloseToFinishLatch.await(5, TimeUnit.SECONDS)
                        isClosed = true
                        handle1Closed.set(true)
                    } else {
                        isClosed = true
                    }
                }
            }
        }

        provider.start(ProviderConfig(useGpu = false))
        assertTrue(provider.awaitAnalysisIdle())
        assertEquals(1, handleCount)

        // Switch delegate so handle #1 is retired onto reaperExecutor
        provider.switchDelegate(true)
        assertTrue("handle #1 close() must start on reaperExecutor", closeEnteredLatch.await(2, TimeUnit.SECONDS))
        assertTrue("New landmarker must be ready on analysisExecutor while reaper is blocked", provider.awaitAnalysisIdle(500))
        assertFalse("handle #1 close() is still blocked on reaperExecutor", handle1Closed.get())

        // Submit 3 frames while handle #1's close() is blocked on reaperExecutor:
        // analysisExecutor must process all 3 frames without blocking!
        for (i in 1..3) {
            fakeClockNs += 33_000_000L
            val (proxy, closeCount) = createCountedYuvProxy(320, 180, 90 + i)
            provider.submit(AnalysisFrame(imageProxy = proxy, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
            assertEquals(1, closeCount.get())
            assertTrue(provider.awaitAnalysisIdle(500))
        }
        assertEquals(
            "All 3 frames must be dispatched and completed while old landmarker close() is blocked on reaper thread",
            3,
            handle2Dispatches.get()
        )

        // Now release handle #1's blocking close() on the reaper thread
        allowCloseToFinishLatch.countDown()
        assertTrue(provider.awaitReaperIdle(2000))
        assertTrue(handle1Closed.get())
        assertEquals("XelvicLandmarkerReaper", provider.lastReaperCloseThreadName)
        provider.stop()
    }

    @Test
    fun testThrowableAndErrorClearInFlightFlagAndBothDelegatesFailHudError() {
        var fakeClockNs = 30_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val provider = MediaPipeFaceGeometryProvider(createMockContext())
        provider.enforceCadenceLimit = false
        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        val pipeline = SignalPipeline()
        val calibrator = ClockOffsetCalibrator(warmupFrames = 5)
        val probe = LatencyProbe()
        val session = CameraSession(provider, pipeline, gov, calibrator, probe)

        var throwErrorInDetect = true
        var detectSuccessCount = 0

        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    if (throwErrorInDetect) {
                        // Throw a java.lang.Error (not an Exception)
                        throw AssertionError("Simulated native/GPU Error in detectAsync")
                    }
                    detectSuccessCount++
                    resultListener(null, mpImage)
                }
                override fun close() {
                    isClosed = true
                }
            }
        }

        session.startProvider(ProviderConfig(useGpu = false))
        assertTrue(provider.awaitAnalysisIdle())

        // 1. Frame 1 throws AssertionError inside detectAsync -> must catch Throwable, release slot, and clear in-flight flag
        val (p1, c1) = createCountedYuvProxy(320, 180, 100)
        provider.submit(AnalysisFrame(imageProxy = p1, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        assertEquals(1, c1.get())
        assertTrue(provider.awaitAnalysisIdle())
        assertEquals("inFlightSlotIndex must be cleared when detectAsync throws Error", -1, provider.inFlightSlotIndex)
        assertFalse(provider.slots[0].inUseByInference.get())

        // 2. Frame 2 must succeed without being stuck behind a wedged in-flight flag
        throwErrorInDetect = false
        fakeClockNs += 33_000_000L
        val (p2, c2) = createCountedYuvProxy(320, 180, 110)
        provider.submit(AnalysisFrame(imageProxy = p2, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        assertEquals(1, c2.get())
        assertTrue(provider.awaitAnalysisIdle())
        assertEquals(1, detectSuccessCount)

        // 3. Both GPU and CPU creation fail with UnsatisfiedLinkError -> HUD shows "ERROR: landmarker init failed"
        provider.customLandmarkerFactory = { _, _, _, _ ->
            throw UnsatisfiedLinkError("Simulated native JNI failure on both GPU and CPU")
        }
        session.setUseGpuDelegate(true)
        assertTrue(provider.awaitAnalysisIdle())

        assertTrue(provider.hasInitError)
        assertEquals(MediaPipeFaceGeometryProvider.STATUS_INIT_ERROR, provider.activeDelegateLabel)
        assertEquals(MediaPipeFaceGeometryProvider.STATUS_INIT_ERROR, provider.statusMessage)
        assertEquals(MediaPipeFaceGeometryProvider.STATUS_INIT_ERROR, probe.delegateLabel)
        probe.refreshMetrics(fakeClockNs + 1_000_000_000L)
        assertTrue(
            "HUD line must display ERROR: landmarker init failed, was: ${probe.cachedHudLine}",
            probe.cachedHudLine.contains(MediaPipeFaceGeometryProvider.STATUS_INIT_ERROR)
        )
        provider.stop()
    }

    @Test
    fun testMonotonicTimestampsAcrossDelegateSwitchesAndRebinds() {
        val provider = MediaPipeFaceGeometryProvider(createMockContext())
        provider.enforceCadenceLimit = false
        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        val pipeline = SignalPipeline()
        val calibrator = ClockOffsetCalibrator(warmupFrames = 5)
        val probe = LatencyProbe()
        val session = CameraSession(provider, pipeline, gov, calibrator, probe)

        val recordedTimestampsMs = ArrayList<Long>()
        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    synchronized(recordedTimestampsMs) {
                        recordedTimestampsMs.add(timestampMs)
                    }
                    resultListener(null, mpImage)
                }
                override fun close() {
                    isClosed = true
                }
            }
        }

        session.startProvider(ProviderConfig(useGpu = false))
        assertTrue(provider.awaitAnalysisIdle())

        // Submit frame with timestamp = 5000 ms
        provider.submit(AnalysisFrame(imageProxy = createCountedYuvProxy(320, 180, 90).first, timestampNs = 5_000_000_000L, rotationDegrees = 0, isFrontCamera = true))
        provider.awaitAnalysisIdle()

        // Switch delegate to GPU and submit frame with equal/earlier timestamp (4000 ms)
        session.setUseGpuDelegate(true)
        provider.awaitAnalysisIdle()
        provider.submit(AnalysisFrame(imageProxy = createCountedYuvProxy(320, 180, 95).first, timestampNs = 4_000_000_000L, rotationDegrees = 0, isFrontCamera = true))
        provider.awaitAnalysisIdle()

        // Simulate analysis rebind and submit frame with timestamp = 1000 ms
        session.onAnalysisRebound("RGBA_8888", "SMALL")
        provider.submit(AnalysisFrame(imageProxy = createCountedYuvProxy(320, 180, 100).first, timestampNs = 1_000_000_000L, rotationDegrees = 0, isFrontCamera = true))
        provider.awaitAnalysisIdle()

        // Switch delegate back to CPU and submit frame with timestamp = 5002 ms
        session.setUseGpuDelegate(false)
        provider.awaitAnalysisIdle()
        provider.submit(AnalysisFrame(imageProxy = createCountedYuvProxy(320, 180, 105).first, timestampNs = 5_002_000_000L, rotationDegrees = 0, isFrontCamera = true))
        provider.awaitAnalysisIdle()

        synchronized(recordedTimestampsMs) {
            assertEquals(listOf(5000L, 5001L, 5002L, 5003L), recordedTimestampsMs)
            for (i in 1 until recordedTimestampsMs.size) {
                assertTrue(
                    "Timestamps must be strictly monotonic across delegate switches and rebinds",
                    recordedTimestampsMs[i] > recordedTimestampsMs[i - 1]
                )
            }
        }
        provider.stop()
    }

    @Test
    fun testGovernorTrackerCadenceLimitEnforcedBetweenDetectAsyncDispatches() {
        var fakeClockNs = 1_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val provider = MediaPipeFaceGeometryProvider(createMockContext())
        provider.enforceCadenceLimit = true
        provider.updateTargetCadenceHz(15) // 15 Hz -> ~56.7 ms minimum interval (0.85 * 66.6ms) between detectAsync dispatches

        val dispatchTimesMs = ArrayList<Long>()
        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    dispatchTimesMs.add(timestampMs)
                    resultListener(null, mpImage)
                }
                override fun close() {
                    isClosed = true
                }
            }
        }

        provider.start(ProviderConfig(useGpu = false))
        assertTrue(provider.awaitAnalysisIdle())

        // 1. Frame 1 at t = 1000 ms -> dispatched immediately
        val (p1, c1) = createCountedYuvProxy(320, 180, 100)
        provider.submit(AnalysisFrame(imageProxy = p1, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        provider.awaitAnalysisIdle()
        assertEquals(1, c1.get())
        assertEquals(1, dispatchTimesMs.size)

        // 2. Frame 2 at t = 1020 ms (only 20 ms later, < 56.7 ms cadence interval):
        // ImageProxy is converted and closed immediately, but detectAsync is throttled!
        fakeClockNs += 20_000_000L
        val (p2, c2) = createCountedYuvProxy(320, 180, 120)
        provider.submit(AnalysisFrame(imageProxy = p2, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        provider.awaitAnalysisIdle()
        assertEquals("ImageProxy must still be closed synchronously even when cadence-throttled", 1, c2.get())
        assertEquals("detectAsync must NOT dispatch after only 20ms when cadence is 15 Hz", 1, dispatchTimesMs.size)

        // 3. Frame 3 at t = 1070 ms (70 ms after Frame 1, >= 56.7 ms interval) -> dispatched!
        fakeClockNs += 50_000_000L
        val (p3, c3) = createCountedYuvProxy(320, 180, 140)
        provider.submit(AnalysisFrame(imageProxy = p3, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        provider.awaitAnalysisIdle()
        assertEquals(1, c3.get())
        assertEquals("Frame 3 at +70ms must be dispatched under 15 Hz cadence", 2, dispatchTimesMs.size)
        provider.stop()
    }

    @Test
    fun testEveryImageProxyClosedSynchronouslyOnceInSubmitAndNeverHeldAcrossCalls() {
        var fakeClockNs = 10_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val provider = MediaPipeFaceGeometryProvider(createMockContext())
        provider.enforceCadenceLimit = false

        var pendingCb: (() -> Unit)? = null
        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    pendingCb = { resultListener(null, mpImage) }
                }
                override fun close() {
                    isClosed = true
                }
            }
        }
        provider.start(ProviderConfig(useGpu = false))
        assertTrue(provider.awaitAnalysisIdle())

        // Case 1: Normal in-flight submission
        val (p1, c1) = createCountedYuvProxy(320, 180, 80)
        provider.submit(AnalysisFrame(imageProxy = p1, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        assertEquals("ImageProxy 1 must be closed exactly once before submit() returns", 1, c1.get())
        assertEquals("Zero ImageProxies may be held after submit() returns", 0, provider.activeHeldImageProxyCount)
        assertTrue(provider.awaitAnalysisIdle())

        // Case 2: Overlapped submission while Frame 1 is still in flight
        fakeClockNs += 33_000_000L
        val (p2, c2) = createCountedYuvProxy(320, 180, 90)
        provider.submit(AnalysisFrame(imageProxy = p2, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        assertEquals("ImageProxy 2 must be closed synchronously inside submit() even while Frame 1 is in flight", 1, c2.get())
        assertEquals(0, provider.activeHeldImageProxyCount)

        // Case 3: Submission while paused
        provider.setPaused(true)
        fakeClockNs += 33_000_000L
        val (p3, c3) = createCountedYuvProxy(320, 180, 100)
        provider.submit(AnalysisFrame(imageProxy = p3, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        assertEquals("ImageProxy 3 must be closed synchronously when provider is paused", 1, c3.get())
        assertEquals(0, provider.activeHeldImageProxyCount)
        provider.setPaused(false)

        // Case 4: Submission where ImageProxy.getPlanes() throws OutOfMemoryError (Throwable)
        val closeCountBroken = AtomicInteger(0)
        val brokenProxy = Proxy.newProxyInstance(
            ImageProxy::class.java.classLoader,
            arrayOf(ImageProxy::class.java)
        ) { _, method, _ ->
            when (method.name) {
                "getWidth" -> 320
                "getHeight" -> 180
                "getPlanes" -> throw OutOfMemoryError("Simulated OOM in getPlanes")
                "close" -> {
                    closeCountBroken.incrementAndGet()
                    Unit
                }
                else -> null
            }
        } as ImageProxy

        fakeClockNs += 33_000_000L
        provider.submit(AnalysisFrame(imageProxy = brokenProxy, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        assertEquals("Broken ImageProxy must still be closed exactly once in try/finally", 1, closeCountBroken.get())
        assertEquals(0, provider.activeHeldImageProxyCount)
        assertEquals(4, provider.totalImageProxyClosedCount)

        pendingCb?.invoke()
        provider.awaitAnalysisIdle()
        provider.stop()
    }

    @Test
    fun testStaleCallbacksIgnoredByDispatchIdAndNeverReleaseNewerSlotOrFlag() {
        var fakeClockNs = 50_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val provider = MediaPipeFaceGeometryProvider(createMockContext())
        provider.enforceCadenceLimit = false

        var handle1ResultCb: (() -> Unit)? = null
        var handle1ErrorCb: (() -> Unit)? = null
        var handle2ResultCb: (() -> Unit)? = null
        var createdCount = 0

        provider.customLandmarkerFactory = { _, delegate, resultListener, errorListener ->
            createdCount++
            val idx = createdCount
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    if (idx == 1) {
                        handle1ResultCb = { resultListener(null, mpImage) }
                        handle1ErrorCb = { errorListener(RuntimeException("Late error from wedged handle 1")) }
                    } else {
                        handle2ResultCb = { resultListener(null, mpImage) }
                    }
                }
                override fun close() {
                    isClosed = true
                }
            }
        }

        val receivedSnapshots = ArrayList<FaceGeometrySnapshot>()
        provider.setListener { snap -> receivedSnapshots.add(snap) }

        provider.start(ProviderConfig(useGpu = true))
        assertTrue(provider.awaitAnalysisIdle())

        // 1. Dispatch Frame 1 on handle #1 (dispatchId = 1, slot = 0)
        val (p1, _) = createCountedYuvProxy(320, 180, 100)
        provider.submit(AnalysisFrame(imageProxy = p1, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        assertTrue(provider.awaitAnalysisIdle())
        assertEquals(1L, provider.activeInFlightDispatchId)
        assertEquals(0, provider.inFlightSlotIndex)
        assertNotNull(handle1ResultCb)

        // 2. Wedge handle #1 past the 5s first-inference watchdog timeout -> watchdog resets and creates handle #2
        fakeClockNs += 5_500_000_000L
        provider.triggerWatchdogCheck(fakeClockNs)
        provider.awaitReaperIdle()
        assertEquals(1, provider.watchdogResetCount)
        assertEquals(2, createdCount)

        // 3. Submit Frame 2 on handle #2 -> now in flight with dispatchId = 2 in Slot 1
        fakeClockNs += 33_000_000L
        val (p2, _) = createCountedYuvProxy(320, 180, 150)
        provider.submit(AnalysisFrame(imageProxy = p2, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        assertTrue(provider.awaitAnalysisIdle())
        assertEquals(2L, provider.activeInFlightDispatchId)
        assertEquals(1, provider.inFlightSlotIndex)
        assertTrue(provider.slots[1].inUseByInference.get())
        assertNotNull(handle2ResultCb)

        // 4. Fire late result callback AND late error callback from retired handle #1 (dispatchId = 1):
        // Both MUST be ignored and must NEVER clear activeInFlightDispatchId or release Slot 1!
        handle1ResultCb!!.invoke()
        handle1ErrorCb!!.invoke()

        assertEquals("Both stale callbacks (result + error) must be ignored", 2, provider.staleCallbackIgnoredCount)
        assertEquals("No snapshot may be emitted for stale callback", 0, receivedSnapshots.size)
        assertEquals("Newer dispatchId (2) must remain active", 2L, provider.activeInFlightDispatchId)
        assertEquals("Newer inFlightSlotIndex (1) must remain in flight", 1, provider.inFlightSlotIndex)
        assertTrue("Slot 1 must still be marked inUseByInference", provider.slots[1].inUseByInference.get())

        // 5. Now fire valid callback from handle #2 (dispatchId = 2) -> succeeds and releases Slot 1
        fakeClockNs += 10_000_000L
        handle2ResultCb!!.invoke()
        provider.awaitAnalysisIdle()

        assertEquals(1, receivedSnapshots.size)
        assertEquals(-1, provider.inFlightSlotIndex)
        assertFalse(provider.slots[1].inUseByInference.get())
        provider.stop()
    }

    private fun createMockContext(): Context {
        return object : android.content.ContextWrapper(null) {
            override fun getAssets(): AssetManager? = null
        }
    }

    private fun createCountedYuvProxy(width: Int, height: Int, yFill: Int): Pair<ImageProxy, AtomicInteger> {
        val closeCount = AtomicInteger(0)
        val yBuf = ByteBuffer.allocateDirect(width * height)
        val uBuf = ByteBuffer.allocateDirect((width / 2) * (height / 2))
        val vBuf = ByteBuffer.allocateDirect((width / 2) * (height / 2))
        for (i in 0 until yBuf.capacity()) yBuf.put(i, yFill.toByte())
        for (i in 0 until uBuf.capacity()) uBuf.put(i, 128.toByte())
        for (i in 0 until vBuf.capacity()) vBuf.put(i, 128.toByte())

        val cropRect = Rect().apply {
            left = 0
            top = 0
            right = width
            bottom = height
        }

        val planes = arrayOf(
            createMockPlaneProxy(yBuf, width, 1),
            createMockPlaneProxy(uBuf, width / 2, 1),
            createMockPlaneProxy(vBuf, width / 2, 1)
        )
        val imageInfo = Proxy.newProxyInstance(
            ImageInfo::class.java.classLoader,
            arrayOf(ImageInfo::class.java)
        ) { _, method, _ ->
            when (method.name) {
                "getTimestamp" -> 1_000_000_000L
                "getRotationDegrees" -> 0
                else -> null
            }
        } as ImageInfo

        val proxy = Proxy.newProxyInstance(
            ImageProxy::class.java.classLoader,
            arrayOf(ImageProxy::class.java)
        ) { _, method, _ ->
            when (method.name) {
                "getWidth" -> width
                "getHeight" -> height
                "getFormat" -> ImageFormat.YUV_420_888
                "getCropRect" -> cropRect
                "getPlanes" -> planes
                "getImageInfo" -> imageInfo
                "close" -> {
                    closeCount.incrementAndGet()
                    Unit
                }
                else -> null
            }
        } as ImageProxy

        return proxy to closeCount
    }

    private fun createMockPlaneProxy(
        buffer: ByteBuffer,
        rowStride: Int,
        pixelStride: Int
    ): ImageProxy.PlaneProxy {
        return Proxy.newProxyInstance(
            ImageProxy.PlaneProxy::class.java.classLoader,
            arrayOf(ImageProxy.PlaneProxy::class.java)
        ) { _, method, _ ->
            when (method.name) {
                "getBuffer" -> buffer
                "getRowStride" -> rowStride
                "getPixelStride" -> pixelStride
                else -> 0
            }
        } as ImageProxy.PlaneProxy
    }
}
