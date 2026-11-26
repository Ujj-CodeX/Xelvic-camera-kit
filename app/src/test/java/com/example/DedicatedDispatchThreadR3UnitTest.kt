package com.example

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import com.example.engine.capture.AnalysisFrame
import com.example.engine.capture.CameraSession
import com.example.engine.capture.ClockOffsetCalibrator
import com.example.engine.capture.FrameClock
import com.example.engine.debug.DebugOverlay
import com.example.engine.debug.LatencyProbe
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.perception.MediaPipeFaceGeometryProvider
import com.example.engine.perception.MediaPipeFaceGeometryProvider.SlotState
import com.example.engine.perception.ProviderConfig
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import com.example.engine.signal.SignalPipeline
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.Category
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import org.junit.After
import org.junit.Assert.*
import org.junit.Test
import java.lang.reflect.Proxy
import java.nio.ByteBuffer
import java.util.Optional
import java.util.Random
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

/**
 * Pure-JVM unit tests for Chunk R3: Dedicated Dispatch Thread & 3-Slot State Machine:
 * 1. Dedicated dispatch thread (`XelvicDispatchThread`) dispatches a staged frame immediately
 *    when inference N finishes even while the camera analysis thread is mid-conversion on frame N+2.
 * 2. Fake-clock steady-state throughput with `conv = 40 ms` and `inf = 59 ms` equals
 *    `1000 / max(conv, inf)` (`59 ms` period, ~16.95 Hz), with `pumpDelayMs == 0` and
 *    `idleGapPendingMs == 0`.
 * 3. Deterministic scheduler stress test across seeded random interleavings of camera-thread
 *    conversion and dispatch-thread pop/result on the 3-slot state machine (`FREE`, `CONVERTING`,
 *    `STAGED`, `IN_FLIGHT`): no write to an `IN_FLIGHT` or popped slot, no frame dispatched twice,
 *    no slot stuck, and every `ImageProxy` closed once.
 * 4. `BLEND:OFF/ON` runtime landmarker recreation on the dispatch thread (`swapMs` on HUD and CSV),
 *    pose + `FaceLifecycle` continuity when blendshapes are off, and watchdog recovery when a
 *    retiring entry never calls back during a `BLEND` swap (without closing the new entry).
 * 5. Non-blocking `start()` and immediate `ImageProxy` close without staging while there is no
 *    active landmarker (`activeEntry == null`).
 * 6. Single cadence gate (`0.85` at provider, `0.50` pre-filter at session) and
 *    `skipConversionIfPendingFresh` flag + `framesSkippedBeforeConversionCount` on HUD.
 */
class DedicatedDispatchThreadR3UnitTest {

    @After
    fun tearDown() {
        FrameClock.customElapsedRealtimeSource = null
    }

    @Test
    fun testInferenceFinishDuringSlowConversionDispatchesNextStagedFrameImmediatelyWithoutWaitingForConversion() {
        var fakeClockNs = 1_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val provider = MediaPipeFaceGeometryProvider(createMockContext())
        provider.enforceCadenceLimit = false

        val dispatchedFrames = ArrayList<Int>()
        val callbacksByCall = ArrayList<() -> Unit>()
        val frame2DispatchedLatch = CountDownLatch(1)

        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    val callNum: Int
                    synchronized(dispatchedFrames) {
                        dispatchedFrames.add(timestampMs.toInt())
                        callNum = dispatchedFrames.size
                        callbacksByCall.add {
                            resultListener(null, mpImage)
                        }
                    }
                    if (callNum == 2) {
                        frame2DispatchedLatch.countDown()
                    }
                }
                override fun close() {
                    isClosed = true
                }
            }
        }

        val emittedSnapshots = ArrayList<FaceGeometrySnapshot>()
        provider.setListener { snap ->
            synchronized(emittedSnapshots) {
                emittedSnapshots.add(snap)
            }
        }

        provider.start(ProviderConfig(analysisLongEdgePx = 320))
        assertTrue(provider.awaitDispatchIdle())

        // 1. Submit Frame 1 at t = 1000ms -> converts in Slot 0 and dispatches on XelvicDispatchThread
        val (p1, c1) = createCountedYuvProxy(320, 180, 50)
        provider.submit(AnalysisFrame(imageProxy = p1, timestampNs = fakeClockNs, frameNumber = 1L, rotationDegrees = 0, isFrontCamera = true))
        assertEquals(1, c1.get())
        assertTrue(provider.awaitDispatchIdle())
        assertEquals(SlotState.IN_FLIGHT, provider.slots[0].state)

        // 2. Submit Frame 2 at t = 1020ms while Frame 1 is in flight -> converts in Slot 1 and stages (STAGED)
        fakeClockNs += 20_000_000L
        val (p2, c2) = createCountedYuvProxy(320, 180, 100)
        provider.submit(AnalysisFrame(imageProxy = p2, timestampNs = fakeClockNs, frameNumber = 2L, rotationDegrees = 0, isFrontCamera = true))
        assertEquals(1, c2.get())
        assertTrue(provider.awaitDispatchIdle())
        assertEquals(SlotState.IN_FLIGHT, provider.slots[0].state)
        assertEquals(SlotState.STAGED, provider.slots[1].state)
        assertEquals(SlotState.FREE, provider.slots[2].state)

        // 3. Start converting Frame 3 on a separate camera analysis thread into FREE Slot 2,
        //    and block Frame 3 mid-conversion!
        val conversion3EnteredLatch = CountDownLatch(1)
        val allowConversion3ToFinishLatch = CountDownLatch(1)
        provider.customSlotConverter = { proxy, rot, slot ->
            conversion3EnteredLatch.countDown()
            assertTrue(allowConversion3ToFinishLatch.await(5, TimeUnit.SECONDS))
            provider.convertAndTransformIntoSlot(proxy, rot, slot)
        }

        fakeClockNs += 20_000_000L
        val cameraExecutor = Executors.newSingleThreadExecutor { r ->
            Thread(r, "CameraAnalysisExecutor").apply { isDaemon = true }
        }
        val (p3, c3) = createCountedYuvProxy(320, 180, 150)
        val submit3Future = cameraExecutor.submit {
            provider.submit(AnalysisFrame(imageProxy = p3, timestampNs = fakeClockNs, frameNumber = 3L, rotationDegrees = 0, isFrontCamera = true))
        }

        // Wait until CameraAnalysisExecutor is blocked inside Frame 3's conversion (Slot 2 == CONVERTING)
        assertTrue(conversion3EnteredLatch.await(2, TimeUnit.SECONDS))
        assertEquals(SlotState.IN_FLIGHT, provider.slots[0].state)
        assertEquals(SlotState.STAGED, provider.slots[1].state)
        assertEquals(SlotState.CONVERTING, provider.slots[2].state)

        // 4. While CameraAnalysisExecutor is blocked mid-conversion of Frame 3, complete Frame 1's inference!
        //    Because XelvicDispatchThread is dedicated and NOT blocked behind conversion, Frame 2 (Slot 1)
        //    must be popped (STAGED -> IN_FLIGHT) and dispatched IMMEDIATELY!
        fakeClockNs += 19_000_000L // Frame 1 finishes at t = 1059ms
        val cb1 = synchronized(dispatchedFrames) { callbacksByCall[0] }
        cb1.invoke()

        assertTrue(
            "Frame 2 must be dispatched on XelvicDispatchThread WHILE Frame 3 conversion is still running on CameraAnalysisExecutor",
            frame2DispatchedLatch.await(2, TimeUnit.SECONDS)
        )
        assertEquals(1, provider.inFlightSlotIndex)
        assertEquals(SlotState.FREE, provider.slots[0].state)
        assertEquals(SlotState.IN_FLIGHT, provider.slots[1].state)
        assertEquals(SlotState.CONVERTING, provider.slots[2].state)
        assertEquals("XelvicDispatchThread", provider.lastDetectAsyncThreadName)
        assertEquals("CameraAnalysisExecutor", provider.lastConversionThreadName)
        assertNotEquals(provider.lastConversionThreadId, provider.lastDetectAsyncThreadId)

        // 5. Now allow Frame 3's conversion on CameraAnalysisExecutor to finish -> stages in Slot 2
        allowConversion3ToFinishLatch.countDown()
        submit3Future.get(2, TimeUnit.SECONDS)
        assertEquals(1, c3.get())
        assertEquals(SlotState.IN_FLIGHT, provider.slots[1].state)
        assertEquals(SlotState.STAGED, provider.slots[2].state)

        // Complete Frame 2 -> dispatches Frame 3 from Slot 2
        fakeClockNs += 59_000_000L
        val cb2 = synchronized(dispatchedFrames) { callbacksByCall[1] }
        cb2.invoke()
        assertTrue(provider.awaitDispatchIdle())
        assertEquals(2, provider.inFlightSlotIndex)
        assertEquals(SlotState.IN_FLIGHT, provider.slots[2].state)

        val snap2 = synchronized(emittedSnapshots) { emittedSnapshots[1] }
        assertEquals("pumpDelayMs must be ~0ms when dispatch thread is dedicated", 0.0f, snap2.pumpDelayMs, 0.01f)
        assertEquals("idleGapPendingMs must be ~0ms when dispatch thread is dedicated", 0.0f, snap2.idleGapPendingMs, 0.01f)

        // Complete Frame 3
        val cb3 = synchronized(dispatchedFrames) { callbacksByCall[2] }
        cb3.invoke()
        assertTrue(provider.awaitDispatchIdle())
        for (slot in provider.slots) {
            assertEquals(0, slot.writeViolationCount)
            assertEquals(SlotState.FREE, slot.state)
        }

        cameraExecutor.shutdownNow()
        provider.stop()
    }

    @Test
    fun testFakeClockSteadyStateThroughputEqualsMaxConvInfWithZeroPumpAndIdleGapPending() {
        // Phone numbers from prompt: conv = 40 ms, inf = 59 ms -> steady-state period must be max(40, 59) = 59 ms
        var fakeClockNs = 10_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val queuedDispatchTasks = ArrayDeque<Runnable>()
        fun drainDispatch() {
            while (queuedDispatchTasks.isNotEmpty()) {
                queuedDispatchTasks.removeFirst().run()
            }
        }

        val provider = MediaPipeFaceGeometryProvider(
            context = createMockContext(),
            dispatchExecutorOverride = Executor { r ->
                queuedDispatchTasks.addLast(r)
            },
            reaperExecutorOverride = Executor { r -> r.run() }
        )
        provider.enforceCadenceLimit = false

        val dispatchTimestampsNs = ArrayList<Long>()
        val dispatchedFrameNums = ArrayList<Long>()
        var inFlightCompleteAtNs = -1L
        var activeCallback: (() -> Unit)? = null

        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    dispatchTimestampsNs.add(fakeClockNs)
                    val slotIdx = provider.inFlightSlotIndex
                    dispatchedFrameNums.add(provider.slots[slotIdx].lastDispatchedFrameNumber)
                    val finishNs = fakeClockNs + 59_000_000L // 59 ms inference
                    inFlightCompleteAtNs = finishNs
                    activeCallback = {
                        fakeClockNs = finishNs
                        resultListener(null, mpImage)
                        // Dedicated dispatch thread is idle, so pump runs immediately at finishNs (0ms pump delay)
                        drainDispatch()
                    }
                }
                override fun close() {
                    isClosed = true
                }
            }
        }

        provider.customSlotConverter = { proxy, rot, slot ->
            // Simulate 40 ms conversion on the camera analysis thread.
            // If the in-flight inference finishes during this 40 ms window, fire its callback at the exact
            // fake-clock completion timestamp while slot is still CONVERTING!
            val convStartNs = fakeClockNs
            val convEndNs = convStartNs + 40_000_000L
            val bmp = provider.convertAndTransformIntoSlot(proxy, rot, slot)
            if (activeCallback != null && inFlightCompleteAtNs in convStartNs..convEndNs) {
                val cb = activeCallback
                activeCallback = null
                inFlightCompleteAtNs = -1L
                cb?.invoke()
            }
            fakeClockNs = convEndNs
            bmp
        }

        val snapshots = ArrayList<FaceGeometrySnapshot>()
        provider.setListener { snap -> snapshots.add(snap) }

        provider.start(ProviderConfig(analysisLongEdgePx = 320))
        drainDispatch()

        // Submit 8 back-to-back camera frames (each taking 40 ms to convert on camera thread while inference takes 59 ms)
        for (frameNum in 1L..8L) {
            val (proxy, closeCount) = createCountedYuvProxy(320, 180, (50 + frameNum).toInt())
            provider.submit(
                AnalysisFrame(
                    imageProxy = proxy,
                    timestampNs = fakeClockNs,
                    frameNumber = frameNum,
                    rotationDegrees = 0,
                    isFrontCamera = true
                )
            )
            assertEquals(1, closeCount.get())
            drainDispatch()
        }
        // Finish the last in-flight inference if still active
        activeCallback?.invoke()
        drainDispatch()

        assertTrue("Multiple inferences must complete", dispatchTimestampsNs.size >= 5)
        // Verify steady-state inter-dispatch period after Frame 2 is staged is exactly 59 ms = max(40ms conv, 59ms inf)
        for (i in 2 until dispatchTimestampsNs.size) {
            val periodMs = (dispatchTimestampsNs[i] - dispatchTimestampsNs[i - 1]) / 1_000_000L
            assertEquals(
                "Steady-state tracker period between dispatch #${i - 1} and #$i must equal max(conv=40ms, inf=59ms) = 59ms",
                59L,
                periodMs
            )
        }
        // Verify no frame was dispatched twice
        assertEquals(dispatchedFrameNums.size, dispatchedFrameNums.distinct().size)
        // Verify pumpDelayMs and idleGapPendingMs are 0 in steady state
        for (i in 2 until snapshots.size) {
            assertEquals(0.0f, snapshots[i].pumpDelayMs, 0.01f)
            assertEquals(0.0f, snapshots[i].idleGapPendingMs, 0.01f)
        }
        provider.stop()
    }

    @Test
    fun testThreeSlotStateMachineDeterministicStressScheduler() {
        var fakeClockNs = 1_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val dispatchQueue = ArrayDeque<Runnable>()
        val provider = MediaPipeFaceGeometryProvider(
            context = createMockContext(),
            dispatchExecutorOverride = Executor { r -> dispatchQueue.addLast(r) },
            reaperExecutorOverride = Executor { r -> r.run() }
        )
        provider.enforceCadenceLimit = false

        val dispatchedFrameNumbers = ArrayList<Long>()
        var inFlightCallback: (() -> Unit)? = null
        var inFlightDispatchedSlotIndex: Int = -1
        var inFlightExpectedFrameNumber: Long = -1L

        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    val slotIdx = provider.inFlightSlotIndex
                    assertTrue("inFlightSlotIndex must be 0, 1, or 2", slotIdx in 0..2)
                    val slot = provider.slots[slotIdx]
                    assertEquals(SlotState.IN_FLIGHT, slot.state)
                    assertTrue(slot.inUseByInference.get())

                    val frameNum = slot.lastDispatchedFrameNumber
                    assertEquals(
                        "Slot written frameNumber must match dispatched frameNumber at detectAsync entry",
                        frameNum,
                        slot.lastWrittenFrameNumber
                    )
                    dispatchedFrameNumbers.add(frameNum)
                    inFlightDispatchedSlotIndex = slotIdx
                    inFlightExpectedFrameNumber = frameNum

                    inFlightCallback = {
                        // Verify that the IN_FLIGHT slot was NEVER overwritten while in flight!
                        assertEquals(
                            "IN_FLIGHT slot must never be written to while inference is in flight",
                            inFlightExpectedFrameNumber,
                            provider.slots[inFlightDispatchedSlotIndex].lastWrittenFrameNumber
                        )
                        assertEquals(
                            SlotState.IN_FLIGHT,
                            provider.slots[inFlightDispatchedSlotIndex].state
                        )
                        resultListener(null, mpImage)
                    }
                }
                override fun close() {
                    isClosed = true
                }
            }
        }

        // Allow pausing conversion mid-flight on a dedicated camera worker thread so dispatch pop/result
        // can interleave arbitrarily while a slot is in SlotState.CONVERTING
        val cameraWorker = Executors.newSingleThreadExecutor { r ->
            Thread(r, "StressCameraWorker").apply { isDaemon = true }
        }
        val midConversionEntered = AtomicReference<CountDownLatch?>(null)
        val midConversionRelease = AtomicReference<CountDownLatch?>(null)

        provider.customSlotConverter = { proxy, rot, slot ->
            assertEquals("Slot must be in CONVERTING state during conversion", SlotState.CONVERTING, slot.state)
            assertFalse("Slot must not be inUseByInference during conversion", slot.inUseByInference.get())
            val entered = midConversionEntered.get()
            val release = midConversionRelease.get()
            if (entered != null && release != null) {
                entered.countDown()
                assertTrue(release.await(5, TimeUnit.SECONDS))
            }
            assertEquals("Slot must still be in CONVERTING state after interleaving", SlotState.CONVERTING, slot.state)
            assertFalse("Slot must still not be inUseByInference after interleaving", slot.inUseByInference.get())
            provider.convertAndTransformIntoSlot(proxy, rot, slot)
        }

        provider.start(ProviderConfig(analysisLongEdgePx = 320))
        while (dispatchQueue.isNotEmpty()) {
            dispatchQueue.removeFirst().run()
        }

        var nextFrameNumber = 1L
        val allCloseCounts = ArrayList<AtomicInteger>()
        var activeConversionFuture: java.util.concurrent.Future<*>? = null

        for (seed in 1L..5L) {
            val rng = Random(seed * 987654321L)
            for (step in 0 until 150) {
                fakeClockNs += 5_000_000L
                val action = rng.nextInt(4)
                when (action) {
                    0 -> {
                        // Start or complete a camera conversion
                        if (activeConversionFuture == null) {
                            val enteredLatch = CountDownLatch(1)
                            val releaseLatch = CountDownLatch(1)
                            midConversionEntered.set(enteredLatch)
                            midConversionRelease.set(releaseLatch)
                            val fNum = nextFrameNumber++
                            val (proxy, closeCount) = createCountedYuvProxy(160, 90, (fNum % 200).toInt())
                            allCloseCounts.add(closeCount)
                            val ts = fakeClockNs
                            activeConversionFuture = cameraWorker.submit {
                                provider.submit(
                                    AnalysisFrame(
                                        imageProxy = proxy,
                                        timestampNs = ts,
                                        frameNumber = fNum,
                                        rotationDegrees = 0,
                                        isFrontCamera = true
                                    )
                                )
                            }
                            assertTrue(enteredLatch.await(2, TimeUnit.SECONDS))
                        } else {
                            midConversionRelease.get()?.countDown()
                            activeConversionFuture!!.get(2, TimeUnit.SECONDS)
                            activeConversionFuture = null
                            midConversionEntered.set(null)
                            midConversionRelease.set(null)
                        }
                    }
                    1 -> {
                        // Synchronous full camera conversion (start + finish) if none is currently mid-conversion
                        if (activeConversionFuture == null) {
                            val fNum = nextFrameNumber++
                            val (proxy, closeCount) = createCountedYuvProxy(160, 90, (fNum % 200).toInt())
                            allCloseCounts.add(closeCount)
                            provider.submit(
                                AnalysisFrame(
                                    imageProxy = proxy,
                                    timestampNs = fakeClockNs,
                                    frameNumber = fNum,
                                    rotationDegrees = 0,
                                    isFrontCamera = true
                                )
                            )
                        }
                    }
                    2 -> {
                        // Run one queued task on dispatch thread (e.g. pump / pop STAGED -> IN_FLIGHT)
                        if (dispatchQueue.isNotEmpty()) {
                            dispatchQueue.removeFirst().run()
                        }
                    }
                    3 -> {
                        // Complete in-flight inference callback if one is in flight
                        val cb = inFlightCallback
                        if (cb != null) {
                            inFlightCallback = null
                            cb.invoke()
                        }
                    }
                }

                // Invariant check at every step:
                // At most 1 slot in CONVERTING, at most 1 slot in STAGED, at most 1 slot in IN_FLIGHT
                var convertingCount = 0
                var stagedCount = 0
                var inFlightCount = 0
                for (slot in provider.slots) {
                    assertEquals("Zero write violations allowed", 0, slot.writeViolationCount)
                    when (slot.state) {
                        SlotState.FREE -> assertFalse(slot.inUseByInference.get())
                        SlotState.CONVERTING -> {
                            convertingCount++
                            assertFalse(slot.inUseByInference.get())
                        }
                        SlotState.STAGED -> {
                            stagedCount++
                            assertFalse(slot.inUseByInference.get())
                        }
                        SlotState.IN_FLIGHT -> {
                            inFlightCount++
                            assertTrue(slot.inUseByInference.get())
                        }
                    }
                }
                assertTrue("At most 1 slot may be CONVERTING", convertingCount <= 1)
                assertTrue("At most 1 slot may be STAGED", stagedCount <= 1)
                assertTrue("At most 1 slot may be IN_FLIGHT", inFlightCount <= 1)
            }
        }

        // Drain any remaining mid-conversion, dispatch tasks, and in-flight callbacks
        if (activeConversionFuture != null) {
            midConversionRelease.get()?.countDown()
            activeConversionFuture!!.get(2, TimeUnit.SECONDS)
        }
        while (dispatchQueue.isNotEmpty() || inFlightCallback != null) {
            while (dispatchQueue.isNotEmpty()) {
                dispatchQueue.removeFirst().run()
            }
            val cb = inFlightCallback
            if (cb != null) {
                inFlightCallback = null
                cb.invoke()
            }
        }

        // Final invariants:
        // 1. No slot stuck: all 3 slots must be FREE
        for (slot in provider.slots) {
            assertEquals("Slot ${slot.index} must return to FREE after drain", SlotState.FREE, slot.state)
            assertFalse(slot.inUseByInference.get())
            assertEquals(0, slot.writeViolationCount)
        }
        // 2. No frame dispatched twice and frames dispatched in strictly increasing order
        assertTrue("Stress test must dispatch many frames", dispatchedFrameNumbers.size >= 50)
        assertEquals(
            "No frame may ever be dispatched twice",
            dispatchedFrameNumbers.size,
            dispatchedFrameNumbers.distinct().size
        )
        for (i in 1 until dispatchedFrameNumbers.size) {
            assertTrue(
                "Dispatched frames must be strictly monotonic (${dispatchedFrameNumbers[i - 1]} -> ${dispatchedFrameNumbers[i]})",
                dispatchedFrameNumbers[i] > dispatchedFrameNumbers[i - 1]
            )
        }
        // 3. Every ImageProxy closed exactly once
        assertEquals(allCloseCounts.size, provider.totalImageProxyClosedCount)
        assertEquals(0, provider.activeHeldImageProxyCount)
        for ((idx, c) in allCloseCounts.withIndex()) {
            assertEquals("ImageProxy #$idx must be closed exactly once", 1, c.get())
        }

        cameraWorker.shutdownNow()
        provider.stop()
    }

    @Test
    fun testBlendToggleAndRetiringWatchdogRecoveryWithoutClosingNewEntryAndSwapMsTelemetry() {
        var fakeClockNs = 5_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val queuedDispatch = ArrayDeque<Runnable>()
        fun drainDispatch() {
            while (queuedDispatch.isNotEmpty()) {
                queuedDispatch.removeFirst().run()
            }
        }

        val provider = MediaPipeFaceGeometryProvider(
            context = createMockContext(),
            dispatchExecutorOverride = Executor { r -> queuedDispatch.addLast(r) },
            reaperExecutorOverride = Executor { r -> r.run() }
        )
        provider.enforceCadenceLimit = false

        val createdHandles = ArrayList<MediaPipeFaceGeometryProvider.LandmarkerHandle>()
        val createdConfigs = ArrayList<ProviderConfig>()
        var handle1Callback: (() -> Unit)? = null
        var handle2Callback: (() -> Unit)? = null

        val mockResultWithPoseNoBlend = createMockResultWithLandmarks(includeBlendshapes = false)
        val mockResultWithPoseAndBlend = createMockResultWithLandmarks(includeBlendshapes = true)

        provider.customLandmarkerFactory = { config, delegate, resultListener, _ ->
            // Simulate 28.5 ms landmarker creation cost on the dispatch thread
            fakeClockNs += 28_500_000L
            createdConfigs.add(config)
            val handleIdx = createdHandles.size + 1
            val handle = object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    if (handleIdx == 1) {
                        handle1Callback = {
                            fakeClockNs += 15_000_000L
                            resultListener(mockResultWithPoseAndBlend, mpImage)
                        }
                    } else {
                        handle2Callback = {
                            fakeClockNs += 11_000_000L
                            resultListener(
                                if (config.outputBlendshapes) mockResultWithPoseAndBlend else mockResultWithPoseNoBlend,
                                mpImage
                            )
                        }
                    }
                }
                override fun close() {
                    isClosed = true
                }
            }
            createdHandles.add(handle)
            handle
        }

        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        val pipeline = SignalPipeline()
        val calibrator = ClockOffsetCalibrator(warmupFrames = 1)
        val probe = LatencyProbe()
        val session = CameraSession(provider, pipeline, gov, calibrator, probe)
        val overlay = DebugOverlay(
            latencyProbe = probe,
            governor = gov,
            faceProvider = provider,
            cameraSession = session
        )

        session.startProvider(ProviderConfig(analysisLongEdgePx = 320, outputBlendshapes = true))
        drainDispatch()
        assertEquals(1, createdHandles.size)
        val entry1Handle = createdHandles[0]
        assertEquals(28.5f, provider.lastSwapMs, 0.1f)
        assertEquals(28.5f, probe.lastSwapMs, 0.1f)

        // 1. Complete first inference on entry #1 so entry #1 is in steady state (hasCompletedFirstInference = true, 500ms timeout)
        provider.submit(AnalysisFrame(imageProxy = createCountedYuvProxy(320, 180, 90).first, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        drainDispatch()
        assertNotNull(handle1Callback)
        handle1Callback!!.invoke()
        drainDispatch()
        assertEquals(0.75f, pipeline.getLatestState().blendshapes["jawOpen"] ?: 0f, 0.01f)

        // 2. Dispatch a second inference on entry #1 at t = 6.0s and leave it in flight (retiring entry will NEVER call back!)
        fakeClockNs = 6_000_000_000L
        handle1Callback = null
        provider.submit(AnalysisFrame(imageProxy = createCountedYuvProxy(320, 180, 100).first, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        drainDispatch()
        assertEquals(1, provider.inFlightSlotIndex)

        // 3. While entry #1's inference is in flight, toggle BLEND:OFF!
        //    Entry #2 (outputBlendshapes = false, hasCompletedFirstInference = false) is created and installed as activeEntry,
        //    while entry #1 is moved to retiringEntries and remains currentInFlightEntry!
        session.setDebugBlendshapesEnabled(false)
        drainDispatch()
        assertEquals(2, createdHandles.size)
        val entry2Handle = createdHandles[1]
        assertFalse(createdConfigs[1].outputBlendshapes)
        assertFalse("Retiring entry #1 must not be closed while its inference is in flight", entry1Handle.isClosed)
        assertFalse("New entry #2 must be open", entry2Handle.isClosed)

        // 4. Advance clock by 200 ms (< 500 ms steady-state timeout of currentInFlightEntry = entry #1):
        //    checkWatchdogOnDispatchThread must evaluate entry #1's 500 ms timeout (not entry #2's 5,000 ms timeout)
        //    and must not reset yet.
        val afterSwapNs = fakeClockNs
        fakeClockNs = 6_000_000_000L + 200_000_000L
        assertFalse(provider.checkWatchdogOnDispatchThread(fakeClockNs))
        assertEquals(0, provider.watchdogResetCount)

        // 5. Advance clock to 6.0s + 550 ms (> 500 ms steady-state timeout of wedged retiring entry #1,
        //    far less than 5,000 ms!). Because entry #1 never called back during the BLEND swap,
        //    the watchdog MUST recover at 550 ms, close wedged entry #1 on reaperExecutor, release the slot,
        //    and MUST NOT close new entry #2!
        fakeClockNs = 6_000_000_000L + 550_000_000L
        assertTrue(
            "Watchdog must time out wedged retiring entry #1 after its 500ms steady-state timeout",
            provider.checkWatchdogOnDispatchThread(fakeClockNs)
        )
        assertEquals(1, provider.watchdogResetCount)
        assertTrue("Wedged retiring entry #1 must be closed on reaperExecutor", entry1Handle.isClosed)
        assertFalse("New entry #2 installed by BLEND:OFF swap must NOT be closed by retiring entry's watchdog", entry2Handle.isClosed)
        assertEquals("No extra landmarker creation needed since entry #2 is already ready", 2, createdHandles.size)
        assertEquals(-1, provider.inFlightSlotIndex)
        assertTrue(afterSwapNs > 0L)

        // 6. Now submit frames with BLEND:OFF on entry #2 -> tracking and lifecycle reach TRACKING and pose updates,
        //    while held blendshapes remain 0.75f without crashing!
        for (i in 1..3) {
            fakeClockNs += 100_000_000L
            provider.submit(AnalysisFrame(imageProxy = createCountedYuvProxy(320, 180, 110 + i).first, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
            drainDispatch()
            assertNotNull("Entry #2 must dispatch frame #$i after watchdog recovery", handle2Callback)
            val cb = handle2Callback!!
            handle2Callback = null
            cb.invoke()
            drainDispatch()
        }

        val stateAfterBlendOff = pipeline.getLatestState()
        assertEquals("FaceLifecycle must reach TRACKING when BLEND is OFF", FaceLifecycle.State.TRACKING, stateAfterBlendOff.lifecycle)
        assertTrue("Filter visibility must remain true when BLEND is OFF", stateAfterBlendOff.isVisible)
        assertEquals("Held jawOpen value must be preserved when BLEND is OFF", 0.75f, stateAfterBlendOff.blendshapes["jawOpen"] ?: 0f, 0.01f)

        // 7. Verify swapMs on HUD and in CSV
        probe.recordRenderFrame(renderTimestampNs = fakeClockNs, faceState = stateAfterBlendOff)
        fakeClockNs += LatencyProbe.DEFAULT_HUD_REFRESH_INTERVAL_NS
        overlay.refreshCachedHudStringsIfNeeded(faceState = stateAfterBlendOff, fps = 60f, tier = "A")

        assertTrue(
            "HUD cachedBlendAndPrioLine must display swap duration (swap: 28.5ms), was: ${overlay.cachedBlendAndPrioLine}",
            overlay.cachedBlendAndPrioLine.contains("swap: 28.5ms")
        )
        val csvLines = probe.dumpCsv().trim().lines()
        assertTrue(csvLines.first().contains(",swapMs"))
        val lastRowCols = csvLines.last().split(",")
        assertEquals(32, lastRowCols.size)
        assertEquals("28.500", lastRowCols[22])

        provider.stop()
    }

    @Test
    fun testNonBlockingStartAndImmediateImageProxyCloseWhileNoActiveLandmarker() {
        var fakeClockNs = 2_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val queuedDispatchTasks = ArrayDeque<Runnable>()
        val provider = MediaPipeFaceGeometryProvider(
            context = createMockContext(),
            dispatchExecutorOverride = Executor { r -> queuedDispatchTasks.addLast(r) },
            reaperExecutorOverride = Executor { r -> r.run() }
        )

        val factoryInvoked = AtomicBoolean(false)
        var failCreation = false
        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            factoryInvoked.set(true)
            if (failCreation) {
                null
            } else {
                object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                    override val delegate: Delegate = delegate
                    override var isClosed: Boolean = false
                    override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                        resultListener(null, mpImage)
                    }
                    override fun close() {
                        isClosed = true
                    }
                }
            }
        }

        // 1. start() must post creation to dispatchExecutor and return immediately without blocking caller thread
        provider.start(ProviderConfig(analysisLongEdgePx = 320))
        assertFalse("start() must NOT block caller thread or invoke factory synchronously", factoryInvoked.get())
        assertNull("activeEntry is null until dispatchExecutor runs creation task", provider.activeEntry)

        // 2. While activeEntry == null during startup, submit() must close each ImageProxy immediately
        //    and must NOT convert or stage frames
        val (startupProxy, startupCloseCount) = createCountedYuvProxy(320, 180, 100)
        provider.submit(AnalysisFrame(imageProxy = startupProxy, timestampNs = fakeClockNs, rotationDegrees = 0, isFrontCamera = true))
        assertEquals("ImageProxy must be closed immediately while activeEntry == null", 1, startupCloseCount.get())
        assertEquals(0, provider.activeHeldImageProxyCount)
        for (slot in provider.slots) {
            assertEquals("No slot may be staged while activeEntry == null", SlotState.FREE, slot.state)
            assertEquals(0, slot.bitmapAllocationCount)
        }

        // 3. Drain dispatchExecutor so landmarker becomes active
        while (queuedDispatchTasks.isNotEmpty()) {
            queuedDispatchTasks.removeFirst().run()
        }
        assertTrue(factoryInvoked.get())
        assertNotNull(provider.activeEntry)

        // 4. Now simulate failed landmarker creation (e.g. init error -> activeEntry == null)
        failCreation = true
        provider.start(ProviderConfig(analysisLongEdgePx = 320))
        while (queuedDispatchTasks.isNotEmpty()) {
            queuedDispatchTasks.removeFirst().run()
        }
        assertTrue(provider.hasInitError)
        assertNull(provider.activeEntry)

        val (failedInitProxy, failedInitCloseCount) = createCountedYuvProxy(320, 180, 120)
        provider.submit(AnalysisFrame(imageProxy = failedInitProxy, timestampNs = fakeClockNs + 33_000_000L, rotationDegrees = 0, isFrontCamera = true))
        assertEquals(1, failedInitCloseCount.get())
        for (slot in provider.slots) {
            assertEquals(SlotState.FREE, slot.state)
        }
        provider.stop()
    }

    @Test
    fun testSingleCadenceGateAndSkipConversionIfPendingFresh() {
        var fakeClockNs = 10_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val provider = MediaPipeFaceGeometryProvider(
            context = createMockContext(),
            dispatchExecutorOverride = Executor { r -> r.run() },
            reaperExecutorOverride = Executor { r -> r.run() }
        )
        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        val pipeline = SignalPipeline()
        val calibrator = ClockOffsetCalibrator(warmupFrames = 1)
        val probe = LatencyProbe()
        val session = CameraSession(provider, pipeline, gov, calibrator, probe)
        session.configureTimestampSource(ClockOffsetCalibrator.TIMESTAMP_SOURCE_REALTIME)
        val overlay = DebugOverlay(
            latencyProbe = probe,
            governor = gov,
            faceProvider = provider,
            cameraSession = session
        )

        var pendingCb: (() -> Unit)? = null
        var dispatchCount = 0
        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    dispatchCount++
                    pendingCb = {
                        resultListener(null, mpImage)
                    }
                }
                override fun close() {
                    isClosed = true
                }
            }
        }

        // Part A: Verify single cadence gate (0.85 at provider, 0.50 pre-filter at session)
        // For 30 Hz (period = 33.33 ms):
        // - Session pre-filter (0.50 * 33.33ms = 16.66ms) drops burst frames at +10ms
        // - Frame arriving at +29ms (>= 0.85 * 33.33ms = 28.33ms) passes both session and provider with 0 provider deferrals!
        session.startProvider(ProviderConfig(analysisLongEdgePx = 320))
        provider.updateTargetCadenceHz(30)

        val (f1, c1) = createCountedYuvProxy(320, 180, 80, timestampNs = fakeClockNs)
        assertTrue(session.submitAnalysisFrame(f1, fakeClockNs))
        assertEquals(1, c1.get())
        assertEquals(1, dispatchCount)
        pendingCb!!.invoke()

        // Burst frame at +10ms (< 0.50 * 33.33ms = 16.66ms) -> dropped cheaply by session pre-filter before conversion
        fakeClockNs += 10_000_000L
        val (fBurst, cBurst) = createCountedYuvProxy(320, 180, 85, timestampNs = fakeClockNs)
        assertFalse(session.submitAnalysisFrame(fBurst, fakeClockNs))
        assertEquals(1, cBurst.get())
        assertEquals(1, session.sessionCadenceThrottledCount)

        // Frame at +29ms after f1 (>= 0.85 * 33.33ms = 28.33ms, even though < 0.92 * 33.33ms!) ->
        // passes session pre-filter AND dispatches immediately at provider with 0 provider deferrals!
        fakeClockNs += 19_000_000L // total +29ms since f1
        val (f2, c2) = createCountedYuvProxy(320, 180, 90, timestampNs = fakeClockNs)
        assertTrue(session.submitAnalysisFrame(f2, fakeClockNs))
        assertEquals(1, c2.get())
        assertEquals("Frame at 0.87 * period must dispatch without deferral", 2, dispatchCount)
        assertEquals("Provider cadence deferred count must be 0", 0, provider.providerCadenceDeferredCount)

        // Part B: Verify skipConversionIfPendingFresh (default OFF -> ON)
        assertFalse("skipConversionIfPendingFresh must default to false", provider.skipConversionIfPendingFresh)
        session.setSkipConversionIfPendingFresh(true)
        assertTrue(provider.skipConversionIfPendingFresh)

        // While f2 is still in flight (Slot 1 == IN_FLIGHT), submit f3 at +33ms -> converts and stages (Slot 2 == STAGED)
        fakeClockNs += 33_000_000L
        val (f3, c3) = createCountedYuvProxy(320, 180, 100, timestampNs = fakeClockNs)
        assertTrue(session.submitAnalysisFrame(f3, fakeClockNs))
        assertEquals(1, c3.get())
        assertEquals(SlotState.STAGED, provider.slots[2].state)
        assertEquals(0, provider.framesSkippedBeforeConversionCount)

        // Submit f4 only 20ms later (< 1.5 * ~33ms camera interval) while f2 is STILL in flight and f3 is STAGED:
        // Because skipConversionIfPendingFresh is ON and staged f3 is fresh (< 1.5 camera intervals),
        // f4 must skip conversion, close its ImageProxy immediately, and increment framesSkippedBeforeConversionCount!
        fakeClockNs += 20_000_000L
        val (f4, c4) = createCountedYuvProxy(320, 180, 110, timestampNs = fakeClockNs)
        assertTrue(session.submitAnalysisFrame(f4, fakeClockNs))
        assertEquals(1, c4.get())
        assertEquals(1, provider.framesSkippedBeforeConversionCount)

        fakeClockNs += LatencyProbe.DEFAULT_HUD_REFRESH_INTERVAL_NS
        overlay.refreshCachedHudStringsIfNeeded(
            faceState = FaceState(timestampNs = fakeClockNs, lifecycle = FaceLifecycle.State.TRACKING, isVisible = true),
            fps = 60f,
            tier = "A"
        )
        assertTrue(
            "HUD cachedCountersLine must show skipFresh:ON(1), was: ${overlay.cachedCountersLine}",
            overlay.cachedCountersLine.contains("skipFresh:ON(1)")
        )

        pendingCb?.invoke()
        provider.stop()
    }

    private fun createMockContext(): Context {
        return object : android.content.ContextWrapper(null) {
            override fun getAssets(): AssetManager? = null
        }
    }

    private fun createMockResultWithLandmarks(includeBlendshapes: Boolean): FaceLandmarkerResult {
        val lm = NormalizedLandmark.create(0.5f, 0.5f, -0.02f)
        val landmarks478 = List(478) { idx ->
            when (idx) {
                234 -> NormalizedLandmark.create(0.35f, 0.50f, -0.02f)
                454 -> NormalizedLandmark.create(0.65f, 0.50f, -0.02f)
                10 -> NormalizedLandmark.create(0.50f, 0.30f, -0.02f)
                152 -> NormalizedLandmark.create(0.50f, 0.70f, -0.02f)
                else -> lm
            }
        }
        val poseMatrix = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, -0.5f, 1f
        )
        val blendList = if (includeBlendshapes) {
            Optional.of(listOf(listOf(Category.create(0.75f, 0, "jawOpen", "jawOpen"))))
        } else {
            Optional.empty()
        }
        val autoValueClass = Class.forName("com.google.mediapipe.tasks.vision.facelandmarker.AutoValue_FaceLandmarkerResult")
        val ctor = autoValueClass.declaredConstructors.first()
        ctor.isAccessible = true
        return ctor.newInstance(
            1000L,
            listOf(landmarks478),
            blendList,
            Optional.of(listOf(poseMatrix))
        ) as FaceLandmarkerResult
    }

    private fun createCountedYuvProxy(
        width: Int,
        height: Int,
        yFill: Int,
        timestampNs: Long = 1_000_000_000L
    ): Pair<ImageProxy, AtomicInteger> {
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
                "getTimestamp" -> timestampNs
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
