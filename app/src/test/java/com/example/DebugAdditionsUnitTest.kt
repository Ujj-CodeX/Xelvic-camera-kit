package com.example

import android.content.Context
import android.content.res.AssetManager
import android.graphics.ImageFormat
import android.graphics.Rect
import android.os.Process
import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import com.example.engine.capture.AnalysisFrame
import com.example.engine.capture.CameraSession
import com.example.engine.capture.ClockOffsetCalibrator
import com.example.engine.capture.FrameClock
import com.example.engine.debug.DebugOverlay
import com.example.engine.debug.LatencyProbe
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.governor.ThermalMonitor
import com.example.engine.perception.AnalysisResolutionMode
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.perception.MediaPipeFaceGeometryProvider
import com.example.engine.perception.ProviderConfig
import com.example.engine.scene.BehaviorRunner
import com.example.engine.scene.PropertySlots
import com.example.engine.scene.SignalBus
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import com.example.engine.signal.SignalPipeline
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.Delegate
import org.junit.After
import org.junit.Assert.*
import org.junit.Test
import java.lang.reflect.Proxy
import java.nio.ByteBuffer
import java.util.concurrent.Executor

/**
 * Pure-JVM JUnit 4 unit tests for Chunk: Debug Additions (UI + config + telemetry):
 * 1. Default `analysisResolutionMode == AnalysisResolutionMode.SMALL` (`LEGACY` still selectable).
 * 2. Split `idleGap` into `idleGapPendingMs` (inference finished while a converted frame was already pending)
 *    and `idleGapNoPendingMs` (waiting for camera/conversion), plus `pumpDelayMs` (time from `handleResult`
 *    posting the pump to the pump starting execution on `analysisExecutor`, rolling mean and max, warm-up excluded)
 *    on HUD and CSV.
 * 3. `BLEND:ON/OFF` runtime toggle: recreates the landmarker on `analysisExecutor` with a 3s warm-up window,
 *    shows `"blendshapes: held (stale)"` on the HUD when `OFF`, reports rolling inference time for `BLEND:ON`
 *    and `BLEND:OFF` separately (warm-up excluded), and ensures blendshape-driven behaviors (`SignalBus` +
 *    `BehaviorRunner`) hold their last value and do not crash.
 * 4. `PRIO:NORM/LOW` (analysis/conversion thread priority) and `RPRIO:NORM/HIGH` (GL render thread priority)
 *    toggles and HUD display.
 * 5. Latency HUD cached ~2 Hz telemetry: thermal status + headroom, `targetTrackerCadenceHz`,
 *    `targetAnalysisSizePx`, `framesDroppedCount`, `pendingReplacedCount`, `watchdogResetCount`,
 *    and `providerCadenceDeferredCount`.
 */
class DebugAdditionsUnitTest {

    @After
    fun tearDown() {
        FrameClock.customElapsedRealtimeSource = null
    }

    @Test
    fun testDefaultAnalysisResolutionModeIsSmallAndLegacyRemainsSelectable() {
        val probe = LatencyProbe()
        val overlay = DebugOverlay(latencyProbe = probe)

        assertEquals(
            "LatencyProbe default resolutionModeLabel must be SMALL",
            AnalysisResolutionMode.SMALL.name,
            probe.resolutionModeLabel
        )
        assertTrue(
            "DebugOverlay initial cachedFovLine must show Analysis: SMALL",
            overlay.cachedFovLine.startsWith("Analysis: SMALL")
        )
        // Verify both SMALL and LEGACY exist in AnalysisResolutionMode
        assertEquals(
            listOf(AnalysisResolutionMode.LEGACY, AnalysisResolutionMode.SMALL),
            AnalysisResolutionMode.values().toList()
        )
    }

    @Test
    fun testIdleGapSplitIntoPendingAndNoPendingAndPumpDelayMsOnHudAndCsv() {
        var fakeClockNs = 1_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        // Queue that lets us control when tasks posted to analysisExecutor actually start running,
        // so we can deterministically simulate a non-zero pumpDelayMs between handleResult() and pumpOnAnalysisThread().
        val queuedAnalysisTasks = ArrayDeque<Runnable>()
        val manualAnalysisExecutor = Executor { r -> queuedAnalysisTasks.addLast(r) }
        fun drainAnalysisExecutor() {
            while (queuedAnalysisTasks.isNotEmpty()) {
                queuedAnalysisTasks.removeFirst().run()
            }
        }

        val provider = MediaPipeFaceGeometryProvider(
            context = createMockContext(),
            analysisExecutorOverride = manualAnalysisExecutor,
            reaperExecutorOverride = Executor { r -> r.run() }
        )
        provider.enforceCadenceLimit = false

        var pendingDetectCallback: (() -> Unit)? = null
        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    pendingDetectCallback = {
                        fakeClockNs += 15_000_000L // 15.0 ms inference
                        resultListener(null, mpImage)
                    }
                }
                override fun close() {
                    isClosed = true
                }
            }
        }

        val probe = LatencyProbe()
        val overlay = DebugOverlay(latencyProbe = probe, faceProvider = provider)
        val emittedSnapshots = ArrayList<FaceGeometrySnapshot>()
        provider.setListener { snap ->
            emittedSnapshots.add(snap)
            probe.recordTrackerSnapshot(snap)
        }

        provider.start(ProviderConfig(analysisLongEdgePx = 320))
        drainAnalysisExecutor()

        // 1. Submit Frame 1 at t = 1000ms
        provider.submit(createYuvFrame(timestampNs = fakeClockNs))
        drainAnalysisExecutor()
        assertNotNull(pendingDetectCallback)

        // 2. While Frame 1 is in flight, submit Frame 2 at t = 1006ms -> converts immediately into
        //    the free ping-pong slot BEFORE Frame 1 finishes inference!
        fakeClockNs += 6_000_000L
        provider.submit(createYuvFrame(timestampNs = fakeClockNs))
        drainAnalysisExecutor()
        assertEquals(1, provider.overlappedConversionCount)

        // 3. Complete Frame 1 inference (adds 15ms -> finishes at t = 1021ms).
        //    handleResult posts pumpOnAnalysisThread() to queuedAnalysisTasks.
        val cb1 = pendingDetectCallback!!
        pendingDetectCallback = null
        cb1.invoke()

        // Simulate 2.5 ms scheduling delay before analysisExecutor runs the posted pump task!
        fakeClockNs += 2_500_000L
        drainAnalysisExecutor() // Dispatches Frame 2 at t = 1023.5ms!

        // Complete Frame 2 inference so Frame 2's snapshot is emitted with idleGapPendingMs and pumpDelayMs
        assertNotNull("Frame 2 must have been dispatched by the pump", pendingDetectCallback)
        val cb2 = pendingDetectCallback!!
        pendingDetectCallback = null
        cb2.invoke()
        drainAnalysisExecutor()

        assertEquals(2, emittedSnapshots.size)
        val snap2 = emittedSnapshots[1]
        // Because Frame 2 finished conversion at t = 1006ms (<= Frame 1 inferenceDone at t = 1021ms),
        // the 2.5ms gap between Frame 1 finishing (1021ms) and Frame 2 dispatching (1023.5ms) must be classified
        // 100% as idleGapPendingMs (threading/pump delay) and 0.0ms as idleGapNoPendingMs!
        assertEquals(2.5f, snap2.idleGapPendingMs, 0.05f)
        assertEquals(0.0f, snap2.idleGapNoPendingMs, 0.01f)
        assertEquals(2.5f, snap2.pumpDelayMs, 0.05f)

        // 4. Now wait 9.0 ms with NO frame pending (camera interval), then submit Frame 3.
        fakeClockNs += 9_000_000L
        provider.submit(createYuvFrame(timestampNs = fakeClockNs))
        drainAnalysisExecutor() // Converts and dispatches Frame 3
        val cb3 = pendingDetectCallback!!
        pendingDetectCallback = null
        cb3.invoke()
        drainAnalysisExecutor()

        assertEquals(3, emittedSnapshots.size)
        val snap3 = emittedSnapshots[2]
        // Because Frame 3 finished conversion AFTER Frame 2's inference finished, the 9.0ms gap is
        // classified 100% as idleGapNoPendingMs (waiting for camera/conversion) and 0.0ms as idleGapPendingMs!
        assertEquals(0.0f, snap3.idleGapPendingMs, 0.01f)
        assertTrue("idleGapNoPendingMs must be >= 9.0ms", snap3.idleGapNoPendingMs >= 9.0f)

        // 5. Record a render frame and refresh HUD + CSV
        probe.recordRenderFrame(
            renderTimestampNs = fakeClockNs,
            faceState = FaceState(
                timestampNs = fakeClockNs,
                poseTimestampNs = fakeClockNs - 20_000_000L,
                lifecycle = FaceLifecycle.State.TRACKING,
                isVisible = true
            )
        )
        fakeClockNs += LatencyProbe.DEFAULT_HUD_REFRESH_INTERVAL_NS
        overlay.refreshCachedHudStringsIfNeeded(
            faceState = FaceState(timestampNs = fakeClockNs, lifecycle = FaceLifecycle.State.TRACKING, isVisible = true),
            fps = 60f,
            tier = "A"
        )

        assertEquals(2.5f / 3f, probe.rollingIdleGapPendingMs, 0.1f)
        assertTrue(probe.rollingIdleGapNoPendingMs >= 3.0f)
        assertEquals(2.5f, probe.rollingPumpDelayMaxMs, 0.05f)
        assertTrue(
            "HUD cachedPumpAndIdleLine must show Pump mean/max and split IdleGap: ${overlay.cachedPumpAndIdleLine}",
            overlay.cachedPumpAndIdleLine.contains("Pump:") &&
                overlay.cachedPumpAndIdleLine.contains("IdleGap: pend") &&
                overlay.cachedPumpAndIdleLine.contains("noPend")
        )

        val csvLines = probe.dumpCsv().trim().lines()
        assertTrue(csvLines.first().contains("idleGapPendingMs,idleGapNoPendingMs,pumpDelayMs,swapMs"))
        val cols = csvLines.last().split(",")
        assertEquals(32, cols.size)
        assertEquals("0.000", cols[19]) // snap3 idleGapPendingMs
        assertEquals("9.000", cols[20]) // snap3 idleGapNoPendingMs
        provider.stop()
    }

    @Test
    fun testBlendshapesRuntimeToggleHoldsLastValueInBehaviorsSeparatesInferenceMetricsAndShowsHudStaleLabel() {
        var fakeClockNs = 5_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val createdConfigs = ArrayList<ProviderConfig>()
        val createdHandles = ArrayList<MediaPipeFaceGeometryProvider.LandmarkerHandle>()
        var providerListener: ((FaceGeometrySnapshot) -> Unit)? = null

        val realProvider = MediaPipeFaceGeometryProvider(
            context = createMockContext(),
            analysisExecutorOverride = Executor { r -> r.run() },
            reaperExecutorOverride = Executor { r -> r.run() }
        )

        realProvider.customLandmarkerFactory = { config, delegate, resultListener, _ ->
            createdConfigs.add(config)
            val handle = object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    fakeClockNs += if (config.outputBlendshapes) 18_000_000L else 11_000_000L
                    resultListener(null, mpImage)
                }
                override fun close() {
                    isClosed = true
                }
            }
            createdHandles.add(handle)
            handle
        }

        val providerProxy = object : com.example.engine.perception.FaceGeometryProvider by realProvider {
            override fun setListener(listener: (FaceGeometrySnapshot) -> Unit) {
                realProvider.setListener(listener)
                providerListener = listener
            }
        }

        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        gov.reset(nowMs = fakeClockNs / 1_000_000L)
        val pipeline = SignalPipeline()
        val calibrator = ClockOffsetCalibrator(warmupFrames = 1)
        val probe = LatencyProbe()
        val session = CameraSession(providerProxy, pipeline, gov, calibrator, probe)
        val overlay = DebugOverlay(
            latencyProbe = probe,
            governor = gov,
            faceProvider = realProvider,
            cameraSession = session
        )

        // Wire a real SignalBus + BehaviorRunner with a CompiledMap driven by "face.blend.jawOpen"
        // writing to node 0's Y position (0.0 -> 0.50 m) with zero smoothing so we verify exact held values!
        val slotIndexMap = mapOf(
            "face.present" to 0,
            "face.fade" to 1,
            "face.pose.yaw" to 2,
            "face.pose.pitch" to 3,
            "face.pose.roll" to 4,
            "time.render" to 5,
            "time.sinceAcquired" to 6,
            "input.tap" to 7,
            "face.blend.jawOpen" to 8
        )
        val signalBus = SignalBus(
            totalSlots = 9,
            derivedSignals = emptyList(),
            slotIndexMap = slotIndexMap
        )
        val jawMap = BehaviorRunner.CompiledMap(
            id = "jaw_to_pos_y",
            inputSlotIndex = 8, // face.blend.jawOpen
            inMin = 0f,
            inMax = 1f,
            outMin = 0f,
            outMax = 0.50f,
            easing = "linear",
            clamp = true,
            smoothingTau = 0f,
            initial = 0f,
            writes = listOf(
                BehaviorRunner.PropertyWrite(
                    nodeIndex = 0,
                    propertyType = BehaviorRunner.PropertyType.POSITION,
                    componentIndex = 1, // Y
                    mode = BehaviorRunner.WriteMode.SET
                )
            )
        )
        val behaviorRunner = BehaviorRunner(
            clips = emptyList(),
            maps = listOf(jawMap),
            triggers = emptyList(),
            springs = emptyList(),
            nodeCount = 1
        )
        val propertySlots = PropertySlots(nodeCount = 1)

        session.startProvider(ProviderConfig(analysisLongEdgePx = 320, outputBlendshapes = true))
        assertEquals(1, createdHandles.size)
        assertTrue(createdConfigs[0].outputBlendshapes)
        assertEquals("ON", probe.blendshapesEnabledLabel)

        // Advance past initial start warm-up (3.5s)
        fakeClockNs += 3_500_000_000L

        // 1. Emit a snapshot with BLEND:ON and jawOpen = 0.64f (inferenceMs = 18.0ms)
        val snapBlendOn = FaceGeometrySnapshot.create(
            timestampNs = fakeClockNs,
            imageWidth = 180,
            imageHeight = 320,
            rotationDegrees = 0,
            isFrontCamera = true,
            isTracking = true,
            landmarks = createValidLandmarks(),
            rawPoseMatrix = floatArrayOf(
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f,
                0f, 0f, 1f, 0f,
                0f, 0f, -0.5f, 1f
            ),
            blendshapes = mapOf("jawOpen" to 0.64f, "mouthSmileLeft" to 0.50f, "mouthSmileRight" to 0.50f),
            conversionMs = 3.0f,
            inferenceMs = 18.0f,
            blendshapesEnabled = true
        )
        providerListener!!.invoke(snapBlendOn)

        val stateOn = pipeline.getLatestState()
        assertEquals(0.64f, stateOn.blendshapes["jawOpen"] ?: 0f, 0.001f)
        signalBus.evaluate(stateOn, renderTimeSec = 1.0f)
        behaviorRunner.step(dt = 0.016f, signalBus = signalBus, faceState = stateOn, slots = propertySlots)
        val posYWhenBlendOn = propertySlots.position[1]
        assertEquals(0.32f, posYWhenBlendOn, 0.001f) // 0.64 * 0.50

        // 2. Toggle BLEND:OFF at runtime -> must recreate landmarker on analysisExecutor with outputBlendshapes=false
        //    and start a 3s warm-up window!
        session.setDebugBlendshapesEnabled(false)
        assertEquals(2, createdHandles.size)
        assertTrue("Previous BLEND:ON handle must be retired", createdHandles[0].isClosed)
        assertFalse("New handle must have outputBlendshapes = false", createdConfigs[1].outputBlendshapes)
        assertEquals("OFF (held)", probe.blendshapesEnabledLabel)
        assertTrue("Toggling BLEND must start a 3s warm-up window", gov.isInWarmup(fakeClockNs / 1_000_000L))

        // Advance past the 3s warm-up window
        fakeClockNs += 3_200_000_000L
        assertFalse(gov.isInWarmup(fakeClockNs / 1_000_000L))

        // 3. Emit a snapshot with BLEND:OFF (empty blendshapes map, inferenceMs = 11.0ms)
        val snapBlendOff = FaceGeometrySnapshot.create(
            timestampNs = fakeClockNs,
            imageWidth = 180,
            imageHeight = 320,
            rotationDegrees = 0,
            isFrontCamera = true,
            isTracking = true,
            landmarks = createValidLandmarks(),
            rawPoseMatrix = floatArrayOf(
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f,
                0f, 0f, 1f, 0f,
                0f, 0f, -0.5f, 1f
            ),
            blendshapes = emptyMap(),
            conversionMs = 3.0f,
            inferenceMs = 11.0f,
            blendshapesEnabled = false
        )
        providerListener!!.invoke(snapBlendOff)

        // Verify that FaceState and blendshape-driven BehaviorRunner hold their last value (0.64 -> 0.32m) without crashing!
        val stateOff = pipeline.getLatestState()
        assertEquals(0.64f, stateOff.blendshapes["jawOpen"] ?: 0f, 0.001f)
        signalBus.evaluate(stateOff, renderTimeSec = 4.2f)
        behaviorRunner.step(dt = 0.016f, signalBus = signalBus, faceState = stateOff, slots = propertySlots)
        assertEquals(
            "Blendshape-driven behavior output must hold its exact last value when BLEND is OFF",
            posYWhenBlendOn,
            propertySlots.position[1],
            0.001f
        )

        // 4. Verify separate rolling inference times for BLEND:ON (18.0ms) and BLEND:OFF (11.0ms) and HUD stale label
        probe.recordRenderFrame(renderTimestampNs = fakeClockNs, faceState = stateOff)
        fakeClockNs += LatencyProbe.DEFAULT_HUD_REFRESH_INTERVAL_NS
        overlay.refreshCachedHudStringsIfNeeded(stateOff, fps = 60f, tier = "A")
        assertEquals(18.0f, probe.rollingInferenceBlendOnMs, 0.1f)
        assertEquals(11.0f, probe.rollingInferenceBlendOffMs, 0.1f)
        assertTrue(
            "HUD cachedBlendAndPrioLine must show 'blendshapes: held (stale)' and separate inf(ON/OFF): 18.0/11.0ms, got: ${overlay.cachedBlendAndPrioLine}",
            overlay.cachedBlendAndPrioLine.contains("blendshapes: held (stale)") &&
                overlay.cachedBlendAndPrioLine.contains("inf(ON/OFF): 18.0/11.0ms")
        )
        realProvider.stop()
    }

    @Test
    fun testThreadPriorityTogglesAndLatencyHudThermalGovernorAndCounters() {
        var fakeClockNs = 10_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val provider = MediaPipeFaceGeometryProvider(
            context = createMockContext(),
            analysisExecutorOverride = Executor { r -> r.run() },
            reaperExecutorOverride = Executor { r -> r.run() }
        )
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

        // 1. Verify default priorities (NORM / NORM)
        assertFalse(provider.useLowAnalysisThreadPriority)
        assertEquals(Process.THREAD_PRIORITY_DEFAULT, provider.lastAppliedAnalysisThreadPriority)
        assertEquals("NORM", probe.analysisThreadPriorityLabel)
        assertEquals("NORM", probe.renderThreadPriorityLabel)

        // 2. Toggle PRIO:LOW and RPRIO:HIGH
        session.setUseLowAnalysisThreadPriority(true)
        probe.renderThreadPriorityLabel = "HIGH"
        assertTrue(provider.useLowAnalysisThreadPriority)
        assertEquals("LOW", probe.analysisThreadPriorityLabel)
        assertEquals("HIGH", probe.renderThreadPriorityLabel)

        // 3. Report thermal status + headroom on ThermalMonitor and refresh HUD
        gov.thermalMonitor.reportThermalStatus(ThermalMonitor.THERMAL_STATUS_MODERATE)
        gov.thermalMonitor.reportThermalHeadroom(0.85f)

        fakeClockNs += LatencyProbe.DEFAULT_HUD_REFRESH_INTERVAL_NS
        overlay.refreshCachedHudStringsIfNeeded(
            faceState = FaceState(timestampNs = fakeClockNs, lifecycle = FaceLifecycle.State.TRACKING, isVisible = true),
            fps = 60f,
            tier = "A"
        )

        assertTrue(
            "HUD cachedBlendAndPrioLine must display Prio A:LOW R:HIGH: ${overlay.cachedBlendAndPrioLine}",
            overlay.cachedBlendAndPrioLine.contains("Prio A:LOW R:HIGH")
        )
        assertTrue(
            "HUD cachedGovWarmupLine must display thermal status, headroom, and governor target cadence/size: ${overlay.cachedGovWarmupLine}",
            overlay.cachedGovWarmupLine.contains("Therm: MODERATE(2) hdrm:0.85") &&
                overlay.cachedGovWarmupLine.contains("Gov: 30Hz@320px")
        )
        assertTrue(
            "HUD cachedCountersLine must display drop, repl, wdog, ovlp, cadDrop: ${overlay.cachedCountersLine}",
            overlay.cachedCountersLine.contains("Counters: drop=0 repl=0 wdog=0") &&
                overlay.cachedCountersLine.contains("ovlp=0 cadDrop=0/0")
        )
        provider.stop()
    }

    private fun createValidLandmarks(): FloatArray {
        val arr = FloatArray(478 * 3)
        // Populate canonical landmarks (234 right cheek, 454 left cheek, 10 forehead, 152 chin, 1 nose, 33 right eye, 263 left eye)
        arr[234 * 3] = 0.30f; arr[234 * 3 + 1] = 0.50f; arr[234 * 3 + 2] = -0.02f
        arr[454 * 3] = 0.70f; arr[454 * 3 + 1] = 0.50f; arr[454 * 3 + 2] = -0.02f
        arr[10 * 3] = 0.50f;  arr[10 * 3 + 1] = 0.30f;  arr[10 * 3 + 2] = -0.01f
        arr[152 * 3] = 0.50f; arr[152 * 3 + 1] = 0.72f; arr[152 * 3 + 2] = -0.01f
        arr[1 * 3] = 0.50f;   arr[1 * 3 + 1] = 0.52f;   arr[1 * 3 + 2] = -0.05f
        arr[33 * 3] = 0.38f;  arr[33 * 3 + 1] = 0.42f;  arr[33 * 3 + 2] = -0.02f
        arr[263 * 3] = 0.62f; arr[263 * 3 + 1] = 0.42f; arr[263 * 3 + 2] = -0.02f
        return arr
    }

    private fun createMockContext(): Context {
        return object : android.content.ContextWrapper(null) {
            override fun getAssets(): AssetManager? = null
        }
    }

    private fun createYuvFrame(timestampNs: Long, width: Int = 320, height: Int = 180): AnalysisFrame {
        val yBuf = ByteBuffer.allocateDirect(width * height)
        val uBuf = ByteBuffer.allocateDirect((width / 2) * (height / 2))
        val vBuf = ByteBuffer.allocateDirect((width / 2) * (height / 2))
        for (i in 0 until yBuf.capacity()) yBuf.put(i, 96.toByte())
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
                "close" -> Unit
                else -> null
            }
        } as ImageProxy

        return AnalysisFrame(
            imageProxy = proxy,
            timestampNs = timestampNs,
            rotationDegrees = 0,
            isFrontCamera = true,
            submittedAtNs = timestampNs - 1_000_000L
        )
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
