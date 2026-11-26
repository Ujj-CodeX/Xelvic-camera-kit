package com.example

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import androidx.test.core.app.ApplicationProvider
import com.example.engine.capture.AnalysisFrame
import com.example.engine.capture.CameraSession
import com.example.engine.capture.ClockOffsetCalibrator
import com.example.engine.capture.FrameClock
import com.example.engine.debug.LatencyProbe
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.perception.FaceGeometryProvider
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.perception.ProviderConfig
import com.example.engine.render.XelvicGlSurfaceView
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import com.example.engine.signal.SignalPipeline
import org.junit.After
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.lang.reflect.Proxy

/**
 * Unit tests for Chunk: LatencyProbe (AUDIT.md Section 3.2):
 * 1. ClockOffsetCalibrator: REALTIME (zero offset) vs UNKNOWN (minimum / lowest ~5th percentile
 *    over warm-up frames so delivery delay is never baked into capture timestamp).
 * 2. End-to-end sensor timestamp propagation: ImageProxy -> AnalysisFrame -> FaceGeometrySnapshot
 *    -> SignalPipeline -> FaceState.poseTimestampNs (preserved across getPredictedState).
 * 3. LatencyProbe 5-second rolling percentile window, sample eviction, TRACKING-only ageAtRender
 *    recording, ~2 Hz HUD refresh caching, and on-demand CSV dump.
 * 4. XelvicGlSurfaceView startup renderMode logging, Choreographer main-thread lifecycle, and
 *    PerformanceGovernor continuous-render gating + drawCpuDurationMs debug stat.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class LatencyProbeUnitTest {

    @After
    fun tearDown() {
        FrameClock.customElapsedRealtimeSource = null
    }

    @Test
    fun testClockOffsetCalibratorRealtimeVsUnknownMinimumOffset() {
        val calibrator = ClockOffsetCalibrator(warmupFrames = 15)

        // 1. REALTIME source: sensor timestamps are already in elapsedRealtimeNanos domain (0 offset)
        calibrator.configureSource(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME)
        assertTrue(calibrator.isCalibrated)
        val sensorTs = 5_000_000_000L
        val arrivalTs = 5_025_000_000L // +25ms delivery delay
        val alignedRealtime = calibrator.toElapsedRealtimeNs(sensorTs, arrivalTs)
        assertEquals(
            "REALTIME source must preserve exact sensor timestamp without baking delivery delay",
            sensorTs,
            alignedRealtime
        )
        assertEquals(0L, calibrator.calibratedOffsetNs)

        // 2. UNKNOWN source: must use the MINIMUM of (arrivalElapsedRealtime - sensorTimestamp)
        // over warm-up frames, NOT the median, so variable delivery delays are not baked into capture time
        calibrator.configureSource(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN)
        assertFalse(calibrator.isCalibrated)

        // True clock offset between sensor clock and elapsedRealtimeNanos is +10_000_000_000L (10s).
        // Simulate 15 warm-up frames where 14 frames have heavy delivery delay (30ms..80ms)
        // and 1 fast frame has the minimal delivery delay (5ms = 5_000_000ns).
        // A median calibrator would wrongly pick ~50ms (+10_050_000_000L), hiding 45ms of latency!
        // Our minimum/5th-percentile calibrator must pick +10_005_000_000L.
        val trueClockBaseDiffNs = 10_000_000_000L
        val deliveryDelaysMs = longArrayOf(
            45L, 50L, 60L, 40L, 55L,
            5L,  // minimum delivery delay frame
            70L, 65L, 50L, 48L,
            52L, 80L, 42L, 58L, 62L
        )

        var lastAligned = 0L
        for (i in deliveryDelaysMs.indices) {
            val rawSensor = 1_000_000_000L + i * 33_333_333L
            val delayNs = deliveryDelaysMs[i] * 1_000_000L
            val arrival = rawSensor + trueClockBaseDiffNs + delayNs
            lastAligned = calibrator.toElapsedRealtimeNs(rawSensor, arrival)
        }

        assertTrue("Calibrator must be calibrated after 15 warm-up frames", calibrator.isCalibrated)
        val expectedMinOffsetNs = trueClockBaseDiffNs + 5_000_000L
        assertEquals(
            "Calibrator must use the minimum offset, not the median",
            expectedMinOffsetNs,
            calibrator.calibratedOffsetNs
        )

        // Subsequent frame after warm-up uses the locked minimum offset
        val postWarmupSensor = 2_000_000_000L
        val postWarmupArrival = postWarmupSensor + trueClockBaseDiffNs + 90_000_000L // 90ms spike
        val calibratedCapture = calibrator.toElapsedRealtimeNs(postWarmupSensor, postWarmupArrival)
        assertEquals(postWarmupSensor + expectedMinOffsetNs, calibratedCapture)
        // Resulting age at arrival reflects the true ~85ms delivery delay rather than being hidden
        val measuredDelayMs = (postWarmupArrival - calibratedCapture) / 1_000_000L
        assertEquals(85L, measuredDelayMs)
        assertTrue(lastAligned > 0L)
    }

    @Test
    fun testClockOffsetCalibrator5thPercentileOnLargeWarmupWindow() {
        val calibrator = ClockOffsetCalibrator(warmupFrames = 30)
        calibrator.configureSource(ClockOffsetCalibrator.TIMESTAMP_SOURCE_UNKNOWN)

        val baseOffsetNs = 2_000_000_000L
        for (i in 0 until 30) {
            val rawSensor = 100_000_000L + i * 33_333_333L
            // Delays from 10ms to 39ms (1ms increments)
            val delayNs = (10L + i) * 1_000_000L
            val arrival = rawSensor + baseOffsetNs + delayNs
            calibrator.toElapsedRealtimeNs(rawSensor, arrival)
        }
        assertTrue(calibrator.isCalibrated)
        // For 30 samples, index = ((30 - 1) * 0.05).toInt() = 1 -> 11ms (near minimum, far below 24.5ms median)
        val expectedOffsetNs = baseOffsetNs + 11_000_000L
        assertEquals(expectedOffsetNs, calibrator.calibratedOffsetNs)
    }

    @Test
    fun testEndToEndSensorTimestampPropagationToFaceStatePoseTimestampNs() {
        var fakeClockNs = 10_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        val pipeline = SignalPipeline()
        val probe = LatencyProbe()

        var providerListener: ((FaceGeometrySnapshot) -> Unit)? = null
        val fakeProvider = object : FaceGeometryProvider {
            override val modelSha256: String = FaceGeometrySnapshot.EXPECTED_MODEL_SHA256
            override fun start(config: ProviderConfig) {}
            override fun stop() {}
            override fun setListener(listener: (FaceGeometrySnapshot) -> Unit) {
                providerListener = listener
            }
            override fun submit(frame: AnalysisFrame) {
                // Simulate 4ms queue, 6ms conversion, 12ms inference
                val analysisStart = frame.submittedAtNs + 4_000_000L
                val inferenceDone = analysisStart + 18_000_000L
                fakeClockNs = inferenceDone

                val identity = FloatArray(16) { if (it % 5 == 0) 1f else 0f }
                identity[14] = -0.5f
                val snapshot = FaceGeometrySnapshot.create(
                    timestampNs = frame.timestampNs,
                    imageWidth = 640,
                    imageHeight = 480,
                    rotationDegrees = frame.rotationDegrees,
                    isFrontCamera = frame.isFrontCamera,
                    isTracking = true,
                    landmarks = FloatArray(478 * 3),
                    rawPoseMatrix = identity,
                    blendshapes = mapOf("jawOpen" to 0.2f),
                    sensorTimestampNs = frame.sensorTimestampNs,
                    analysisStartNs = analysisStart,
                    inferenceDoneNs = inferenceDone,
                    queueMs = 4.0f,
                    conversionMs = 6.0f,
                    inferenceMs = 12.0f
                )
                frame.imageProxy?.close()
                providerListener?.invoke(snapshot)
            }
        }

        val session = CameraSession(
            provider = fakeProvider,
            signalPipeline = pipeline,
            governor = gov,
            latencyProbe = probe
        )
        session.configureTimestampSource(ClockOffsetCalibrator.TIMESTAMP_SOURCE_REALTIME)
        session.startProvider()

        val sensorCaptureNs = 9_980_000_000L // captured 20ms before arrival at 10.000s
        val proxy = createMockImageProxy(sensorCaptureNs)

        val submitted = session.submitAnalysisFrame(
            imageProxy = proxy,
            arrivalElapsedRealtimeNs = 10_000_000_000L
        )
        assertTrue(submitted)

        val state = pipeline.getLatestState()
        assertEquals(
            "FaceState.poseTimestampNs must equal calibrated ImageProxy sensor timestamp",
            sensorCaptureNs,
            state.poseTimestampNs
        )
        assertEquals(4.0f, state.queueMs, 0.001f)
        assertEquals(6.0f, state.conversionMs, 0.001f)
        assertEquals(12.0f, state.inferenceMs, 0.001f)

        // Advance past easeInMs so lifecycle reaches TRACKING, then test getPredictedState
        val sensorCapture2Ns = sensorCaptureNs + 120_000_000L
        fakeClockNs = 10_140_000_000L
        session.submitAnalysisFrame(
            imageProxy = createMockImageProxy(sensorCapture2Ns),
            arrivalElapsedRealtimeNs = 10_120_000_000L
        )
        val trackingState = pipeline.getLatestState()
        assertEquals(FaceLifecycle.State.TRACKING, trackingState.lifecycle)
        assertEquals(sensorCapture2Ns, trackingState.poseTimestampNs)

        // Predicted state at render time updates timestampNs but preserves poseTimestampNs
        val renderTimeNs = sensorCapture2Ns + 35_000_000L
        val predicted = pipeline.getPredictedState(renderTimeNs)
        assertEquals(renderTimeNs, predicted.timestampNs)
        assertEquals(
            "getPredictedState must preserve underlying capture poseTimestampNs",
            sensorCapture2Ns,
            predicted.poseTimestampNs
        )
    }

    @Test
    fun testRollingWindowPercentilesEvictionTrackingFilter2HzHudAndCsvDump() {
        val probe = LatencyProbe(
            windowDurationNs = 5_000_000_000L,
            hudRefreshIntervalNs = 500_000_000L
        )

        var nowNs = 1_000_000_000L

        // 1. Non-TRACKING frames (NO_FACE, ACQUIRING, HOLD, FADING) must NOT record ageAtRender samples
        for (nonTrackingState in listOf(
            FaceLifecycle.State.NO_FACE,
            FaceLifecycle.State.ACQUIRING,
            FaceLifecycle.State.HOLD,
            FaceLifecycle.State.FADING
        )) {
            val fs = FaceState(
                timestampNs = nowNs,
                poseTimestampNs = nowNs - 50_000_000L,
                lifecycle = nonTrackingState,
                isVisible = nonTrackingState != FaceLifecycle.State.NO_FACE
            )
            probe.recordRenderFrame(renderTimestampNs = nowNs, faceState = fs)
            nowNs += 16_666_667L
        }
        assertEquals("Non-TRACKING states must not record ageAtRender samples", 0, probe.ageSampleCount())

        // 2. Feed 20 TRACKING frames at t = 2.0s .. 2.32s with known high age (100ms)
        nowNs = 2_000_000_000L
        for (i in 0 until 20) {
            val captureNs = nowNs - 100_000_000L // 100ms age
            val fs = FaceState(
                timestampNs = nowNs,
                poseTimestampNs = captureNs,
                lifecycle = FaceLifecycle.State.TRACKING,
                isVisible = true,
                queueMs = 2.0f,
                conversionMs = 5.0f,
                inferenceMs = 15.0f
            )
            probe.recordCameraFrame(nowNs)
            probe.recordRenderFrame(renderTimestampNs = nowNs, faceState = fs)
            nowNs += 16_666_667L
        }
        probe.refreshMetrics(nowNs)
        assertEquals(20, probe.ageSampleCount())
        assertEquals(100.0f, probe.ageP50Ms, 0.5f)
        assertEquals(100.0f, probe.ageP95Ms, 0.5f)

        val initialHud = probe.cachedHudLine
        assertTrue(initialHud.contains("100.0/100.0ms"))

        // 3. Verify ~2 Hz HUD caching: recording a new frame 100ms later (< 500ms refresh interval)
        // must NOT re-sort or re-format cachedHudLine
        val earlyNs = nowNs + 100_000_000L
        val fastState = FaceState(
            timestampNs = earlyNs,
            poseTimestampNs = earlyNs - 20_000_000L, // 20ms age
            lifecycle = FaceLifecycle.State.TRACKING,
            isVisible = true
        )
        probe.recordRenderFrame(renderTimestampNs = earlyNs, faceState = fastState)
        assertSame(
            "HUD string reference must remain cached between ~2 Hz refreshes",
            initialHud,
            probe.cachedHudLine
        )

        // 4. Advance time to t = 8.0s (> 5 seconds after the 2.0s..2.32s batch) so the 100ms samples expire,
        // and feed 100 TRACKING samples with ages 11ms .. 110ms (1ms increments)
        nowNs = 8_000_000_000L
        for (i in 1..100) {
            val ageMs = (10 + i).toFloat() // 11ms .. 110ms
            val captureNs = nowNs - (ageMs * 1_000_000f).toLong()
            val snapshot = FaceGeometrySnapshot.empty(timestampNs = captureNs).copy(
                inferenceDoneNs = nowNs,
                queueMs = 1.5f,
                conversionMs = 4.5f,
                inferenceMs = 14.0f
            )
            probe.recordCameraFrame(nowNs)
            probe.recordTrackerSnapshot(snapshot)
            val fs = FaceState(
                timestampNs = nowNs,
                poseTimestampNs = captureNs,
                lifecycle = FaceLifecycle.State.TRACKING,
                isVisible = true,
                queueMs = 1.5f,
                conversionMs = 4.5f,
                inferenceMs = 14.0f
            )
            probe.recordRenderFrame(renderTimestampNs = nowNs, faceState = fs)
            nowNs += 16_666_667L // ~60 fps
        }

        probe.refreshMetrics(nowNs - 16_666_667L)
        assertEquals("Expired samples older than 5s must be evicted", 100, probe.ageSampleCount())
        // For 100 samples [11..110]: p50Idx = (99 * 0.50).toInt() = 49 -> 60ms; p95Idx = (99 * 0.95).toInt() = 94 -> 105ms
        assertEquals(60.0f, probe.ageP50Ms, 1.0f)
        assertEquals(105.0f, probe.ageP95Ms, 1.0f)
        assertEquals(14.0f, probe.rollingInferenceMs, 0.5f)
        assertTrue("Render FPS should be ~60 Hz", probe.renderFps in 58.0f..62.0f)
        assertTrue(probe.cachedHudLine.contains("age p50/p95: 60.0/105.0ms"))

        // 5. Verify on-demand CSV dump format and columns (including pendingWaitMs, warmup, govReason, idleGapPendingMs, idleGapNoPendingMs, pumpDelayMs, swapMs, convMode, per-stage timings & source size/rotation)
        val csv = probe.dumpCsv()
        val lines = csv.trim().lines()
        assertEquals(
            "timestamp,ageMs,conversionMs,pendingWaitMs,inferenceMs,queueMs,cameraFps,trackerFps,renderFps,predictionOn,predErrPosMm,predErrRotDeg,fovYDeg,fovSource,focalLengthMm,scalerCropRegion,depthBits,warmup,govReason,idleGapPendingMs,idleGapNoPendingMs,pumpDelayMs,swapMs,convMode,unpackMs,yuvToRgbMs,bitmapCopyMs,rotateScaleMs,mpImageBuildMs,srcWidth,srcHeight,srcRotationDeg",
            lines.first()
        )
        assertTrue("CSV must contain all recorded frames", lines.size > 100)
        val lastCols = lines.last().split(",")
        assertEquals(32, lastCols.size)
        assertEquals("false", lastCols[9]) // predictionOn = false
        assertEquals("false", lastCols[17]) // warmup = false
        assertEquals("none", lastCols[18]) // govReason = none
        assertEquals("FAST", lastCols[23]) // convMode = FAST
    }

    @Test
    fun testGovernorContinuousRenderModeGuardAndGlSurfaceViewChoreographerLifecycle() {
        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        var tMs = 10_000L
        gov.reset(nowMs = tMs)

        // 1. In non-continuous render mode (RENDERMODE_WHEN_DIRTY), slow frame-to-frame intervals (e.g. 66.7ms / 15fps)
        // must be ignored so a slow tracker cannot cause false tier demotions!
        tMs += PerformanceGovernor.DWELL_WINDOW_MS + 200L
        for (i in 0 until 40) {
            gov.recordFrameDuration(
                durationMs = 66.7f,
                nowMs = tMs + i * 67L,
                resultAgeAtRenderMs = 45.0f,
                currentTrackerHz = 15.0f,
                drawCpuDurationMs = 2.4f,
                isContinuousRender = false
            )
        }
        assertEquals(
            "Non-continuous render mode must NOT demote on slow inter-frame intervals",
            30,
            gov.targetTrackerCadenceHz
        )
        assertEquals(PerformanceGovernor.Tier.A, gov.currentTier)
        assertEquals("drawCpuDurationMs must be kept as debug stat", 2.4f, gov.lastDrawCpuDurationMs, 0.01f)

        // 2. In continuous render mode (RENDERMODE_CONTINUOUSLY), sustained >33ms frame-to-frame intervals
        // DO trigger degradation after the 1.5s hysteresis window
        tMs += 5_000L
        for (i in 0 until 40) {
            gov.recordFrameDuration(
                durationMs = 45.0f,
                nowMs = tMs + i * 50L,
                resultAgeAtRenderMs = 45.0f,
                currentTrackerHz = 30.0f,
                drawCpuDurationMs = 3.1f,
                isContinuousRender = true
            )
        }
        assertEquals(24, gov.targetTrackerCadenceHz)
        assertEquals(3.1f, gov.lastDrawCpuDurationMs, 0.01f)

        // 3. Verify XelvicGlSurfaceView logs startup renderMode and manages Choreographer registration
        val context = ApplicationProvider.getApplicationContext<Context>()
        val glView = XelvicGlSurfaceView(context)
        assertTrue("Startup renderMode must be logged", glView.startupRenderModeLogged)
        assertTrue("Choreographer must be registered at startup on main thread", glView.isChoreographerRegistered)

        glView.onPause()
        assertFalse("Choreographer must be unregistered on pause", glView.isChoreographerRegistered)

        glView.onResume()
        assertTrue("Choreographer must be re-registered on resume", glView.isChoreographerRegistered)

        glView.release()
        assertFalse("Choreographer must be unregistered on release", glView.isChoreographerRegistered)
    }

    private fun createMockImageProxy(timestampNs: Long): ImageProxy {
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

        return Proxy.newProxyInstance(
            ImageProxy::class.java.classLoader,
            arrayOf(ImageProxy::class.java)
        ) { _, method, _ ->
            when (method.name) {
                "getImageInfo" -> imageInfo
                "getWidth" -> 640
                "getHeight" -> 480
                "close" -> Unit
                else -> null
            }
        } as ImageProxy
    }
}
