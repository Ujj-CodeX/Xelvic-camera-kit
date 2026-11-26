package com.example

import com.example.engine.debug.LatencyProbe
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.LatencyCompensator
import com.example.engine.signal.PoseDecomposer
import com.example.engine.signal.SignalPipeline
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.*

/**
 * Unit tests for Chunk: prediction wiring:
 * 1. Constant velocity extrapolation (X/Y full gain, Z reduced gain, scale & blendshapes unpredicted,
 *    zero per-frame allocation in `consumeLatestPredictedState`).
 * 2. Horizon config (default 50 ms, max 80 ms clamp) & smooth taper gain on extrapolation delta.
 * 3. Stop-after-motion (zero overshoot) & oscillatory velocity-noise blend-down.
 * 4. Quaternion hemisphere continuity across antipodal sign flips (`dot >= 0`).
 * 5. Lifecycle event delivery (`faceAcquired`, `faceLost`) via `consumeLatestPredictedState`.
 * 6. Velocity reset on `> 150 ms` sample gap and on `ACQUIRING -> TRACKING` transition, plus speed clamps.
 * 7. Extrapolation delta from `poseTimestampNs` + `extraLeadMs`, and prediction-error metric logged into
 *    the `LatencyProbe` CSV ring buffer while `ageAtRender` remains `renderElapsedNs - poseTimestampNs`.
 */
class PredictionWiringUnitTest {

    @Test
    fun testConstantVelocityExtrapolationZGainAndZeroAllocationReuse() {
        val comp = LatencyCompensator(
            maxHorizonSec = 0.050f,
            zTranslationGain = 0.25f
        )

        val qIdentity = floatArrayOf(0f, 0f, 0f, 1f)
        val t0 = 1_000_000_000L
        val dtNs = 33_333_333L // ~30 fps (33.33 ms)

        // Feed steady motion: vx = 0.30 m/s, vy = -0.15 m/s, vz = 0.40 m/s
        for (i in 0..4) {
            val ts = t0 + i * dtNs
            val sec = (i * dtNs).toFloat() / 1_000_000_000f
            val pos = floatArrayOf(0.30f * sec, -0.15f * sec, -0.50f + 0.40f * sec)
            val scale = floatArrayOf(1.0f + 0.1f * i, 1.0f, 1.0f)
            comp.update(pos, qIdentity, ts, scale)
        }

        val lastCaptureNs = t0 + 4 * dtNs
        val lastSec = (4 * dtNs).toFloat() / 1_000_000_000f
        val baseX = 0.30f * lastSec
        val baseY = -0.15f * lastSec
        val baseZ = -0.50f + 0.40f * lastSec

        // Predict 25 ms ahead of lastCaptureNs
        val renderNs = lastCaptureNs + 25_000_000L
        val outT = FloatArray(3)
        val outQ = FloatArray(4)
        comp.predict(renderNs, outT, outQ, lastCaptureNs)

        val deltaX = outT[0] - baseX
        val deltaY = outT[1] - baseY
        val deltaZ = outT[2] - baseZ

        assertTrue("Predicted X must advance in +X direction", deltaX > 0.004f)
        assertTrue("Predicted Y must advance in -Y direction", deltaY < -0.002f)
        assertTrue("Predicted Z must advance in +Z direction", deltaZ > 0f)

        // Verify Z translation uses reduced gain (0.25) relative to X/Y full gain:
        // Since vz (0.40) = (4/3) * vx (0.30), with zGain = 0.25: deltaZ / deltaX == (0.40 * 0.25) / 0.30 == 1/3
        val expectedRatio = (0.40f * 0.25f) / 0.30f
        assertEquals(
            "Z depth translation must be scaled by zTranslationGain (0.25)",
            expectedRatio,
            deltaZ / deltaX,
            0.03f
        )

        // Verify SignalPipeline.consumeLatestPredictedState reuses preallocated buffers with zero allocation
        // and does NOT predict scale or blendshapes
        val pipeline = SignalPipeline(latencyCompensator = comp)
        var simTimeNs = 2_000_000_000L
        // Reach TRACKING (past 100ms easeInMs)
        for (i in 0..5) {
            val mat = PoseDecomposer.recompose(
                floatArrayOf(0.01f * i, 0f, -0.5f),
                qIdentity,
                floatArrayOf(1.25f, 1.30f, 1.35f)
            )
            val snap = createTrackingSnapshot(
                timestampNs = simTimeNs,
                poseMatrix = mat,
                blendshapes = mapOf("jawOpen" to (0.1f * i))
            )
            pipeline.processSnapshot(snap, nowNs = simTimeNs)
            simTimeNs += 30_000_000L
        }

        val latestPublished = pipeline.getLatestState()
        assertEquals(FaceLifecycle.State.TRACKING, latestPublished.lifecycle)

        val vsync1 = pipeline.consumeLatestPredictedState(latestPublished.poseTimestampNs + 16_000_000L, true)
        val vsync1PoseRef = vsync1.poseMatrix
        val vsync1TransRef = vsync1.translation
        val vsync1RotRef = vsync1.rotation
        val vsync1ScaleRef = vsync1.scale
        val vsync1LandmarksRef = vsync1.landmarks
        val vsync1BlendshapesRef = vsync1.blendshapes

        val vsync2 = pipeline.consumeLatestPredictedState(latestPublished.poseTimestampNs + 32_000_000L, true)

        // Zero-allocation verification: identical instance and array references across vsync ticks
        assertSame("consumeLatestPredictedState must reuse the same FaceState instance", vsync1, vsync2)
        assertSame("poseMatrix FloatArray must be reused", vsync1PoseRef, vsync2.poseMatrix)
        assertSame("translation FloatArray must be reused", vsync1TransRef, vsync2.translation)
        assertSame("rotation FloatArray must be reused", vsync1RotRef, vsync2.rotation)
        assertSame("scale FloatArray must be reused", vsync1ScaleRef, vsync2.scale)
        assertSame("landmarks reference must not be copied", vsync1LandmarksRef, vsync2.landmarks)
        assertSame("blendshapes map reference must not be copied", vsync1BlendshapesRef, vsync2.blendshapes)

        // Scale and blendshapes must match the latest filtered state without extrapolation
        assertArrayEquals(latestPublished.scale, vsync2.scale, 1e-6f)
        assertEquals(latestPublished.blendshapes["jawOpen"]!!, vsync2.blendshapes["jawOpen"]!!, 1e-6f)
    }

    @Test
    fun testHorizonConfig50msDefault80msMaxAndSmoothTaper() {
        val comp = LatencyCompensator()
        assertEquals(0.050f, comp.maxHorizonSec, 1e-6f)

        // Attempting to set horizon > 80ms clamps to 80ms (0.080f)
        comp.maxHorizonSec = 0.200f
        assertEquals(0.080f, comp.maxHorizonSec, 1e-6f)

        comp.maxHorizonSec = 0.050f
        val qIdentity = floatArrayOf(0f, 0f, 0f, 1f)
        val t0 = 5_000_000_000L
        for (i in 0..4) {
            val ts = t0 + i * 25_000_000L
            val x = 0.40f * ((i * 25_000_000L).toFloat() / 1_000_000_000f)
            comp.update(floatArrayOf(x, 0f, -0.5f), qIdentity, ts)
        }
        val lastTs = t0 + 4 * 25_000_000L

        val out50ms = FloatArray(3)
        val out80ms = FloatArray(3)
        val out500ms = FloatArray(3)
        val dummyQ = FloatArray(4)

        comp.predict(lastTs + 50_000_000L, out50ms, dummyQ, lastTs)
        comp.predict(lastTs + 80_000_000L, out80ms, dummyQ, lastTs)
        comp.predict(lastTs + 500_000_000L, out500ms, dummyQ, lastTs)

        // Beyond the 50ms horizon, extrapolation is clamped so 80ms and 500ms equal 50ms
        assertEquals(out50ms[0], out80ms[0], 1e-6f)
        assertEquals(out50ms[0], out500ms[0], 1e-6f)

        // Verify smooth taper near horizon: gain at 50ms is tapered (0.65 * linear) so it does not
        // advance by the full raw untapered delta
        val baseAtLast = 0.40f * 0.100f
        val untaperedAdvance = 0.40f * 0.050f
        val actualAdvance = out50ms[0] - baseAtLast
        assertTrue("Actual advance must be positive", actualAdvance > 0.008f)
        assertTrue(
            "Horizon taper must smoothly reduce gain below raw untapered linear advance",
            actualAdvance < untaperedAdvance
        )
    }

    @Test
    fun testStopAfterMotionNoOvershootAndVelocityNoiseBlendDown() {
        val comp = LatencyCompensator(maxHorizonSec = 0.050f)
        val qIdentity = floatArrayOf(0f, 0f, 0f, 1f)
        var ts = 10_000_000_000L

        // 1. Move steadily to x = 0.10m over 4 frames (25ms steps -> 1.0 m/s)
        for (i in 0..4) {
            val x = 0.025f * i
            comp.update(floatArrayOf(x, 0f, -0.5f), qIdentity, ts)
            ts += 25_000_000L
        }

        // 2. Sudden stop at x = 0.10m on the next frame
        val stopX = 0.10f
        comp.update(floatArrayOf(stopX, 0f, -0.5f), qIdentity, ts)

        // Predict 35ms ahead after stopping: must NOT overshoot past stopX = 0.10f!
        val predT = FloatArray(3)
        val predQ = FloatArray(4)
        comp.predict(ts + 35_000_000L, predT, predQ, ts)
        assertEquals(
            "Stop-after-motion must zero velocity within deadzone with zero overshoot",
            stopX,
            predT[0],
            1e-5f
        )

        // 3. Oscillatory velocity noise (+1mm, -1mm, +1mm at 30Hz -> reversal every frame)
        // must blend down velocity rather than amplifying jitter
        for (i in 1..6) {
            ts += 33_333_333L
            val jitterX = stopX + (if (i % 2 == 0) 0.001f else -0.001f)
            comp.update(floatArrayOf(jitterX, 0f, -0.5f), qIdentity, ts)
        }
        val lastJitterX = stopX + 0.001f
        comp.predict(ts + 40_000_000L, predT, predQ, ts)
        assertTrue(
            "High-frequency reversal noise must be blended down (< 0.5mm extrapolation)",
            abs(predT[0] - lastJitterX) < 0.0005f
        )
    }

    @Test
    fun testQuaternionHemisphereContinuity() {
        val comp = LatencyCompensator(maxHorizonSec = 0.050f)
        val t = floatArrayOf(0f, 0f, -0.5f)
        var ts = 20_000_000_000L

        // Rotation around Y axis by 10 deg, then 12 deg with antipodal negated quaternion (-q)
        val q10 = eulerYawToQuaternion(10f)
        val q12 = eulerYawToQuaternion(12f)
        val q12Negated = floatArrayOf(-q12[0], -q12[1], -q12[2], -q12[3])

        comp.update(t, q10, ts)
        ts += 33_333_333L
        comp.update(t, q12Negated, ts)

        val predT = FloatArray(3)
        val predQ = FloatArray(4)
        comp.predict(ts + 25_000_000L, predT, predQ, ts)

        // Predicted quaternion must have unit norm and stay in the same hemisphere as q10 (dot > 0.98)
        val norm = sqrt(predQ[0] * predQ[0] + predQ[1] * predQ[1] + predQ[2] * predQ[2] + predQ[3] * predQ[3])
        assertEquals(1.0f, norm, 1e-4f)
        val dotWithQ10 = predQ[0] * q10[0] + predQ[1] * q10[1] + predQ[2] * q10[2] + predQ[3] * q10[3]
        assertTrue(
            "Quaternion hemisphere continuity must prevent sign flip on antipodal input",
            dotWithQ10 > 0.98f
        )
    }

    @Test
    fun testLifecycleEventDeliveryThroughConsumeLatestPredictedState() {
        val pipeline = SignalPipeline()
        val identity = FloatArray(16) { if (it % 5 == 0) 1f else 0f }
        identity[14] = -0.5f

        var ts = 30_000_000_000L
        // First valid pose -> triggers "faceAcquired"
        val snap1 = createTrackingSnapshot(ts, identity)
        pipeline.processSnapshot(snap1, nowNs = ts)

        // Subsequent snapshot before vsync consumes the event -> event must remain latched
        ts += 16_000_000L
        val snap2 = createTrackingSnapshot(ts, identity)
        pipeline.processSnapshot(snap2, nowNs = ts)

        // Vsync 1 consumes the latched "faceAcquired" event
        val stateVsync1 = pipeline.consumeLatestPredictedState(ts + 5_000_000L, true)
        assertEquals("faceAcquired must be delivered on vsync", "faceAcquired", stateVsync1.triggeredEvent)

        // Vsync 2 does not re-deliver the already consumed event
        val stateVsync2 = pipeline.consumeLatestPredictedState(ts + 21_000_000L, true)
        assertNull("Consumed event must not fire twice", stateVsync2.triggeredEvent)

        // Advance past easeInMs (100ms) into TRACKING
        ts += 120_000_000L
        pipeline.processSnapshot(createTrackingSnapshot(ts, identity), nowNs = ts)
        assertEquals(FaceLifecycle.State.TRACKING, pipeline.getLatestState().lifecycle)

        // Lose face and advance past holdMs (200ms) + fadeMs (250ms) -> triggers "faceLost"
        ts += 50_000_000L
        pipeline.processSnapshot(FaceGeometrySnapshot.empty(ts), nowNs = ts) // enters HOLD
        ts += 210_000_000L
        pipeline.processSnapshot(FaceGeometrySnapshot.empty(ts), nowNs = ts) // enters FADING
        ts += 260_000_000L
        pipeline.processSnapshot(FaceGeometrySnapshot.empty(ts), nowNs = ts) // enters NO_FACE + faceLost

        val stateLost = pipeline.consumeLatestPredictedState(ts + 10_000_000L, true)
        assertEquals("faceLost must be delivered on vsync", "faceLost", stateLost.triggeredEvent)
        val stateLostSecondVsync = pipeline.consumeLatestPredictedState(ts + 26_000_000L, true)
        assertNull("faceLost must not fire twice on consecutive vsyncs", stateLostSecondVsync.triggeredEvent)
    }

    @Test
    fun testVelocityResetOn150msGapAndAcquiringToTrackingAndSpeedClamps() {
        val comp = LatencyCompensator(
            maxHorizonSec = 0.050f,
            maxLinearSpeedMps = 1.5f,
            maxAngularSpeedRadPerSec = 5.0f
        )
        val qIdentity = floatArrayOf(0f, 0f, 0f, 1f)
        var ts = 40_000_000_000L

        // 1. Test linear speed clamp: 1.0m jump in 20ms = 50 m/s -> clamped to 1.5 m/s
        comp.update(floatArrayOf(0f, 0f, -0.5f), qIdentity, ts)
        ts += 20_000_000L
        comp.update(floatArrayOf(1.0f, 0f, -0.5f), qIdentity, ts)
        val linVel = comp.getLinearVelocity()
        assertTrue("Linear velocity must be clamped to maxLinearSpeedMps (1.5 m/s)", abs(linVel[0]) <= 1.5001f)

        // 2. Test > 150ms sample gap resets velocity to zero
        ts += 180_000_000L // 180ms gap (> 150ms threshold)
        comp.update(floatArrayOf(1.2f, 0f, -0.5f), qIdentity, ts)
        val velAfterGap = comp.getLinearVelocity()
        assertEquals("Velocity must reset to 0 after >150ms sample gap", 0f, velAfterGap[0], 1e-6f)

        // 3. Test ACQUIRING -> TRACKING resets velocity in SignalPipeline
        val pipeline = SignalPipeline()
        var pTs = 50_000_000_000L
        // Frame 1: enters ACQUIRING at x = 0.0
        pipeline.processSnapshot(
            createTrackingSnapshot(pTs, PoseDecomposer.recompose(floatArrayOf(0f, 0f, -0.5f), qIdentity, floatArrayOf(1f, 1f, 1f))),
            nowNs = pTs
        )
        assertEquals(FaceLifecycle.State.ACQUIRING, pipeline.getLatestState().lifecycle)

        // Frame 2 (50ms later, still ACQUIRING): moves to x = 0.20m
        pTs += 50_000_000L
        pipeline.processSnapshot(
            createTrackingSnapshot(pTs, PoseDecomposer.recompose(floatArrayOf(0.20f, 0f, -0.5f), qIdentity, floatArrayOf(1f, 1f, 1f))),
            nowNs = pTs
        )
        assertEquals(FaceLifecycle.State.ACQUIRING, pipeline.getLatestState().lifecycle)

        // Frame 3 (60ms later, total 110ms > 100ms easeInMs): transitions ACQUIRING -> TRACKING
        pTs += 60_000_000L
        val trackingState = pipeline.processSnapshot(
            createTrackingSnapshot(pTs, PoseDecomposer.recompose(floatArrayOf(0.40f, 0f, -0.5f), qIdentity, floatArrayOf(1f, 1f, 1f))),
            nowNs = pTs
        )
        assertEquals(FaceLifecycle.State.TRACKING, trackingState.lifecycle)
        assertEquals(
            "Velocity must be reset to zero on the ACQUIRING -> TRACKING transition frame",
            0f,
            trackingState.velocity[0],
            1e-6f
        )
    }

    @Test
    fun testDeltaFromPoseTimestampNsExtraLeadMsAndPredictionErrorInCsv() {
        val pipeline = SignalPipeline()
        val probe = LatencyProbe()
        val qIdentity = floatArrayOf(0f, 0f, 0f, 1f)

        var captureNs = 60_000_000_000L
        // Feed frames to reach TRACKING and establish steady +X velocity
        for (i in 0..6) {
            val x = 0.01f * i
            val mat = PoseDecomposer.recompose(floatArrayOf(x, 0f, -0.5f), qIdentity, floatArrayOf(1f, 1f, 1f))
            val snap = createTrackingSnapshot(captureNs, mat)
            // Call processSnapshot with a much later `nowNs` (e.g. +25ms) to verify prediction delta
            // is computed from `poseTimestampNs` (captureNs), NOT `nowNs`
            val state = pipeline.processSnapshot(snap, nowNs = captureNs + 25_000_000L)
            probe.recordTrackerSnapshot(snap, state.predErrPosMm, state.predErrRotDeg)
            if (i < 6) {
                captureNs += 30_000_000L
            }
        }

        val latest = pipeline.getLatestState()
        assertEquals(FaceLifecycle.State.TRACKING, latest.lifecycle)
        assertEquals(captureNs, latest.poseTimestampNs)

        // Render at exactly poseTimestampNs with extraLeadMs = 0 -> zero extrapolation delta
        pipeline.latencyCompensator.extraLeadMs = 0f
        val atCapture = pipeline.consumeLatestPredictedState(captureNs, true)
        val baseTx = atCapture.translation[0]
        assertEquals(latest.translation[0], baseTx, 1e-6f)

        // Add extraLeadMs = 20ms at the exact same renderTimestampNs = captureNs -> extrapolates 20ms ahead!
        pipeline.latencyCompensator.extraLeadMs = 20f
        val withExtraLead = pipeline.consumeLatestPredictedState(captureNs, true)
        assertTrue(
            "extraLeadMs must advance prediction target time even when renderElapsedNs == poseTimestampNs",
            withExtraLead.translation[0] > baseTx
        )

        // Verify ageAtRender in LatencyProbe remains renderElapsedNs - poseTimestampNs regardless of predictionOn
        val renderTimeNs = captureNs + 35_000_000L
        val predOnState = pipeline.consumeLatestPredictedState(renderTimeNs, true)
        probe.recordRenderFrame(renderTimeNs, predOnState, drawCpuMs = 1.2f, predictionOn = true)
        assertEquals(35.0f, probe.lastAgeAtRenderMs, 0.01f)

        val predOffState = pipeline.consumeLatestPredictedState(renderTimeNs, false)
        probe.recordRenderFrame(renderTimeNs + 16_000_000L, predOffState, drawCpuMs = 1.1f, predictionOn = false)
        assertEquals(51.0f, probe.lastAgeAtRenderMs, 0.01f)

        // Verify CSV dump records predictionOn (index 9 after pendingWaitMs), predErrPosMm (index 10), and predErrRotDeg
        val csvLines = probe.dumpCsv().trim().lines()
        assertEquals(LatencyProbe.CSV_HEADER, csvLines.first())
        val rowPredOn = csvLines[1].split(",")
        val rowPredOff = csvLines[2].split(",")
        assertEquals("true", rowPredOn[9])
        assertEquals("false", rowPredOff[9])
        val loggedPosErrMm = rowPredOn[10].toFloat()
        assertTrue("Prediction error in mm must be finite and >= 0", loggedPosErrMm.isFinite() && loggedPosErrMm >= 0f)
    }

    private fun eulerYawToQuaternion(yawDeg: Float): FloatArray {
        val halfRad = (yawDeg * PI.toFloat() / 180f) * 0.5f
        return floatArrayOf(0f, sin(halfRad), 0f, cos(halfRad))
    }

    private fun createTrackingSnapshot(
        timestampNs: Long,
        poseMatrix: FloatArray,
        blendshapes: Map<String, Float> = emptyMap()
    ): FaceGeometrySnapshot {
        val landmarks = FloatArray(478 * 3)
        // Populate cheek landmarks 234 and 454 so FW calculation succeeds
        landmarks[234 * 3] = 0.35f; landmarks[234 * 3 + 1] = 0.5f; landmarks[234 * 3 + 2] = 0f
        landmarks[454 * 3] = 0.65f; landmarks[454 * 3 + 1] = 0.5f; landmarks[454 * 3 + 2] = 0f
        return FaceGeometrySnapshot.create(
            timestampNs = timestampNs,
            imageWidth = 640,
            imageHeight = 480,
            rotationDegrees = 0,
            isFrontCamera = true,
            isTracking = true,
            landmarks = landmarks,
            rawPoseMatrix = poseMatrix,
            blendshapes = blendshapes,
            sensorTimestampNs = timestampNs
        )
    }
}
