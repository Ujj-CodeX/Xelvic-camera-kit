package com.example.engine.debug

import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.PoseDecomposer
import com.example.engine.signal.SignalPipeline
import kotlin.math.*

/**
 * ReplayHarness provides deterministic replay verification for the Signal layer (README Section 20 & 21).
 *
 * Validates:
 * 1. Jitter reduction: compares standard deviation of raw tracker pose vs One-Euro filtered pose.
 * 2. Fast motion response: verifies bounded latency extrapolation without detachment or overshoot.
 * 3. Lifecycle transitions: validates NO_FACE -> ACQUIRING -> TRACKING -> HOLD -> FADING -> NO_FACE and re-acquisition.
 */
object ReplayHarness {

    data class JitterResult(
        val rawStdDevMm: Float,
        val filteredStdDevMm: Float,
        val jitterReductionPercent: Float,
        val passed: Boolean
    )

    data class FastMotionResult(
        val maxLagMm: Float,
        val maxAngleErrorDeg: Float,
        val maxExtrapolationHorizonMs: Float,
        val passed: Boolean
    )

    data class LifecycleResult(
        val observedStates: List<FaceLifecycle.State>,
        val reacquisitionCanceledFade: Boolean,
        val resetTriggered: Boolean,
        val passed: Boolean
    )

    data class FullReplayReport(
        val jitter: JitterResult,
        val fastMotion: FastMotionResult,
        val lifecycle: LifecycleResult,
        val allPassed: Boolean
    )

    /**
     * Runs jitter comparison between raw tracker and filtered pose.
     */
    fun verifyJitter(numFrames: Int = 100, noiseStdDevMm: Float = 2.0f): JitterResult {
        val pipeline = SignalPipeline()
        val rawPositionsX = FloatArray(numFrames)
        val filteredPositionsX = FloatArray(numFrames)

        val baseMatrix = FloatArray(16) { if (it % 5 == 0) 1f else 0f }
        baseMatrix[12] = 0.0f  // x = 0m
        baseMatrix[13] = 0.0f  // y = 0m
        baseMatrix[14] = -0.5f // z = -0.5m

        // Pseudo-random deterministic noise sequence
        var seed = 123456789L
        fun nextGaussian(): Float {
            seed = (seed * 1103515245L + 12345L) and 0x7fffffffL
            val u1 = (seed.toFloat() / 0x7fffffffL).coerceIn(1e-6f, 1f)
            seed = (seed * 1103515245L + 12345L) and 0x7fffffffL
            val u2 = (seed.toFloat() / 0x7fffffffL).coerceIn(1e-6f, 1f)
            return (sqrt(-2.0 * ln(u1.toDouble())) * cos(2.0 * PI * u2.toDouble())).toFloat()
        }

        var timestampNs = 1_000_000_000L
        val dtNs = 33_333_333L // ~30 fps tracker rate

        for (i in 0 until numFrames) {
            val noiseX = nextGaussian() * (noiseStdDevMm / 1000f) // convert mm to meters
            val matrix = baseMatrix.clone()
            matrix[12] += noiseX

            val snapshot = FaceGeometrySnapshot.create(
                timestampNs = timestampNs,
                imageWidth = 640,
                imageHeight = 480,
                rotationDegrees = 0,
                isFrontCamera = true,
                isTracking = true,
                landmarks = FloatArray(478 * 3),
                rawPoseMatrix = matrix,
                blendshapes = mapOf("jawOpen" to 0.0f),
                modelSha256 = FaceGeometrySnapshot.EXPECTED_MODEL_SHA256,
                imageToViewTransform = FloatArray(9)
            )

            val state = pipeline.processSnapshot(snapshot, timestampNs)
            rawPositionsX[i] = matrix[12] * 1000f // mm
            filteredPositionsX[i] = state.translation[0] * 1000f // mm

            timestampNs += dtNs
        }

        // Calculate standard deviations (skipping initial 15 warm-up frames)
        val warmUp = 15
        val n = numFrames - warmUp
        var rawSum = 0f
        var filteredSum = 0f
        for (i in warmUp until numFrames) {
            rawSum += rawPositionsX[i]
            filteredSum += filteredPositionsX[i]
        }
        val rawMean = rawSum / n
        val filteredMean = filteredSum / n

        var rawVar = 0f
        var filteredVar = 0f
        for (i in warmUp until numFrames) {
            rawVar += (rawPositionsX[i] - rawMean) * (rawPositionsX[i] - rawMean)
            filteredVar += (filteredPositionsX[i] - filteredMean) * (filteredPositionsX[i] - filteredMean)
        }
        val rawStdDev = sqrt(rawVar / n)
        val filteredStdDev = sqrt(filteredVar / n)

        val reduction = if (rawStdDev > 0f) {
            ((rawStdDev - filteredStdDev) / rawStdDev) * 100f
        } else 0f

        // Pass condition: filtered jitter must be significantly lower than raw jitter
        val passed = filteredStdDev < rawStdDev && reduction > 40f

        return JitterResult(
            rawStdDevMm = rawStdDev,
            filteredStdDevMm = filteredStdDev,
            jitterReductionPercent = reduction,
            passed = passed
        )
    }

    /**
     * Runs fast-motion verification to confirm bounded latency compensation without detachment.
     */
    fun verifyFastMotion(degreesPerSec: Float = 300f, numFrames: Int = 30): FastMotionResult {
        val pipeline = SignalPipeline()
        var timestampNs = 1_000_000_000L
        val dtNs = 33_333_333L // 30 fps
        val displayDtNs = 16_666_666L // 60 fps render query (+16.6ms)

        var maxLagMm = 0f
        var maxAngleError = 0f

        val maxTargetYawDeg = 45f
        for (i in 0 until numFrames) {
            // Rapid yaw rotation and lateral translation sweep (0 to 45 deg)
            val timeSec = i * 0.0333f
            val currentYawDeg = (timeSec * degreesPerSec).coerceAtMost(maxTargetYawDeg)
            val yawRad = Math.toRadians(currentYawDeg.toDouble()).toFloat()
            val posX = sin(yawRad) * 0.15f // 15cm lateral movement

            val matrix = FloatArray(16) { if (it % 5 == 0) 1f else 0f }
            matrix[0] = cos(yawRad)
            matrix[2] = -sin(yawRad)
            matrix[8] = sin(yawRad)
            matrix[10] = cos(yawRad)
            matrix[12] = posX
            matrix[14] = -0.5f

            val snapshot = FaceGeometrySnapshot.create(
                timestampNs = timestampNs,
                imageWidth = 640,
                imageHeight = 480,
                rotationDegrees = 0,
                isFrontCamera = true,
                isTracking = true,
                landmarks = FloatArray(478 * 3),
                rawPoseMatrix = matrix,
                blendshapes = emptyMap(),
                modelSha256 = FaceGeometrySnapshot.EXPECTED_MODEL_SHA256,
                imageToViewTransform = FloatArray(9)
            )

            pipeline.processSnapshot(snapshot, timestampNs)

            // Query predicted state at display time (+16.6ms)
            val renderTimeNs = timestampNs + displayDtNs
            val predState = pipeline.getPredictedState(renderTimeNs)

            // Predict ground truth at renderTime (+16.6ms)
            val trueTimeSec = timeSec + 0.0166f
            val trueYawDeg = (trueTimeSec * degreesPerSec).coerceAtMost(maxTargetYawDeg)
            val trueYawRad = Math.toRadians(trueYawDeg.toDouble()).toFloat()
            val truePosX = sin(trueYawRad) * 0.15f

            val lagMm = abs(predState.translation[0] - truePosX) * 1000f
            if (i > 3 && lagMm > maxLagMm) maxLagMm = lagMm

            val angleErr = abs(predState.yawDeg - trueYawDeg)
            if (i > 3 && angleErr > maxAngleError) maxAngleError = angleErr

            timestampNs += dtNs
        }

        // Fast motion attachment check: verifies no visible detachment (lag < 80mm, angle error < 35 deg at 250 deg/s)
        val passed = maxLagMm < 80f && maxAngleError < 35f

        return FastMotionResult(
            maxLagMm = maxLagMm,
            maxAngleErrorDeg = maxAngleError,
            maxExtrapolationHorizonMs = pipeline.latencyCompensator.maxHorizonSec * 1000f,
            passed = passed
        )
    }

    /**
     * Runs full lifecycle sequence verification:
     * NO_FACE -> ACQUIRING -> TRACKING -> HOLD -> FADING -> NO_FACE and re-acquisition.
     */
    fun verifyLifecycle(): LifecycleResult {
        val pipeline = SignalPipeline()
        val observedStates = mutableListOf<FaceLifecycle.State>()

        var timestampNs = 1_000_000_000L
        val dtNs = 33_333_333L // 33ms step

        val validMatrix = FloatArray(16) { if (it % 5 == 0) 1f else 0f }

        fun feed(hasFace: Boolean) {
            val snapshot = FaceGeometrySnapshot.create(
                timestampNs = timestampNs,
                imageWidth = 640,
                imageHeight = 480,
                rotationDegrees = 0,
                isFrontCamera = true,
                isTracking = hasFace,
                landmarks = if (hasFace) FloatArray(478 * 3) else null,
                rawPoseMatrix = if (hasFace) validMatrix else null,
                blendshapes = emptyMap(),
                modelSha256 = FaceGeometrySnapshot.EXPECTED_MODEL_SHA256,
                imageToViewTransform = FloatArray(9)
            )
            val state = pipeline.processSnapshot(snapshot, timestampNs)
            observedStates.add(state.lifecycle)
            timestampNs += dtNs
        }

        // 1. Initial state NO_FACE
        observedStates.add(pipeline.getLatestState().lifecycle)

        // 2. Face appears -> ACQUIRING
        feed(true)

        // 3. Keep tracking past easeInMs (100ms = 3 frames) -> TRACKING
        feed(true); feed(true); feed(true); feed(true)

        // 4. Face disappears -> enters HOLD (holdMs = 200ms = 6 frames)
        feed(false); feed(false); feed(false); feed(false); feed(false); feed(false); feed(false)

        // 5. Continues into FADING (fadeMs = 250ms = 8 frames)
        feed(false); feed(false); feed(false); feed(false)

        // 6. Re-acquire during fade!
        val stateDuringFade = pipeline.getLatestState().lifecycle
        feed(true)
        val stateAfterReacquire = pipeline.getLatestState().lifecycle
        val reacquireSuccess = stateDuringFade == FaceLifecycle.State.FADING &&
                stateAfterReacquire == FaceLifecycle.State.TRACKING &&
                pipeline.getLatestState().fadeWeight == 1.0f

        // 7. Let face disappear completely to NO_FACE
        for (i in 0 until 18) feed(false) // 18 * 33ms = ~600ms > hold(200ms) + fade(250ms)
        val finalState = pipeline.getLatestState().lifecycle
        val finalFade = pipeline.getLatestState().fadeWeight

        val passed = observedStates.contains(FaceLifecycle.State.ACQUIRING) &&
                observedStates.contains(FaceLifecycle.State.TRACKING) &&
                observedStates.contains(FaceLifecycle.State.HOLD) &&
                observedStates.contains(FaceLifecycle.State.FADING) &&
                finalState == FaceLifecycle.State.NO_FACE &&
                finalFade == 0.0f &&
                reacquireSuccess

        return LifecycleResult(
            observedStates = observedStates.distinct(),
            reacquisitionCanceledFade = reacquireSuccess,
            resetTriggered = finalState == FaceLifecycle.State.NO_FACE,
            passed = passed
        )
    }

    /**
     * Executes the complete Step 2 Replay verification suite.
     */
    fun runFullVerification(): FullReplayReport {
        val jitter = verifyJitter()
        val fastMotion = verifyFastMotion()
        val lifecycle = verifyLifecycle()
        val allPassed = jitter.passed && fastMotion.passed && lifecycle.passed
        return FullReplayReport(
            jitter = jitter,
            fastMotion = fastMotion,
            lifecycle = lifecycle,
            allPassed = allPassed
        )
    }
}
