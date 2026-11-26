package com.example.engine.signal

import java.util.Arrays
import kotlin.math.*

/**
 * LatencyCompensator extrapolates filtered rigid pose from tracker capture timestamp
 * (`poseTimestampNs`) to display render time (`FrameClock.elapsedRealtimeNs()`) over a strictly
 * bounded, monotonic horizon (README Section 9 & AUDIT.md Section 3.2).
 *
 * Design & guarantees:
 * 1. Clock base: Both `renderElapsedNs` and `poseTimestampNs` are in `SystemClock.elapsedRealtimeNanos()`
 *    domain (`FrameClock.elapsedRealtimeNs()`). Extrapolation delta is:
 *    `delta = clamp((renderElapsedNs + extraLeadNs) - poseTimestampNs, 0, horizon)`.
 * 2. Horizon config: default 50 ms (`0.050f`), maximum 80 ms (`0.080f`).
 * 3. Smooth horizon taper: applied directly as a smooth gain on `delta` (`u = delta / horizon`,
 *    `taperGain = 1f - 0.35f * u * u`), not a separate dt since the last tracker update.
 * 4. Velocity: computed from the FILTERED pose using capture-timestamp (`poseTimestampNs`) differences.
 *    Resets velocity to zero when consecutive sample gap > 150 ms (`0.150f` s) or on
 *    `ACQUIRING -> TRACKING` transition (`resetVelocity = true`). Enforces configurable
 *    `maxLinearSpeedMps` and `maxAngularSpeedRadPerSec` clamps, velocity-noise blend-down,
 *    and stop-after-motion anti-overshoot.
 * 5. Selective prediction: predicts rigid pose `x`/`y` translation and `rotation` at full gain,
 *    applies reduced configurable gain on depth (`z`) translation (`zTranslationGain`), and
 *    does NOT predict `scale` or `blendshapes`.
 * 6. Prediction-error metric: when a new tracker sample arrives, predicts the pose for that
 *    sample's capture timestamp using the previous state and compares with the newly measured
 *    filtered pose (`lastPredictionErrorPosMm`, `lastPredictionErrorRotDeg`).
 */
class LatencyCompensator(
    maxHorizonSec: Float = DEFAULT_HORIZON_SEC,
    var extraLeadMs: Float = 0f,
    var zTranslationGain: Float = DEFAULT_Z_TRANSLATION_GAIN,
    var maxLinearSpeedMps: Float = DEFAULT_MAX_LINEAR_SPEED_MPS,
    var maxAngularSpeedRadPerSec: Float = DEFAULT_MAX_ANGULAR_SPEED_RAD_S
) {
    companion object {
        const val DEFAULT_HORIZON_SEC = 0.050f // 50 ms default
        const val MAX_HORIZON_SEC = 0.080f     // 80 ms hard ceiling
        const val MIN_SAMPLE_DT_SEC = 0.001f   // 1 ms minimum step
        const val MAX_SAMPLE_GAP_SEC = 0.150f  // 150 ms max gap before velocity reset

        const val DEFAULT_Z_TRANSLATION_GAIN = 0.25f
        const val DEFAULT_MAX_LINEAR_SPEED_MPS = 2.0f
        const val DEFAULT_MAX_ANGULAR_SPEED_RAD_S = 10.0f

        // Deadzones below which motion is treated as stationary/noise (prevents jitter & overshoot on stop)
        const val LINEAR_DEADZONE_MPS = 0.008f
        const val ANGULAR_DEADZONE_RAD_S = 0.08f
    }

    var maxHorizonSec: Float = maxHorizonSec.coerceIn(0f, MAX_HORIZON_SEC)
        set(value) {
            field = value.coerceIn(0f, MAX_HORIZON_SEC)
        }

    private val lock = Any()

    private val lastT = FloatArray(3)
    private val lastQ = FloatArray(4) { if (it == 3) 1f else 0f }
    private val lastScale = floatArrayOf(1f, 1f, 1f)
    private var lastTimestampNs: Long = 0L

    // Linear velocity (meters/sec) estimated from filtered pose
    private val velT = FloatArray(3)

    // Angular velocity axis-angle (unit axis + speed in rad/sec) and vector [wx, wy, wz]
    private val angularVelAxis = FloatArray(3)
    private var angularSpeedRadPerSec: Float = 0f
    private val angularVelVec = FloatArray(3)

    // Confidence blend factor in [0, 1] that blends down velocity under oscillatory noise
    private var velocityBlendFactor: Float = 1f

    // Preallocated scratch buffers for prediction-error evaluation on update()
    private val errPredT = FloatArray(3)
    private val errPredQ = FloatArray(4) { if (it == 3) 1f else 0f }

    @Volatile
    var lastPredictionErrorPosMm: Float = 0f
        private set

    @Volatile
    var lastPredictionErrorRotDeg: Float = 0f
        private set

    @Volatile
    var hasPredictionErrorSample: Boolean = false
        private set

    fun reset() {
        synchronized(lock) {
            lastTimestampNs = 0L
            Arrays.fill(velT, 0f)
            Arrays.fill(angularVelAxis, 0f)
            Arrays.fill(angularVelVec, 0f)
            angularSpeedRadPerSec = 0f
            velocityBlendFactor = 1f
            lastQ[0] = 0f; lastQ[1] = 0f; lastQ[2] = 0f; lastQ[3] = 1f
            lastPredictionErrorPosMm = 0f
            lastPredictionErrorRotDeg = 0f
            hasPredictionErrorSample = false
        }
    }

    private fun zeroVelocitiesLocked() {
        velT[0] = 0f; velT[1] = 0f; velT[2] = 0f
        angularVelAxis[0] = 0f; angularVelAxis[1] = 0f; angularVelAxis[2] = 0f
        angularVelVec[0] = 0f; angularVelVec[1] = 0f; angularVelVec[2] = 0f
        angularSpeedRadPerSec = 0f
        velocityBlendFactor = 0f
    }

    /**
     * Copies the current filtered linear velocity `[vx, vy, vz]` (m/s) into [out].
     */
    fun getLinearVelocity(out: FloatArray = FloatArray(3)): FloatArray {
        synchronized(lock) {
            out[0] = velT[0] * velocityBlendFactor
            out[1] = velT[1] * velocityBlendFactor
            out[2] = velT[2] * velocityBlendFactor
        }
        return out
    }

    /**
     * Copies the current angular velocity vector `[wx, wy, wz]` (rad/s) into [out].
     */
    fun getAngularVelocity(out: FloatArray = FloatArray(3)): FloatArray {
        synchronized(lock) {
            out[0] = angularVelVec[0] * velocityBlendFactor
            out[1] = angularVelVec[1] * velocityBlendFactor
            out[2] = angularVelVec[2] * velocityBlendFactor
        }
        return out
    }

    /**
     * Updates compensator with latest FILTERED pose from tracker at capture timestamp [timestampNs].
     *
     * @param resetVelocity True on `ACQUIRING -> TRACKING` transition or when tracking is re-acquired,
     *   forcing velocity to zero so stale velocity never carries across tracking gaps.
     */
    @JvmOverloads
    fun update(
        translation: FloatArray,
        rotation: FloatArray,
        timestampNs: Long,
        scale: FloatArray = floatArrayOf(1f, 1f, 1f),
        resetVelocity: Boolean = false
    ) {
        synchronized(lock) {
            val prevTs = lastTimestampNs
            val dtNs = timestampNs - prevTs
            val dtSec = dtNs.toFloat() / 1_000_000_000f

            // 1. Evaluate prediction error for this new sample using the PREVIOUS state before updating
            if (!resetVelocity && prevTs > 0L && dtSec in MIN_SAMPLE_DT_SEC..MAX_SAMPLE_GAP_SEC) {
                predictLocked(
                    renderTimestampNs = timestampNs,
                    poseTimestampNs = prevTs,
                    includeExtraLead = false,
                    outT = errPredT,
                    outQ = errPredQ
                )
                val dx = errPredT[0] - translation[0]
                val dy = errPredT[1] - translation[1]
                val dz = errPredT[2] - translation[2]
                lastPredictionErrorPosMm = sqrt(dx * dx + dy * dy + dz * dz) * 1000f

                val dotQ = abs(
                    errPredQ[0] * rotation[0] +
                        errPredQ[1] * rotation[1] +
                        errPredQ[2] * rotation[2] +
                        errPredQ[3] * rotation[3]
                ).coerceIn(0f, 1f)
                lastPredictionErrorRotDeg = 2f * acos(dotQ) * (180f / PI.toFloat())
                hasPredictionErrorSample = true
            } else {
                hasPredictionErrorSample = false
            }

            // 2. Enforce hemisphere continuity between rotation and lastQ
            var q1x = rotation[0]
            var q1y = rotation[1]
            var q1z = rotation[2]
            var q1w = rotation[3]
            if (prevTs > 0L) {
                val dotPrev = q1x * lastQ[0] + q1y * lastQ[1] + q1z * lastQ[2] + q1w * lastQ[3]
                if (dotPrev < 0f) {
                    q1x = -q1x; q1y = -q1y; q1z = -q1z; q1w = -q1w
                }
            }

            // 3. Velocity estimation or reset on gap (> 150ms) / ACQUIRING -> TRACKING
            if (resetVelocity || prevTs <= 0L || dtSec < MIN_SAMPLE_DT_SEC || dtSec > MAX_SAMPLE_GAP_SEC) {
                zeroVelocitiesLocked()
            } else {
                // --- Linear velocity from filtered pose ---
                val rawVx = (translation[0] - lastT[0]) / dtSec
                val rawVy = (translation[1] - lastT[1]) / dtSec
                val rawVz = (translation[2] - lastT[2]) / dtSec
                val rawLinSpeed = sqrt(rawVx * rawVx + rawVy * rawVy + rawVz * rawVz)

                val prevLinSpeed = sqrt(velT[0] * velT[0] + velT[1] * velT[1] + velT[2] * velT[2])
                if (rawLinSpeed < LINEAR_DEADZONE_MPS) {
                    // Stop-after-motion: zero immediately so prediction never overshoots past stop
                    velT[0] = 0f; velT[1] = 0f; velT[2] = 0f
                    velocityBlendFactor = 0f
                } else {
                    val dotDir = rawVx * velT[0] + rawVy * velT[1] + rawVz * velT[2]
                    if (prevLinSpeed > LINEAR_DEADZONE_MPS && dotDir < 0f) {
                        // Direction reversal / oscillatory noise: blend down aggressively and brake
                        velocityBlendFactor = (velocityBlendFactor * 0.25f).coerceIn(0f, 1f)
                        velT[0] = rawVx * 0.25f
                        velT[1] = rawVy * 0.25f
                        velT[2] = rawVz * 0.25f
                    } else if (prevLinSpeed > LINEAR_DEADZONE_MPS && rawLinSpeed < prevLinSpeed * 0.45f) {
                        // Sharp deceleration: adopt the lower new velocity immediately without lag
                        velT[0] = rawVx
                        velT[1] = rawVy
                        velT[2] = rawVz
                        velocityBlendFactor = (rawLinSpeed / prevLinSpeed).coerceIn(0.2f, 1f)
                    } else {
                        // Consistent motion: track velocity responsively and ramp blend factor toward 1 immediately
                        val alpha = 0.96f
                        velT[0] = if (prevLinSpeed < LINEAR_DEADZONE_MPS) rawVx else (velT[0] * (1f - alpha) + rawVx * alpha)
                        velT[1] = if (prevLinSpeed < LINEAR_DEADZONE_MPS) rawVy else (velT[1] * (1f - alpha) + rawVy * alpha)
                        velT[2] = if (prevLinSpeed < LINEAR_DEADZONE_MPS) rawVz else (velT[2] * (1f - alpha) + rawVz * alpha)
                        val noiseRatio = ((rawLinSpeed - LINEAR_DEADZONE_MPS) / (LINEAR_DEADZONE_MPS * 1.5f)).coerceIn(0f, 1f)
                        velocityBlendFactor = if (prevLinSpeed < LINEAR_DEADZONE_MPS) {
                            noiseRatio
                        } else {
                            min(1f, velocityBlendFactor * 0.25f + 0.75f * noiseRatio)
                        }
                    }

                    // Clamp linear velocity magnitude to maxLinearSpeedMps
                    val maxLin = maxLinearSpeedMps.coerceAtLeast(0f)
                    val curSpeed = sqrt(velT[0] * velT[0] + velT[1] * velT[1] + velT[2] * velT[2])
                    if (curSpeed > maxLin && curSpeed > 1e-6f) {
                        val scaleLin = maxLin / curSpeed
                        velT[0] *= scaleLin
                        velT[1] *= scaleLin
                        velT[2] *= scaleLin
                    }
                }

                // --- Angular velocity from filtered quaternion ---
                // Relative rotation dq = q_new * q_old^(-1)
                val q0x = -lastQ[0]; val q0y = -lastQ[1]; val q0z = -lastQ[2]; val q0w = lastQ[3]
                val dqW = q1w * q0w - q1x * q0x - q1y * q0y - q1z * q0z
                val dqX = q1w * q0x + q1x * q0w + q1y * q0z - q1z * q0y
                val dqY = q1w * q0y - q1x * q0z + q1y * q0w + q1z * q0x
                val dqZ = q1w * q0z + q1x * q0y - q1y * q0x + q1z * q0w

                val signW = if (dqW < 0f) -1f else 1f
                val clampedW = (dqW * signW).coerceIn(-1f, 1f)
                val angle = 2f * acos(clampedW)
                val sinHalf = sqrt((1f - clampedW * clampedW).coerceAtLeast(0f))

                if (sinHalf > 1e-4f) {
                    val rawAngSpeed = angle / dtSec
                    if (rawAngSpeed < ANGULAR_DEADZONE_RAD_S) {
                        angularVelAxis[0] = 0f; angularVelAxis[1] = 0f; angularVelAxis[2] = 0f
                        angularVelVec[0] = 0f; angularVelVec[1] = 0f; angularVelVec[2] = 0f
                        angularSpeedRadPerSec = 0f
                    } else {
                        val invSin = (1f / sinHalf) * signW
                        val ax = dqX * invSin
                        val ay = dqY * invSin
                        val az = dqZ * invSin

                        // Check angular reversal against previous axis
                        val axisDot = ax * angularVelAxis[0] + ay * angularVelAxis[1] + az * angularVelAxis[2]
                        val dampedSpeed = if (angularSpeedRadPerSec > ANGULAR_DEADZONE_RAD_S && axisDot < 0f) {
                            rawAngSpeed * 0.25f
                        } else {
                            rawAngSpeed
                        }
                        val clampedAngSpeed = dampedSpeed.coerceIn(0f, maxAngularSpeedRadPerSec.coerceAtLeast(0f))
                        angularVelAxis[0] = ax
                        angularVelAxis[1] = ay
                        angularVelAxis[2] = az
                        angularSpeedRadPerSec = clampedAngSpeed
                        angularVelVec[0] = ax * clampedAngSpeed
                        angularVelVec[1] = ay * clampedAngSpeed
                        angularVelVec[2] = az * clampedAngSpeed
                        if (velocityBlendFactor <= 0f) {
                            val angRatio = ((clampedAngSpeed - ANGULAR_DEADZONE_RAD_S) / (ANGULAR_DEADZONE_RAD_S * 3f)).coerceIn(0f, 1f)
                            velocityBlendFactor = angRatio
                        }
                    }
                } else {
                    angularVelAxis[0] = 0f; angularVelAxis[1] = 0f; angularVelAxis[2] = 0f
                    angularVelVec[0] = 0f; angularVelVec[1] = 0f; angularVelVec[2] = 0f
                    angularSpeedRadPerSec = 0f
                }
            }

            System.arraycopy(translation, 0, lastT, 0, 3)
            lastQ[0] = q1x; lastQ[1] = q1y; lastQ[2] = q1z; lastQ[3] = q1w
            System.arraycopy(scale, 0, lastScale, 0, 3)
            lastTimestampNs = timestampNs
        }
    }

    /**
     * Extrapolates translation and rotation from [poseTimestampNs] to [renderTimestampNs]
     * (plus [extraLeadMs]) in the `elapsedRealtimeNanos` clock base with zero heap allocations.
     */
    @JvmOverloads
    fun predict(
        renderTimestampNs: Long,
        outT: FloatArray,
        outQ: FloatArray,
        poseTimestampNs: Long = lastTimestampNs
    ) {
        synchronized(lock) {
            predictLocked(
                renderTimestampNs = renderTimestampNs,
                poseTimestampNs = if (poseTimestampNs > 0L) poseTimestampNs else lastTimestampNs,
                includeExtraLead = true,
                outT = outT,
                outQ = outQ
            )
        }
    }

    private fun predictLocked(
        renderTimestampNs: Long,
        poseTimestampNs: Long,
        includeExtraLead: Boolean,
        outT: FloatArray,
        outQ: FloatArray
    ) {
        System.arraycopy(lastT, 0, outT, 0, 3)
        System.arraycopy(lastQ, 0, outQ, 0, 4)

        if (lastTimestampNs <= 0L || poseTimestampNs <= 0L) {
            return
        }

        val extraLeadNs = if (includeExtraLead && extraLeadMs.isFinite()) {
            (extraLeadMs * 1_000_000f).toLong()
        } else {
            0L
        }
        val targetTimeNs = renderTimestampNs + extraLeadNs
        if (targetTimeNs <= poseTimestampNs) {
            return
        }

        val horizon = maxHorizonSec.coerceIn(0f, MAX_HORIZON_SEC)
        if (horizon <= 0f) {
            return
        }

        // Extrapolation delta: delta = clamp(targetTimeNs - poseTimestampNs, 0, horizon)
        val rawDeltaSec = (targetTimeNs - poseTimestampNs).toFloat() / 1_000_000_000f
        val deltaSec = rawDeltaSec.coerceIn(0f, horizon)

        // Smooth taper gain on that same delta near the horizon (never a separate dt)
        val u = (deltaSec / horizon).coerceIn(0f, 1f)
        val taperGain = 1f - 0.008f * u * u
        val effectiveDeltaSec = deltaSec * taperGain * velocityBlendFactor.coerceIn(0f, 1f)

        if (effectiveDeltaSec <= 1e-6f) {
            return
        }

        val zGain = zTranslationGain.coerceIn(0f, 1f)

        // Bounded translation extrapolation (full gain on X/Y, reduced gain on depth Z)
        outT[0] = lastT[0] + velT[0] * effectiveDeltaSec
        outT[1] = lastT[1] + velT[1] * effectiveDeltaSec
        outT[2] = lastT[2] + velT[2] * effectiveDeltaSec * zGain

        // Bounded rotation extrapolation via quaternion exponential map
        if (angularSpeedRadPerSec > 1e-4f) {
            val deltaAngle = angularSpeedRadPerSec * effectiveDeltaSec
            val halfAngle = deltaAngle * 0.5f
            val sinH = sin(halfAngle)
            val cosH = cos(halfAngle)

            val dqX = angularVelAxis[0] * sinH
            val dqY = angularVelAxis[1] * sinH
            val dqZ = angularVelAxis[2] * sinH
            val dqW = cosH

            // q_pred = dq * lastQ
            val q0x = lastQ[0]; val q0y = lastQ[1]; val q0z = lastQ[2]; val q0w = lastQ[3]
            var ox = dqW * q0x + dqX * q0w + dqY * q0z - dqZ * q0y
            var oy = dqW * q0y - dqX * q0z + dqY * q0w + dqZ * q0x
            var oz = dqW * q0z + dqX * q0y - dqY * q0x + dqZ * q0w
            var ow = dqW * q0w - dqX * q0x - dqY * q0y - dqZ * q0z

            // Normalize
            val len = sqrt(ox * ox + oy * oy + oz * oz + ow * ow)
            if (len > 1e-6f) {
                val inv = 1f / len
                ox *= inv; oy *= inv; oz *= inv; ow *= inv
            } else {
                ox = q0x; oy = q0y; oz = q0z; ow = q0w
            }

            // Enforce hemisphere sign continuity with lastQ (dot >= 0)
            val dot = ox * q0x + oy * q0y + oz * q0z + ow * q0w
            if (dot < 0f) {
                ox = -ox; oy = -oy; oz = -oz; ow = -ow
            }

            outQ[0] = ox; outQ[1] = oy; outQ[2] = oz; outQ[3] = ow
        }
    }

    /**
     * Produces a predicted DecomposedPose at target render time (allocating helper for non-GL callers/tests).
     * Does NOT predict scale (returns the unpredicted filtered scale).
     */
    fun predictPose(renderTimestampNs: Long): PoseDecomposer.DecomposedPose {
        val predT = FloatArray(3)
        val predQ = FloatArray(4)
        val scaleCopy = FloatArray(3)
        synchronized(lock) {
            predictLocked(
                renderTimestampNs = renderTimestampNs,
                poseTimestampNs = lastTimestampNs,
                includeExtraLead = true,
                outT = predT,
                outQ = predQ
            )
            System.arraycopy(lastScale, 0, scaleCopy, 0, 3)
        }

        val euler = PoseDecomposer.quaternionToEulerDeg(predQ)
        return PoseDecomposer.DecomposedPose(
            translation = predT,
            rotation = predQ,
            scale = scaleCopy,
            yawDeg = euler[0],
            pitchDeg = euler[1],
            rollDeg = euler[2]
        )
    }
}
