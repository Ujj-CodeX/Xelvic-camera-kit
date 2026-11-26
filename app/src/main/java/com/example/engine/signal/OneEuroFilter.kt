package com.example.engine.signal

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * 1€ (One-Euro) filter for noisy signals.
 * Smooth when still (low cutoff fcmin), responsive when moving (cutoff adapts with beta * speed).
 */
class OneEuroFilter @JvmOverloads constructor(
    private var minCutoff: Float = 1.0f,
    private var beta: Float = 0.007f,
    private var dCutoff: Float = 1.0f,
    private var highVelocityBeta: Float = 0.0f,
    private var highSpeedThreshold: Float = 0.28f
) {
    private var xPrev: Float? = null
    private var xRawPrev: Float? = null
    private var dxPrev: Float = 0f
    private var tPrev: Long? = null
    private var lastRawDxSign: Int = 0
    private var coherentSteps: Int = 0

    fun reset() {
        xPrev = null
        xRawPrev = null
        dxPrev = 0f
        tPrev = null
        lastRawDxSign = 0
        coherentSteps = 0
    }

    fun filter(x: Float, timestampNs: Long): Float {
        val prevX = xPrev
        val prevRawX = xRawPrev ?: x
        val prevT = tPrev
        if (prevX == null || prevT == null) {
            xPrev = x
            xRawPrev = x
            dxPrev = 0f
            tPrev = timestampNs
            lastRawDxSign = 0
            coherentSteps = 0
            return x
        }

        val dt = (timestampNs - prevT).coerceAtLeast(1_000_000L).toFloat() / 1_000_000_000f
        tPrev = timestampNs
        xRawPrev = x

        // Track consecutive same-direction RAW input steps above the stationary jitter threshold
        val rawStepSpeed = (x - prevRawX) / dt
        val rawSign = when {
            rawStepSpeed > highSpeedThreshold * 0.75f -> 1
            rawStepSpeed < -highSpeedThreshold * 0.75f -> -1
            else -> 0
        }
        if (rawSign != 0 && rawSign == lastRawDxSign) {
            coherentSteps = (coherentSteps + 1).coerceAtMost(16)
        } else {
            coherentSteps = if (rawSign != 0) 1 else 0
            lastRawDxSign = rawSign
        }

        // Estimate derivative
        val dx = (x - prevX) / dt
        val edx = exponentialSmoothing(dx, dxPrev, alpha(dt, dCutoff))
        dxPrev = edx

        // Adaptive cutoff with coherent high-velocity boost above stationary jitter floor
        // (activates only after >= 4 consecutive same-direction raw steps so Gaussian stationary jitter is never boosted)
        val speed = abs(edx)
        val highSpeedBoost = if (highVelocityBeta > 0f && coherentSteps >= 4 && speed > highSpeedThreshold) {
            val excess = speed - highSpeedThreshold
            val normExcess = excess / highSpeedThreshold.coerceAtLeast(0.04f)
            highVelocityBeta * excess * (1.0f + 10.0f * normExcess * normExcess)
        } else {
            0f
        }
        val cutoff = minCutoff + beta * speed + highSpeedBoost
        val filtered = exponentialSmoothing(x, prevX, alpha(dt, cutoff))
        xPrev = filtered
        return filtered
    }

    private fun alpha(dt: Float, cutoff: Float): Float {
        val tau = 1f / (2f * PI.toFloat() * cutoff)
        return 1f / (1f + tau / dt)
    }

    private fun exponentialSmoothing(x: Float, prev: Float, alpha: Float): Float {
        return alpha * x + (1f - alpha) * prev
    }
}

/**
 * 3D Vector One-Euro filter.
 */
class OneEuroFilterVec3 @JvmOverloads constructor(
    minCutoff: Float = 1.2f,
    beta: Float = 0.015f,
    dCutoff: Float = 1.0f,
    highVelocityBeta: Float = 0.0f,
    highSpeedThreshold: Float = 0.12f
) {
    private val fx = OneEuroFilter(minCutoff, beta, dCutoff, highVelocityBeta, highSpeedThreshold)
    private val fy = OneEuroFilter(minCutoff, beta, dCutoff, highVelocityBeta, highSpeedThreshold)
    private val fz = OneEuroFilter(minCutoff, beta, dCutoff, highVelocityBeta, highSpeedThreshold)

    fun reset() {
        fx.reset()
        fy.reset()
        fz.reset()
    }

    fun filter(x: Float, y: Float, z: Float, timestampNs: Long, out: FloatArray = FloatArray(3)): FloatArray {
        out[0] = fx.filter(x, timestampNs)
        out[1] = fy.filter(y, timestampNs)
        out[2] = fz.filter(z, timestampNs)
        return out
    }
}

/**
 * Quaternion One-Euro filter with sign continuity.
 * Preserves unit norm and avoids 180-degree flip artifacts when q and -q represent the same rotation.
 */
class OneEuroFilterQuaternion @JvmOverloads constructor(
    minCutoff: Float = 1.0f,
    beta: Float = 0.01f,
    dCutoff: Float = 1.0f,
    highVelocityBeta: Float = 0.0f,
    highSpeedThreshold: Float = 0.25f
) {
    private val fx = OneEuroFilter(minCutoff, beta, dCutoff, highVelocityBeta, highSpeedThreshold)
    private val fy = OneEuroFilter(minCutoff, beta, dCutoff, highVelocityBeta, highSpeedThreshold)
    private val fz = OneEuroFilter(minCutoff, beta, dCutoff, highVelocityBeta, highSpeedThreshold)
    private val fw = OneEuroFilter(minCutoff, beta, dCutoff, highVelocityBeta, highSpeedThreshold)

    private var prevX = 0f
    private var prevY = 0f
    private var prevZ = 0f
    private var prevW = 1f
    private var hasPrev = false

    fun reset() {
        fx.reset()
        fy.reset()
        fz.reset()
        fw.reset()
        hasPrev = false
    }

    fun filter(x: Float, y: Float, z: Float, w: Float, timestampNs: Long, out: FloatArray = FloatArray(4)): FloatArray {
        var qx = x
        var qy = y
        var qz = z
        var qw = w

        // Enforce quaternion sign continuity (dot product < 0 means opposite hemisphere)
        if (hasPrev) {
            val dot = prevX * qx + prevY * qy + prevZ * qz + prevW * qw
            if (dot < 0f) {
                qx = -qx
                qy = -qy
                qz = -qz
                qw = -qw
            }
        }

        var ox = fx.filter(qx, timestampNs)
        var oy = fy.filter(qy, timestampNs)
        var oz = fz.filter(qz, timestampNs)
        var ow = fw.filter(qw, timestampNs)

        // Normalize output quaternion
        val len = sqrt(ox * ox + oy * oy + oz * oz + ow * ow)
        if (len > 1e-6f) {
            val inv = 1f / len
            ox *= inv
            oy *= inv
            oz *= inv
            ow *= inv
        } else {
            ox = 0f; oy = 0f; oz = 0f; ow = 1f
        }

        prevX = ox
        prevY = oy
        prevZ = oz
        prevW = ow
        hasPrev = true

        out[0] = ox
        out[1] = oy
        out[2] = oz
        out[3] = ow
        return out
    }
}
