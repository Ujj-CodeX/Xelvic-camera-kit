package com.example.engine.signal

import com.example.engine.pkg.FaceRegions
import com.example.engine.pkg.UvRegistry

/**
 * ControlPointSmoother (Flow 2 — Chunk R1):
 *
 * Zero-allocation 1-Euro landmark smoother operating selectively on [FaceRegions.CONTROL_POINT_INDICES]
 * (~118 anatomical contour & iris landmarks out of 478).
 *
 * Why selective control-point smoothing:
 * - Region masks (`RegionMaskPass`), mouth-inner cutouts, eyelid contours, and iris circles amplify
 *   raw frame-to-frame landmark jitter when stationary, whereas filtering all 478 * 3 coordinates
 *   every frame wastes CPU cycles.
 * - Pre-allocates [OneEuroFilter] pairs `(x, y)` only for [controlPointIndices] and writes into
 *   a pre-allocated `smoothedLandmarks` float array (`478 * 3` floats) with zero per-frame heap allocations.
 */
class ControlPointSmoother @JvmOverloads constructor(
    val controlPointIndices: IntArray = FaceRegions.CONTROL_POINT_INDICES,
    minCutoff: Float = DEFAULT_MIN_CUTOFF,
    beta: Float = DEFAULT_BETA,
    dCutoff: Float = DEFAULT_D_CUTOFF,
    highVelocityBeta: Float = DEFAULT_HIGH_VELOCITY_BETA,
    highSpeedThreshold: Float = DEFAULT_HIGH_SPEED_THRESHOLD
) {
    companion object {
        const val DEFAULT_MIN_CUTOFF = 1.6f
        const val DEFAULT_BETA = 0.025f
        const val DEFAULT_D_CUTOFF = 1.0f
        const val DEFAULT_HIGH_VELOCITY_BETA = 650.0f
        const val DEFAULT_HIGH_SPEED_THRESHOLD = 0.08f
        const val MAX_GAP_RESET_NS = 500_000_000L // 500 ms gap triggers filter reset
    }

    private val filtersX = Array(controlPointIndices.size) {
        OneEuroFilter(
            minCutoff = minCutoff,
            beta = beta,
            dCutoff = dCutoff,
            highVelocityBeta = highVelocityBeta,
            highSpeedThreshold = highSpeedThreshold
        )
    }
    private val filtersY = Array(controlPointIndices.size) {
        OneEuroFilter(
            minCutoff = minCutoff,
            beta = beta,
            dCutoff = dCutoff,
            highVelocityBeta = highVelocityBeta,
            highSpeedThreshold = highSpeedThreshold
        )
    }

    /**
     * Pre-allocated output buffer (`478 * 3` floats) holding the full landmark array with smoothed
     * `(x, y)` values at [controlPointIndices].
     */
    val smoothedLandmarks = FloatArray(UvRegistry.TRACKER_LANDMARK_COUNT_WITH_IRIS * 3)

    private var lastTimestampNs: Long = 0L
    private var hasActiveTrack: Boolean = false

    var smoothedFrameCount: Int = 0
        private set

    private var lastRawLandmarksRef: FloatArray? = null

    fun reset() {
        for (i in controlPointIndices.indices) {
            filtersX[i].reset()
            filtersY[i].reset()
        }
        lastTimestampNs = 0L
        lastRawLandmarksRef = null
        hasActiveTrack = false
    }

    /**
     * Smooths the control-point landmarks in [rawLandmarks] at [timestampNs] into [outLandmarks]
     * (defaults to [smoothedLandmarks]) with zero heap allocations.
     *
     * Returns `null` if [rawLandmarks] has fewer than `468 * 3` elements or if `!isTracking`.
     */
    @JvmOverloads
    fun smooth(
        rawLandmarks: FloatArray?,
        timestampNs: Long,
        isTracking: Boolean = true,
        outLandmarks: FloatArray = smoothedLandmarks
    ): FloatArray? {
        if (!isTracking || rawLandmarks == null || rawLandmarks.size < UvRegistry.VERTEX_COUNT * 3) {
            if (hasActiveTrack) {
                reset()
            }
            return null
        }

        if (hasActiveTrack &&
            timestampNs > 0L &&
            timestampNs == lastTimestampNs &&
            rawLandmarks === lastRawLandmarksRef
        ) {
            if (outLandmarks !== smoothedLandmarks) {
                val copyLen = minOf(smoothedLandmarks.size, outLandmarks.size)
                System.arraycopy(smoothedLandmarks, 0, outLandmarks, 0, copyLen)
            }
            return outLandmarks
        }

        if (!hasActiveTrack || (lastTimestampNs > 0L && (timestampNs - lastTimestampNs) > MAX_GAP_RESET_NS) || timestampNs < lastTimestampNs) {
            for (i in controlPointIndices.indices) {
                filtersX[i].reset()
                filtersY[i].reset()
            }
            hasActiveTrack = true
        }
        lastTimestampNs = timestampNs
        lastRawLandmarksRef = rawLandmarks

        val internalCopyLen = minOf(rawLandmarks.size, smoothedLandmarks.size)
        System.arraycopy(rawLandmarks, 0, smoothedLandmarks, 0, internalCopyLen)
        val availableLandmarks = internalCopyLen / 3

        for (k in controlPointIndices.indices) {
            val lmIdx = controlPointIndices[k]
            if (lmIdx in 0 until availableLandmarks) {
                val base = lmIdx * 3
                val rx = rawLandmarks[base]
                val ry = rawLandmarks[base + 1]
                if (rx.isFinite() && ry.isFinite()) {
                    smoothedLandmarks[base] = filtersX[k].filter(rx, timestampNs)
                    smoothedLandmarks[base + 1] = filtersY[k].filter(ry, timestampNs)
                }
            }
        }

        if (outLandmarks !== smoothedLandmarks) {
            val outCopyLen = minOf(internalCopyLen, outLandmarks.size)
            System.arraycopy(smoothedLandmarks, 0, outLandmarks, 0, outCopyLen)
        }

        smoothedFrameCount++
        return outLandmarks
    }
}
