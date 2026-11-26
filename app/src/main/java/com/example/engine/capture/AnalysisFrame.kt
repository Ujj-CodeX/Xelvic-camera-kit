package com.example.engine.capture

import android.hardware.camera2.CameraMetadata
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import com.example.engine.signal.ProjectionParams
import java.util.Arrays

/**
 * Immutable wrapper around an analysis frame submitted from CameraX.
 *
 * @property imageProxy The CameraX ImageProxy to analyze (closed once after conversion or skip).
 * @property timestampNs Authoritative capture timestamp aligned to `FrameClock.elapsedRealtimeNs()`
 *   (`SystemClock.elapsedRealtimeNanos()`).
 * @property rotationDegrees Display/sensor rotation degrees.
 * @property isFrontCamera True if front-facing camera (captured per frame so lens switches never mutate old frames).
 * @property imageToViewTransform 3x3 normalized image-to-view transform matrix values.
 * @property sensorTimestampNs Raw `ImageProxy.imageInfo.timestamp` in sensor clock domain.
 * @property submittedAtNs Timestamp (in `elapsedRealtimeNs` domain) when the frame entered the provider queue.
 * @property projectionParams Per-frame camera projection parameters (analysis-crop FOV and intrinsics).
 */
data class AnalysisFrame @JvmOverloads constructor(
    val imageProxy: ImageProxy?,
    val timestampNs: Long,
    val rotationDegrees: Int,
    val isFrontCamera: Boolean,
    val imageToViewTransform: FloatArray? = null,
    val sensorTimestampNs: Long = timestampNs,
    val submittedAtNs: Long = timestampNs,
    val projectionParams: ProjectionParams = ProjectionParams.DEFAULT,
    val frameNumber: Long = timestampNs
)

/**
 * Unified monotonic clock in the `SystemClock.elapsedRealtimeNanos()` (`CLOCK_BOOTTIME`) domain.
 * Never mixes camera time with wall-clock time (`System.currentTimeMillis`).
 */
object FrameClock {
    @Volatile
    var customElapsedRealtimeSource: (() -> Long)? = null

    fun elapsedRealtimeNs(): Long {
        customElapsedRealtimeSource?.let { return it() }
        return try {
            val ert = SystemClock.elapsedRealtimeNanos()
            if (ert > 0L) ert else System.nanoTime()
        } catch (_: Throwable) {
            System.nanoTime()
        }
    }

    fun monotonicNs(): Long = elapsedRealtimeNs()
}

/**
 * ClockOffsetCalibrator (AUDIT.md Section 3.2):
 *
 * Aligns `ImageProxy.imageInfo.timestamp` to `SystemClock.elapsedRealtimeNanos()`.
 *
 * - If `SENSOR_INFO_TIMESTAMP_SOURCE == SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME` (1):
 *   Sensor timestamps already share the `elapsedRealtimeNanos()` base (`calibratedOffsetNs = 0L`).
 * - If `SENSOR_INFO_TIMESTAMP_SOURCE != REALTIME` (e.g., `SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN` = 0):
 *   Estimates the clock base offset over the first `warmupFrames` frames using the **MINIMUM**
 *   (or lowest ~5th percentile when `sampleCount >= 20`) of `(arrivalElapsedRealtimeNs - sensorTimestampNs)`,
 *   NOT the median. Using the minimum/lowest-5th-percentile captures the closest lower bound to zero
 *   delivery delay, whereas the median would bake camera-to-analyzer delivery delay into the capture
 *   timestamp and artificially hide latency.
 *
 * Note on metric semantics:
 * The calibrated timestamp measures **capture-timestamp -> render** (from sensor exposure timestamp
 * to GL render submission), NOT full glass-to-glass latency.
 */
class ClockOffsetCalibrator(
    val warmupFrames: Int = DEFAULT_WARMUP_FRAMES
) {
    companion object {
        const val TIMESTAMP_SOURCE_UNKNOWN: Int = CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN
        const val TIMESTAMP_SOURCE_REALTIME: Int = CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
        const val DEFAULT_WARMUP_FRAMES: Int = 30
    }

    private val lock = Any()
    private val diffSamplesNs = LongArray(warmupFrames.coerceAtLeast(1))
    private val sortScratchNs = LongArray(warmupFrames.coerceAtLeast(1))

    @Volatile
    var timestampSource: Int = TIMESTAMP_SOURCE_UNKNOWN
        private set

    @Volatile
    var sampleCount: Int = 0
        private set

    @Volatile
    var calibratedOffsetNs: Long = 0L
        private set

    @Volatile
    var minObservedOffsetNs: Long = Long.MAX_VALUE
        private set

    val isCalibrated: Boolean
        get() = timestampSource == TIMESTAMP_SOURCE_REALTIME || sampleCount >= warmupFrames

    fun configureSource(source: Int) {
        synchronized(lock) {
            timestampSource = source
            sampleCount = 0
            calibratedOffsetNs = 0L
            minObservedOffsetNs = Long.MAX_VALUE
        }
    }

    fun reset() {
        synchronized(lock) {
            sampleCount = 0
            calibratedOffsetNs = 0L
            minObservedOffsetNs = Long.MAX_VALUE
        }
    }

    /**
     * Converts `sensorTimestampNs` into the `elapsedRealtimeNanos` clock domain.
     *
     * @param sensorTimestampNs Raw timestamp from `ImageProxy.imageInfo.timestamp`.
     * @param arrivalElapsedRealtimeNs Current `FrameClock.elapsedRealtimeNs()` when the frame arrived.
     * @return Capture timestamp in `elapsedRealtimeNanos` clock base.
     */
    fun toElapsedRealtimeNs(
        sensorTimestampNs: Long,
        arrivalElapsedRealtimeNs: Long = FrameClock.elapsedRealtimeNs()
    ): Long {
        if (timestampSource == TIMESTAMP_SOURCE_REALTIME) {
            return sensorTimestampNs
        }
        synchronized(lock) {
            if (timestampSource == TIMESTAMP_SOURCE_REALTIME) {
                return sensorTimestampNs
            }
            if (sampleCount < warmupFrames) {
                val diff = arrivalElapsedRealtimeNs - sensorTimestampNs
                diffSamplesNs[sampleCount] = diff
                sampleCount++
                if (diff < minObservedOffsetNs) {
                    minObservedOffsetNs = diff
                }
                // Use exact minimum for <= 20 samples, or lowest ~5th percentile for larger warmup windows
                if (sampleCount < 20) {
                    calibratedOffsetNs = minObservedOffsetNs
                } else {
                    System.arraycopy(diffSamplesNs, 0, sortScratchNs, 0, sampleCount)
                    Arrays.sort(sortScratchNs, 0, sampleCount)
                    val p05Index = ((sampleCount - 1) * 0.05f).toInt().coerceIn(0, sampleCount - 1)
                    calibratedOffsetNs = sortScratchNs[p05Index]
                }
            }
            return sensorTimestampNs + calibratedOffsetNs
        }
    }
}

