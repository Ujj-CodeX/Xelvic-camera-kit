package com.example.engine.perception

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.Rect
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import androidx.camera.core.ImageProxy
import com.example.engine.capture.AnalysisFrame
import com.example.engine.capture.FrameClock
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.ProjectionParams
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import java.security.MessageDigest
import java.util.Collections
import java.util.IdentityHashMap
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * MediaPipe Live Stream Face Landmarker provider (README Section 8 & AUDIT.md Section 3.3).
 *
 * Implements the dedicated dispatch thread (`XelvicDispatchThread`) and 3-slot triple-buffered
 * conversion/inference pipeline:
 *
 * ### Shared State Thread-Ownership Table
 * | State / Field | Writer Thread(s) | Reader Thread(s) | Guard / Synchronization |
 * |---|---|---|---|
 * | `slots[0..2].state` (`FREE`, `CONVERTING`, `STAGED`, `IN_FLIGHT`) | Camera analysis (`FREE -> CONVERTING`, `CONVERTING -> STAGED`, `STAGED -> FREE`), Dispatch (`STAGED -> IN_FLIGHT`), Callback/Dispatch/Stop (`IN_FLIGHT -> FREE`) | Camera analysis, Dispatch | `synchronized(slotLock)` |
 * | `slots[i]` pixel buffers (`sourceBitmap`, `targetBitmap`, `argbScratch`) | Camera analysis thread ONLY while `slots[i].state == CONVERTING` | Dispatch thread & MediaPipe native inference ONLY while `slots[i].state == IN_FLIGHT` | `slotLock` state transitions (`FREE -> CONVERTING -> STAGED -> IN_FLIGHT -> FREE`) |
 * | `stagedPreparedFrame` / `pendingConvertedFrameRef` | Camera analysis (stage converted frame), Dispatch (pop for `detectAsync`), UI/Stop (clear) | Camera analysis, Dispatch | `synchronized(slotLock)` |
 * | `isInferenceInFlight` | Dispatch (set `true` on pop/dispatch, `false` on watchdog/error), Callback (`false` on result/error) | Camera analysis, Dispatch, Callback | `AtomicBoolean` + `synchronized(slotLock)` on pop/release |
 * | `nextDispatchId`, `activeInFlightDispatchId`, `dispatchIdByMpImage` | Dispatch (increment & assign before `detectAsync`), Callback/Watchdog (validate & clear) | Dispatch, Callback | `AtomicLong`, `@Volatile`, `Collections.synchronizedMap` |
 * | `lastDetectTimestampMs` | Dispatch thread ONLY (`tryDispatchPendingConvertedFrameOnDispatchThread`) | Dispatch thread, Test inspection | Dispatch-thread confinement + `AtomicLong` |
 * | `lastDispatchedFrameTimestampNs`, `lastDispatchedClockNs` | Dispatch thread ONLY (and `stop()`) | Dispatch thread | `@Volatile`, Dispatch-thread confinement |
 * | `activeEntry`, `currentInFlightEntry`, `retiringEntries` | Dispatch thread (create/swap/watchdog), Callback (retire after completion), `stop()` | Camera analysis (`activeEntry != null` check), Dispatch, Callback | `synchronized(landmarkerLock)` + `@Volatile` |
 * | `pendingDelegateSwitchRef`, `pendingBlendshapesSwitchRef` | UI / Caller thread (`set`) | Dispatch thread (`getAndSet(null)`) | `AtomicReference` |
 */
class MediaPipeFaceGeometryProvider @JvmOverloads constructor(
    private val context: Context,
    private val modelAssetPath: String = "face_landmarker.task",
    analysisExecutorOverride: Executor? = null,
    reaperExecutorOverride: Executor? = null,
    watchdogSchedulerOverride: ScheduledExecutorService? = null,
    dispatchExecutorOverride: Executor? = null
) : FaceGeometryProvider {

    companion object {
        private const val TAG = "MediaPipeProvider"

        const val FIRST_INFERENCE_WATCHDOG_TIMEOUT_MS = 5_000L
        const val FIRST_INFERENCE_WATCHDOG_TIMEOUT_NS = 5_000_000_000L
        const val STEADY_WATCHDOG_TIMEOUT_MS = 500L
        const val STEADY_WATCHDOG_TIMEOUT_NS = 500_000_000L

        const val WATCHDOG_FALLBACK_MAX_RESETS = 3
        const val WATCHDOG_FALLBACK_WINDOW_NS = 30_000_000_000L

        const val STATUS_TRACKER_RESET = "tracker reset"
        const val STATUS_GPU_TO_CPU_WATCHDOG = "GPU->CPU (watchdog)"
        const val STATUS_GPU_TO_CPU_FALLBACK = "GPU->CPU fallback"
        const val STATUS_INIT_ERROR = "ERROR: landmarker init failed"

        /**
         * Computes theoretical steady-state tracker throughput (Hz) with overlapped conversion and
         * inference (`1000 / max(conversionMs, inferenceMs)`) vs sequential (`1000 / (conversionMs + inferenceMs)`).
         */
        @JvmStatic
        fun computeExpectedOverlappedThroughputHz(conversionMs: Float, inferenceMs: Float): Float {
            val bottleneckMs = max(conversionMs, inferenceMs).coerceAtLeast(0.1f)
            return 1000f / bottleneckMs
        }

        /**
         * Converts a `YUV_420_888` [ImageProxy] directly from its plane `ByteBuffer`s into packed
         * `ARGB_8888` ints in [outArgb], honoring per-plane `rowStride`, `pixelStride` (1 or 2),
         * and [cropRect] without allocating or copying any intermediate `ByteArray`.
         */
        fun convertYuv420ToArgbPixels(
            imageProxy: ImageProxy,
            cropRect: Rect,
            outArgb: IntArray
        ) {
            val yPlane = imageProxy.planes[0]
            val uPlane = imageProxy.planes[1]
            val vPlane = imageProxy.planes[2]

            val yBuf = yPlane.buffer
            val uBuf = uPlane.buffer
            val vBuf = vPlane.buffer

            val rowStrideY = yPlane.rowStride
            val pixelStrideY = yPlane.pixelStride
            val rowStrideU = uPlane.rowStride
            val pixelStrideU = uPlane.pixelStride
            val rowStrideV = vPlane.rowStride
            val pixelStrideV = vPlane.pixelStride

            val left = cropRect.left
            val top = cropRect.top
            val cropW = (cropRect.right - cropRect.left).let { if (it > 0) it else cropRect.width() }
            val cropH = (cropRect.bottom - cropRect.top).let { if (it > 0) it else cropRect.height() }

            for (r in 0 until cropH) {
                val srcY = top + r
                val yRowOffset = srcY * rowStrideY
                val uvRow = srcY shr 1
                val uRowOffset = uvRow * rowStrideU
                val vRowOffset = uvRow * rowStrideV
                val outRowOffset = r * cropW

                for (c in 0 until cropW) {
                    val srcX = left + c
                    val uvCol = srcX shr 1

                    val yVal = yBuf.get(yRowOffset + srcX * pixelStrideY).toInt() and 0xFF
                    val uVal = (uBuf.get(uRowOffset + uvCol * pixelStrideU).toInt() and 0xFF) - 128
                    val vVal = (vBuf.get(vRowOffset + uvCol * pixelStrideV).toInt() and 0xFF) - 128

                    // ITU-R BT.601 integer conversion
                    val y1192 = 1192 * (yVal - 16).coerceAtLeast(0)
                    val rVal = ((y1192 + 1634 * vVal) shr 10).coerceIn(0, 255)
                    val gVal = ((y1192 - 833 * vVal - 400 * uVal) shr 10).coerceIn(0, 255)
                    val bVal = ((y1192 + 2066 * uVal) shr 10).coerceIn(0, 255)

                    outArgb[outRowOffset + c] =
                        (0xFF shl 24) or (rVal shl 16) or (gVal shl 8) or bVal
                }
            }
        }

        /**
         * Converts an `RGBA_8888` single-plane [ImageProxy] directly from `planes[0].buffer` into
         * [destBitmap], honoring `rowStride`, `pixelStride`, and [cropRect] with zero per-frame allocations.
         */
        fun convertRgba8888ToBitmap(
            imageProxy: ImageProxy,
            cropRect: Rect,
            destBitmap: Bitmap?,
            scratchArgb: IntArray
        ) {
            val plane = imageProxy.planes[0]
            val buf = plane.buffer
            val rowStride = plane.rowStride
            val pixelStride = plane.pixelStride.coerceAtLeast(4)

            val left = cropRect.left
            val top = cropRect.top
            val cropW = (cropRect.right - cropRect.left).let { if (it > 0) it else cropRect.width() }
            val cropH = (cropRect.bottom - cropRect.top).let { if (it > 0) it else cropRect.height() }

            // Zero-copy fast path when uncropped and tightly packed
            if (destBitmap != null &&
                left == 0 && top == 0 &&
                cropW == imageProxy.width && cropH == imageProxy.height &&
                pixelStride == 4 && rowStride == cropW * 4 &&
                buf.capacity() >= cropW * cropH * 4
            ) {
                val prevPos = buf.position()
                buf.position(0)
                destBitmap.copyPixelsFromBuffer(buf)
                buf.position(prevPos)
                return
            }

            for (r in 0 until cropH) {
                val srcY = top + r
                val rowOffset = srcY * rowStride
                val outRowOffset = r * cropW
                for (c in 0 until cropW) {
                    val srcX = left + c
                    val pxOffset = rowOffset + srcX * pixelStride
                    val red = buf.get(pxOffset).toInt() and 0xFF
                    val green = buf.get(pxOffset + 1).toInt() and 0xFF
                    val blue = buf.get(pxOffset + 2).toInt() and 0xFF
                    val alpha = buf.get(pxOffset + 3).toInt() and 0xFF
                    scratchArgb[outRowOffset + c] =
                        (alpha shl 24) or (red shl 16) or (green shl 8) or blue
                }
            }
            destBitmap?.setPixels(scratchArgb, 0, cropW, 0, 0, cropW, cropH)
        }

        /**
         * Copies `[startOffset .. startOffset + length)` from [buf] into [dest] with zero heap allocations
         * by saving and restoring `buf.position()`.
         */
        internal fun copyByteBufferSliceZeroAlloc(
            buf: java.nio.ByteBuffer,
            startOffset: Int,
            dest: ByteArray,
            length: Int
        ) {
            if (length <= 0) return
            val safeStart = startOffset.coerceIn(0, (buf.limit() - 1).coerceAtLeast(0))
            val avail = (buf.limit() - safeStart).coerceAtLeast(0)
            val toCopy = length.coerceAtMost(avail).coerceAtMost(dest.size)
            if (toCopy <= 0) return
            val prevPos = buf.position()
            try {
                buf.position(safeStart)
                buf.get(dest, 0, toCopy)
            } finally {
                buf.position(prevPos)
            }
        }

        /**
         * Reference two-stage rotation and downscale from unpacked [srcArgb] (`cropW x cropH`)
         * into [outArgb] (`dstW x dstH`) using center-aligned sampling matching the Stage-2 transform.
         */
        fun rotateAndScaleReferenceArgb(
            srcArgb: IntArray,
            cropW: Int,
            cropH: Int,
            dstW: Int,
            dstH: Int,
            rot: Int,
            outArgb: IntArray
        ) {
            if (cropW <= 0 || cropH <= 0 || dstW <= 0 || dstH <= 0) return
            val normalizedRot = ((rot % 360) + 360) % 360
            if (normalizedRot == 0 && dstW == cropW && dstH == cropH) {
                System.arraycopy(srcArgb, 0, outArgb, 0, dstW * dstH)
                return
            }
            val orientedW = if (normalizedRot == 90 || normalizedRot == 270) cropH else cropW
            val orientedH = if (normalizedRot == 90 || normalizedRot == 270) cropW else cropH

            for (dy in 0 until dstH) {
                val v = (((2 * dy + 1) * orientedH) / (2 * dstH)).coerceIn(0, orientedH - 1)
                val outRowOff = dy * dstW
                for (dx in 0 until dstW) {
                    val u = (((2 * dx + 1) * orientedW) / (2 * dstW)).coerceIn(0, orientedW - 1)
                    val cx: Int
                    val cy: Int
                    when (normalizedRot) {
                        90 -> {
                            cx = v
                            cy = (cropH - 1) - u
                        }
                        180 -> {
                            cx = (cropW - 1) - u
                            cy = (cropH - 1) - v
                        }
                        270 -> {
                            cx = (cropW - 1) - v
                            cy = u
                        }
                        else -> {
                            cx = u
                            cy = v
                        }
                    }
                    outArgb[outRowOff + dx] = srcArgb[cy * cropW + cx]
                }
            }
        }

        private fun isFiniteMatrixStatic(m: FloatArray): Boolean {
            for (v in m) {
                if (v.isNaN() || v.isInfinite()) return false
            }
            return true
        }

        /**
         * Calibrates the 4x4 facial pose matrix in metric camera space (meters) aligned with
         * the ANALYSIS-CROP vertical FOV ([fovYCropDeg]), because MediaPipe landmarks [lm]
         * are normalized `[0..1]` to the upright analysis crop (`imageWidth x imageHeight`).
         */
        @JvmStatic
        @JvmOverloads
        fun calibratePoseMatrix(
            rawMpMatrix: FloatArray?,
            lm: FloatArray,
            imageWidth: Int,
            imageHeight: Int,
            fovYCropDeg: Float = ProjectionParams.DEFAULT_FOV_Y_DEG
        ): FloatArray {
            val out = FloatArray(16) { if (it % 5 == 0) 1f else 0f }
            if (lm.size < 455 * 3) return out

            val aspect = imageWidth.toFloat() / imageHeight.toFloat().coerceAtLeast(1f)
            val f = FacePoseAdapter.focalFactorFromFovY(fovYCropDeg)

            val iRight = 234 * 3 // subject-right cheek
            val iLeft = 454 * 3  // subject-left cheek
            val iTop = 10 * 3    // forehead
            val iChin = 152 * 3  // chin

            val x234 = lm[iRight]; val y234 = lm[iRight + 1]; val z234 = lm[iRight + 2]
            val x454 = lm[iLeft];  val y454 = lm[iLeft + 1];  val z454 = lm[iLeft + 2]

            // 1. Extract or derive orthonormal 3x3 rotation basis.
            // On live camera frames where full 468 facial landmarks have real vertical/horizontal span
            // (chin #152 below forehead #10 and eye/nose landmarks populated), derive the 3D orthonormal
            // frame directly from the unprojected landmarks in the aspect-corrected camera frustum so
            // +X_pose (#234 -> #454) and +Y_pose (#152 -> #10) match the real face roll, pitch, and yaw
            // without MediaPipe square-tensor aspect distortion.
            val isFullTrackedFace = (lm[iChin + 1] - lm[iTop + 1]) > 0.08f &&
                kotlin.math.abs(x454 - x234) > 0.02f &&
                lm[33 * 3 + 1] != 0.50f &&
                lm[1 * 3 + 1] != 0.50f
            if (!isFullTrackedFace && rawMpMatrix != null && rawMpMatrix.size == 16 && isFiniteMatrixStatic(rawMpMatrix)) {
                var c0x = rawMpMatrix[0]; var c0y = rawMpMatrix[1]; var c0z = rawMpMatrix[2]
                var c1x = rawMpMatrix[4]; var c1y = rawMpMatrix[5]; var c1z = rawMpMatrix[6]
                var c2x = rawMpMatrix[8]; var c2y = rawMpMatrix[9]; var c2z = rawMpMatrix[10]

                val s0 = kotlin.math.sqrt(c0x * c0x + c0y * c0y + c0z * c0z).coerceAtLeast(1e-6f)
                val s1 = kotlin.math.sqrt(c1x * c1x + c1y * c1y + c1z * c1z).coerceAtLeast(1e-6f)
                val s2 = kotlin.math.sqrt(c2x * c2x + c2y * c2y + c2z * c2z).coerceAtLeast(1e-6f)

                out[0] = c0x / s0; out[1] = c0y / s0; out[2] = c0z / s0
                out[4] = c1x / s1; out[5] = c1y / s1; out[6] = c1z / s1
                out[8] = c2x / s2; out[9] = c2y / s2; out[10] = c2z / s2
            } else {
                var rx = (x454 - x234) * aspect
                var ry = -(y454 - y234)
                var rz = -(z454 - z234) * aspect
                val rLen = kotlin.math.sqrt(rx * rx + ry * ry + rz * rz).coerceAtLeast(1e-5f)
                rx /= rLen; ry /= rLen; rz /= rLen

                var ux = (lm[iTop] - lm[iChin]) * aspect
                var uy = -(lm[iTop + 1] - lm[iChin + 1])
                var uz = -(lm[iTop + 2] - lm[iChin + 2]) * aspect
                val uLen = kotlin.math.sqrt(ux * ux + uy * uy + uz * uz).coerceAtLeast(1e-5f)
                ux /= uLen; uy /= uLen; uz /= uLen

                if (rawMpMatrix != null && rawMpMatrix.size == 16 && isFiniteMatrixStatic(rawMpMatrix)) {
                    // Blend out-of-plane Z component with MediaPipe's 3D matrix solver for smooth pitch/yaw
                    // while keeping XY strictly aligned with the live unprojected landmarks on screen
                    val mpS0 = kotlin.math.sqrt(
                        rawMpMatrix[0] * rawMpMatrix[0] +
                            rawMpMatrix[1] * rawMpMatrix[1] +
                            rawMpMatrix[2] * rawMpMatrix[2]
                    ).coerceAtLeast(1e-6f)
                    val mpS1 = kotlin.math.sqrt(
                        rawMpMatrix[4] * rawMpMatrix[4] +
                            rawMpMatrix[5] * rawMpMatrix[5] +
                            rawMpMatrix[6] * rawMpMatrix[6]
                    ).coerceAtLeast(1e-6f)
                    rz = 0.65f * rz + 0.35f * (rawMpMatrix[2] / mpS0)
                    uz = 0.65f * uz + 0.35f * (rawMpMatrix[6] / mpS1)
                    val rNorm = kotlin.math.sqrt(rx * rx + ry * ry + rz * rz).coerceAtLeast(1e-5f)
                    rx /= rNorm; ry /= rNorm; rz /= rNorm
                    val uNorm = kotlin.math.sqrt(ux * ux + uy * uy + uz * uz).coerceAtLeast(1e-5f)
                    ux /= uNorm; uy /= uNorm; uz /= uNorm
                }

                var fx = ry * uz - rz * uy
                var fy = rz * ux - rx * uz
                var fz = rx * uy - ry * ux
                val fLen = kotlin.math.sqrt(fx * fx + fy * fy + fz * fz).coerceAtLeast(1e-5f)
                fx /= fLen; fy /= fLen; fz /= fLen

                ux = fy * rz - fz * ry
                uy = fz * rx - fx * rz
                uz = fx * ry - fy * rx

                out[0] = rx; out[1] = ry; out[2] = rz
                out[4] = ux; out[5] = uy; out[6] = uz
                out[8] = fx; out[9] = fy; out[10] = fz
            }

            // 2. Metric depth tz in meters
            val rawTz = if (rawMpMatrix != null && rawMpMatrix.size == 16 && isFiniteMatrixStatic(rawMpMatrix)) {
                val zVal = rawMpMatrix[14]
                when {
                    kotlin.math.abs(zVal) > 2.0f -> zVal * 0.01f
                    kotlin.math.abs(zVal) > 0.15f -> zVal
                    else -> -0.50f
                }
            } else {
                -0.50f
            }
            val tz = (-kotlin.math.abs(rawTz)).coerceIn(-1.20f, -0.25f)
            val absZ = kotlin.math.abs(tz)

            // 3. Frustum dimensions in meters at depth absZ for the analysis-crop vertical FOV
            val wCam = 2.0f * absZ * aspect / f
            val hCam = 2.0f * absZ / f

            val cheekMidX = (x234 + x454) * 0.5f
            val cheekMidY = (y234 + y454) * 0.5f

            out[12] = (cheekMidX - 0.5f) * wCam
            out[13] = -(cheekMidY - 0.5f) * hCam
            out[14] = tz
            out[15] = 1.0f

            return out
        }
    }

    /**
     * Explicit state machine for each of the 3 frame buffer slots (`slots[0..2]`), guarded by `slotLock`:
     * - [FREE]: Available to be acquired by the camera analysis thread for conversion.
     * - [CONVERTING]: Owned exclusively by the camera analysis thread while unpacking/rotating/downscaling pixels.
     * - [STAGED]: Holds the latest converted frame waiting to be popped by the dispatch thread.
     * - [IN_FLIGHT]: Owned exclusively by the dispatch thread / MediaPipe `detectAsync` until callback or watchdog reset.
     */
    enum class SlotState {
        FREE,
        CONVERTING,
        STAGED,
        IN_FLIGHT
    }

    /**
     * Triple-buffered bitmap slot. Three slots (`0`, `1`, `2`) are managed via [SlotState] under
     * `slotLock` so the camera thread can always convert into a [SlotState.FREE] slot even while one
     * slot is [SlotState.IN_FLIGHT] and another slot is [SlotState.STAGED].
     *
     * On resolution or rotation change, new `Bitmap`s are allocated and the old references are
     * dropped without calling `Bitmap.recycle()` during active execution.
     */
    class FrameBufferSlot(val index: Int) {
        @Volatile
        var state: SlotState = SlotState.FREE

        var sourceBitmap: Bitmap? = null
        var targetBitmap: Bitmap? = null
        var targetCanvas: Canvas? = null
        val transformMatrix = Matrix()
        var argbScratch: IntArray = IntArray(0)
        var targetArgbScratch: IntArray = IntArray(0)
        var referenceTargetArgb: IntArray = IntArray(0)
        var lastUnpackedPixelArgb: Int = 0

        // Reusable plane byte arrays for zero-allocation bulk unpack in CONV:FAST
        var yPlaneBytes: ByteArray = ByteArray(0)
        var uPlaneBytes: ByteArray = ByteArray(0)
        var vPlaneBytes: ByteArray = ByteArray(0)
        var rgbaPlaneBytes: ByteArray = ByteArray(0)

        // Precomputed 1D byte-offset lookup tables for CONV:FAST (rebuilt ONLY when geometry/strides change)
        var dxYOffsetTable: IntArray = IntArray(0)
        var dxUOffsetTable: IntArray = IntArray(0)
        var dxVOffsetTable: IntArray = IntArray(0)
        var dyYOffsetTable: IntArray = IntArray(0)
        var dyUOffsetTable: IntArray = IntArray(0)
        var dyVOffsetTable: IntArray = IntArray(0)
        var dxRgbaOffsetTable: IntArray = IntArray(0)
        var dyRgbaOffsetTable: IntArray = IntArray(0)

        private var fastTableCropLeft: Int = -1
        private var fastTableCropTop: Int = -1
        private var fastTableCropW: Int = 0
        private var fastTableCropH: Int = 0
        private var fastTableDstW: Int = 0
        private var fastTableDstH: Int = 0
        private var fastTableRot: Int = -1
        private var fastTableRowStrideY: Int = 0
        private var fastTablePixelStrideY: Int = 0
        private var fastTableRowStrideU: Int = 0
        private var fastTablePixelStrideU: Int = 0
        private var fastTableRowStrideV: Int = 0
        private var fastTablePixelStrideV: Int = 0

        private var fastRgbaCropLeft: Int = -1
        private var fastRgbaCropTop: Int = -1
        private var fastRgbaCropW: Int = 0
        private var fastRgbaCropH: Int = 0
        private var fastRgbaDstW: Int = 0
        private var fastRgbaDstH: Int = 0
        private var fastRgbaRot: Int = -1
        private var fastRgbaRowStride: Int = 0
        private var fastRgbaPixelStride: Int = 0

        var fastTableRebuildCount: Int = 0
            private set
        var byteBufferAllocationCount: Int = 0
            private set
        var intBufferAllocationCount: Int = 0
            private set

        // Per-stage conversion timings (ms) recorded during the latest conversion in this slot
        @Volatile
        var lastUnpackMs: Float = 0f
            internal set
        @Volatile
        var lastYuvToRgbMs: Float = 0f
            internal set
        @Volatile
        var lastBitmapCopyMs: Float = 0f
            internal set
        @Volatile
        var lastRotateScaleMs: Float = 0f
            internal set
        @Volatile
        var lastMpImageBuildMs: Float = 0f
            internal set
        @Volatile
        var lastRawSourceWidth: Int = 0
            internal set
        @Volatile
        var lastRawSourceHeight: Int = 0
            internal set
        @Volatile
        var lastCropWidth: Int = 0
            internal set
        @Volatile
        var lastCropHeight: Int = 0
            internal set
        @Volatile
        var lastRotationDeg: Int = 0
            internal set

        var sourceWidth: Int = 0
            private set
        var sourceHeight: Int = 0
            private set
        var targetWidth: Int = 0
            private set
        var targetHeight: Int = 0
            private set
        private var hasAllocatedSource: Boolean = false
        private var hasAllocatedTarget: Boolean = false

        var cachedSrcW: Int = 0
        var cachedSrcH: Int = 0
        var cachedDstW: Int = 0
        var cachedDstH: Int = 0
        var cachedRotationDeg: Int = -1

        val inUseByInference = AtomicBoolean(false)
        var bitmapAllocationCount: Int = 0
            private set
        var recycleCallCount: Int = 0
            private set
        var lastConversionSucceeded: Boolean = false
            internal set

        /**
         * Diagnostic safety counter incremented if any conversion write is attempted when
         * `state != SlotState.CONVERTING` or `inUseByInference.get() == true`. Must remain 0.
         */
        @Volatile
        var writeViolationCount: Int = 0
            internal set

        @Volatile
        var lastWrittenFrameNumber: Long = -1L
            internal set

        @Volatile
        var lastDispatchedFrameNumber: Long = -1L
            internal set

        fun ensureYuvPlaneBuffers(yLen: Int, uLen: Int, vLen: Int) {
            if (yPlaneBytes.size < yLen) {
                yPlaneBytes = ByteArray(yLen)
                byteBufferAllocationCount++
            }
            if (uPlaneBytes.size < uLen) {
                uPlaneBytes = ByteArray(uLen)
                byteBufferAllocationCount++
            }
            if (vPlaneBytes.size < vLen) {
                vPlaneBytes = ByteArray(vLen)
                byteBufferAllocationCount++
            }
        }

        fun ensureRgbaPlaneBuffer(rgbaLen: Int) {
            if (rgbaPlaneBytes.size < rgbaLen) {
                rgbaPlaneBytes = ByteArray(rgbaLen)
                byteBufferAllocationCount++
            }
        }

        fun ensureTargetArgbScratch(pixelCount: Int): IntArray {
            if (targetArgbScratch.size < pixelCount) {
                targetArgbScratch = IntArray(pixelCount)
                intBufferAllocationCount++
            }
            return targetArgbScratch
        }

        fun ensureArgbScratch(pixelCount: Int): IntArray {
            if (argbScratch.size < pixelCount) {
                argbScratch = IntArray(pixelCount)
                intBufferAllocationCount++
            }
            return argbScratch
        }

        fun ensureReferenceTargetArgb(pixelCount: Int): IntArray {
            if (referenceTargetArgb.size < pixelCount) {
                referenceTargetArgb = IntArray(pixelCount)
                intBufferAllocationCount++
            }
            return referenceTargetArgb
        }

        /**
         * Precomputes 1D source byte-offset tables for `CONV:FAST` YUV_420_888 conversion.
         * Rebuilds ONLY when crop, target dimensions, rotation, or plane strides change.
         */
        fun ensureFastYuvLookupTables(
            cropLeft: Int,
            cropTop: Int,
            cropW: Int,
            cropH: Int,
            dstW: Int,
            dstH: Int,
            rot: Int,
            rowStrideY: Int,
            pixelStrideY: Int,
            rowStrideU: Int,
            pixelStrideU: Int,
            rowStrideV: Int,
            pixelStrideV: Int
        ) {
            if (fastTableCropLeft == cropLeft &&
                fastTableCropTop == cropTop &&
                fastTableCropW == cropW &&
                fastTableCropH == cropH &&
                fastTableDstW == dstW &&
                fastTableDstH == dstH &&
                fastTableRot == rot &&
                fastTableRowStrideY == rowStrideY &&
                fastTablePixelStrideY == pixelStrideY &&
                fastTableRowStrideU == rowStrideU &&
                fastTablePixelStrideU == pixelStrideU &&
                fastTableRowStrideV == rowStrideV &&
                fastTablePixelStrideV == pixelStrideV &&
                dxYOffsetTable.size >= dstW &&
                dyYOffsetTable.size >= dstH
            ) {
                return
            }

            if (dxYOffsetTable.size < dstW) {
                dxYOffsetTable = IntArray(dstW)
                dxUOffsetTable = IntArray(dstW)
                dxVOffsetTable = IntArray(dstW)
                intBufferAllocationCount++
            }
            if (dyYOffsetTable.size < dstH) {
                dyYOffsetTable = IntArray(dstH)
                dyUOffsetTable = IntArray(dstH)
                dyVOffsetTable = IntArray(dstH)
                intBufferAllocationCount++
            }

            val orientedW = if (rot == 90 || rot == 270) cropH else cropW
            val orientedH = if (rot == 90 || rot == 270) cropW else cropH
            val uvTop = cropTop shr 1

            if (rot == 0 || rot == 180) {
                for (dx in 0 until dstW) {
                    val u = (((2 * dx + 1) * orientedW) / (2 * dstW)).coerceIn(0, orientedW - 1)
                    val cx = if (rot == 0) u else (cropW - 1 - u)
                    val sx = cropLeft + cx
                    dxYOffsetTable[dx] = sx * pixelStrideY
                    val uvX = sx shr 1
                    dxUOffsetTable[dx] = uvX * pixelStrideU
                    dxVOffsetTable[dx] = uvX * pixelStrideV
                }
                for (dy in 0 until dstH) {
                    val v = (((2 * dy + 1) * orientedH) / (2 * dstH)).coerceIn(0, orientedH - 1)
                    val cy = if (rot == 0) v else (cropH - 1 - v)
                    val sy = cropTop + cy
                    dyYOffsetTable[dy] = (sy - cropTop) * rowStrideY
                    val uvY = (sy shr 1) - uvTop
                    dyUOffsetTable[dy] = uvY * rowStrideU
                    dyVOffsetTable[dy] = uvY * rowStrideV
                }
            } else {
                // rot == 90 or rot == 270: sx depends on dy, sy depends on dx
                for (dx in 0 until dstW) {
                    val u = (((2 * dx + 1) * orientedW) / (2 * dstW)).coerceIn(0, orientedW - 1)
                    val cy = if (rot == 90) (cropH - 1 - u) else u
                    val sy = cropTop + cy
                    dxYOffsetTable[dx] = (sy - cropTop) * rowStrideY
                    val uvY = (sy shr 1) - uvTop
                    dxUOffsetTable[dx] = uvY * rowStrideU
                    dxVOffsetTable[dx] = uvY * rowStrideV
                }
                for (dy in 0 until dstH) {
                    val v = (((2 * dy + 1) * orientedH) / (2 * dstH)).coerceIn(0, orientedH - 1)
                    val cx = if (rot == 90) v else (cropW - 1 - v)
                    val sx = cropLeft + cx
                    dyYOffsetTable[dy] = sx * pixelStrideY
                    val uvX = sx shr 1
                    dyUOffsetTable[dy] = uvX * pixelStrideU
                    dyVOffsetTable[dy] = uvX * pixelStrideV
                }
            }

            fastTableCropLeft = cropLeft
            fastTableCropTop = cropTop
            fastTableCropW = cropW
            fastTableCropH = cropH
            fastTableDstW = dstW
            fastTableDstH = dstH
            fastTableRot = rot
            fastTableRowStrideY = rowStrideY
            fastTablePixelStrideY = pixelStrideY
            fastTableRowStrideU = rowStrideU
            fastTablePixelStrideU = pixelStrideU
            fastTableRowStrideV = rowStrideV
            fastTablePixelStrideV = pixelStrideV
            fastTableRebuildCount++
        }

        /**
         * Precomputes 1D source byte-offset tables for `CONV:FAST` RGBA_8888 conversion.
         */
        fun ensureFastRgbaLookupTables(
            cropLeft: Int,
            cropTop: Int,
            cropW: Int,
            cropH: Int,
            dstW: Int,
            dstH: Int,
            rot: Int,
            rowStride: Int,
            pixelStride: Int
        ) {
            if (fastRgbaCropLeft == cropLeft &&
                fastRgbaCropTop == cropTop &&
                fastRgbaCropW == cropW &&
                fastRgbaCropH == cropH &&
                fastRgbaDstW == dstW &&
                fastRgbaDstH == dstH &&
                fastRgbaRot == rot &&
                fastRgbaRowStride == rowStride &&
                fastRgbaPixelStride == pixelStride &&
                dxRgbaOffsetTable.size >= dstW &&
                dyRgbaOffsetTable.size >= dstH
            ) {
                return
            }

            if (dxRgbaOffsetTable.size < dstW) {
                dxRgbaOffsetTable = IntArray(dstW)
                intBufferAllocationCount++
            }
            if (dyRgbaOffsetTable.size < dstH) {
                dyRgbaOffsetTable = IntArray(dstH)
                intBufferAllocationCount++
            }

            val orientedW = if (rot == 90 || rot == 270) cropH else cropW
            val orientedH = if (rot == 90 || rot == 270) cropW else cropH

            if (rot == 0 || rot == 180) {
                for (dx in 0 until dstW) {
                    val u = (((2 * dx + 1) * orientedW) / (2 * dstW)).coerceIn(0, orientedW - 1)
                    val cx = if (rot == 0) u else (cropW - 1 - u)
                    val sx = cropLeft + cx
                    dxRgbaOffsetTable[dx] = sx * pixelStride
                }
                for (dy in 0 until dstH) {
                    val v = (((2 * dy + 1) * orientedH) / (2 * dstH)).coerceIn(0, orientedH - 1)
                    val cy = if (rot == 0) v else (cropH - 1 - v)
                    val sy = cropTop + cy
                    dyRgbaOffsetTable[dy] = (sy - cropTop) * rowStride
                }
            } else {
                for (dx in 0 until dstW) {
                    val u = (((2 * dx + 1) * orientedW) / (2 * dstW)).coerceIn(0, orientedW - 1)
                    val cy = if (rot == 90) (cropH - 1 - u) else u
                    val sy = cropTop + cy
                    dxRgbaOffsetTable[dx] = (sy - cropTop) * rowStride
                }
                for (dy in 0 until dstH) {
                    val v = (((2 * dy + 1) * orientedH) / (2 * dstH)).coerceIn(0, orientedH - 1)
                    val cx = if (rot == 90) v else (cropW - 1 - v)
                    val sx = cropLeft + cx
                    dyRgbaOffsetTable[dy] = sx * pixelStride
                }
            }

            fastRgbaCropLeft = cropLeft
            fastRgbaCropTop = cropTop
            fastRgbaCropW = cropW
            fastRgbaCropH = cropH
            fastRgbaDstW = dstW
            fastRgbaDstH = dstH
            fastRgbaRot = rot
            fastRgbaRowStride = rowStride
            fastRgbaPixelStride = pixelStride
            fastTableRebuildCount++
        }

        fun ensureSourceBitmap(width: Int, height: Int): Bitmap? {
            val existing = sourceBitmap
            if (hasAllocatedSource &&
                sourceWidth == width &&
                sourceHeight == height &&
                (existing == null || !existing.isRecycled)
            ) {
                return existing
            }
            // Never call existing.recycle() here: an in-flight or retiring native GPU packet
            // must never encounter a recycled Bitmap. Drop the reference and let GC collect it.
            val created: Bitmap? = try {
                Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            } catch (_: Throwable) {
                null
            }
            sourceBitmap = created
            sourceWidth = width
            sourceHeight = height
            hasAllocatedSource = true
            bitmapAllocationCount++
            return created
        }

        fun ensureTargetBitmap(width: Int, height: Int): Bitmap? {
            val existing = targetBitmap
            if (hasAllocatedTarget &&
                targetWidth == width &&
                targetHeight == height &&
                (existing == null || !existing.isRecycled)
            ) {
                return existing
            }
            // Never call existing.recycle() here: drop the old reference safely without recycling.
            val created: Bitmap? = try {
                Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            } catch (_: Throwable) {
                null
            }
            targetBitmap = created
            targetCanvas = if (created != null) Canvas(created) else null
            targetWidth = width
            targetHeight = height
            hasAllocatedTarget = true
            cachedSrcW = 0 // force matrix recomputation
            bitmapAllocationCount++
            return created
        }

        internal fun recordDirectTargetDimensions(width: Int, height: Int) {
            targetWidth = width
            targetHeight = height
        }

        fun release() {
            sourceBitmap = null
            targetBitmap = null
            targetCanvas = null
            hasAllocatedSource = false
            hasAllocatedTarget = false
            sourceWidth = 0
            sourceHeight = 0
            targetWidth = 0
            targetHeight = 0
            inUseByInference.set(false)
            state = SlotState.FREE
        }
    }

    /**
     * Converted frame staged in a [SlotState.STAGED] slot waiting for `detectAsync` dispatch.
     * Holds ZERO reference to any `ImageProxy` (`frame.imageProxy` is always `null`).
     */
    internal class PreparedFrame(
        val frame: AnalysisFrame,
        val slotIndex: Int,
        val mpImage: MPImage,
        val analysisStartNs: Long,
        val conversionDoneNs: Long,
        val queueMs: Float,
        val conversionMs: Float,
        val unpackMs: Float = 0f,
        val yuvToRgbMs: Float = 0f,
        val bitmapCopyMs: Float = 0f,
        val rotateScaleMs: Float = 0f,
        val mpImageBuildMs: Float = 0f,
        val sourceWidth: Int = 0,
        val sourceHeight: Int = 0,
        val cropWidth: Int = 0,
        val cropHeight: Int = 0,
        val conversionModeLabel: String = "FAST"
    )

    /**
     * Abstraction over `FaceLandmarker` lifecycle for deterministic testing of async delegate
     * switching, blendshape toggling, watchdog recovery, and GPU->CPU fallback without requiring native JNI.
     */
    interface LandmarkerHandle {
        val delegate: Delegate
        val isClosed: Boolean
        fun detectAsync(mpImage: MPImage, timestampMs: Long)
        fun close()
    }

    internal class ManagedLandmarkerEntry(
        val handle: LandmarkerHandle,
        @Volatile var config: ProviderConfig,
        val delegate: Delegate
    ) {
        @Volatile
        var hasCompletedFirstInference: Boolean = false

        @Volatile
        var latestDispatchId: Long = 0L
    }

    private class RealLandmarkerHandle(
        override val delegate: Delegate,
        private val landmarker: FaceLandmarker
    ) : LandmarkerHandle {
        @Volatile
        override var isClosed: Boolean = false
            private set

        override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
            landmarker.detectAsync(mpImage, timestampMs)
        }

        override fun close() {
            if (!isClosed) {
                isClosed = true
                landmarker.close()
            }
        }
    }

    @Volatile
    private var dispatchThread: Thread? = null

    private var dispatchHandlerThread: HandlerThread? = null

    private val defaultDispatchExecutor: Executor = createDefaultDispatchExecutor()

    private fun createDefaultDispatchExecutor(): Executor {
        return try {
            val ht = HandlerThread("XelvicDispatchThread").apply {
                isDaemon = true
                start()
            }
            val looper = ht.looper
            if (looper != null) {
                dispatchHandlerThread = ht
                val handler = Handler(looper)
                Executor { r ->
                    handler.post {
                        dispatchThread = Thread.currentThread()
                        r.run()
                    }
                }
            } else {
                // Pure-JVM unit test fallback where android.os.HandlerThread returns null looper
                Executors.newSingleThreadExecutor { r ->
                    Thread({
                        dispatchThread = Thread.currentThread()
                        r.run()
                    }, "XelvicDispatchThread").apply { isDaemon = true }
                }
            }
        } catch (_: Throwable) {
            Executors.newSingleThreadExecutor { r ->
                Thread({
                    dispatchThread = Thread.currentThread()
                    r.run()
                }, "XelvicDispatchThread").apply { isDaemon = true }
            }
        }
    }

    private val defaultReaperExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "XelvicLandmarkerReaper").apply { isDaemon = true }
    }

    private val defaultWatchdogScheduler = Executors.newSingleThreadScheduledExecutor { r ->
        Thread(r, "XelvicWatchdogScheduler").apply { isDaemon = true }
    }

    /**
     * Dedicated single-thread dispatch executor (`XelvicDispatchThread`) that exclusively owns:
     * - `detectAsync` calls
     * - `FaceLandmarker` creation and swaps (delegate, blendshapes, watchdog reset)
     * - Watchdog timeout checks
     * - The dispatch pump (`pumpOnDispatchThread`)
     */
    @Volatile
    var dispatchExecutor: Executor = dispatchExecutorOverride ?: analysisExecutorOverride ?: defaultDispatchExecutor

    /**
     * Alias for [dispatchExecutor] kept for test compatibility.
     */
    var analysisExecutor: Executor
        get() = dispatchExecutor
        set(value) {
            dispatchExecutor = value
        }

    @Volatile
    var reaperExecutor: Executor = reaperExecutorOverride ?: defaultReaperExecutor

    @Volatile
    var watchdogScheduler: ScheduledExecutorService = watchdogSchedulerOverride ?: defaultWatchdogScheduler

    internal var customLandmarkerFactory: ((ProviderConfig, Delegate, (FaceLandmarkerResult?, MPImage) -> Unit, (RuntimeException) -> Unit) -> LandmarkerHandle?)? = null

    /**
     * Optional test hook allowing deterministic tests to intercept slot conversion (e.g., to simulate
     * conversion duration or inspect slot states mid-conversion).
     */
    internal var customSlotConverter: ((ImageProxy, Int, FrameBufferSlot) -> Bitmap?)? = null

    private val landmarkerLock = Any()
    private val conversionLock = Any()
    private val slotLock = Any()

    @Volatile
    internal var activeEntry: ManagedLandmarkerEntry? = null
        private set

    @Volatile
    internal var currentInFlightEntry: ManagedLandmarkerEntry? = null
        private set

    private val retiringEntries = ArrayList<ManagedLandmarkerEntry>(2)

    @Volatile
    private var providerGeneration: Int = 0

    private var listener: ((FaceGeometrySnapshot) -> Unit)? = null
    var onStatusChanged: ((delegateLabel: String, statusMsg: String) -> Unit)? = null

    private val isInferenceInFlight = AtomicBoolean(false)
    @Volatile
    private var stagedPreparedFrame: PreparedFrame? = null
    private val pendingConvertedFrameRef = AtomicReference<PreparedFrame?>(null)
    private val pendingDelegateSwitchRef = AtomicReference<Boolean?>(null)
    private val pendingBlendshapesSwitchRef = AtomicReference<Boolean?>(null)

    @Volatile
    var useLowAnalysisThreadPriority: Boolean = false
        private set

    @Volatile
    var lastAppliedAnalysisThreadPriority: Int = android.os.Process.THREAD_PRIORITY_DEFAULT
        private set

    // Monotonic dispatch IDs and MPImage -> dispatchId correlation for stale callback rejection
    private val nextDispatchId = AtomicLong(0L)
    @Volatile
    var activeInFlightDispatchId: Long = 0L
        private set
    private val dispatchIdByMpImage = Collections.synchronizedMap(IdentityHashMap<MPImage, Long>())

    // Monotonic timestamp (ms) passed to detectAsync — confined to dispatch thread, never resets on delegate switch or rebind
    private val lastDetectTimestampMs = AtomicLong(0L)

    // Watchdog state
    @Volatile
    var inferenceDispatchedAtNs: Long = 0L
        private set
    private var scheduledWatchdogFuture: ScheduledFuture<*>? = null
    private var scheduledCadenceFuture: ScheduledFuture<*>? = null
    private val watchdogResetTimestampsNs = ArrayDeque<Long>(8)

    @Volatile
    var isWatchdogForcedCpuForSession: Boolean = false
        private set

    // Governor cadence state
    @Volatile
    var targetTrackerCadenceHz: Int = PerformanceGovernor.MAX_TRACKER_CADENCE_HZ
        private set
    @Volatile
    var enforceCadenceLimit: Boolean = true
    @Volatile
    private var lastDispatchedFrameTimestampNs: Long = 0L
    @Volatile
    private var lastDispatchedClockNs: Long = 0L

    // Skip-conversion-if-pending-fresh state (default OFF)
    @Volatile
    override var skipConversionIfPendingFresh: Boolean = false
        private set
    @Volatile
    private var lastSubmittedFrameTimestampNs: Long = 0L
    @Volatile
    private var estimatedCameraIntervalNs: Long = 33_333_333L

    // Telemetry & diagnostic counters
    private val framesDroppedCounter = AtomicInteger(0)
    private val pendingReplacedCounter = AtomicInteger(0)
    private val overlappedConversionCounter = AtomicInteger(0)
    private val providerCadenceDeferredCounter = AtomicInteger(0)
    private val framesSkippedBeforeConversionCounter = AtomicInteger(0)
    private val watchdogResetCounter = AtomicInteger(0)
    private val staleCallbackIgnoredCounter = AtomicInteger(0)
    private val reaperCloseCounter = AtomicInteger(0)
    private val totalImageProxyClosedCounter = AtomicInteger(0)
    private val activeHeldImageProxyCounter = AtomicInteger(0)

    val framesDroppedCount: Int get() = framesDroppedCounter.get()
    val pendingReplacedCount: Int get() = pendingReplacedCounter.get()
    val overlappedConversionCount: Int get() = overlappedConversionCounter.get()
    val providerCadenceDeferredCount: Int get() = providerCadenceDeferredCounter.get()
    override val framesSkippedBeforeConversionCount: Int get() = framesSkippedBeforeConversionCounter.get()
    val watchdogResetCount: Int get() = watchdogResetCounter.get()
    val staleCallbackIgnoredCount: Int get() = staleCallbackIgnoredCounter.get()
    val reaperCloseCount: Int get() = reaperCloseCounter.get()
    val totalImageProxyClosedCount: Int get() = totalImageProxyClosedCounter.get()
    val activeHeldImageProxyCount: Int get() = activeHeldImageProxyCounter.get()

    // Idle-gap, pump-delay, and swap duration telemetry
    @Volatile
    var lastInferenceDoneNs: Long = 0L
        private set
    @Volatile
    private var pumpPostedAtNs: Long = 0L
    @Volatile
    var lastPumpDelayMs: Float = 0f
        private set
    @Volatile
    var lastIdleGapPendingMs: Float = 0f
        private set
    @Volatile
    var lastIdleGapNoPendingMs: Float = 0f
        private set
    @Volatile
    override var lastSwapMs: Float = 0f
        private set

    @Volatile
    var lastDetectAsyncThreadId: Long = -1L
        private set
    @Volatile
    var lastDetectAsyncThreadName: String = ""
        private set
    @Volatile
    var lastResultCallbackThreadId: Long = -1L
        private set
    @Volatile
    var lastResultCallbackThreadName: String = ""
        private set
    @Volatile
    var lastCreateLandmarkerThreadId: Long = -1L
        private set
    @Volatile
    var lastCreateLandmarkerThreadName: String = ""
        private set
    @Volatile
    var lastConversionThreadId: Long = -1L
        private set
    @Volatile
    var lastConversionThreadName: String = ""
        private set
    @Volatile
    var lastReaperCloseThreadName: String = ""
        private set

    // Three bitmap slots (0, 1, 2) managed by explicit SlotState machine under slotLock
    internal val slots = arrayOf(FrameBufferSlot(0), FrameBufferSlot(1), FrameBufferSlot(2))
    @Volatile
    var nextSlotIndex: Int = 0
        private set
    @Volatile
    var inFlightSlotIndex: Int = -1
        private set
    @Volatile
    var lastUsedSlotIndex: Int = -1
        private set

    private val reusableCropRect = Rect()

    // Stage-2 downscale Paint with bilinear filtering enabled
    internal val downscalePaint = Paint(Paint.FILTER_BITMAP_FLAG).apply {
        isFilterBitmap = true
    }

    @Volatile
    var activeConfig: ProviderConfig = ProviderConfig()
        private set

    @Volatile
    var analysisLongEdgePx: Int = 320
        private set

    @Volatile
    override var activeDelegateLabel: String = "CPU"
        private set

    @Volatile
    override var statusMessage: String = ""
        private set

    @Volatile
    override var hasInitError: Boolean = false
        private set

    @Volatile
    override var conversionMode: ConversionMode = ConversionMode.FAST
        private set

    @Volatile
    var currentResolutionMode: AnalysisResolutionMode = AnalysisResolutionMode.SMALL
        private set

    @Volatile
    var lastRawSourceWidth: Int = 0
        private set

    @Volatile
    var lastRawSourceHeight: Int = 0
        private set

    @Volatile
    var lastCropWidth: Int = 0
        private set

    @Volatile
    var lastCropHeight: Int = 0
        private set

    @Volatile
    var lastRotationDegrees: Int = 0
        private set

    @Volatile
    var smallModeRawSourceWidth: Int = 0
        private set
    @Volatile
    var smallModeRawSourceHeight: Int = 0
        private set
    @Volatile
    var smallModeCropWidth: Int = 0
        private set
    @Volatile
    var smallModeCropHeight: Int = 0
        private set
    @Volatile
    var smallModeRotationDeg: Int = -1
        private set

    @Volatile
    var legacyModeRawSourceWidth: Int = 0
        private set
    @Volatile
    var legacyModeRawSourceHeight: Int = 0
        private set
    @Volatile
    var legacyModeCropWidth: Int = 0
        private set
    @Volatile
    var legacyModeCropHeight: Int = 0
        private set
    @Volatile
    var legacyModeRotationDeg: Int = -1
        private set

    // Latest per-stage conversion timings (ms)
    @Volatile
    var lastUnpackMs: Float = 0f
        private set
    @Volatile
    var lastYuvToRgbMs: Float = 0f
        private set
    @Volatile
    var lastBitmapCopyMs: Float = 0f
        private set
    @Volatile
    var lastRotateScaleMs: Float = 0f
        private set
    @Volatile
    var lastMpImageBuildMs: Float = 0f
        private set

    @Volatile
    var lastSourceWidth: Int = 0
        private set

    @Volatile
    var lastSourceHeight: Int = 0
        private set

    @Volatile
    var lastScaledWidth: Int = 0
        private set

    @Volatile
    var lastScaledHeight: Int = 0
        private set

    private var currentModelSha256: String = ""

    override val modelSha256: String
        get() = currentModelSha256

    init {
        currentModelSha256 = calculateAssetSha256(modelAssetPath)
        val modelVerify = com.example.engine.pkg.UvRegistry.verifyModelSha256(currentModelSha256)
        if (!modelVerify.isValid) {
            Log.e(TAG, "Task model hash mismatch: ${modelVerify.errorReason}")
        }
    }

    private fun calculateAssetSha256(assetPath: String): String {
        return try {
            val digest = MessageDigest.getInstance("SHA-256")
            context.assets.open(assetPath).use { input ->
                val buffer = ByteArray(8192)
                var read: Int
                while (input.read(buffer).also { read = it } != -1) {
                    digest.update(buffer, 0, read)
                }
            }
            digest.digest().joinToString("") { "%02x".format(it) }
        } catch (e: Throwable) {
            Log.w(TAG, "Could not hash asset $assetPath: ${e.message}")
            com.example.engine.pkg.UvRegistry.PINNED_MODEL_SHA256
        }
    }

    fun getLastDetectTimestampMs(): Long = lastDetectTimestampMs.get()

    override fun setListener(listener: (FaceGeometrySnapshot) -> Unit) {
        this.listener = listener
    }

    override fun setSkipConversionIfPendingFresh(enabled: Boolean) {
        this.skipConversionIfPendingFresh = enabled
    }

    override fun setConversionMode(mode: ConversionMode) {
        this.conversionMode = mode
    }

    override fun setResolutionModeForTelemetry(mode: AnalysisResolutionMode) {
        this.currentResolutionMode = mode
    }

    fun formatModeSourceSummary(mode: AnalysisResolutionMode): String {
        val rawW: Int
        val rawH: Int
        val cropW: Int
        val cropH: Int
        val rot: Int
        if (mode == AnalysisResolutionMode.SMALL) {
            rawW = smallModeRawSourceWidth
            rawH = smallModeRawSourceHeight
            cropW = smallModeCropWidth
            cropH = smallModeCropHeight
            rot = smallModeRotationDeg
        } else {
            rawW = legacyModeRawSourceWidth
            rawH = legacyModeRawSourceHeight
            cropW = legacyModeCropWidth
            cropH = legacyModeCropHeight
            rot = legacyModeRotationDeg
        }
        if (rawW <= 0 || rawH <= 0 || rot < 0) return "--"
        return if (cropW > 0 && cropH > 0 && (cropW != rawW || cropH != rawH)) {
            "${rawW}x${rawH}(crop ${cropW}x${cropH})@${rot}°"
        } else {
            "${rawW}x${rawH}@${rot}°"
        }
    }

    override fun updateConfig(config: ProviderConfig) {
        val prevUseGpu = this.activeConfig.useGpu
        val effectiveGpu = config.useGpu && !isWatchdogForcedCpuForSession
        val nextConfig = config.copy(useGpu = effectiveGpu)
        this.activeConfig = nextConfig
        this.analysisLongEdgePx = config.analysisLongEdgePx.coerceIn(160, 640)
        synchronized(landmarkerLock) {
            val current = activeEntry
            if (current != null) {
                current.config = nextConfig.copy(
                    useGpu = current.config.useGpu,
                    outputBlendshapes = current.config.outputBlendshapes
                )
            }
        }
        if (prevUseGpu != config.useGpu && activeEntry != null) {
            switchDelegate(config.useGpu)
        }
    }

    override fun updateAnalysisSize(longEdgePx: Int) {
        this.analysisLongEdgePx = longEdgePx.coerceIn(160, 640)
    }

    override fun updateTargetCadenceHz(cadenceHz: Int) {
        this.targetTrackerCadenceHz = cadenceHz.coerceIn(
            PerformanceGovernor.MIN_TRACKER_CADENCE_HZ,
            PerformanceGovernor.MAX_TRACKER_CADENCE_HZ
        )
    }

    /**
     * Starts the provider asynchronously on [dispatchExecutor] without blocking the calling (main) thread.
     * While `activeEntry == null` during startup (or if creation fails), `submit()` closes each
     * `ImageProxy` immediately and does not stage frames.
     */
    override fun start(config: ProviderConfig) {
        val effectiveGpu = config.useGpu && !isWatchdogForcedCpuForSession
        val startConfig = config.copy(useGpu = effectiveGpu)
        this.activeConfig = startConfig
        this.analysisLongEdgePx = config.analysisLongEdgePx.coerceIn(160, 640)
        stop()
        val gen = ++providerGeneration
        postToDispatchThread {
            if (gen == providerGeneration) {
                createAndInstallLandmarkerOnDispatchThread(
                    config = startConfig,
                    requestedGpu = config.useGpu,
                    isWatchdogReset = false
                )
                pumpOnDispatchThread()
            }
        }
    }

    /**
     * Switches between CPU and GPU delegates on [dispatchExecutor] (never blocking the main UI thread
     * or the camera conversion thread):
     * 1. Posts delegate creation/swap onto [dispatchExecutor].
     * 2. Keeps the old landmarker active until the new one is ready, then swaps.
     * 3. Retires the old `FaceLandmarker` on [reaperExecutor] only after its in-flight inference
     *    callback finishes (or after the watchdog timeout).
     */
    override fun switchDelegate(useGpu: Boolean) {
        if (isWatchdogForcedCpuForSession && useGpu) {
            activeDelegateLabel = STATUS_GPU_TO_CPU_WATCHDOG
            statusMessage = STATUS_GPU_TO_CPU_WATCHDOG
            onStatusChanged?.invoke(activeDelegateLabel, statusMessage)
            return
        }
        pendingDelegateSwitchRef.set(useGpu)
        postToDispatchThread {
            processPendingDelegateSwitchOnDispatchThread()
            pumpOnDispatchThread()
        }
    }

    /**
     * Enables or disables `outputBlendshapes` on the `FaceLandmarker` at runtime by recreating
     * the landmarker on [dispatchExecutor] without blocking conversion, keeping the old landmarker
     * active until the new one is ready, and retiring the previous handle on [reaperExecutor]
     * after any in-flight inference finishes.
     */
    override fun setBlendshapesEnabled(enabled: Boolean) {
        pendingBlendshapesSwitchRef.set(enabled)
        postToDispatchThread {
            processPendingBlendshapesSwitchOnDispatchThread()
            pumpOnDispatchThread()
        }
    }

    /**
     * Adjusts the thread priority of the camera analysis/conversion worker between default
     * (`Process.THREAD_PRIORITY_DEFAULT`, `0`) and background (`Process.THREAD_PRIORITY_BACKGROUND`, `10`).
     */
    override fun setUseLowThreadPriority(lowPriority: Boolean) {
        useLowAnalysisThreadPriority = lowPriority
        lastAppliedAnalysisThreadPriority = if (lowPriority) {
            android.os.Process.THREAD_PRIORITY_BACKGROUND
        } else {
            android.os.Process.THREAD_PRIORITY_DEFAULT
        }
    }

    @Volatile
    private var lastOsAppliedThreadId: Long = -1L
    @Volatile
    private var lastOsAppliedThreadPrio: Int = Int.MIN_VALUE

    private fun applyCurrentThreadPriority() {
        val targetPrio = if (useLowAnalysisThreadPriority) {
            android.os.Process.THREAD_PRIORITY_BACKGROUND
        } else {
            android.os.Process.THREAD_PRIORITY_DEFAULT
        }
        lastAppliedAnalysisThreadPriority = targetPrio
        val tid = Thread.currentThread().id
        if (tid == lastOsAppliedThreadId && targetPrio == lastOsAppliedThreadPrio) {
            return
        }
        lastOsAppliedThreadId = tid
        lastOsAppliedThreadPrio = targetPrio
        try {
            android.os.Process.setThreadPriority(targetPrio)
        } catch (_: Throwable) {
        }
    }

    /**
     * Waits until all currently queued tasks on [dispatchExecutor] have completed (for tests).
     */
    fun awaitAnalysisIdle(timeoutMs: Long = 2000L): Boolean = awaitDispatchIdle(timeoutMs)

    /**
     * Waits until all currently queued tasks on [dispatchExecutor] have completed (for tests).
     */
    fun awaitDispatchIdle(timeoutMs: Long = 2000L): Boolean {
        if (Thread.currentThread() === dispatchThread) return true
        val latch = CountDownLatch(1)
        return try {
            dispatchExecutor.execute {
                dispatchThread = Thread.currentThread()
                latch.countDown()
            }
            latch.await(timeoutMs, TimeUnit.MILLISECONDS)
        } catch (_: Throwable) {
            false
        }
    }

    /**
     * Waits until all currently queued tasks on [reaperExecutor] have completed (for tests).
     */
    fun awaitReaperIdle(timeoutMs: Long = 2000L): Boolean {
        val latch = CountDownLatch(1)
        return try {
            reaperExecutor.execute {
                latch.countDown()
            }
            latch.await(timeoutMs, TimeUnit.MILLISECONDS)
        } catch (_: Throwable) {
            false
        }
    }

    private fun runOnDispatchThreadBlockingOrInline(block: () -> Unit) {
        if (Thread.currentThread() === dispatchThread) {
            block()
            return
        }
        val latch = CountDownLatch(1)
        var thrown: Throwable? = null
        try {
            dispatchExecutor.execute {
                dispatchThread = Thread.currentThread()
                try {
                    block()
                } catch (t: Throwable) {
                    thrown = t
                } finally {
                    latch.countDown()
                }
            }
            if (!latch.await(5000L, TimeUnit.MILLISECONDS)) {
                block()
            }
        } catch (_: Throwable) {
            block()
        }
        thrown?.let { Log.e(TAG, "Error on dispatch thread: ${it.message}", it) }
    }

    private fun postToDispatchThread(block: () -> Unit) {
        try {
            dispatchExecutor.execute {
                dispatchThread = Thread.currentThread()
                try {
                    block()
                } catch (t: Throwable) {
                    Log.e(TAG, "Unhandled throwable on dispatchExecutor: ${t.message}", t)
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to post to dispatchExecutor: ${t.message}", t)
        }
    }

    private fun postPumpToDispatchThread(fromResultCallback: Boolean) {
        if (fromResultCallback) {
            pumpPostedAtNs = FrameClock.elapsedRealtimeNs()
        } else if (pumpPostedAtNs == 0L) {
            pumpPostedAtNs = FrameClock.elapsedRealtimeNs()
        }
        postToDispatchThread {
            pumpOnDispatchThread()
        }
    }

    private fun closeHandleOnReaper(handle: LandmarkerHandle?) {
        if (handle == null) return
        try {
            reaperExecutor.execute {
                lastReaperCloseThreadName = Thread.currentThread().name
                reaperCloseCounter.incrementAndGet()
                try {
                    handle.close()
                } catch (t: Throwable) {
                    Log.w(TAG, "Error closing landmarker on reaper thread: ${t.message}")
                }
            }
        } catch (t: Throwable) {
            Log.w(TAG, "Reaper executor rejected close: ${t.message}")
        }
    }

    private fun processPendingDelegateSwitchOnDispatchThread() {
        val requestedGpu = pendingDelegateSwitchRef.getAndSet(null) ?: return
        val cfg = activeConfig.copy(useGpu = requestedGpu && !isWatchdogForcedCpuForSession)
        createAndInstallLandmarkerOnDispatchThread(cfg, requestedGpu, isWatchdogReset = false)
    }

    private fun processPendingBlendshapesSwitchOnDispatchThread() {
        val requestedBlendshapes = pendingBlendshapesSwitchRef.getAndSet(null) ?: return
        val cfg = activeConfig.copy(outputBlendshapes = requestedBlendshapes)
        createAndInstallLandmarkerOnDispatchThread(
            config = cfg,
            requestedGpu = cfg.useGpu && !isWatchdogForcedCpuForSession,
            isWatchdogReset = false
        )
    }

    private fun createAndInstallLandmarkerOnDispatchThread(
        config: ProviderConfig,
        requestedGpu: Boolean,
        isWatchdogReset: Boolean
    ) {
        val swapStartNs = FrameClock.elapsedRealtimeNs()
        lastCreateLandmarkerThreadId = Thread.currentThread().id
        lastCreateLandmarkerThreadName = Thread.currentThread().name

        var newEntry: ManagedLandmarkerEntry? = null
        var delegateLabelResult = "CPU"

        val effectiveRequestGpu = requestedGpu && !isWatchdogForcedCpuForSession
        if (isWatchdogForcedCpuForSession) {
            newEntry = tryCreateLandmarkerEntry(config.copy(useGpu = false), Delegate.CPU)
            delegateLabelResult = if (newEntry != null) STATUS_GPU_TO_CPU_WATCHDOG else STATUS_INIT_ERROR
        } else if (effectiveRequestGpu) {
            newEntry = tryCreateLandmarkerEntry(config.copy(useGpu = true), Delegate.GPU)
            if (newEntry != null) {
                delegateLabelResult = "GPU"
            } else {
                Log.w(TAG, "GPU delegate unavailable, falling back to CPU delegate")
                newEntry = tryCreateLandmarkerEntry(config.copy(useGpu = false), Delegate.CPU)
                delegateLabelResult = if (newEntry != null) STATUS_GPU_TO_CPU_FALLBACK else STATUS_INIT_ERROR
            }
        } else {
            newEntry = tryCreateLandmarkerEntry(config.copy(useGpu = false), Delegate.CPU)
            delegateLabelResult = if (newEntry != null) "CPU" else STATUS_INIT_ERROR
        }

        if (newEntry == null) {
            hasInitError = true
            activeDelegateLabel = STATUS_INIT_ERROR
            statusMessage = STATUS_INIT_ERROR
        } else {
            hasInitError = false
            activeConfig = newEntry.config
            activeDelegateLabel = delegateLabelResult
            if (isWatchdogForcedCpuForSession) {
                statusMessage = STATUS_GPU_TO_CPU_WATCHDOG
            } else if (isWatchdogReset) {
                statusMessage = STATUS_TRACKER_RESET
            } else if (delegateLabelResult == STATUS_GPU_TO_CPU_FALLBACK) {
                statusMessage = STATUS_GPU_TO_CPU_FALLBACK
            } else if (statusMessage == STATUS_INIT_ERROR) {
                statusMessage = ""
            }
        }

        val oldEntry: ManagedLandmarkerEntry?
        val closeImmediately: Boolean
        synchronized(landmarkerLock) {
            oldEntry = activeEntry
            activeEntry = newEntry
            if (oldEntry != null && oldEntry !== newEntry) {
                if (isInferenceInFlight.get() && currentInFlightEntry === oldEntry) {
                    if (!retiringEntries.contains(oldEntry)) {
                        retiringEntries.add(oldEntry)
                    }
                    closeImmediately = false
                } else {
                    closeImmediately = true
                }
            } else {
                closeImmediately = false
            }
        }

        if (closeImmediately && oldEntry != null) {
            closeHandleOnReaper(oldEntry.handle)
        }

        val swapDoneNs = FrameClock.elapsedRealtimeNs()
        lastSwapMs = if (swapDoneNs >= swapStartNs) {
            (swapDoneNs - swapStartNs).toFloat() / 1_000_000f
        } else {
            0f
        }

        onStatusChanged?.invoke(activeDelegateLabel, statusMessage)
    }

    private fun tryCreateLandmarkerEntry(config: ProviderConfig, delegate: Delegate): ManagedLandmarkerEntry? {
        val entryRef = AtomicReference<ManagedLandmarkerEntry?>(null)

        val resultCallback: (FaceLandmarkerResult?, MPImage) -> Unit = { res, img ->
            val entry = entryRef.get()
            val cbDispatchId = dispatchIdByMpImage.remove(img) ?: (entry?.latestDispatchId ?: -1L)
            handleResult(cbDispatchId, entry, res, img)
        }
        val errorCallback: (RuntimeException) -> Unit = { err ->
            val entry = entryRef.get()
            val cbDispatchId = entry?.latestDispatchId ?: -1L
            handleInferenceError(cbDispatchId, entry, err)
        }

        customLandmarkerFactory?.let { factory ->
            return try {
                val handle = factory.invoke(config, delegate, resultCallback, errorCallback)
                if (handle != null) {
                    val createdEntry = ManagedLandmarkerEntry(handle, config, delegate)
                    entryRef.set(createdEntry)
                    createdEntry
                } else {
                    null
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Custom landmarker factory failed for $delegate: ${t.message}")
                null
            }
        }

        return try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelAssetPath)
                .setDelegate(delegate)
                .build()

            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setOutputFaceBlendshapes(config.outputBlendshapes)
                .setOutputFacialTransformationMatrixes(true)
                .setNumFaces(config.maxFaces)
                .setResultListener { result, mpImage ->
                    resultCallback(result, mpImage)
                }
                .setErrorListener { error ->
                    errorCallback(error)
                }
                .build()

            val created = FaceLandmarker.createFromOptions(context, options)
            Log.i(TAG, "MediaPipe FaceLandmarker started successfully (delegate: $delegate)")
            val createdEntry = ManagedLandmarkerEntry(RealLandmarkerHandle(delegate, created), config, delegate)
            entryRef.set(createdEntry)
            createdEntry
        } catch (t: Throwable) {
            Log.e(TAG, "Failed to create FaceLandmarker with $delegate: ${t.message}")
            null
        }
    }

    private fun releaseInFlightSlotAndRetiringLandmarkers(expectedDispatchId: Long = 0L) {
        var toClose: List<ManagedLandmarkerEntry>? = null
        val shouldReleaseSlot: Boolean
        synchronized(landmarkerLock) {
            if (expectedDispatchId > 0L && activeInFlightDispatchId != expectedDispatchId) {
                shouldReleaseSlot = false
            } else {
                shouldReleaseSlot = true
                currentInFlightPrepared = null
                activeInFlightDispatchId = 0L
                currentInFlightEntry = null
                isInferenceInFlight.set(false)
                if (retiringEntries.isNotEmpty()) {
                    toClose = ArrayList(retiringEntries)
                    retiringEntries.clear()
                }
            }
        }
        if (!shouldReleaseSlot) return

        synchronized(slotLock) {
            val slotIdx = inFlightSlotIndex
            if (slotIdx in slots.indices) {
                slots[slotIdx].inUseByInference.set(false)
                slots[slotIdx].state = SlotState.FREE
            }
            inFlightSlotIndex = -1
        }

        toClose?.forEach { entry ->
            closeHandleOnReaper(entry.handle)
        }
    }

    private fun clearStagedSlot() {
        synchronized(slotLock) {
            val prev = stagedPreparedFrame
            if (prev != null) {
                val s = slots[prev.slotIndex]
                if (s.state == SlotState.STAGED) {
                    s.state = SlotState.FREE
                }
            }
            stagedPreparedFrame = null
            pendingConvertedFrameRef.set(null)
        }
    }

    /**
     * Schedules a watchdog check via [watchdogScheduler] that ONLY posts the check to [dispatchExecutor].
     */
    private fun scheduleWatchdogTimer(timeoutMs: Long, expectedDispatchId: Long) {
        try {
            scheduledWatchdogFuture?.cancel(false)
            scheduledWatchdogFuture = watchdogScheduler.schedule({
                postToDispatchThread {
                    if (activeInFlightDispatchId == expectedDispatchId && isInferenceInFlight.get()) {
                        checkWatchdogOnDispatchThread()
                        pumpOnDispatchThread()
                    }
                }
            }, timeoutMs.coerceAtLeast(1L), TimeUnit.MILLISECONDS)
        } catch (_: Throwable) {
        }
    }

    private fun scheduleCadenceWakeup(delayMs: Long) {
        try {
            scheduledCadenceFuture?.cancel(false)
            scheduledCadenceFuture = watchdogScheduler.schedule({
                postToDispatchThread {
                    pumpOnDispatchThread()
                }
            }, delayMs.coerceAtLeast(1L), TimeUnit.MILLISECONDS)
        } catch (_: Throwable) {
        }
    }

    /**
     * Explicitly triggers a watchdog evaluation on [dispatchExecutor] (useful when advancing
     * a fake clock in unit tests).
     */
    fun triggerWatchdogCheck(nowNs: Long = FrameClock.elapsedRealtimeNs()) {
        runOnDispatchThreadBlockingOrInline {
            checkWatchdogOnDispatchThread(nowNs)
            pumpOnDispatchThread()
        }
    }

    internal fun checkWatchdogOnAnalysisThread(nowNs: Long = FrameClock.elapsedRealtimeNs()): Boolean =
        checkWatchdogOnDispatchThread(nowNs)

    /**
     * Checks whether the current in-flight inference has exceeded its timeout:
     * - Evaluates first-inference grace (`5,000 ms`) vs steady-state (`500 ms`) based on
     *   [currentInFlightEntry] (never `activeEntry`, which may have been replaced by a runtime
     *   `BLEND` or `DEL` swap while `currentInFlightEntry` was still in flight).
     * - If an inference is still in flight and has not yet timed out, ALWAYS reschedules the
     *   watchdog timer for the remaining timeout so the watchdog cannot die silently.
     * - If timed out:
     *   - Clears `isInferenceInFlight`, invalidates `activeInFlightDispatchId`, and releases the
     *     `IN_FLIGHT` slot to `FREE`.
     *   - Closes the wedged `inFlightEntry` on [reaperExecutor].
     *   - If a newer `activeEntry` was already installed by a `BLEND` or `DEL` swap (`activeEntry !== inFlightEntry`),
     *     keeps the new `activeEntry` intact! Only recreates `activeEntry` if `activeEntry === inFlightEntry`
     *     or `activeEntry == null` (or if 3-strike CPU fallback triggered).
     */
    internal fun checkWatchdogOnDispatchThread(nowNs: Long = FrameClock.elapsedRealtimeNs()): Boolean {
        if (!isInferenceInFlight.get()) return false
        val dispatchedAt = inferenceDispatchedAtNs
        if (dispatchedAt <= 0L) return false

        val inFlightEntry = currentInFlightEntry ?: activeEntry
        val isFirstInference = (inFlightEntry == null || !inFlightEntry.hasCompletedFirstInference)
        val timeoutNs = if (isFirstInference) {
            FIRST_INFERENCE_WATCHDOG_TIMEOUT_NS
        } else {
            STEADY_WATCHDOG_TIMEOUT_NS
        }

        val elapsedNs = (nowNs - dispatchedAt).coerceAtLeast(0L)
        if (elapsedNs < timeoutNs) {
            val remainingMs = ((timeoutNs - elapsedNs + 999_999L) / 1_000_000L).coerceAtLeast(1L)
            scheduleWatchdogTimer(remainingMs, activeInFlightDispatchId)
            return false
        }

        // Watchdog timeout exceeded!
        scheduledWatchdogFuture?.cancel(false)
        val timedOutDispatchId: Long
        val timedOutEntry: ManagedLandmarkerEntry?
        synchronized(landmarkerLock) {
            timedOutDispatchId = activeInFlightDispatchId
            activeInFlightDispatchId = 0L
            currentInFlightPrepared?.let { dispatchIdByMpImage.remove(it.mpImage) }
            currentInFlightPrepared = null
            timedOutEntry = currentInFlightEntry
            currentInFlightEntry = null
            isInferenceInFlight.set(false)
        }

        synchronized(slotLock) {
            val slotIdx = inFlightSlotIndex
            if (slotIdx in slots.indices) {
                slots[slotIdx].inUseByInference.set(false)
                slots[slotIdx].state = SlotState.FREE
            }
            inFlightSlotIndex = -1
        }

        val resetNum = watchdogResetCounter.incrementAndGet()

        // Maintain 30-second rolling window of watchdog resets
        val cutoffNs = nowNs - WATCHDOG_FALLBACK_WINDOW_NS
        while (watchdogResetTimestampsNs.isNotEmpty() && watchdogResetTimestampsNs.first() < cutoffNs) {
            watchdogResetTimestampsNs.removeFirst()
        }
        watchdogResetTimestampsNs.addLast(nowNs)

        val forcedCpuNow: Boolean
        if (watchdogResetTimestampsNs.size >= WATCHDOG_FALLBACK_MAX_RESETS) {
            isWatchdogForcedCpuForSession = true
            activeConfig = activeConfig.copy(useGpu = false)
            forcedCpuNow = true
            Log.w(
                TAG,
                "Watchdog reset #$resetNum (dispatchId=$timedOutDispatchId): 3 resets within 30s -> permanent CPU fallback ($STATUS_GPU_TO_CPU_WATCHDOG)"
            )
        } else {
            forcedCpuNow = false
            Log.w(
                TAG,
                "Watchdog reset #$resetNum (dispatchId=$timedOutDispatchId, firstInference=$isFirstInference): inference exceeded ${timeoutNs / 1_000_000L}ms -> resetting landmarker"
            )
        }

        val entriesToClose = ArrayList<ManagedLandmarkerEntry>(3)
        var shouldRecreateActive = false
        synchronized(landmarkerLock) {
            if (timedOutEntry != null) {
                entriesToClose.add(timedOutEntry)
                retiringEntries.remove(timedOutEntry)
            }
            if (retiringEntries.isNotEmpty()) {
                entriesToClose.addAll(retiringEntries)
                retiringEntries.clear()
            }
            val currentActive = activeEntry
            if (currentActive == null || currentActive === timedOutEntry) {
                if (currentActive != null && !entriesToClose.contains(currentActive)) {
                    entriesToClose.add(currentActive)
                }
                activeEntry = null
                shouldRecreateActive = true
            } else if (forcedCpuNow && currentActive.delegate == Delegate.GPU) {
                entriesToClose.add(currentActive)
                activeEntry = null
                shouldRecreateActive = true
            }
        }

        for (entry in entriesToClose) {
            closeHandleOnReaper(entry.handle)
        }

        if (shouldRecreateActive) {
            createAndInstallLandmarkerOnDispatchThread(
                config = activeConfig,
                requestedGpu = activeConfig.useGpu && !isWatchdogForcedCpuForSession,
                isWatchdogReset = true
            )
        } else {
            statusMessage = if (isWatchdogForcedCpuForSession) {
                STATUS_GPU_TO_CPU_WATCHDOG
            } else {
                STATUS_TRACKER_RESET
            }
            onStatusChanged?.invoke(activeDelegateLabel, statusMessage)
        }
        return true
    }

    private val stillLock = Any()
    private var stillLandmarker: FaceLandmarker? = null

    private fun getOrCreateStillLandmarker(): FaceLandmarker? {
        synchronized(stillLock) {
            stillLandmarker?.let { return it }
            return try {
                val baseOptions = BaseOptions.builder()
                    .setModelAssetPath(modelAssetPath)
                    .setDelegate(Delegate.CPU)
                    .build()
                val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setRunningMode(RunningMode.IMAGE)
                    .setOutputFaceBlendshapes(true)
                    .setOutputFacialTransformationMatrixes(true)
                    .setNumFaces(1)
                    .build()
                val created = FaceLandmarker.createFromOptions(context, options)
                stillLandmarker = created
                created
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to create still-image FaceLandmarker: ${t.message}")
                null
            }
        }
    }

    override fun detectStillImage(
        bitmap: Bitmap,
        rotationDegrees: Int,
        isFrontCamera: Boolean,
        timestampNs: Long
    ): FaceGeometrySnapshot {
        val stillCropAspect = if (bitmap.width > 0 && bitmap.height > 0) {
            bitmap.width.toFloat() / bitmap.height.toFloat()
        } else {
            9f / 16f
        }
        val stillProjParams = ProjectionParams.DEFAULT.copy(cropAspectRatio = stillCropAspect)

        val detector = getOrCreateStillLandmarker()
        if (detector == null) {
            return FaceGeometrySnapshot(
                timestampNs = timestampNs,
                imageWidth = bitmap.width,
                imageHeight = bitmap.height,
                rotationDegrees = rotationDegrees,
                isFrontCamera = isFrontCamera,
                isTracking = false,
                landmarks = null,
                rawPoseMatrix = null,
                blendshapes = emptyMap(),
                modelSha256 = currentModelSha256,
                projectionParams = stillProjParams
            )
        }

        return synchronized(stillLock) {
            try {
                val mpImage = BitmapImageBuilder(bitmap).build()
                val result = detector.detect(mpImage)
                val landmarksList = result.faceLandmarks()
                val hasFace = landmarksList.isNotEmpty() && landmarksList[0].isNotEmpty()
                if (!hasFace) {
                    return@synchronized FaceGeometrySnapshot(
                        timestampNs = timestampNs,
                        imageWidth = bitmap.width,
                        imageHeight = bitmap.height,
                        rotationDegrees = rotationDegrees,
                        isFrontCamera = isFrontCamera,
                        isTracking = false,
                        landmarks = null,
                        rawPoseMatrix = null,
                        blendshapes = emptyMap(),
                        modelSha256 = currentModelSha256,
                        projectionParams = stillProjParams
                    )
                }

                val landmarksObj = landmarksList[0]
                val lms = FloatArray(landmarksObj.size * 3)
                for (i in landmarksObj.indices) {
                    val lm = landmarksObj[i]
                    lms[i * 3] = lm.x()
                    lms[i * 3 + 1] = lm.y()
                    lms[i * 3 + 2] = lm.z()
                }

                var rawMpMatrix: FloatArray? = null
                if (result.facialTransformationMatrixes().isPresent) {
                    val matrixList = result.facialTransformationMatrixes().get()
                    if (matrixList.isNotEmpty()) {
                        rawMpMatrix = matrixList[0].clone()
                    }
                }
                val poseMatrix = calibratePoseMatrix(
                    rawMpMatrix = rawMpMatrix,
                    lm = lms,
                    imageWidth = bitmap.width,
                    imageHeight = bitmap.height,
                    fovYCropDeg = stillProjParams.fovYDeg
                )

                val blendshapesMap = HashMap<String, Float>()
                if (result.faceBlendshapes().isPresent) {
                    val bsList = result.faceBlendshapes().get()
                    if (bsList.isNotEmpty()) {
                        for (category in bsList[0]) {
                            blendshapesMap[category.categoryName()] = category.score()
                        }
                    }
                }

                FaceGeometrySnapshot.create(
                    timestampNs = timestampNs,
                    imageWidth = bitmap.width,
                    imageHeight = bitmap.height,
                    rotationDegrees = rotationDegrees,
                    isFrontCamera = isFrontCamera,
                    isTracking = true,
                    landmarks = lms,
                    rawPoseMatrix = poseMatrix,
                    blendshapes = blendshapesMap,
                    modelSha256 = currentModelSha256,
                    projectionParams = stillProjParams
                )
            } catch (t: Throwable) {
                Log.e(TAG, "Error running still-image face inference: ${t.message}", t)
                FaceGeometrySnapshot(
                    timestampNs = timestampNs,
                    imageWidth = bitmap.width,
                    imageHeight = bitmap.height,
                    rotationDegrees = rotationDegrees,
                    isFrontCamera = isFrontCamera,
                    isTracking = false,
                    landmarks = null,
                    rawPoseMatrix = null,
                    blendshapes = emptyMap(),
                    modelSha256 = currentModelSha256,
                    projectionParams = stillProjParams
                )
            }
        }
    }

    @Volatile
    override var isPaused: Boolean = false
        private set

    @Volatile
    var pauseGeneration: Int = 0
        private set

    @Volatile
    var droppedInFlightResultCount: Int = 0
        private set

    override fun setPaused(paused: Boolean) {
        if (paused) {
            isPaused = true
            pauseGeneration++
            scheduledWatchdogFuture?.cancel(false)
            scheduledCadenceFuture?.cancel(false)
            clearStagedSlot()
        } else {
            isPaused = false
        }
    }

    override fun stop() {
        isPaused = false
        pauseGeneration++
        providerGeneration++
        scheduledWatchdogFuture?.cancel(false)
        scheduledCadenceFuture?.cancel(false)
        pendingDelegateSwitchRef.set(null)
        pendingBlendshapesSwitchRef.set(null)
        dispatchIdByMpImage.clear()

        val toClose = ArrayList<ManagedLandmarkerEntry>(3)
        synchronized(landmarkerLock) {
            activeInFlightDispatchId = 0L
            currentInFlightPrepared = null
            currentInFlightEntry = null
            isInferenceInFlight.set(false)
            activeEntry?.let { toClose.add(it) }
            activeEntry = null
            toClose.addAll(retiringEntries)
            retiringEntries.clear()
        }
        for (entry in toClose) {
            closeHandleOnReaper(entry.handle)
        }
        synchronized(stillLock) {
            try {
                stillLandmarker?.close()
            } catch (t: Throwable) {
                Log.w(TAG, "Error closing stillLandmarker: ${t.message}")
            }
            stillLandmarker = null
        }
        synchronized(slotLock) {
            stagedPreparedFrame = null
            pendingConvertedFrameRef.set(null)
            for (slot in slots) {
                slot.inUseByInference.set(false)
                slot.state = SlotState.FREE
            }
            inFlightSlotIndex = -1
            isInferenceInFlight.set(false)
        }
        lastDispatchedFrameTimestampNs = 0L
        lastDispatchedClockNs = 0L
        lastSubmittedFrameTimestampNs = 0L
        lastInferenceDoneNs = 0L
        pumpPostedAtNs = 0L
        lastPumpDelayMs = 0f
        lastIdleGapPendingMs = 0f
        lastIdleGapNoPendingMs = 0f
    }

    /**
     * Submits a camera [AnalysisFrame] on the camera analysis thread for conversion and staging:
     * - Never calls `detectAsync` or creates/closes a landmarker on the conversion thread.
     * - Guarantees that `frame.imageProxy` is closed **synchronously and exactly once** inside
     *   this call's `try/finally` block (`activeHeldImageProxyCount` returns to `0` before return).
     * - While there is no active landmarker (`activeEntry == null`), closes `frame.imageProxy`
     *   immediately without converting or staging frames.
     * - Converts into a [SlotState.FREE] slot (`FREE -> CONVERTING`) and atomically swaps it into
     *   [SlotState.STAGED] under `slotLock` (moving any replaced `STAGED` slot back to `FREE`).
     */
    override fun submit(frame: AnalysisFrame) {
        val proxy = frame.imageProxy
        var proxyClosed = false
        if (proxy != null) {
            activeHeldImageProxyCounter.incrementAndGet()
        }
        try {
            applyCurrentThreadPriority()
            val gen = providerGeneration
            if (isPaused || activeEntry == null || proxy == null) {
                framesDroppedCounter.incrementAndGet()
                return
            }

            // Track rolling camera frame interval for skipConversionIfPendingFresh
            val prevSubTs = lastSubmittedFrameTimestampNs
            if (frame.timestampNs > prevSubTs && prevSubTs > 0L) {
                val deltaNs = frame.timestampNs - prevSubTs
                if (deltaNs in 5_000_000L..200_000_000L) {
                    estimatedCameraIntervalNs = ((estimatedCameraIntervalNs * 3L) + deltaNs) / 4L
                }
            }
            if (frame.timestampNs > 0L) {
                lastSubmittedFrameTimestampNs = frame.timestampNs
            }

            val analysisStartNs = FrameClock.elapsedRealtimeNs()

            // Optional skipConversionIfPendingFresh gate:
            // Do not convert a new frame while a converted frame is pending (STAGED) and younger
            // than 1.5 camera intervals and an inference is in flight.
            if (skipConversionIfPendingFresh) {
                val skipNow = synchronized(slotLock) {
                    val staged = stagedPreparedFrame
                    if (isInferenceInFlight.get() && staged != null) {
                        val stagedAgeNs = (analysisStartNs - staged.conversionDoneNs).coerceAtLeast(0L)
                        val freshnessThresholdNs = (estimatedCameraIntervalNs * 3L) / 2L
                        stagedAgeNs < freshnessThresholdNs
                    } else {
                        false
                    }
                }
                if (skipNow) {
                    framesSkippedBeforeConversionCounter.incrementAndGet()
                    framesDroppedCounter.incrementAndGet()
                    return
                }
            }

            val queueMs = if (frame.submittedAtNs > 0L && analysisStartNs >= frame.submittedAtNs) {
                (analysisStartNs - frame.submittedAtNs).toFloat() / 1_000_000f
            } else {
                0f
            }

            // Acquire a FREE slot under slotLock and transition it to CONVERTING
            val targetSlot: FrameBufferSlot = synchronized(slotLock) {
                if (isPaused || activeEntry == null || gen != providerGeneration) {
                    framesDroppedCounter.incrementAndGet()
                    return
                }
                var chosen: FrameBufferSlot? = null
                for (i in slots.indices) {
                    val idx = (nextSlotIndex + i) % slots.size
                    val candidate = slots[idx]
                    if (candidate.state == SlotState.FREE && !candidate.inUseByInference.get()) {
                        chosen = candidate
                        nextSlotIndex = (idx + 1) % slots.size
                        break
                    }
                }
                if (chosen == null) {
                    framesDroppedCounter.incrementAndGet()
                    return
                }
                chosen.state = SlotState.CONVERTING
                if (isInferenceInFlight.get()) {
                    overlappedConversionCounter.incrementAndGet()
                }
                chosen
            }

            var prepared: PreparedFrame? = null
            var conversionSucceeded = false
            try {
                synchronized(conversionLock) {
                    lastConversionThreadId = Thread.currentThread().id
                    lastConversionThreadName = Thread.currentThread().name

                    if (targetSlot.state != SlotState.CONVERTING || targetSlot.inUseByInference.get()) {
                        targetSlot.writeViolationCount++
                        return
                    }

                    val bitmap = customSlotConverter?.invoke(proxy, frame.rotationDegrees, targetSlot)
                        ?: convertAndTransformIntoSlot(proxy, frame.rotationDegrees, targetSlot)

                    if (targetSlot.state != SlotState.CONVERTING || targetSlot.inUseByInference.get()) {
                        targetSlot.writeViolationCount++
                        return
                    }
                    targetSlot.lastWrittenFrameNumber = frame.frameNumber

                    // Close ImageProxy immediately once pixel unpack completes
                    if (!proxyClosed) {
                        proxyClosed = true
                        try {
                            proxy.close()
                        } finally {
                            totalImageProxyClosedCounter.incrementAndGet()
                            activeHeldImageProxyCounter.decrementAndGet()
                        }
                    }

                    if (targetSlot.lastConversionSucceeded && !isPaused && activeEntry != null && gen == providerGeneration) {
                        val mpBuildStartNs = FrameClock.elapsedRealtimeNs()
                        val mpImage = if (bitmap != null) {
                            BitmapImageBuilder(bitmap).build()
                        } else {
                            val w = targetSlot.targetWidth.coerceAtLeast(1)
                            val h = targetSlot.targetHeight.coerceAtLeast(1)
                            com.google.mediapipe.framework.image.ByteBufferImageBuilder(
                                java.nio.ByteBuffer.allocateDirect(w * h * 4),
                                w,
                                h,
                                MPImage.IMAGE_FORMAT_RGBA
                            ).build()
                        }
                        val conversionDoneNs = FrameClock.elapsedRealtimeNs()
                        val mpBuildMs = if (conversionDoneNs >= mpBuildStartNs) {
                            (conversionDoneNs - mpBuildStartNs).toFloat() / 1_000_000f
                        } else {
                            0f
                        }
                        targetSlot.lastMpImageBuildMs = mpBuildMs
                        lastMpImageBuildMs = mpBuildMs

                        val conversionMs = if (conversionDoneNs >= analysisStartNs) {
                            (conversionDoneNs - analysisStartNs).toFloat() / 1_000_000f
                        } else {
                            0f
                        }
                        prepared = PreparedFrame(
                            frame = frame.copy(imageProxy = null),
                            slotIndex = targetSlot.index,
                            mpImage = mpImage,
                            analysisStartNs = analysisStartNs,
                            conversionDoneNs = conversionDoneNs,
                            queueMs = queueMs,
                            conversionMs = conversionMs,
                            unpackMs = targetSlot.lastUnpackMs,
                            yuvToRgbMs = targetSlot.lastYuvToRgbMs,
                            bitmapCopyMs = targetSlot.lastBitmapCopyMs,
                            rotateScaleMs = targetSlot.lastRotateScaleMs,
                            mpImageBuildMs = mpBuildMs,
                            sourceWidth = targetSlot.lastRawSourceWidth,
                            sourceHeight = targetSlot.lastRawSourceHeight,
                            cropWidth = targetSlot.lastCropWidth,
                            cropHeight = targetSlot.lastCropHeight,
                            conversionModeLabel = conversionMode.name
                        )
                        conversionSucceeded = true
                    }
                }
            } finally {
                // Atomically transition targetSlot from CONVERTING -> STAGED (or back to FREE on abort)
                synchronized(slotLock) {
                    val readyFrame = prepared
                    if (!conversionSucceeded || readyFrame == null || isPaused || activeEntry == null || gen != providerGeneration) {
                        if (targetSlot.state == SlotState.CONVERTING) {
                            targetSlot.state = SlotState.FREE
                        }
                        prepared = null
                    } else {
                        val prevStaged = stagedPreparedFrame
                        if (prevStaged != null) {
                            val prevSlot = slots[prevStaged.slotIndex]
                            if (prevSlot.state == SlotState.STAGED) {
                                prevSlot.state = SlotState.FREE
                            }
                            pendingReplacedCounter.incrementAndGet()
                            framesDroppedCounter.incrementAndGet()
                        }
                        targetSlot.state = SlotState.STAGED
                        stagedPreparedFrame = readyFrame
                        pendingConvertedFrameRef.set(readyFrame)
                    }
                }
            }

            if (prepared != null) {
                postPumpToDispatchThread(fromResultCallback = false)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Throwable in submit: ${t.message}", t)
        } finally {
            if (!proxyClosed && proxy != null) {
                proxyClosed = true
                try {
                    proxy.close()
                } catch (_: Throwable) {
                } finally {
                    totalImageProxyClosedCounter.incrementAndGet()
                    activeHeldImageProxyCounter.decrementAndGet()
                }
            }
        }
    }

    @Volatile
    private var currentInFlightPrepared: PreparedFrame? = null
    @Volatile
    private var currentInFlightPendingWaitMs: Float = 0f
    @Volatile
    private var currentInFlightDispatchNs: Long = 0L
    @Volatile
    private var currentInFlightIdleGapPendingMs: Float = 0f
    @Volatile
    private var currentInFlightIdleGapNoPendingMs: Float = 0f
    @Volatile
    private var currentInFlightPumpDelayMs: Float = 0f
    @Volatile
    private var currentInFlightBlendshapesEnabled: Boolean = true
    @Volatile
    private var currentDispatchPauseGeneration: Int = 0

    internal fun pumpOnAnalysisThread() = pumpOnDispatchThread()

    internal fun pumpOnDispatchThread() {
        dispatchThread = Thread.currentThread()
        val postedNs = pumpPostedAtNs
        if (postedNs > 0L) {
            pumpPostedAtNs = 0L
            val nowPumpNs = FrameClock.elapsedRealtimeNs()
            lastPumpDelayMs = if (nowPumpNs >= postedNs) {
                (nowPumpNs - postedNs).toFloat() / 1_000_000f
            } else {
                0f
            }
        }
        try {
            checkWatchdogOnDispatchThread()
            processPendingDelegateSwitchOnDispatchThread()
            processPendingBlendshapesSwitchOnDispatchThread()
            tryDispatchPendingConvertedFrameOnDispatchThread()
        } catch (t: Throwable) {
            Log.e(TAG, "Throwable in pumpOnDispatchThread: ${t.message}", t)
            releaseInFlightSlotAndRetiringLandmarkers()
            activeInFlightDispatchId = 0L
            isInferenceInFlight.set(false)
        }
    }

    private fun tryDispatchPendingConvertedFrameOnDispatchThread() {
        if (isPaused) {
            clearStagedSlot()
            return
        }
        if (isInferenceInFlight.get()) {
            return
        }

        val peeked = synchronized(slotLock) {
            if (isPaused || isInferenceInFlight.get()) return
            stagedPreparedFrame ?: return
        }

        val entryCheck = activeEntry
        if (entryCheck == null) {
            clearStagedSlot()
            return
        }

        // Single cadence gate at the provider: 0.85 of the target period.
        // CameraSession uses a 0.50 cheap pre-filter so frames passing >= 0.85 * period are never deferred here.
        val nowNs = FrameClock.elapsedRealtimeNs()
        if (enforceCadenceLimit) {
            val cadenceHz = targetTrackerCadenceHz.coerceIn(
                PerformanceGovernor.MIN_TRACKER_CADENCE_HZ,
                PerformanceGovernor.MAX_TRACKER_CADENCE_HZ
            )
            val minIntervalNs = ((1_000_000_000L / cadenceHz) * 85L) / 100L
            val prevFrameTs = lastDispatchedFrameTimestampNs
            val prevClockTs = lastDispatchedClockNs
            if (prevFrameTs > 0L) {
                val frameTs = peeked.frame.timestampNs
                val deltaNs = if (frameTs > prevFrameTs) {
                    frameTs - prevFrameTs
                } else {
                    nowNs - prevClockTs
                }
                if (deltaNs < minIntervalNs) {
                    providerCadenceDeferredCounter.incrementAndGet()
                    val remainingMs = ((minIntervalNs - deltaNs) / 1_000_000L).coerceAtLeast(1L)
                    scheduleCadenceWakeup(remainingMs)
                    return
                }
            }
        }

        val currentEntry = activeEntry
        if (currentEntry == null || isPaused) {
            clearStagedSlot()
            return
        }

        // Atomically pop STAGED -> IN_FLIGHT under slotLock
        val prepared: PreparedFrame = synchronized(slotLock) {
            if (isPaused || isInferenceInFlight.get()) {
                return
            }
            val p = stagedPreparedFrame ?: return
            val slot = slots[p.slotIndex]
            if (slot.state != SlotState.STAGED) {
                stagedPreparedFrame = null
                pendingConvertedFrameRef.set(null)
                return
            }
            stagedPreparedFrame = null
            pendingConvertedFrameRef.set(null)
            slot.state = SlotState.IN_FLIGHT
            slot.inUseByInference.set(true)
            slot.lastDispatchedFrameNumber = p.frame.frameNumber
            inFlightSlotIndex = slot.index
            lastUsedSlotIndex = slot.index
            isInferenceInFlight.set(true)
            p
        }

        try {
            val dispatchNs = FrameClock.elapsedRealtimeNs()
            val pendingWaitMs = if (dispatchNs >= prepared.conversionDoneNs) {
                (dispatchNs - prepared.conversionDoneNs).toFloat() / 1_000_000f
            } else {
                0f
            }

            // Split idleGap since previous inference completion into:
            // - idleGapPendingMs: a converted frame was already pending when the previous inference
            //   completed (conversionDoneNs <= prevInfDoneNs, i.e., delay caused by threading/pump)
            // - idleGapNoPendingMs: no converted frame was ready when the previous inference
            //   completed (conversionDoneNs > prevInfDoneNs, i.e., waiting for camera/conversion)
            val prevInfDoneNs = lastInferenceDoneNs
            val gapPendingMs: Float
            val gapNoPendingMs: Float
            if (prevInfDoneNs > 0L && dispatchNs >= prevInfDoneNs) {
                val gapMs = (dispatchNs - prevInfDoneNs).toFloat() / 1_000_000f
                if (prepared.conversionDoneNs <= prevInfDoneNs) {
                    gapPendingMs = gapMs
                    gapNoPendingMs = 0f
                } else {
                    gapPendingMs = 0f
                    gapNoPendingMs = gapMs
                }
            } else {
                gapPendingMs = 0f
                gapNoPendingMs = 0f
            }
            lastIdleGapPendingMs = gapPendingMs
            lastIdleGapNoPendingMs = gapNoPendingMs

            // Strictly monotonic timestampMs confined to the dispatch thread
            val prevMs = lastDetectTimestampMs.get()
            val candidateMs = (prepared.frame.timestampNs / 1_000_000L).coerceAtLeast(1L)
            val timestampMs = if (candidateMs > prevMs) candidateMs else prevMs + 1L
            lastDetectTimestampMs.set(timestampMs)

            val dispatchId = nextDispatchId.incrementAndGet()
            synchronized(landmarkerLock) {
                activeInFlightDispatchId = dispatchId
                inferenceDispatchedAtNs = dispatchNs
                currentInFlightEntry = currentEntry
                currentInFlightPrepared = prepared
            }
            currentInFlightPendingWaitMs = pendingWaitMs
            currentInFlightDispatchNs = dispatchNs
            currentInFlightIdleGapPendingMs = gapPendingMs
            currentInFlightIdleGapNoPendingMs = gapNoPendingMs
            currentInFlightPumpDelayMs = lastPumpDelayMs
            currentInFlightBlendshapesEnabled = currentEntry.config.outputBlendshapes
            currentDispatchPauseGeneration = pauseGeneration
            lastDispatchedFrameTimestampNs = prepared.frame.timestampNs
            lastDispatchedClockNs = dispatchNs

            currentEntry.latestDispatchId = dispatchId
            dispatchIdByMpImage[prepared.mpImage] = dispatchId

            val timeoutMs = if (currentEntry.hasCompletedFirstInference) {
                STEADY_WATCHDOG_TIMEOUT_MS
            } else {
                FIRST_INFERENCE_WATCHDOG_TIMEOUT_MS
            }
            scheduleWatchdogTimer(timeoutMs, dispatchId)

            lastDetectAsyncThreadId = Thread.currentThread().id
            lastDetectAsyncThreadName = Thread.currentThread().name

            currentEntry.handle.detectAsync(prepared.mpImage, timestampMs)
        } catch (t: Throwable) {
            Log.e(TAG, "Throwable in detectAsync dispatch: ${t.message}", t)
            scheduledWatchdogFuture?.cancel(false)
            dispatchIdByMpImage.remove(prepared.mpImage)
            releaseInFlightSlotAndRetiringLandmarkers()
        }
    }

    /**
     * Converts [imageProxy] into [slot]'s target bitmap according to [conversionMode] (`FAST` default vs `OLD`):
     * - `FAST` (`CONV:FAST`): Single-pass direct YUV/RGBA resampling + rotation + integer BT.601 conversion
     *   using precomputed 1D byte-offset lookup tables into `slot.targetArgbScratch` (`dstW * dstH`) and
     *   a single `targetBitmap.setPixels(...)` call. Zero intermediate full-size `sourceBitmap` and zero `Canvas` draw.
     * - `OLD` (`CONV:OLD`): Reference two-stage converter unpacking full `cropW * cropH` into `slot.argbScratch`
     *   and `sourceBitmap`, then rotating and downscaling via `targetCanvas.drawBitmap` (and `rotateAndScaleReferenceArgb`).
     * - Records per-stage timings (`unpackMs`, `yuvToRgbMs`, `bitmapCopyMs`, `rotateScaleMs`) and source/crop/rotation
     *   telemetry per resolution mode (`SMALL` / `LEGACY`).
     */
    internal fun convertAndTransformIntoSlot(
        imageProxy: ImageProxy,
        rotationDegrees: Int,
        slot: FrameBufferSlot
    ): Bitmap? {
        slot.lastConversionSucceeded = false
        slot.lastUnpackMs = 0f
        slot.lastYuvToRgbMs = 0f
        slot.lastBitmapCopyMs = 0f
        slot.lastRotateScaleMs = 0f
        return try {
            val imgW = imageProxy.width
            val imgH = imageProxy.height
            if (imgW <= 0 || imgH <= 0) return null

            val rawCrop = try {
                imageProxy.cropRect
            } catch (_: Throwable) {
                null
            }
            val rawRight = if (rawCrop != null) {
                val r = rawCrop.right
                if (r > 0) r else (rawCrop.left + rawCrop.width())
            } else {
                imgW
            }
            val rawBottom = if (rawCrop != null) {
                val b = rawCrop.bottom
                if (b > 0) b else (rawCrop.top + rawCrop.height())
            } else {
                imgH
            }
            val left = (rawCrop?.left ?: 0).coerceIn(0, imgW - 1)
            val top = (rawCrop?.top ?: 0).coerceIn(0, imgH - 1)
            val right = (if (rawRight > left) rawRight else imgW).coerceIn(left + 1, imgW)
            val bottom = (if (rawBottom > top) rawBottom else imgH).coerceIn(top + 1, imgH)
            reusableCropRect.left = left
            reusableCropRect.top = top
            reusableCropRect.right = right
            reusableCropRect.bottom = bottom
            reusableCropRect.set(left, top, right, bottom)

            val cropW = right - left
            val cropH = bottom - top
            val rot = ((rotationDegrees % 360) + 360) % 360

            // Record source and crop dimensions globally, in the slot, and per resolution mode
            lastRawSourceWidth = imgW
            lastRawSourceHeight = imgH
            lastCropWidth = cropW
            lastCropHeight = cropH
            lastSourceWidth = cropW
            lastSourceHeight = cropH
            lastRotationDegrees = rot

            slot.lastRawSourceWidth = imgW
            slot.lastRawSourceHeight = imgH
            slot.lastCropWidth = cropW
            slot.lastCropHeight = cropH
            slot.lastRotationDeg = rot

            if (currentResolutionMode == AnalysisResolutionMode.SMALL) {
                smallModeRawSourceWidth = imgW
                smallModeRawSourceHeight = imgH
                smallModeCropWidth = cropW
                smallModeCropHeight = cropH
                smallModeRotationDeg = rot
            } else {
                legacyModeRawSourceWidth = imgW
                legacyModeRawSourceHeight = imgH
                legacyModeCropWidth = cropW
                legacyModeCropHeight = cropH
                legacyModeRotationDeg = rot
            }

            val orientedW = if (rot == 90 || rot == 270) cropH else cropW
            val orientedH = if (rot == 90 || rot == 270) cropW else cropH
            val orientedLongEdge = max(orientedW, orientedH)
            val targetLongEdge = analysisLongEdgePx

            val dstW: Int
            val dstH: Int
            if (targetLongEdge > 0 && orientedLongEdge > targetLongEdge) {
                val scale = targetLongEdge.toFloat() / orientedLongEdge.toFloat()
                dstW = (orientedW * scale).roundToInt().coerceAtLeast(1)
                dstH = (orientedH * scale).roundToInt().coerceAtLeast(1)
            } else {
                dstW = orientedW
                dstH = orientedH
            }

            lastScaledWidth = dstW
            lastScaledHeight = dstH

            val resultBitmap = if (conversionMode == ConversionMode.FAST) {
                convertAndTransformFastIntoSlot(
                    imageProxy = imageProxy,
                    cropLeft = left,
                    cropTop = top,
                    cropW = cropW,
                    cropH = cropH,
                    dstW = dstW,
                    dstH = dstH,
                    rot = rot,
                    slot = slot
                )
            } else {
                convertAndTransformOldIntoSlot(
                    imageProxy = imageProxy,
                    cropW = cropW,
                    cropH = cropH,
                    orientedW = orientedW,
                    orientedH = orientedH,
                    dstW = dstW,
                    dstH = dstH,
                    rot = rot,
                    slot = slot
                )
            }

            lastUnpackMs = slot.lastUnpackMs
            lastYuvToRgbMs = slot.lastYuvToRgbMs
            lastBitmapCopyMs = slot.lastBitmapCopyMs
            lastRotateScaleMs = slot.lastRotateScaleMs
            resultBitmap
        } catch (t: Throwable) {
            Log.e(TAG, "Throwable converting ImageProxy in slot ${slot.index}: ${t.message}", t)
            slot.lastConversionSucceeded = false
            null
        }
    }

    /**
     * Single-pass fast converter (`CONV:FAST`):
     * 1. Unpacks the active crop row span from each plane `ByteBuffer` into reusable `ByteArray`s
     *    in [slot] and validates/reuses the precomputed 1D `(dx, dy)` byte-offset lookup tables (`unpackMs`).
     * 2. Samples directly at the target `dstW x dstH` resolution with rotation (`0`, `90`, `180`, `270`)
     *    applied through the 1D offset tables and integer ITU-R BT.601 YUV->ARGB conversion, writing
     *    `slot.targetArgbScratch` (`dstW * dstH`) once (`yuvToRgbMs`).
     * 3. Copies `slot.targetArgbScratch` once into `slot.ensureTargetBitmap(dstW, dstH)` (`bitmapCopyMs`).
     *    No intermediate full-size `sourceBitmap` and no `Canvas` (`rotateScaleMs = 0f`).
     */
    private fun convertAndTransformFastIntoSlot(
        imageProxy: ImageProxy,
        cropLeft: Int,
        cropTop: Int,
        cropW: Int,
        cropH: Int,
        dstW: Int,
        dstH: Int,
        rot: Int,
        slot: FrameBufferSlot
    ): Bitmap? {
        val t0Ns = FrameClock.elapsedRealtimeNs()
        val planes = imageProxy.planes
        val isRgba = imageProxy.format == PixelFormat.RGBA_8888 || planes.size == 1
        val outArgb = slot.ensureTargetArgbScratch(dstW * dstH)

        if (isRgba) {
            val plane = planes[0]
            val buf = plane.buffer
            val rowStride = plane.rowStride
            val pixelStride = plane.pixelStride.coerceAtLeast(4)

            val startByte = cropTop * rowStride
            val lastByte = (cropTop + cropH - 1) * rowStride + (cropLeft + cropW - 1) * pixelStride + 3
            val spanLen = (lastByte - startByte + 1).coerceAtLeast(4)

            slot.ensureRgbaPlaneBuffer(spanLen)
            copyByteBufferSliceZeroAlloc(buf, startByte, slot.rgbaPlaneBytes, spanLen)
            slot.ensureFastRgbaLookupTables(
                cropLeft = cropLeft,
                cropTop = cropTop,
                cropW = cropW,
                cropH = cropH,
                dstW = dstW,
                dstH = dstH,
                rot = rot,
                rowStride = rowStride,
                pixelStride = pixelStride
            )
            val t1Ns = FrameClock.elapsedRealtimeNs()
            slot.lastUnpackMs = ((t1Ns - t0Ns).coerceAtLeast(0L)).toFloat() / 1_000_000f

            val rgbaBytes = slot.rgbaPlaneBytes
            val dxTable = slot.dxRgbaOffsetTable
            val dyTable = slot.dyRgbaOffsetTable

            if (rot == 90 || rot == 270) {
                val tileW = 16
                var dx0 = 0
                while (dx0 < dstW) {
                    val dxEnd = (dx0 + tileW).coerceAtMost(dstW)
                    for (dy in 0 until dstH) {
                        val dyOff = dyTable[dy]
                        val outRowOff = dy * dstW
                        for (dx in dx0 until dxEnd) {
                            val pxOffset = dyOff + dxTable[dx]
                            val red = rgbaBytes[pxOffset].toInt() and 0xFF
                            val green = rgbaBytes[pxOffset + 1].toInt() and 0xFF
                            val blue = rgbaBytes[pxOffset + 2].toInt() and 0xFF
                            val alpha = rgbaBytes[pxOffset + 3].toInt() and 0xFF
                            outArgb[outRowOff + dx] =
                                (alpha shl 24) or (red shl 16) or (green shl 8) or blue
                        }
                    }
                    dx0 += tileW
                }
            } else {
                for (dy in 0 until dstH) {
                    val dyOff = dyTable[dy]
                    val outRowOff = dy * dstW
                    for (dx in 0 until dstW) {
                        val pxOffset = dyOff + dxTable[dx]
                        val red = rgbaBytes[pxOffset].toInt() and 0xFF
                        val green = rgbaBytes[pxOffset + 1].toInt() and 0xFF
                        val blue = rgbaBytes[pxOffset + 2].toInt() and 0xFF
                        val alpha = rgbaBytes[pxOffset + 3].toInt() and 0xFF
                        outArgb[outRowOff + dx] =
                            (alpha shl 24) or (red shl 16) or (green shl 8) or blue
                    }
                }
            }
            val t2Ns = FrameClock.elapsedRealtimeNs()
            slot.lastYuvToRgbMs = ((t2Ns - t1Ns).coerceAtLeast(0L)).toFloat() / 1_000_000f
        } else {
            val yPlane = planes[0]
            val uPlane = planes[1]
            val vPlane = planes[2]

            val rowStrideY = yPlane.rowStride
            val pixelStrideY = yPlane.pixelStride
            val rowStrideU = uPlane.rowStride
            val pixelStrideU = uPlane.pixelStride
            val rowStrideV = vPlane.rowStride
            val pixelStrideV = vPlane.pixelStride

            val cropRight = cropLeft + cropW
            val cropBottom = cropTop + cropH

            val yStartByte = cropTop * rowStrideY
            val yLastByte = (cropBottom - 1) * rowStrideY + (cropRight - 1) * pixelStrideY
            val ySpanLen = (yLastByte - yStartByte + 1).coerceAtLeast(1)

            val uvTop = cropTop shr 1
            val uvLastRow = (cropBottom - 1) shr 1
            val uvLastCol = (cropRight - 1) shr 1

            val uStartByte = uvTop * rowStrideU
            val uLastByte = uvLastRow * rowStrideU + uvLastCol * pixelStrideU
            val uSpanLen = (uLastByte - uStartByte + 1).coerceAtLeast(1)

            val vStartByte = uvTop * rowStrideV
            val vLastByte = uvLastRow * rowStrideV + uvLastCol * pixelStrideV
            val vSpanLen = (vLastByte - vStartByte + 1).coerceAtLeast(1)

            slot.ensureYuvPlaneBuffers(ySpanLen, uSpanLen, vSpanLen)
            copyByteBufferSliceZeroAlloc(yPlane.buffer, yStartByte, slot.yPlaneBytes, ySpanLen)
            copyByteBufferSliceZeroAlloc(uPlane.buffer, uStartByte, slot.uPlaneBytes, uSpanLen)
            copyByteBufferSliceZeroAlloc(vPlane.buffer, vStartByte, slot.vPlaneBytes, vSpanLen)

            slot.ensureFastYuvLookupTables(
                cropLeft = cropLeft,
                cropTop = cropTop,
                cropW = cropW,
                cropH = cropH,
                dstW = dstW,
                dstH = dstH,
                rot = rot,
                rowStrideY = rowStrideY,
                pixelStrideY = pixelStrideY,
                rowStrideU = rowStrideU,
                pixelStrideU = pixelStrideU,
                rowStrideV = rowStrideV,
                pixelStrideV = pixelStrideV
            )
            val t1Ns = FrameClock.elapsedRealtimeNs()
            slot.lastUnpackMs = ((t1Ns - t0Ns).coerceAtLeast(0L)).toFloat() / 1_000_000f

            val yBytes = slot.yPlaneBytes
            val uBytes = slot.uPlaneBytes
            val vBytes = slot.vPlaneBytes
            val dxY = slot.dxYOffsetTable
            val dxU = slot.dxUOffsetTable
            val dxV = slot.dxVOffsetTable
            val dyY = slot.dyYOffsetTable
            val dyU = slot.dyUOffsetTable
            val dyV = slot.dyVOffsetTable

            // When rot == 90 or 270 (portrait mode), dx steps across source rows (rowStride bytes apart)
            // while dy steps sequentially along a single source row (pixelStride bytes apart).
            // Tiling dx in 16-column strips keeps active source rows and destination cache lines in L1 cache.
            if (rot == 90 || rot == 270) {
                val tileW = 16
                var dx0 = 0
                while (dx0 < dstW) {
                    val dxEnd = (dx0 + tileW).coerceAtMost(dstW)
                    for (dy in 0 until dstH) {
                        val dyYOff = dyY[dy]
                        val dyUOff = dyU[dy]
                        val dyVOff = dyV[dy]
                        val outRowOff = dy * dstW
                        for (dx in dx0 until dxEnd) {
                            val yVal = yBytes[dyYOff + dxY[dx]].toInt() and 0xFF
                            val uVal = (uBytes[dyUOff + dxU[dx]].toInt() and 0xFF) - 128
                            val vVal = (vBytes[dyVOff + dxV[dx]].toInt() and 0xFF) - 128

                            // ITU-R BT.601 integer conversion with fast inline clamping
                            val yDiff = yVal - 16
                            val y1192 = if (yDiff > 0) 1192 * yDiff else 0
                            var rVal = (y1192 + 1634 * vVal) shr 10
                            if (rVal < 0) rVal = 0 else if (rVal > 255) rVal = 255
                            var gVal = (y1192 - 833 * vVal - 400 * uVal) shr 10
                            if (gVal < 0) gVal = 0 else if (gVal > 255) gVal = 255
                            var bVal = (y1192 + 2066 * uVal) shr 10
                            if (bVal < 0) bVal = 0 else if (bVal > 255) bVal = 255

                            outArgb[outRowOff + dx] =
                                (0xFF shl 24) or (rVal shl 16) or (gVal shl 8) or bVal
                        }
                    }
                    dx0 += tileW
                }
            } else {
                for (dy in 0 until dstH) {
                    val dyYOff = dyY[dy]
                    val dyUOff = dyU[dy]
                    val dyVOff = dyV[dy]
                    val outRowOff = dy * dstW
                    for (dx in 0 until dstW) {
                        val yVal = yBytes[dyYOff + dxY[dx]].toInt() and 0xFF
                        val uVal = (uBytes[dyUOff + dxU[dx]].toInt() and 0xFF) - 128
                        val vVal = (vBytes[dyVOff + dxV[dx]].toInt() and 0xFF) - 128

                        // ITU-R BT.601 integer conversion with fast inline clamping
                        val yDiff = yVal - 16
                        val y1192 = if (yDiff > 0) 1192 * yDiff else 0
                        var rVal = (y1192 + 1634 * vVal) shr 10
                        if (rVal < 0) rVal = 0 else if (rVal > 255) rVal = 255
                        var gVal = (y1192 - 833 * vVal - 400 * uVal) shr 10
                        if (gVal < 0) gVal = 0 else if (gVal > 255) gVal = 255
                        var bVal = (y1192 + 2066 * uVal) shr 10
                        if (bVal < 0) bVal = 0 else if (bVal > 255) bVal = 255

                        outArgb[outRowOff + dx] =
                            (0xFF shl 24) or (rVal shl 16) or (gVal shl 8) or bVal
                    }
                }
            }
            val t2Ns = FrameClock.elapsedRealtimeNs()
            slot.lastYuvToRgbMs = ((t2Ns - t1Ns).coerceAtLeast(0L)).toFloat() / 1_000_000f
        }

        if (outArgb.isNotEmpty()) {
            slot.lastUnpackedPixelArgb = outArgb[0]
        }

        val tCopyStartNs = FrameClock.elapsedRealtimeNs()
        val dstBitmap = slot.ensureTargetBitmap(dstW, dstH)
        dstBitmap?.setPixels(outArgb, 0, dstW, 0, 0, dstW, dstH)
        val tCopyDoneNs = FrameClock.elapsedRealtimeNs()
        slot.lastBitmapCopyMs = ((tCopyDoneNs - tCopyStartNs).coerceAtLeast(0L)).toFloat() / 1_000_000f
        slot.lastRotateScaleMs = 0f
        slot.lastConversionSucceeded = true
        return dstBitmap
    }

    /**
     * Reference two-stage converter (`CONV:OLD`):
     * 1. Unpacks full `cropW x cropH` source pixels into `slot.argbScratch` (`unpackMs` + `yuvToRgbMs`).
     * 2. Copies `slot.argbScratch` into `slot.sourceBitmap` (`bitmapCopyMs`).
     * 3. Rotates and downscales `sourceBitmap` into `targetBitmap` via `targetCanvas.drawBitmap`
     *    and populates `slot.referenceTargetArgb` (`rotateScaleMs`).
     */
    private fun convertAndTransformOldIntoSlot(
        imageProxy: ImageProxy,
        cropW: Int,
        cropH: Int,
        orientedW: Int,
        orientedH: Int,
        dstW: Int,
        dstH: Int,
        rot: Int,
        slot: FrameBufferSlot
    ): Bitmap? {
        val t0Ns = FrameClock.elapsedRealtimeNs()
        val pixelCount = cropW * cropH
        val scratch = slot.ensureArgbScratch(pixelCount)
        val srcBitmap = slot.ensureSourceBitmap(cropW, cropH)
        val planes = imageProxy.planes
        val isRgba = imageProxy.format == PixelFormat.RGBA_8888 || planes.size == 1
        val t1Ns = FrameClock.elapsedRealtimeNs()
        slot.lastUnpackMs = ((t1Ns - t0Ns).coerceAtLeast(0L)).toFloat() / 1_000_000f

        if (isRgba) {
            convertRgba8888ToBitmap(imageProxy, reusableCropRect, null, scratch)
            val t2Ns = FrameClock.elapsedRealtimeNs()
            slot.lastYuvToRgbMs = ((t2Ns - t1Ns).coerceAtLeast(0L)).toFloat() / 1_000_000f
            srcBitmap?.setPixels(scratch, 0, cropW, 0, 0, cropW, cropH)
            val t3Ns = FrameClock.elapsedRealtimeNs()
            slot.lastBitmapCopyMs = ((t3Ns - t2Ns).coerceAtLeast(0L)).toFloat() / 1_000_000f
        } else {
            convertYuv420ToArgbPixels(imageProxy, reusableCropRect, scratch)
            val t2Ns = FrameClock.elapsedRealtimeNs()
            slot.lastYuvToRgbMs = ((t2Ns - t1Ns).coerceAtLeast(0L)).toFloat() / 1_000_000f
            srcBitmap?.setPixels(scratch, 0, cropW, 0, 0, cropW, cropH)
            val t3Ns = FrameClock.elapsedRealtimeNs()
            slot.lastBitmapCopyMs = ((t3Ns - t2Ns).coerceAtLeast(0L)).toFloat() / 1_000_000f
        }
        if (scratch.isNotEmpty()) {
            slot.lastUnpackedPixelArgb = scratch[0]
        }

        val tRotStartNs = FrameClock.elapsedRealtimeNs()
        val refOut = slot.ensureReferenceTargetArgb(dstW * dstH)
        rotateAndScaleReferenceArgb(
            srcArgb = scratch,
            cropW = cropW,
            cropH = cropH,
            dstW = dstW,
            dstH = dstH,
            rot = rot,
            outArgb = refOut
        )

        // Unrotated 1:1 path
        if (rot == 0 && dstW == cropW && dstH == cropH) {
            slot.recordDirectTargetDimensions(dstW, dstH)
            val tRotDoneNs = FrameClock.elapsedRealtimeNs()
            slot.lastRotateScaleMs = ((tRotDoneNs - tRotStartNs).coerceAtLeast(0L)).toFloat() / 1_000_000f
            slot.lastConversionSucceeded = true
            return srcBitmap
        }

        val dstBitmap = slot.ensureTargetBitmap(dstW, dstH)
        if (slot.cachedSrcW != cropW ||
            slot.cachedSrcH != cropH ||
            slot.cachedDstW != dstW ||
            slot.cachedDstH != dstH ||
            slot.cachedRotationDeg != rot
        ) {
            slot.cachedSrcW = cropW
            slot.cachedSrcH = cropH
            slot.cachedDstW = dstW
            slot.cachedDstH = dstH
            slot.cachedRotationDeg = rot

            val m = slot.transformMatrix
            m.reset()
            m.postTranslate(-cropW * 0.5f, -cropH * 0.5f)
            if (rot != 0) {
                m.postRotate(rot.toFloat())
            }
            val sx = dstW.toFloat() / orientedW.toFloat()
            val sy = dstH.toFloat() / orientedH.toFloat()
            m.postScale(sx, sy)
            m.postTranslate(dstW * 0.5f, dstH * 0.5f)
        }

        if (srcBitmap != null) {
            slot.targetCanvas?.drawBitmap(srcBitmap, slot.transformMatrix, downscalePaint)
        }
        val tRotDoneNs = FrameClock.elapsedRealtimeNs()
        slot.lastRotateScaleMs = ((tRotDoneNs - tRotStartNs).coerceAtLeast(0L)).toFloat() / 1_000_000f
        slot.lastConversionSucceeded = true
        return dstBitmap
    }

    private fun handleInferenceError(
        callbackDispatchId: Long,
        sourceEntry: ManagedLandmarkerEntry?,
        error: RuntimeException
    ) {
        lastResultCallbackThreadId = Thread.currentThread().id
        lastResultCallbackThreadName = Thread.currentThread().name

        // Stale callback filter: ignore callbacks from closed/reset landmarkers or old dispatches
        val isStaleError = synchronized(landmarkerLock) {
            callbackDispatchId <= 0L || callbackDispatchId != activeInFlightDispatchId || !isInferenceInFlight.get()
        }
        if (isStaleError) {
            staleCallbackIgnoredCounter.incrementAndGet()
            return
        }

        Log.e(TAG, "MediaPipe landmarker error (dispatchId=$callbackDispatchId): ${error.message}", error)
        scheduledWatchdogFuture?.cancel(false)
        sourceEntry?.hasCompletedFirstInference = true
        val doneAtNs = FrameClock.elapsedRealtimeNs()
        lastInferenceDoneNs = doneAtNs
        releaseInFlightSlotAndRetiringLandmarkers(callbackDispatchId)

        // Never run detectAsync on the callback thread; always post to dispatchExecutor
        postPumpToDispatchThread(fromResultCallback = true)
    }

    internal fun handleResult(
        callbackDispatchId: Long,
        sourceEntry: ManagedLandmarkerEntry?,
        result: FaceLandmarkerResult?,
        mpImage: MPImage
    ) {
        lastResultCallbackThreadId = Thread.currentThread().id
        lastResultCallbackThreadName = Thread.currentThread().name

        // Stale callback filter: never release a slot or clear in-flight flag of a newer dispatch
        val isStaleResult = synchronized(landmarkerLock) {
            callbackDispatchId <= 0L || callbackDispatchId != activeInFlightDispatchId || !isInferenceInFlight.get()
        }
        if (isStaleResult) {
            staleCallbackIgnoredCounter.incrementAndGet()
            return
        }

        var doneAtNs = 0L
        try {
            scheduledWatchdogFuture?.cancel(false)
            sourceEntry?.hasCompletedFirstInference = true

            val inferenceDoneNs = FrameClock.elapsedRealtimeNs()
            doneAtNs = inferenceDoneNs
            lastInferenceDoneNs = inferenceDoneNs

            if (isPaused || currentDispatchPauseGeneration != pauseGeneration) {
                droppedInFlightResultCount++
                return
            }

            val entryConfig = sourceEntry?.config ?: activeConfig
            val prepared = currentInFlightPrepared
            val frame = prepared?.frame
            val timestampNs = frame?.timestampNs ?: inferenceDoneNs
            val sensorTimestampNs = frame?.sensorTimestampNs ?: timestampNs
            val analysisStartNs = prepared?.analysisStartNs ?: 0L
            val queueMs = prepared?.queueMs ?: 0f
            val conversionMs = prepared?.conversionMs ?: 0f
            val unpackMs = prepared?.unpackMs ?: lastUnpackMs
            val yuvToRgbMs = prepared?.yuvToRgbMs ?: lastYuvToRgbMs
            val bitmapCopyMs = prepared?.bitmapCopyMs ?: lastBitmapCopyMs
            val rotateScaleMs = prepared?.rotateScaleMs ?: lastRotateScaleMs
            val mpImageBuildMs = prepared?.mpImageBuildMs ?: lastMpImageBuildMs
            val srcW = prepared?.sourceWidth?.takeIf { it > 0 } ?: lastRawSourceWidth.takeIf { it > 0 } ?: mpImage.width
            val srcH = prepared?.sourceHeight?.takeIf { it > 0 } ?: lastRawSourceHeight.takeIf { it > 0 } ?: mpImage.height
            val crpW = prepared?.cropWidth?.takeIf { it > 0 } ?: lastCropWidth.takeIf { it > 0 } ?: mpImage.width
            val crpH = prepared?.cropHeight?.takeIf { it > 0 } ?: lastCropHeight.takeIf { it > 0 } ?: mpImage.height
            val convModeLabel = prepared?.conversionModeLabel ?: conversionMode.name
            val pendingWaitMs = currentInFlightPendingWaitMs
            val dispatchNs = currentInFlightDispatchNs
            val idleGapPendingMs = currentInFlightIdleGapPendingMs
            val idleGapNoPendingMs = currentInFlightIdleGapNoPendingMs
            val pumpDelayMs = currentInFlightPumpDelayMs
            val blendshapesEnabled = currentInFlightBlendshapesEnabled
            val swapMs = lastSwapMs
            val inferenceMs = if (dispatchNs > 0L && inferenceDoneNs >= dispatchNs) {
                (inferenceDoneNs - dispatchNs).toFloat() / 1_000_000f
            } else {
                0f
            }
            val isFront = frame?.isFrontCamera ?: true
            val rot = frame?.rotationDegrees ?: 0
            val rawProjParams = frame?.projectionParams ?: ProjectionParams.DEFAULT
            val projParams = if (rawProjParams === ProjectionParams.DEFAULT && mpImage.width > 0 && mpImage.height > 0) {
                rawProjParams.copy(cropAspectRatio = mpImage.width.toFloat() / mpImage.height.toFloat())
            } else {
                rawProjParams
            }

            val landmarksList = result?.faceLandmarks()
            val hasFace = !landmarksList.isNullOrEmpty() && landmarksList[0].isNotEmpty()

            if (!hasFace || result == null) {
                val snapshot = FaceGeometrySnapshot(
                    timestampNs = timestampNs,
                    imageWidth = mpImage.width,
                    imageHeight = mpImage.height,
                    rotationDegrees = rot,
                    isFrontCamera = isFront,
                    isTracking = false,
                    landmarks = null,
                    rawPoseMatrix = null,
                    blendshapes = emptyMap(),
                    modelSha256 = currentModelSha256,
                    imageToViewTransform = frame?.imageToViewTransform,
                    sensorTimestampNs = sensorTimestampNs,
                    analysisStartNs = analysisStartNs,
                    inferenceDoneNs = inferenceDoneNs,
                    queueMs = queueMs,
                    conversionMs = conversionMs,
                    pendingWaitMs = pendingWaitMs,
                    inferenceMs = inferenceMs,
                    projectionParams = projParams,
                    idleGapPendingMs = idleGapPendingMs,
                    idleGapNoPendingMs = idleGapNoPendingMs,
                    pumpDelayMs = pumpDelayMs,
                    blendshapesEnabled = blendshapesEnabled,
                    swapMs = swapMs,
                    unpackMs = unpackMs,
                    yuvToRgbMs = yuvToRgbMs,
                    bitmapCopyMs = bitmapCopyMs,
                    rotateScaleMs = rotateScaleMs,
                    mpImageBuildMs = mpImageBuildMs,
                    sourceWidth = srcW,
                    sourceHeight = srcH,
                    cropWidth = crpW,
                    cropHeight = crpH,
                    conversionModeLabel = convModeLabel
                )
                listener?.invoke(snapshot)
                return
            }

            val landmarksObj = landmarksList[0]
            val lms = FloatArray(landmarksObj.size * 3)
            for (i in landmarksObj.indices) {
                val lm = landmarksObj[i]
                lms[i * 3] = lm.x()
                lms[i * 3 + 1] = lm.y()
                lms[i * 3 + 2] = lm.z()
            }
            val shouldOutputLandmarks = activeConfig.outputLandmarks || entryConfig.outputLandmarks
            val flatLandmarks: FloatArray? = if (shouldOutputLandmarks) lms else null

            var poseMatrix: FloatArray? = null
            if (activeConfig.outputPose || entryConfig.outputPose || flatLandmarks != null) {
                var rawMpMatrix: FloatArray? = null
                try {
                    val optMatrices = result.facialTransformationMatrixes()
                    if (optMatrices != null && optMatrices.isPresent) {
                        val matrixList = optMatrices.get()
                        if (matrixList.isNotEmpty()) {
                            rawMpMatrix = matrixList[0].clone()
                        }
                    }
                } catch (_: Throwable) {
                }
                poseMatrix = calibratePoseMatrix(
                    rawMpMatrix = rawMpMatrix,
                    lm = lms,
                    imageWidth = mpImage.width,
                    imageHeight = mpImage.height,
                    fovYCropDeg = projParams.fovYDeg
                )
            }

            val blendshapesMap = HashMap<String, Float>()
            if (blendshapesEnabled) {
                try {
                    val optBlendshapes = result.faceBlendshapes()
                    if (optBlendshapes != null && optBlendshapes.isPresent) {
                        val bsList = optBlendshapes.get()
                        if (bsList.isNotEmpty()) {
                            for (category in bsList[0]) {
                                blendshapesMap[category.categoryName()] = category.score()
                            }
                        }
                    }
                } catch (_: Throwable) {
                }
            }

            val snapshot = FaceGeometrySnapshot.create(
                timestampNs = timestampNs,
                imageWidth = mpImage.width,
                imageHeight = mpImage.height,
                rotationDegrees = rot,
                isFrontCamera = isFront,
                isTracking = true,
                landmarks = flatLandmarks,
                rawPoseMatrix = poseMatrix,
                blendshapes = blendshapesMap,
                modelSha256 = currentModelSha256,
                imageToViewTransform = frame?.imageToViewTransform,
                sensorTimestampNs = sensorTimestampNs,
                analysisStartNs = analysisStartNs,
                inferenceDoneNs = inferenceDoneNs,
                queueMs = queueMs,
                conversionMs = conversionMs,
                inferenceMs = inferenceMs,
                projectionParams = projParams,
                pendingWaitMs = pendingWaitMs,
                idleGapPendingMs = idleGapPendingMs,
                idleGapNoPendingMs = idleGapNoPendingMs,
                pumpDelayMs = pumpDelayMs,
                blendshapesEnabled = blendshapesEnabled,
                swapMs = swapMs,
                unpackMs = unpackMs,
                yuvToRgbMs = yuvToRgbMs,
                bitmapCopyMs = bitmapCopyMs,
                rotateScaleMs = rotateScaleMs,
                mpImageBuildMs = mpImageBuildMs,
                sourceWidth = srcW,
                sourceHeight = srcH,
                cropWidth = crpW,
                cropHeight = crpH,
                conversionModeLabel = convModeLabel
            )

            listener?.invoke(snapshot)
        } finally {
            if (doneAtNs == 0L) {
                doneAtNs = FrameClock.elapsedRealtimeNs()
                lastInferenceDoneNs = doneAtNs
            }
            releaseInFlightSlotAndRetiringLandmarkers(callbackDispatchId)
            // Never call detectAsync on MediaPipe's callback thread: post to dispatchExecutor
            postPumpToDispatchThread(fromResultCallback = true)
        }
    }
}
