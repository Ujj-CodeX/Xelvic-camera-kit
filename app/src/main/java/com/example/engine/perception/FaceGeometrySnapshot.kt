package com.example.engine.perception

import com.example.engine.capture.FrameClock
import com.example.engine.signal.ProjectionParams

/**
 * Immutable, validated perception snapshot containing face geometry results.
 * Strictly adheres to README Section 8 perception contract and AUDIT.md 3.2 LatencyProbe:
 * - Defensive copying for all arrays and collections
 * - Validation for finite coordinates (no NaN / Infinity)
 * - Exact image-to-view transform and ProjectionParams captured at frame submission time
 * - Monotonic `timestampNs` (calibrated capture timestamp in `FrameClock.elapsedRealtimeNs()` base)
 * - Per-frame stage timings (`queueMs`, `conversionMs`, `inferenceMs`)
 */
data class FaceGeometrySnapshot(
    val timestampNs: Long,
    val imageWidth: Int,
    val imageHeight: Int,
    val rotationDegrees: Int,
    val isFrontCamera: Boolean,
    val isTracking: Boolean,
    val landmarks: FloatArray?,            // 478 * 3 normalized coordinates [0..1]
    val rawPoseMatrix: FloatArray?,        // 4x4 column-major matrix
    val blendshapes: Map<String, Float>,   // Requested blendshape scores
    val modelSha256: String,
    val topologyId: String = "canonical_478_v1",
    val imageToViewTransform: FloatArray? = null,
    val sensorTimestampNs: Long = timestampNs,
    val analysisStartNs: Long = 0L,
    val inferenceDoneNs: Long = 0L,
    val queueMs: Float = 0f,
    val conversionMs: Float = 0f,
    val pendingWaitMs: Float = 0f,
    val inferenceMs: Float = 0f,
    val projectionParams: ProjectionParams = ProjectionParams.DEFAULT,
    val idleGapPendingMs: Float = 0f,
    val idleGapNoPendingMs: Float = 0f,
    val pumpDelayMs: Float = 0f,
    val blendshapesEnabled: Boolean = true,
    val swapMs: Float = 0f,
    val unpackMs: Float = 0f,
    val yuvToRgbMs: Float = 0f,
    val bitmapCopyMs: Float = 0f,
    val rotateScaleMs: Float = 0f,
    val mpImageBuildMs: Float = 0f,
    val sourceWidth: Int = imageWidth,
    val sourceHeight: Int = imageHeight,
    val cropWidth: Int = imageWidth,
    val cropHeight: Int = imageHeight,
    val conversionModeLabel: String = "FAST"
) {
    val landmarkCount: Int
        get() = (landmarks?.size ?: 0) / 3

    fun hasFiniteLandmarks(): Boolean {
        val lm = landmarks ?: return false
        if (lm.size < 478 * 3) return false
        for (v in lm) {
            if (v.isNaN() || v.isInfinite()) return false
        }
        return true
    }

    fun hasFinitePoseMatrix(): Boolean {
        val m = rawPoseMatrix ?: return false
        if (m.size != 16) return false
        for (v in m) {
            if (v.isNaN() || v.isInfinite()) return false
        }
        return true
    }

    fun isValid(): Boolean {
        if (!isTracking) return true // Untracked frame can be valid with null landmarks
        return hasFiniteLandmarks() && hasFinitePoseMatrix() && modelSha256.isNotEmpty()
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as FaceGeometrySnapshot
        return timestampNs == other.timestampNs && isTracking == other.isTracking &&
                landmarks.contentEquals(other.landmarks) &&
                rawPoseMatrix.contentEquals(other.rawPoseMatrix)
    }

    override fun hashCode(): Int {
        var result = timestampNs.hashCode()
        result = 31 * result + isTracking.hashCode()
        result = 31 * result + (landmarks?.contentHashCode() ?: 0)
        result = 31 * result + (rawPoseMatrix?.contentHashCode() ?: 0)
        return result
    }

    companion object {
        const val EXPECTED_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

        /**
         * Factory function creating snapshot with strict defensive copies.
         */
        @JvmOverloads
        fun create(
            timestampNs: Long,
            imageWidth: Int,
            imageHeight: Int,
            rotationDegrees: Int,
            isFrontCamera: Boolean,
            isTracking: Boolean,
            landmarks: FloatArray?,
            rawPoseMatrix: FloatArray?,
            blendshapes: Map<String, Float>,
            modelSha256: String = EXPECTED_MODEL_SHA256,
            topologyId: String = "canonical_478_v1",
            imageToViewTransform: FloatArray? = null,
            sensorTimestampNs: Long = timestampNs,
            analysisStartNs: Long = 0L,
            inferenceDoneNs: Long = 0L,
            queueMs: Float = 0f,
            conversionMs: Float = 0f,
            inferenceMs: Float = 0f,
            projectionParams: ProjectionParams = ProjectionParams.DEFAULT,
            pendingWaitMs: Float = 0f,
            idleGapPendingMs: Float = 0f,
            idleGapNoPendingMs: Float = 0f,
            pumpDelayMs: Float = 0f,
            blendshapesEnabled: Boolean = true,
            swapMs: Float = 0f,
            unpackMs: Float = 0f,
            yuvToRgbMs: Float = 0f,
            bitmapCopyMs: Float = 0f,
            rotateScaleMs: Float = 0f,
            mpImageBuildMs: Float = 0f,
            sourceWidth: Int = imageWidth,
            sourceHeight: Int = imageHeight,
            cropWidth: Int = imageWidth,
            cropHeight: Int = imageHeight,
            conversionModeLabel: String = "FAST"
        ): FaceGeometrySnapshot {
            return FaceGeometrySnapshot(
                timestampNs = timestampNs,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
                rotationDegrees = rotationDegrees,
                isFrontCamera = isFrontCamera,
                isTracking = isTracking,
                landmarks = landmarks?.clone(),
                rawPoseMatrix = rawPoseMatrix?.clone(),
                blendshapes = HashMap(blendshapes),
                modelSha256 = modelSha256,
                topologyId = topologyId,
                imageToViewTransform = imageToViewTransform?.clone(),
                sensorTimestampNs = sensorTimestampNs,
                analysisStartNs = analysisStartNs,
                inferenceDoneNs = inferenceDoneNs,
                queueMs = queueMs,
                conversionMs = conversionMs,
                pendingWaitMs = pendingWaitMs,
                inferenceMs = inferenceMs,
                projectionParams = projectionParams,
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
                sourceWidth = sourceWidth,
                sourceHeight = sourceHeight,
                cropWidth = cropWidth,
                cropHeight = cropHeight,
                conversionModeLabel = conversionModeLabel
            )
        }

        fun empty(timestampNs: Long = FrameClock.elapsedRealtimeNs()): FaceGeometrySnapshot {
            return FaceGeometrySnapshot(
                timestampNs = timestampNs,
                imageWidth = 0,
                imageHeight = 0,
                rotationDegrees = 0,
                isFrontCamera = true,
                isTracking = false,
                landmarks = null,
                rawPoseMatrix = null,
                blendshapes = emptyMap(),
                modelSha256 = EXPECTED_MODEL_SHA256
            )
        }

        @JvmOverloads
        fun noFace(
            timestampNs: Long = FrameClock.elapsedRealtimeNs(),
            imageWidth: Int = 0,
            imageHeight: Int = 0,
            isFrontCamera: Boolean = true,
            rotationDegrees: Int = 0
        ): FaceGeometrySnapshot {
            return FaceGeometrySnapshot(
                timestampNs = timestampNs,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
                rotationDegrees = rotationDegrees,
                isFrontCamera = isFrontCamera,
                isTracking = false,
                landmarks = null,
                rawPoseMatrix = null,
                blendshapes = emptyMap(),
                modelSha256 = EXPECTED_MODEL_SHA256
            )
        }
    }
}

