package com.example.engine.signal

import android.graphics.Rect

/**
 * Hardware camera intrinsics and sensor array geometry extracted from `CameraCharacteristics`
 * (and updated per frame with `CaptureResult.LENS_FOCAL_LENGTH` and `CaptureResult.SCALER_CROP_REGION`
 * for logical / multi-focal cameras).
 */
data class CameraIntrinsics(
    val focalLengthMm: Float = 0f,
    val sensorWidthMm: Float = 0f,
    val sensorHeightMm: Float = 0f,
    val pixelArrayWidth: Int = 0,
    val pixelArrayHeight: Int = 0,
    val activeArrayRect: Rect? = null,
    val scalerCropRegion: Rect? = null,
    val activeArrayLeft: Int = activeArrayRect?.left ?: 0,
    val activeArrayTop: Int = activeArrayRect?.top ?: 0,
    val activeArrayWidth: Int = activeArrayRect?.let { (it.right - it.left).takeIf { w -> w > 0 } ?: it.width() } ?: 0,
    val activeArrayHeight: Int = activeArrayRect?.let { (it.bottom - it.top).takeIf { h -> h > 0 } ?: it.height() } ?: 0,
    val scalerCropLeft: Int = scalerCropRegion?.left ?: 0,
    val scalerCropTop: Int = scalerCropRegion?.top ?: 0,
    val scalerCropWidth: Int = scalerCropRegion?.let { (it.right - it.left).takeIf { w -> w > 0 } ?: it.width() } ?: 0,
    val scalerCropHeight: Int = scalerCropRegion?.let { (it.bottom - it.top).takeIf { h -> h > 0 } ?: it.height() } ?: 0
) {
    val isValid: Boolean
        get() = focalLengthMm > 0.1f &&
            focalLengthMm.isFinite() &&
            sensorWidthMm > 0.1f &&
            sensorWidthMm.isFinite() &&
            sensorHeightMm > 0.1f &&
            sensorHeightMm.isFinite()

    companion object {
        val UNKNOWN = CameraIntrinsics()
    }
}

/**
 * Per-frame camera projection parameters carried through `AnalysisFrame` -> `FaceGeometrySnapshot` -> `FaceState`
 * (README Sections 7, 9, 10).
 *
 * Distinguishes the TWO FOVs required by the coordinate contract:
 * 1. [fovYDeg] (`fovYCropDeg`): Vertical field-of-view (in degrees) of the upright **analysis crop**.
 *    Used by `MediaPipeFaceGeometryProvider.calibratePoseMatrix`, `FacePoseAdapter.computeFaceWidth`,
 *    and `FaceMeshRenderer.updateVertices` when unprojecting normalized `[0..1]` landmarks (which are
 *    normalized to the upright analysis crop) into 3D metric camera space.
 * 2. [visibleViewFovYDeg]: Vertical field-of-view (in degrees) of the **visible view** after `FILL_CENTER`
 *    scaling/cropping to a target viewport aspect ratio (`viewWidth / viewHeight`), computed via
 *    `FacePoseAdapter.computeVisibleViewFovYDeg(fovYDeg, cropAspectRatio, viewAspectRatio)`.
 *    Used by all render projections (`RenderGraph`, `OffscreenRenderer`, `OffsetTestHarness`,
 *    `DebugOverlay`, and `FaceMeshRenderer`).
 */
data class ProjectionParams(
    val fovYDeg: Float = DEFAULT_FOV_Y_DEG,
    val fovXDeg: Float = DEFAULT_FOV_Y_DEG,
    val cropAspectRatio: Float = DEFAULT_ASPECT_RATIO,
    val near: Float = DEFAULT_NEAR,
    val far: Float = DEFAULT_FAR,
    val focalLengthMm: Float = 0f,
    val sensorWidthMm: Float = 0f,
    val sensorHeightMm: Float = 0f,
    val pixelArrayWidth: Int = 0,
    val pixelArrayHeight: Int = 0,
    val activeArrayWidth: Int = 0,
    val activeArrayHeight: Int = 0,
    val scalerCropLeft: Int = 0,
    val scalerCropTop: Int = 0,
    val scalerCropWidth: Int = 0,
    val scalerCropHeight: Int = 0,
    val cropLeft: Int = 0,
    val cropTop: Int = 0,
    val cropWidth: Int = 0,
    val cropHeight: Int = 0,
    val isCalibratedFromIntrinsics: Boolean = false
) {
    /**
     * `"intrinsics"` when derived from valid hardware `CameraCharacteristics` within the `[30..100]`
     * degree sanity bounds; `"fallback"` when using the default `60.0°` fallback.
     */
    val fovSource: String
        get() = if (isCalibratedFromIntrinsics) FOV_SOURCE_INTRINSICS else FOV_SOURCE_FALLBACK

    /**
     * Compact `SCALER_CROP_REGION` descriptor for HUD and CSV logging.
     */
    val scalerCropDescriptor: String
        get() = if (scalerCropWidth > 0 && scalerCropHeight > 0) {
            "${scalerCropWidth}x${scalerCropHeight}@${scalerCropLeft},${scalerCropTop}"
        } else {
            "none"
        }

    /**
     * Returns the visible-view vertical FOV (in degrees) for a viewport with [viewAspectRatio] (`viewW / viewH`)
     * after `FILL_CENTER` crop/scale.
     */
    fun visibleViewFovYDeg(viewAspectRatio: Float): Float {
        return FacePoseAdapter.computeVisibleViewFovYDeg(
            fovYCropDeg = fovYDeg,
            cropAspectRatio = cropAspectRatio,
            viewAspectRatio = viewAspectRatio
        )
    }

    companion object {
        const val DEFAULT_FOV_Y_DEG = 60.0f
        const val MIN_SANITY_FOV_Y_DEG = 30.0f
        const val MAX_SANITY_FOV_Y_DEG = 100.0f
        const val DEFAULT_ASPECT_RATIO = 9f / 16f
        const val DEFAULT_NEAR = 0.1f
        const val DEFAULT_FAR = 10.0f
        const val FOV_SOURCE_INTRINSICS = "intrinsics"
        const val FOV_SOURCE_FALLBACK = "fallback"

        val DEFAULT = ProjectionParams()
    }
}
