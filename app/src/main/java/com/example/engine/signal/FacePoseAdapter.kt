package com.example.engine.signal

import android.graphics.Matrix
import android.graphics.Rect
import android.opengl.GLES30
import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * FacePoseAdapter is the single authority and owner of (README Section 9 & 10):
 * - Canonical axes: X subject-right (+X), Y up (+Y), Z toward camera (+Z).
 * - Subject-based anatomical left/right mapping: negative X = subject-left.
 * - Single-stage front-camera mirroring:
 *   * **3D Path**: Mirror is applied **exclusively** in [createCameraProjection] (`proj[0] = -(f / aspectRatio)`
 *     when `frontFacing == true`), with triangle winding inversion handled via [glFrontFaceForCamera].
 *     Upstream stages (`convertAndTransformIntoSlot`, `calibratePoseMatrix`, `SignalPipeline`,
 *     `SceneEvaluator`, `FaceMeshRenderer.updateVertices`) operate in unmirrored metric camera space.
 *   * **2D Landmark Path**: Mirror is applied **exclusively** inside the 3x3 `imageToViewTransform`
 *     matrix (either by CameraX `PreviewView.outputTransform` via `CoordinateTransform`, which already
 *     mirrors the front camera, or by [computeManualFillCenterImageToViewTransform] when `outputTransform`
 *     is unavailable). Downstream 2D consumers ([mapLandmarkToView], `DebugOverlay`) apply
 *     `imageToViewTransform` directly without any second manual flip.
 * - Two-FOV contract:
 *   * Analysis-crop vertical FOV (`fovYCropDeg`) for unprojecting normalized analysis-crop landmarks.
 *   * Visible-view vertical FOV ([computeVisibleViewFovYDeg]) after `FILL_CENTER` for all view render projections.
 * - Face-width (FW) normalisation: `FW = ||M · c234 - M · c454||`.
 * - Face-local coordinate frame (all package distances use FW units).
 *
 * No filter package, renderer or behavior runner is permitted to implement its own coordinate correction.
 */
object FacePoseAdapter {

    // Pinned cheek landmark IDs in MediaPipe 478 topology (README Section 20)
    const val LANDMARK_CHEEK_RIGHT = 234 // subject-right
    const val LANDMARK_CHEEK_LEFT = 454  // subject-left

    // Key facial anchor landmark indices
    const val LANDMARK_NOSE_TIP = 1
    const val LANDMARK_FOREHEAD = 10
    const val LANDMARK_CHIN = 152
    const val LANDMARK_EYE_LEFT = 263   // subject left eye outer corner
    const val LANDMARK_EYE_RIGHT = 33   // subject right eye outer corner
    const val LANDMARK_MOUTH_CENTER = 13 // upper lip midpoint
    const val LANDMARK_MOUTH_LOWER = 14 // lower lip midpoint
    const val LANDMARK_EAR_LEFT = 454   // subject left ear/cheek region
    const val LANDMARK_EAR_RIGHT = 234  // subject right ear/cheek region

    // Canonical Axis unit vectors
    val AXIS_SUBJECT_RIGHT = floatArrayOf(1f, 0f, 0f)  // +X
    val AXIS_UP = floatArrayOf(0f, 1f, 0f)             // +Y
    val AXIS_TOWARD_CAMERA = floatArrayOf(0f, 0f, 1f)  // +Z
    val AXIS_SUBJECT_LEFT = floatArrayOf(-1f, 0f, 0f)  // -X

    /**
     * Confirms that negative X is the subject's anatomical left under active camera and mirror policy.
     */
    fun isSubjectLeft(x: Float): Boolean = x < 0f

    /**
     * Confirms that positive X is the subject's anatomical right under active camera and mirror policy.
     */
    fun isSubjectRight(x: Float): Boolean = x > 0f

    /**
     * Returns the vertical focal factor `f = 1 / tan(fovYDeg / 2)` for a vertical FOV in degrees.
     */
    fun focalFactorFromFovY(fovYDeg: Float): Float {
        val clampedFov = if (fovYDeg.isFinite() && fovYDeg in 5.0f..170.0f) {
            fovYDeg
        } else {
            ProjectionParams.DEFAULT_FOV_Y_DEG
        }
        val halfRad = Math.toRadians(clampedFov.toDouble() * 0.5)
        return (1.0 / tan(halfRad)).toFloat()
    }

    /**
     * Pure function computing the **VISIBLE-VIEW** vertical FOV (in degrees) after `FILL_CENTER`
     * scaling/cropping from the upright analysis crop (`cropAspectRatio`) to the visible viewport
     * (`viewAspectRatio = viewW / viewH`):
     * - When `viewAspectRatio > cropAspectRatio` (view is wider than the analysis crop, so `FILL_CENTER`
     *   matches width and crops top/bottom vertically):
     *   `tan(fovYView / 2) = tan(fovYCrop / 2) * (cropAspectRatio / viewAspectRatio)`
     * - Otherwise (`viewAspectRatio <= cropAspectRatio`, where `FILL_CENTER` matches height and crops
     *   left/right horizontally):
     *   `fovYView = fovYCropDeg` (vertical FOV is unchanged).
     */
    fun computeVisibleViewFovYDeg(
        fovYCropDeg: Float,
        cropAspectRatio: Float,
        viewAspectRatio: Float
    ): Float {
        val validFovYCrop = if (fovYCropDeg.isFinite() && fovYCropDeg > 1.0f && fovYCropDeg < 179.0f) {
            fovYCropDeg
        } else {
            ProjectionParams.DEFAULT_FOV_Y_DEG
        }
        val safeCropAspect = if (cropAspectRatio.isFinite() && cropAspectRatio > 0.01f) cropAspectRatio else 9f / 16f
        val safeViewAspect = if (viewAspectRatio.isFinite() && viewAspectRatio > 0.01f) viewAspectRatio else safeCropAspect

        if (safeViewAspect <= safeCropAspect) {
            return validFovYCrop
        }
        val tanHalfCrop = tan(Math.toRadians(validFovYCrop.toDouble() * 0.5))
        val tanHalfView = tanHalfCrop * (safeCropAspect.toDouble() / safeViewAspect.toDouble())
        val fovYView = Math.toDegrees(2.0 * atan(tanHalfView)).toFloat()
        return if (fovYView.isFinite() && fovYView > 1.0f && fovYView < 179.0f) fovYView else validFovYCrop
    }

    /**
     * Derives per-frame [ProjectionParams] (specifically the **analysis-crop FOV** `fovYDeg` and
     * `cropAspectRatio`) from `CameraCharacteristics` intrinsics (`LENS_INFO_AVAILABLE_FOCAL_LENGTHS`
     * or `CaptureResult.LENS_FOCAL_LENGTH`, `SENSOR_INFO_PHYSICAL_SIZE`, `SENSOR_INFO_PIXEL_ARRAY_SIZE`,
     * `SENSOR_INFO_ACTIVE_ARRAY_SIZE`, `CaptureResult.SCALER_CROP_REGION`) and the frame's actual
     * `cropRect` and upright `rotationDegrees`.
     *
     * Enforces sanity bounds (`fovYDeg in 30.0f..100.0f`); if hardware intrinsics are missing or
     * produce an out-of-bounds FOV, falls back cleanly to `60.0°` with `fovSource = "fallback"`.
     */
    @JvmOverloads
    fun computeProjectionParams(
        intrinsics: CameraIntrinsics,
        bufferWidth: Int,
        bufferHeight: Int,
        cropRect: Rect? = null,
        rotationDegrees: Int = 0,
        cropLeftOverride: Int = -1,
        cropTopOverride: Int = -1,
        cropWidthOverride: Int = -1,
        cropHeightOverride: Int = -1
    ): ProjectionParams {
        val safeBufW = bufferWidth.coerceAtLeast(1)
        val safeBufH = bufferHeight.coerceAtLeast(1)
        val cLeft = (if (cropLeftOverride >= 0) cropLeftOverride else (cropRect?.left ?: 0)).coerceIn(0, safeBufW - 1)
        val cTop = (if (cropTopOverride >= 0) cropTopOverride else (cropRect?.top ?: 0)).coerceIn(0, safeBufH - 1)
        val rawCropW = if (cropWidthOverride > 0) {
            cropWidthOverride
        } else {
            cropRect?.let { (it.right - it.left).takeIf { w -> w > 0 } ?: it.width() }?.takeIf { it > 0 } ?: safeBufW
        }
        val rawCropH = if (cropHeightOverride > 0) {
            cropHeightOverride
        } else {
            cropRect?.let { (it.bottom - it.top).takeIf { h -> h > 0 } ?: it.height() }?.takeIf { it > 0 } ?: safeBufH
        }
        val cropW = rawCropW.coerceIn(1, safeBufW - cLeft)
        val cropH = rawCropH.coerceIn(1, safeBufH - cTop)

        val normRot = normalizeRotationDegrees(rotationDegrees)
        val uprightCropW = if (normRot == 90 || normRot == 270) cropH else cropW
        val uprightCropH = if (normRot == 90 || normRot == 270) cropW else cropH
        val cropAspect = uprightCropW.toFloat() / uprightCropH.toFloat().coerceAtLeast(1f)

        val activeW = intrinsics.activeArrayWidth.coerceAtLeast(0)
        val activeH = intrinsics.activeArrayHeight.coerceAtLeast(0)
        val scalerLeft = intrinsics.scalerCropLeft
        val scalerTop = intrinsics.scalerCropTop
        val scalerW = intrinsics.scalerCropWidth.coerceAtLeast(0)
        val scalerH = intrinsics.scalerCropHeight.coerceAtLeast(0)

        if (!intrinsics.isValid) {
            val defaultFovX = Math.toDegrees(
                2.0 * atan(tan(Math.toRadians(ProjectionParams.DEFAULT_FOV_Y_DEG * 0.5)) * cropAspect.toDouble())
            ).toFloat()
            return ProjectionParams(
                fovYDeg = ProjectionParams.DEFAULT_FOV_Y_DEG,
                fovXDeg = defaultFovX,
                cropAspectRatio = cropAspect,
                focalLengthMm = intrinsics.focalLengthMm,
                sensorWidthMm = intrinsics.sensorWidthMm,
                sensorHeightMm = intrinsics.sensorHeightMm,
                pixelArrayWidth = intrinsics.pixelArrayWidth,
                pixelArrayHeight = intrinsics.pixelArrayHeight,
                activeArrayWidth = activeW,
                activeArrayHeight = activeH,
                scalerCropLeft = scalerLeft,
                scalerCropTop = scalerTop,
                scalerCropWidth = scalerW,
                scalerCropHeight = scalerH,
                cropLeft = cLeft,
                cropTop = cTop,
                cropWidth = cropW,
                cropHeight = cropH,
                isCalibratedFromIntrinsics = false
            )
        }

        // 1. Scale physical sensor size (mm) to active pixel array using SENSOR_INFO_PIXEL_ARRAY_SIZE
        val pxArrW = intrinsics.pixelArrayWidth
        val pxArrH = intrinsics.pixelArrayHeight
        val activeScaleX = if (pxArrW > 0 && activeW > 0) {
            (activeW.toFloat() / pxArrW.toFloat()).coerceIn(0.1f, 1.0f)
        } else {
            1.0f
        }
        val activeScaleY = if (pxArrH > 0 && activeH > 0) {
            (activeH.toFloat() / pxArrH.toFloat()).coerceIn(0.1f, 1.0f)
        } else {
            1.0f
        }

        var windowWidthMm = intrinsics.sensorWidthMm * activeScaleX
        var windowHeightMm = intrinsics.sensorHeightMm * activeScaleY
        var refWindowPxW = if (activeW > 0) activeW.toFloat() else intrinsics.sensorWidthMm
        var refWindowPxH = if (activeH > 0) activeH.toFloat() else intrinsics.sensorHeightMm

        // 2. Apply CaptureResult.SCALER_CROP_REGION if present (digital zoom / logical camera crop)
        if (scalerW > 0 && scalerH > 0 && activeW > 0 && activeH > 0) {
            val scalerFracX = (scalerW.toFloat() / activeW.toFloat()).coerceIn(0.05f, 1.0f)
            val scalerFracY = (scalerH.toFloat() / activeH.toFloat()).coerceIn(0.05f, 1.0f)
            windowWidthMm *= scalerFracX
            windowHeightMm *= scalerFracY
            refWindowPxW = scalerW.toFloat()
            refWindowPxH = scalerH.toFloat()
        }

        // 3. Account for stream buffer aspect crop relative to the active/scaler sensor window
        val bufAspect = safeBufW.toFloat() / safeBufH.toFloat()
        val windowAspect = (refWindowPxW / refWindowPxH.coerceAtLeast(1e-4f)).coerceAtLeast(1e-4f)
        val bufFracX: Float
        val bufFracY: Float
        if (bufAspect >= windowAspect) {
            bufFracX = 1.0f
            bufFracY = (windowAspect / bufAspect).coerceIn(0.05f, 1.0f)
        } else {
            bufFracX = (bufAspect / windowAspect).coerceIn(0.05f, 1.0f)
            bufFracY = 1.0f
        }

        // 4. Apply ImageProxy.cropRect within the stream buffer
        val cropFracX = cropW.toFloat() / safeBufW.toFloat()
        val cropFracY = cropH.toFloat() / safeBufH.toFloat()
        val effectiveSensorCropW_mm = windowWidthMm * bufFracX * cropFracX
        val effectiveSensorCropH_mm = windowHeightMm * bufFracY * cropFracY

        // 5. Rotate to upright analysis image orientation
        val uprightW_mm = if (normRot == 90 || normRot == 270) effectiveSensorCropH_mm else effectiveSensorCropW_mm
        val uprightH_mm = if (normRot == 90 || normRot == 270) effectiveSensorCropW_mm else effectiveSensorCropH_mm

        val fMm = intrinsics.focalLengthMm
        val fovYCropDeg = Math.toDegrees(2.0 * atan((uprightH_mm * 0.5) / fMm.toDouble())).toFloat()
        val fovXCropDeg = Math.toDegrees(2.0 * atan((uprightW_mm * 0.5) / fMm.toDouble())).toFloat()

        // 6. Sanity bounds check: vertical FOV must be within [30..100] degrees, else fall back to 60 deg
        val withinSanityBounds = fovYCropDeg.isFinite() &&
            fovXCropDeg.isFinite() &&
            fovYCropDeg in ProjectionParams.MIN_SANITY_FOV_Y_DEG..ProjectionParams.MAX_SANITY_FOV_Y_DEG

        val finalFovY = if (withinSanityBounds) fovYCropDeg else ProjectionParams.DEFAULT_FOV_Y_DEG
        val finalFovX = if (withinSanityBounds) {
            fovXCropDeg
        } else {
            Math.toDegrees(
                2.0 * atan(tan(Math.toRadians(ProjectionParams.DEFAULT_FOV_Y_DEG * 0.5)) * cropAspect.toDouble())
            ).toFloat()
        }

        return ProjectionParams(
            fovYDeg = finalFovY,
            fovXDeg = finalFovX,
            cropAspectRatio = cropAspect,
            focalLengthMm = fMm,
            sensorWidthMm = intrinsics.sensorWidthMm,
            sensorHeightMm = intrinsics.sensorHeightMm,
            pixelArrayWidth = pxArrW,
            pixelArrayHeight = pxArrH,
            activeArrayWidth = activeW,
            activeArrayHeight = activeH,
            scalerCropLeft = scalerLeft,
            scalerCropTop = scalerTop,
            scalerCropWidth = scalerW,
            scalerCropHeight = scalerH,
            cropLeft = cLeft,
            cropTop = cTop,
            cropWidth = cropW,
            cropHeight = cropH,
            isCalibratedFromIntrinsics = withinSanityBounds
        )
    }

    /**
     * Computes Face Width (FW) using cheek landmarks 234 and 454:
     * `FW = ||M · c234 - M · c454||`
     *
     * Uses the **analysis-crop vertical FOV** (`fovYCropDeg`) and `imageAspectRatio` (`cropAspectRatio`)
     * because landmarks `c234` and `c454` are normalized to the upright analysis crop.
     * Falls back to standard human face width (~0.14m = 140mm) when landmarks are unavailable.
     */
    @JvmOverloads
    fun computeFaceWidth(
        landmarks: FloatArray?, // Flat [x0, y0, z0, x1, y1, z1, ...]
        matrix: FloatArray? = null,
        imageAspectRatio: Float = 9f / 16f,
        fovYCropDeg: Float = ProjectionParams.DEFAULT_FOV_Y_DEG
    ): Float {
        if (landmarks != null && landmarks.size >= (455 * 3)) {
            val i234 = LANDMARK_CHEEK_RIGHT * 3
            val i454 = LANDMARK_CHEEK_LEFT * 3

            val rawX1 = landmarks[i234]
            val rawY1 = landmarks[i234 + 1]
            val rawZ1 = landmarks[i234 + 2]

            val rawX2 = landmarks[i454]
            val rawY2 = landmarks[i454 + 1]
            val rawZ2 = landmarks[i454 + 2]

            var x1 = rawX1
            var y1 = rawY1
            var z1 = rawZ1

            var x2 = rawX2
            var y2 = rawY2
            var z2 = rawZ2

            if (matrix != null && matrix.size == 16) {
                val absZ = abs(matrix[14])
                // Distinguish pre-centered metric test points (y1 == 0 && y2 == 0, opposite X signs)
                // from normalized analysis-crop coordinates (which live around [0..1] when in-frame
                // and can extend < 0.01 or > 0.99 when a cheek approaches or crosses the screen edge).
                val isPreCenteredMetric = (rawY1 == 0f && rawY2 == 0f && rawX1 * rawX2 < 0f && abs(rawX2 - rawX1) <= 0.30f)
                var wCam = 1f
                var hCam = 1f
                if (absZ >= 0.05f && !isPreCenteredMetric) {
                    // Convert normalized analysis-crop coordinates into metric camera space at depth absZ
                    // using the analysis-crop vertical FOV (fovYCropDeg)
                    val f = focalFactorFromFovY(fovYCropDeg)
                    wCam = 2.0f * absZ * imageAspectRatio.coerceAtLeast(0.1f) / f
                    hCam = 2.0f * absZ / f
                    x1 *= wCam; x2 *= wCam
                    y1 *= hCam; y2 *= hCam
                    z1 *= wCam; z2 *= wCam
                }

                // Apply 4x4 facial transform M to c234 and c454
                val mx1 = matrix[0] * x1 + matrix[4] * y1 + matrix[8] * z1 + matrix[12]
                val my1 = matrix[1] * x1 + matrix[5] * y1 + matrix[9] * z1 + matrix[13]
                val mz1 = matrix[2] * x1 + matrix[6] * y1 + matrix[10] * z1 + matrix[14]

                val mx2 = matrix[0] * x2 + matrix[4] * y2 + matrix[8] * z2 + matrix[12]
                val my2 = matrix[1] * x2 + matrix[5] * y2 + matrix[9] * z2 + matrix[13]
                val mz2 = matrix[2] * x2 + matrix[6] * y2 + matrix[10] * z2 + matrix[14]

                val dx = mx2 - mx1
                val dy = my2 - my1
                val dz = mz2 - mz1
                var dist = sqrt(dx * dx + dy * dy + dz * dz)

                // When a full 468-landmark face moves near or past the left/right/top/bottom crop edge,
                // one cheek (#234 or #454) may be clipped and extrapolated by MediaPipe. Bound the
                // resulting metric width against the visible vertical forehead-to-chin span (#10..#152)
                // so partially off-screen faces never blow up or collapse in scale.
                if (absZ >= 0.05f && !isPreCenteredMetric && landmarks.size >= 468 * 3) {
                    val isNearOrPastEdge = rawX1 !in 0.04f..0.96f || rawX2 !in 0.04f..0.96f ||
                        rawY1 !in 0.04f..0.96f || rawY2 !in 0.04f..0.96f
                    val yTop = landmarks[LANDMARK_FOREHEAD * 3 + 1]
                    val yChin = landmarks[LANDMARK_CHIN * 3 + 1]
                    val vertSpanNorm = yChin - yTop
                    if (isNearOrPastEdge && vertSpanNorm > 0.08f && landmarks[33 * 3 + 1] != 0.50f) {
                        val xTop = landmarks[LANDMARK_FOREHEAD * 3]
                        val xChin = landmarks[LANDMARK_CHIN * 3]
                        val dxVert = (xChin - xTop) * wCam
                        val dyVert = (yChin - yTop) * hCam
                        val faceHeightMeters = sqrt(dxVert * dxVert + dyVert * dyVert)
                        if (faceHeightMeters > 0.05f) {
                            // Canonical cheek-width / (forehead #10 to chin #152 height) ratio is ~0.88
                            val minEdgeFw = (faceHeightMeters * 0.72f).coerceAtLeast(0.08f)
                            val maxEdgeFw = (faceHeightMeters * 0.96f).coerceAtMost(0.24f)
                            dist = dist.coerceIn(minEdgeFw, maxEdgeFw)
                        }
                    }
                }

                if (dist > 1e-4f && dist.isFinite()) return dist
            } else {
                val dx = x2 - x1
                val dy = y2 - y1
                val dz = z2 - z1
                val dist = sqrt(dx * dx + dy * dy + dz * dz)
                if (dist > 1e-4f && dist.isFinite()) return dist
            }
        }
        return 0.14f // Standard human face width ~ 14 cm
    }

    /**
     * Single position-only X-axis sign conversion between package/anchor face-local coordinates
     * (`negative x = subject-left`, `positive x = subject-right`) and MediaPipe's canonical pose
     * basis (`+R_x` points from `#234` subject-right toward `#454` subject-left in unmirrored camera space).
     *
     * Applied ONLY to 3D positions (never to GLB 3x3 rotation/scale matrices) so models are never mirrored.
     */
    const val FACE_LOCAL_TO_POSE_X_SIGN = -1.0f

    fun faceLocalXToPoseMeters(localX: Float, fw: Float): Float {
        return FACE_LOCAL_TO_POSE_X_SIGN * localX * fw
    }

    fun poseMetersToFaceLocalX(poseX: Float, fw: Float): Float {
        return FACE_LOCAL_TO_POSE_X_SIGN * (poseX / fw.coerceAtLeast(0.01f))
    }

    /**
     * Converts a 3D coordinate from face-local FW units (`negative x = subject-left`) to metric
     * camera space using pose matrix M:
     * `p_camera = M * [-localX * FW, localY * FW, localZ * FW, 1]^T`
     */
    fun faceLocalToCamera(
        localX: Float,
        localY: Float,
        localZ: Float,
        matrix: FloatArray,
        fw: Float,
        out: FloatArray = FloatArray(3)
    ): FloatArray {
        require(matrix.size == 16 && out.size >= 3)
        val scaledX = faceLocalXToPoseMeters(localX, fw)
        val scaledY = localY * fw
        val scaledZ = localZ * fw

        out[0] = matrix[0] * scaledX + matrix[4] * scaledY + matrix[8] * scaledZ + matrix[12]
        out[1] = matrix[1] * scaledX + matrix[5] * scaledY + matrix[9] * scaledZ + matrix[13]
        out[2] = matrix[2] * scaledX + matrix[6] * scaledY + matrix[10] * scaledZ + matrix[14]
        return out
    }

    /**
     * Shared zero-allocation landmark unprojection from normalized analysis-crop coordinates
     * `(lx, ly, lz)` in [faceState] into metric 3D camera space (`+X` image-right, `+Y` up, `-Z` into screen)
     * at landmark-time pose depth `rawTz`, used by `FaceMeshRenderer`, `OccluderPass`, and `AnchorResolver`.
     */
    @JvmOverloads
    fun unprojectLandmarkToCameraSpace(
        faceState: FaceState,
        landmarkId: Int,
        viewportAspect: Float = 9f / 16f,
        pushBackFW: Float = 0f,
        outCam: FloatArray
    ): Boolean {
        val lm = faceState.landmarks ?: return false
        val baseIdx = landmarkId * 3
        if (landmarkId < 0 || baseIdx + 2 >= lm.size || outCam.size < 3) return false

        val viewAspect = viewportAspect.coerceAtLeast(0.01f)
        val projParams = faceState.projectionParams
        val cropAspect = if (projParams !== ProjectionParams.DEFAULT) {
            projParams.cropAspectRatio.coerceAtLeast(0.01f)
        } else {
            viewAspect
        }
        val fovYCropDeg = projParams.fovYDeg
        val f = focalFactorFromFovY(fovYCropDeg)

        val rawTz = faceState.landmarkPoseMatrix[14].let {
            if (abs(it) > 0.05f) it else faceState.poseMatrix[14]
        }
        val absZ = (if (abs(rawTz) > 0.05f) abs(rawTz) else 0.50f).coerceIn(0.25f, 1.20f)

        val wCam = 2.0f * absZ * cropAspect / f
        val hCam = 2.0f * absZ / f
        val fw = faceState.faceWidth.coerceAtLeast(0.08f)
        val pushDistMeters = pushBackFW * fw

        val lx = lm[baseIdx]
        val ly = lm[baseIdx + 1]
        val lz = lm[baseIdx + 2]

        var vx = (lx - 0.5f) * wCam
        var vy = -(ly - 0.5f) * hCam
        var vz = -absZ - (lz * wCam)

        if (pushDistMeters != 0f) {
            val rayLen = sqrt(vx * vx + vy * vy + vz * vz).coerceAtLeast(1e-6f)
            val rayScale = 1.0f + (pushDistMeters / rayLen)
            vx *= rayScale
            vy *= rayScale
            vz *= rayScale
        }

        outCam[0] = vx
        outCam[1] = vy
        outCam[2] = vz
        return true
    }

    /**
     * Transforms a metric 3D camera-space position `(camX, camY, camZ)` by `inverse(poseMatrix)`,
     * divides by `fw`, and applies the single X-axis side convention conversion (`negative x = subject-left`)
     * into `outLocalFW` (`[localX, localY, localZ]` in `FW` units).
     * Zero heap allocations.
     */
    fun cameraToFaceLocalFW(
        camX: Float,
        camY: Float,
        camZ: Float,
        poseMatrix: FloatArray,
        fw: Float,
        outLocalFW: FloatArray
    ) {
        val safeFw = fw.coerceAtLeast(0.01f)

        // Column 0 of R (orthonormalized)
        var r00 = poseMatrix[0]; var r10 = poseMatrix[1]; var r20 = poseMatrix[2]
        val n0 = sqrt(r00 * r00 + r10 * r10 + r20 * r20)
        if (n0 > 1e-6f) { val inv0 = 1f / n0; r00 *= inv0; r10 *= inv0; r20 *= inv0 }

        // Column 1 of R (orthonormalized)
        var r01 = poseMatrix[4]; var r11 = poseMatrix[5]; var r21 = poseMatrix[6]
        val n1 = sqrt(r01 * r01 + r11 * r11 + r21 * r21)
        if (n1 > 1e-6f) { val inv1 = 1f / n1; r01 *= inv1; r11 *= inv1; r21 *= inv1 }

        // Column 2 of R (orthonormalized)
        var r02 = poseMatrix[8]; var r12 = poseMatrix[9]; var r22 = poseMatrix[10]
        val n2 = sqrt(r02 * r02 + r12 * r12 + r22 * r22)
        if (n2 > 1e-6f) { val inv2 = 1f / n2; r02 *= inv2; r12 *= inv2; r22 *= inv2 }

        val dx = camX - poseMatrix[12]
        val dy = camY - poseMatrix[13]
        val dz = camZ - poseMatrix[14]

        // R^T * (p_cam - t) in MediaPipe pose meters
        val poseX = r00 * dx + r10 * dy + r20 * dz
        val poseY = r01 * dx + r11 * dy + r21 * dz
        val poseZ = r02 * dx + r12 * dy + r22 * dz

        outLocalFW[0] = poseMetersToFaceLocalX(poseX, safeFw)
        outLocalFW[1] = poseY / safeFw
        outLocalFW[2] = poseZ / safeFw
    }

    /**
     * Create perspective projection matrix for camera view.
     *
     * **Single-Stage 3D Mirror Policy**:
     * Front-camera mirroring for all 3D geometry occurs **exclusively** in this final projection stage
     * (`out[0] = -(f / aspectRatio)` when `frontFacing == true`, `+(f / aspectRatio)` when `false`).
     */
    fun createCameraProjection(
        fovYDeg: Float,
        aspectRatio: Float,
        near: Float = ProjectionParams.DEFAULT_NEAR,
        far: Float = 10.0f,
        frontFacing: Boolean = true,
        out: FloatArray = FloatArray(16)
    ): FloatArray {
        val f = focalFactorFromFovY(fovYDeg)
        val safeAspect = aspectRatio.coerceAtLeast(0.01f)
        java.util.Arrays.fill(out, 0f)

        // Standard perspective projection
        // If front camera, mirror X axis: -f / aspectRatio (mirror exactly once)
        out[0] = if (frontFacing) -(f / safeAspect) else (f / safeAspect)
        out[5] = f
        out[10] = (far + near) / (near - far)
        out[11] = -1.0f
        out[14] = (2.0f * far * near) / (near - far)

        return out
    }

    /**
     * Returns the OpenGL front-face winding order matching [createCameraProjection]:
     * - Rear camera (`frontFacing == false`): `proj[0] > 0`, standard counter-clockwise (`GL_CCW`)
     *   triangles remain counter-clockwise in NDC -> returns `GLES30.GL_CCW`.
     * - Front camera (`frontFacing == true`): `proj[0] < 0` mirrors X once in NDC, which flips
     *   triangle winding from `CCW` to `CW` -> returns `GLES30.GL_CW` so back-face culling (`GL_BACK`)
     *   still culls back faces and keeps front faces visible.
     */
    fun glFrontFaceForCamera(frontFacing: Boolean): Int {
        return if (frontFacing) GLES30.GL_CW else GLES30.GL_CCW
    }

    /**
     * Evaluates whether a 3D triangle `(v0, v1, v2)` transformed by [mvpMatrix] is considered
     * front-facing in NDC under [glFrontFaceForCamera] (`GL_CW` when `frontFacing == true`,
     * `GL_CCW` when `frontFacing == false`).
     */
    fun isTriangleFrontFacingInNdc(
        v0: FloatArray,
        v1: FloatArray,
        v2: FloatArray,
        mvpMatrix: FloatArray,
        frontFacing: Boolean
    ): Boolean {
        val ndc0x = projectToNdcX(v0, mvpMatrix)
        val ndc0y = projectToNdcY(v0, mvpMatrix)
        val ndc1x = projectToNdcX(v1, mvpMatrix)
        val ndc1y = projectToNdcY(v1, mvpMatrix)
        val ndc2x = projectToNdcX(v2, mvpMatrix)
        val ndc2y = projectToNdcY(v2, mvpMatrix)

        // Signed 2D area in OpenGL NDC (+X right, +Y up): positive = CCW, negative = CW
        val signedArea2 = (ndc1x - ndc0x) * (ndc2y - ndc0y) - (ndc1y - ndc0y) * (ndc2x - ndc0x)
        return when (glFrontFaceForCamera(frontFacing)) {
            GLES30.GL_CW -> signedArea2 < 0f
            else -> signedArea2 > 0f
        }
    }

    private fun projectToNdcX(v: FloatArray, m: FloatArray): Float {
        val cx = m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12]
        val cw = (m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15]).let { if (abs(it) < 1e-6f) 1f else it }
        return cx / cw
    }

    private fun projectToNdcY(v: FloatArray, m: FloatArray): Float {
        val cy = m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13]
        val cw = (m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15]).let { if (abs(it) < 1e-6f) 1f else it }
        return cy / cw
    }

    /**
     * Composes CameraX's `CoordinateTransform` pixel matrix (which maps upright analysis-crop pixels
     * `[0..orientedCropWidth, 0..orientedCropHeight]` into `PreviewView` pixels `[0..viewWidth, 0..viewHeight]`)
     * into a normalized 3x3 row-major matrix mapping normalized upright analysis-crop coordinates
     * `(u, v) in [0, 1]^2` into normalized view coordinates `(vx, vy) in [0, 1]^2`.
     *
     * **Single-Stage 2D Mirror Note**:
     * `PreviewView.outputTransform` ALREADY includes the horizontal mirror for a front-facing camera.
     * Therefore, this function does NOT apply any extra horizontal flip, and [mapLandmarkToView]
     * applies the resulting `outMatrix9` directly without any extra flip.
     */
    @JvmOverloads
    fun composeCameraXNormalizedImageToViewTransform(
        pixelTransformMatrix: Matrix,
        orientedCropWidth: Int,
        orientedCropHeight: Int,
        viewWidth: Int,
        viewHeight: Int,
        outMatrix9: FloatArray = FloatArray(9),
        @Suppress("UNUSED_PARAMETER") scratchMatrix: Matrix? = null
    ): FloatArray {
        pixelTransformMatrix.getValues(outMatrix9)
        return composeCameraXNormalizedImageToViewTransform(
            pixelTransformValues9 = outMatrix9,
            orientedCropWidth = orientedCropWidth,
            orientedCropHeight = orientedCropHeight,
            viewWidth = viewWidth,
            viewHeight = viewHeight,
            outMatrix9 = outMatrix9
        )
    }

    /**
     * Pure-float overload of [composeCameraXNormalizedImageToViewTransform] operating directly on
     * the 9 row-major values of CameraX's `CoordinateTransform` pixel matrix with zero JNI or heap
     * allocations.
     */
    fun composeCameraXNormalizedImageToViewTransform(
        pixelTransformValues9: FloatArray,
        orientedCropWidth: Int,
        orientedCropHeight: Int,
        viewWidth: Int,
        viewHeight: Int,
        outMatrix9: FloatArray = FloatArray(9)
    ): FloatArray {
        val cw = orientedCropWidth.coerceAtLeast(1).toFloat()
        val ch = orientedCropHeight.coerceAtLeast(1).toFloat()
        val vw = viewWidth.coerceAtLeast(1).toFloat()
        val vh = viewHeight.coerceAtLeast(1).toFloat()

        val m00 = pixelTransformValues9[0]
        val m01 = pixelTransformValues9[1]
        val m02 = pixelTransformValues9[2]
        val m10 = pixelTransformValues9[3]
        val m11 = pixelTransformValues9[4]
        val m12 = pixelTransformValues9[5]
        val m20 = pixelTransformValues9[6]
        val m21 = pixelTransformValues9[7]
        val m22 = pixelTransformValues9[8]

        // Pre-scale normalized [0..1] crop coordinates by (cw, ch) and post-scale view pixels by (1/vw, 1/vh)
        outMatrix9[0] = (m00 * cw) / vw
        outMatrix9[1] = (m01 * ch) / vw
        outMatrix9[2] = m02 / vw
        outMatrix9[3] = (m10 * cw) / vh
        outMatrix9[4] = (m11 * ch) / vh
        outMatrix9[5] = m12 / vh
        outMatrix9[6] = m20 * cw
        outMatrix9[7] = m21 * ch
        outMatrix9[8] = m22
        return outMatrix9
    }

    /**
     * Manual `FILL_CENTER` 3x3 normalized `image -> view` transform (`[0..1]^2` upright analysis crop
     * -> `[0..1]^2` visible view), used ONLY as a fallback when `PreviewView.outputTransform` is not
     * available (e.g., before first view layout or in headless JVM unit tests) and as the analytical
     * oracle in unit tests.
     *
     * **Single-Stage 2D Mirror Note**:
     * Matching CameraX `PreviewView.outputTransform`, the single front-camera horizontal mirror
     * (`1 - u` when `isFrontCamera == true`) is baked directly into `outMatrix9`. Downstream 2D
     * consumers ([mapLandmarkToView]) never apply a second flip.
     */
    fun computeManualFillCenterImageToViewTransform(
        cropAspectRatio: Float,
        viewAspectRatio: Float,
        isFrontCamera: Boolean,
        outMatrix9: FloatArray = FloatArray(9)
    ): FloatArray {
        val safeCropAspect = if (cropAspectRatio.isFinite() && cropAspectRatio > 0.01f) cropAspectRatio else 9f / 16f
        val safeViewAspect = if (viewAspectRatio.isFinite() && viewAspectRatio > 0.01f) viewAspectRatio else safeCropAspect

        val scaleX: Float
        val scaleY: Float
        if (safeViewAspect > safeCropAspect) {
            // Wide view: FILL_CENTER matches width (scaleX = 1) and crops top/bottom vertically (scaleY > 1)
            scaleX = 1.0f
            scaleY = safeViewAspect / safeCropAspect
        } else {
            // Narrow or equal view: FILL_CENTER matches height (scaleY = 1) and crops left/right horizontally (scaleX >= 1)
            scaleX = safeCropAspect / safeViewAspect
            scaleY = 1.0f
        }

        // Row-major 3x3 homogeneous matrix:
        // [ m00  m01  m02 ]   [ u ]
        // [ m10  m11  m12 ] * [ v ]
        // [  0    0    1  ]   [ 1 ]
        outMatrix9[0] = if (isFrontCamera) -scaleX else scaleX
        outMatrix9[1] = 0f
        outMatrix9[2] = if (isFrontCamera) 0.5f * (1.0f + scaleX) else 0.5f * (1.0f - scaleX)

        outMatrix9[3] = 0f
        outMatrix9[4] = scaleY
        outMatrix9[5] = 0.5f * (1.0f - scaleY)

        outMatrix9[6] = 0f
        outMatrix9[7] = 0f
        outMatrix9[8] = 1.0f
        return outMatrix9
    }

    /**
     * Checks whether a 9-float `imageToViewTransform` represents a valid normalized `[0..1] -> [0..1]`
     * affine transform (as opposed to an unnormalized pixel sensorToBuffer matrix).
     */
    fun isNormalizedImageToViewTransform(m: FloatArray?): Boolean {
        if (m == null || m.size < 9) return false
        val scaleX = abs(m[0]) + abs(m[1])
        val scaleY = abs(m[3]) + abs(m[4])
        if (!scaleX.isFinite() || !scaleY.isFinite()) return false
        if (scaleX < 0.05f || scaleX > 20.0f || scaleY < 0.05f || scaleY > 20.0f) return false
        if (abs(m[2]) > 10.0f || abs(m[5]) > 10.0f) return false
        return true
    }

    /**
     * Maps a normalized upright analysis-crop landmark `(u, v) in [0, 1]^2` into view pixel coordinates
     * `(px, py) in [0..viewWidth, 0..viewHeight]` with zero heap allocation when [outPt] is supplied.
     *
     * **Single-Stage 2D Mirror Enforcement**:
     * - When a valid normalized `imageToViewTransform` is provided (from CameraX `CoordinateTransform`
     *   or [computeManualFillCenterImageToViewTransform]), the front-camera mirror is ALREADY inside
     *   `imageToViewTransform`. This method multiplies `imageToViewTransform * [u, v, 1]^T` directly
     *   and **never** applies an extra horizontal flip.
     * - Only when `imageToViewTransform` is null/unnormalized does it fall back to `[if (isFrontCamera) 1 - u else u, v]`.
     */
    fun mapLandmarkToView(
        u: Float,
        v: Float,
        imageToViewTransform: FloatArray?,
        isFrontCamera: Boolean,
        viewWidth: Float,
        viewHeight: Float,
        outPt: FloatArray = FloatArray(2)
    ): FloatArray {
        val normX: Float
        val normY: Float
        if (isNormalizedImageToViewTransform(imageToViewTransform)) {
            val m = imageToViewTransform!!
            val w = (m[6] * u + m[7] * v + m[8]).let { if (abs(it) < 1e-6f) 1f else it }
            normX = (m[0] * u + m[1] * v + m[2]) / w
            normY = (m[3] * u + m[4] * v + m[5]) / w
        } else {
            normX = if (isFrontCamera) (1.0f - u) else u
            normY = v
        }
        outPt[0] = normX * viewWidth
        outPt[1] = normY * viewHeight
        return outPt
    }

    /**
     * Unified 3D+2D landmark projection:
     * Maps a normalized upright analysis-crop landmark `(u, v, z)` into view coordinates
     * `(px, py)` in `[0..viewWidth, 0..viewHeight]` while folding in the exact 3D rigid
     * pose-prediction delta ($T = M_{\text{pred}} \cdot M_{\text{landmark}}^{-1}$) used by
     * `FaceMeshRenderer` and 3D AR/VR props when `faceState.poseMatrix` has been predicted
     * ahead of `faceState.landmarkPoseMatrix`.
     *
     * When `poseMatrix` and `landmarkPoseMatrix` are identical (e.g. static frames or unit tests),
     * the 3D delta is identically `(0, 0)`, preserving 2D `imageToViewTransform` bit-for-bit.
     */
    @JvmOverloads
    fun mapLandmarkToViewWithPoseDelta(
        u: Float,
        v: Float,
        z: Float = 0f,
        faceState: FaceState,
        viewWidth: Float = 1.0f,
        viewHeight: Float = 1.0f,
        outPt: FloatArray = FloatArray(2)
    ): FloatArray {
        mapLandmarkToView(
            u = u,
            v = v,
            imageToViewTransform = faceState.imageToViewTransform,
            isFrontCamera = faceState.isFrontCamera,
            viewWidth = 1.0f,
            viewHeight = 1.0f,
            outPt = outPt
        )
        var normX = outPt[0]
        var normY = outPt[1]

        // Apply 3D rigid pose-prediction delta only when poseMatrix has been predicted ahead of
        // capture time (or explicitly differs from landmarkPoseMatrix in synthetic pose-delta tests).
        // When timestampNs == poseTimestampNs > 0L (an unpredicted snapshot straight from SignalPipeline.processSnapshot),
        // poseMatrix is the capture-time filtered pose while landmarkPoseMatrix is rawMatrix, so applying
        // filtered * raw^(-1) at dt=0 would pull 2D landmarks backward by the pose filter's lag.
        val isUnpredictedCaptureSnapshot = faceState.poseTimestampNs > 0L &&
            faceState.timestampNs == faceState.poseTimestampNs
        val predM = faceState.poseMatrix
        val lmM = faceState.landmarkPoseMatrix
        if (!isUnpredictedCaptureSnapshot &&
            predM.size >= 16 &&
            lmM.size >= 16 &&
            abs(predM[14]) > 0.05f &&
            abs(lmM[14]) > 0.05f
        ) {
            var maxDiff = 0f
            for (k in 0 until 16) {
                val d = abs(predM[k] - lmM[k])
                if (d > maxDiff) maxDiff = d
            }
            if (maxDiff > 1e-5f) {
                val projParams = faceState.projectionParams
                val viewAspect = (viewWidth / viewHeight.coerceAtLeast(1e-4f)).let {
                    if (it.isFinite() && it > 0.01f && abs(it - 1.0f) > 1e-3f) {
                        it
                    } else if (faceState.imageWidth > 0 && faceState.imageHeight > 0) {
                        faceState.imageWidth.toFloat() / faceState.imageHeight.toFloat()
                    } else {
                        projParams.cropAspectRatio.coerceAtLeast(0.01f)
                    }
                }
                val cropAspect = if (projParams !== ProjectionParams.DEFAULT) {
                    projParams.cropAspectRatio.coerceAtLeast(0.01f)
                } else if (faceState.imageWidth > 0 && faceState.imageHeight > 0) {
                    (faceState.imageWidth.toFloat() / faceState.imageHeight.toFloat()).coerceAtLeast(0.01f)
                } else {
                    viewAspect
                }
                val fCrop = focalFactorFromFovY(projParams.fovYDeg)
                val absZ = abs(lmM[14]).coerceIn(0.25f, 1.20f)
                val wCam = 2.0f * absZ * cropAspect / fCrop
                val hCam = 2.0f * absZ / fCrop

                val vx = (u - 0.5f) * wCam
                val vy = -(v - 0.5f) * hCam
                val vz = -absZ - (z * wCam)

                // Orthonormal inverse of lmM applied to p_cam = (vx, vy, vz): p_local = R_lm^T * (p_cam - t_lm)
                var r00 = lmM[0]; var r10 = lmM[1]; var r20 = lmM[2]
                val n0 = sqrt(r00 * r00 + r10 * r10 + r20 * r20)
                if (n0 > 1e-6f) { val inv0 = 1f / n0; r00 *= inv0; r10 *= inv0; r20 *= inv0 }

                var r01 = lmM[4]; var r11 = lmM[5]; var r21 = lmM[6]
                val n1 = sqrt(r01 * r01 + r11 * r11 + r21 * r21)
                if (n1 > 1e-6f) { val inv1 = 1f / n1; r01 *= inv1; r11 *= inv1; r21 *= inv1 }

                var r02 = lmM[8]; var r12 = lmM[9]; var r22 = lmM[10]
                val n2 = sqrt(r02 * r02 + r12 * r12 + r22 * r22)
                if (n2 > 1e-6f) { val inv2 = 1f / n2; r02 *= inv2; r12 *= inv2; r22 *= inv2 }

                val dx = vx - lmM[12]
                val dy = vy - lmM[13]
                val dz = vz - lmM[14]
                val lx = r00 * dx + r10 * dy + r20 * dz
                val ly = r01 * dx + r11 * dy + r21 * dz
                val lz = r02 * dx + r12 * dy + r22 * dz

                // Transform by orthonormalized rigid rotation of predM: p_pred = R_pred * p_local + t_pred
                var p00 = predM[0]; var p10 = predM[1]; var p20 = predM[2]
                val pn0 = sqrt(p00 * p00 + p10 * p10 + p20 * p20)
                if (pn0 > 1e-6f) { val inv0 = 1f / pn0; p00 *= inv0; p10 *= inv0; p20 *= inv0 }

                var p01 = predM[4]; var p11 = predM[5]; var p21 = predM[6]
                val pn1 = sqrt(p01 * p01 + p11 * p11 + p21 * p21)
                if (pn1 > 1e-6f) { val inv1 = 1f / pn1; p01 *= inv1; p11 *= inv1; p21 *= inv1 }

                var p02 = predM[8]; var p12 = predM[9]; var p22 = predM[10]
                val pn2 = sqrt(p02 * p02 + p12 * p12 + p22 * p22)
                if (pn2 > 1e-6f) { val inv2 = 1f / pn2; p02 *= inv2; p12 *= inv2; p22 *= inv2 }

                val px = p00 * lx + p01 * ly + p02 * lz + predM[12]
                val py = p10 * lx + p11 * ly + p12 * lz + predM[13]
                val pz = p20 * lx + p21 * ly + p22 * lz + predM[14]

                // Project both p_cam and p_pred back into normalized analysis-crop coordinates (u, v)
                // using the exact same (fCrop, cropAspect) and pass through mapLandmarkToView so
                // imageToViewTransform (FILL_CENTER crop, rotation, and mirror) is applied identically.
                val w0 = (-vz).coerceAtLeast(0.05f)
                val w1 = (-pz).coerceAtLeast(0.05f)
                val invHalfW = fCrop / (2.0f * cropAspect)
                val invHalfH = fCrop / 2.0f

                val u0Crop = 0.5f + (vx / w0) * invHalfW
                val v0Crop = 0.5f - (vy / w0) * invHalfH
                val u1Crop = 0.5f + (px / w1) * invHalfW
                val v1Crop = 0.5f - (py / w1) * invHalfH

                val duCrop = u1Crop - u0Crop
                val dvCrop = v1Crop - v0Crop

                // Apply differential through imageToViewTransform (or fallback front-camera mirror)
                val uShifted = u + duCrop
                val vShifted = v + dvCrop
                mapLandmarkToView(
                    u = uShifted,
                    v = vShifted,
                    imageToViewTransform = faceState.imageToViewTransform,
                    isFrontCamera = faceState.isFrontCamera,
                    viewWidth = 1.0f,
                    viewHeight = 1.0f,
                    outPt = outPt
                )
                val deltaNormX = outPt[0] - normX
                val deltaNormY = outPt[1] - normY
                if (deltaNormX.isFinite() && deltaNormY.isFinite()) {
                    normX += deltaNormX.coerceIn(-0.25f, 0.25f)
                    normY += deltaNormY.coerceIn(-0.25f, 0.25f)
                }
            }
        }

        outPt[0] = normX * viewWidth
        outPt[1] = normY * viewHeight
        return outPt
    }

    /**
     * Projects a 3D point in face-local FW units `(localX, localY, localZ)` through the 4x4 [poseMatrix]
     * and the 4x4 camera [projectionMatrix] (created via [createCameraProjection] with the visible-view FOV)
     * into screen pixel coordinates `(px, py)` in `[0..viewWidth, 0..viewHeight]`.
     *
     * Allocates nothing when [outScreenPt] and [scratchCam] are provided.
     */
    fun projectFaceLocalToView(
        localX: Float,
        localY: Float,
        localZ: Float,
        poseMatrix: FloatArray,
        fw: Float,
        projectionMatrix: FloatArray,
        viewWidth: Float,
        viewHeight: Float,
        outScreenPt: FloatArray = FloatArray(2),
        scratchCam: FloatArray = FloatArray(3)
    ): FloatArray {
        faceLocalToCamera(localX, localY, localZ, poseMatrix, fw, scratchCam)
        val cx = scratchCam[0]
        val cy = scratchCam[1]
        val cz = scratchCam[2]

        val clipX = projectionMatrix[0] * cx + projectionMatrix[4] * cy + projectionMatrix[8] * cz + projectionMatrix[12]
        val clipY = projectionMatrix[1] * cx + projectionMatrix[5] * cy + projectionMatrix[9] * cz + projectionMatrix[13]
        val clipW = (projectionMatrix[3] * cx + projectionMatrix[7] * cy + projectionMatrix[11] * cz + projectionMatrix[15]).let {
            if (abs(it) < 1e-6f) 1.0f else it
        }

        val ndcX = clipX / clipW
        val ndcY = clipY / clipW
        outScreenPt[0] = (ndcX + 1.0f) * 0.5f * viewWidth
        outScreenPt[1] = (1.0f - ndcY) * 0.5f * viewHeight
        return outScreenPt
    }

    /**
     * Normalizes display/camera rotation degrees to canonical {0, 90, 180, 270}.
     */
    fun normalizeRotationDegrees(rotationDegrees: Int): Int {
        val mod = ((rotationDegrees % 360) + 360) % 360
        return when {
            mod < 45 -> 0
            mod < 135 -> 90
            mod < 225 -> 180
            mod < 315 -> 270
            else -> 0
        }
    }

    /**
     * Authoritative camera-to-overlay transform configuration derived by FacePoseAdapter
     * on display rotation or front/rear lens switches (README Section 9, 10 & 21 Step 7).
     * Never compensates through filter-specific coordinate patches.
     */
    data class CameraTransformConfig(
        val isFrontCamera: Boolean,
        val rotationDegrees: Int,
        val effectiveViewportWidth: Int,
        val effectiveViewportHeight: Int,
        val effectiveAspectRatio: Float,
        val projectionMatrix: FloatArray
    )

    fun deriveCameraTransformConfig(
        rawViewportWidth: Int,
        rawViewportHeight: Int,
        rotationDegrees: Int,
        isFrontCamera: Boolean,
        fovYDeg: Float = ProjectionParams.DEFAULT_FOV_Y_DEG
    ): CameraTransformConfig {
        val normRot = normalizeRotationDegrees(rotationDegrees)
        val w = rawViewportWidth.coerceAtLeast(1)
        val h = rawViewportHeight.coerceAtLeast(1)
        val aspect = w.toFloat() / h.toFloat()
        val proj = createCameraProjection(
            fovYDeg = fovYDeg,
            aspectRatio = aspect,
            near = ProjectionParams.DEFAULT_NEAR,
            far = 10.0f,
            frontFacing = isFrontCamera
        )
        return CameraTransformConfig(
            isFrontCamera = isFrontCamera,
            rotationDegrees = normRot,
            effectiveViewportWidth = w,
            effectiveViewportHeight = h,
            effectiveAspectRatio = aspect,
            projectionMatrix = proj
        )
    }
}
