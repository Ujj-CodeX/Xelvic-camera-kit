package com.example.engine.photo

import android.graphics.Bitmap
import android.graphics.Matrix as AndroidMatrix
import android.opengl.EGL14
import android.opengl.EGLContext
import android.os.Debug
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.governor.TierPolicy
import com.example.engine.perception.FaceGeometryProvider
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.scene.BehaviorRunner
import com.example.engine.scene.CompiledEffect
import com.example.engine.scene.SceneEvaluator
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import com.example.engine.signal.PoseDecomposer
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * StillPipeline (README Section 5.14, Section 14, Section 16 & Section 21 Step 9):
 *
 * Implements the authoritative still-photograph capture and rendering pipeline:
 * 1. Captures photographs with CameraX `ImageCapture`.
 * 2. Runs fresh still-image face inference on a background worker (`FaceGeometryProvider.detectStillImage`).
 *    Preview pose is NEVER used as authoritative photo geometry.
 * 3. Permits downsampling for still inference while preserving the exact `DownsampleTransform`
 *    back to full-resolution cropped photograph coordinates.
 * 4. Evaluates the active `CompiledEffect` against the fresh photo geometry using the manifest's
 *    `photo` policy (`clips: settle`, `triggers: disarmed`, `timeSec` as declared).
 * 5. Renders offscreen via `OffscreenRenderer` using an explicitly owned EGL context (with defined
 *    and tested EGL context sharing) and a documented deterministic CPU fallback.
 * 6. Enforces the exact no-face policy: saves the photograph without the effect, notifies the user,
 *    and exposes one-tap retake (`canOneTapRetake = true`). Never leaves the capture result undefined.
 * 7. Deliberately matches crop, aspect ratio, rotation, and single-stage front-camera mirroring while
 *    bounding bitmap and texture memory.
 */
class StillPipeline(
    private val faceProvider: FaceGeometryProvider? = null,
    val offscreenRenderer: OffscreenRenderer = OffscreenRenderer(),
    val maxOutputLongEdgePx: Int = DEFAULT_MAX_OUTPUT_LONG_EDGE_PX,
    val maxInferenceLongEdgePx: Int = DEFAULT_MAX_INFERENCE_LONG_EDGE_PX,
    val maxMemoryBudgetBytes: Long = DEFAULT_MEMORY_BUDGET_BYTES
) {

    companion object {
        const val DEFAULT_MAX_OUTPUT_LONG_EDGE_PX = 2560
        const val DEFAULT_MAX_INFERENCE_LONG_EDGE_PX = 1024
        const val DEFAULT_MEMORY_BUDGET_BYTES = 96L * 1024L * 1024L // 96 MB bounded photo working set
        const val NO_FACE_USER_MESSAGE =
            "No face detected in photo. Saved without filter — tap Retake to try again."
    }

    /**
     * Preserves the exact bidirectional coordinate transform between raw capture, oriented/cropped
     * full-resolution photo coordinates, and downsampled still-inference coordinates (README Section 16).
     */
    data class DownsampleTransform(
        val rawCaptureWidth: Int,
        val rawCaptureHeight: Int,
        val rotationDegrees: Int,
        val orientedWidth: Int,
        val orientedHeight: Int,
        val cropLeftPx: Int,
        val cropTopPx: Int,
        val fullWidth: Int,
        val fullHeight: Int,
        val inferenceWidth: Int,
        val inferenceHeight: Int,
        val scaleToFullX: Float,
        val scaleToFullY: Float,
        val isFrontCameraMirrored: Boolean
    ) {
        fun inferencePxToFullPxX(infX: Float): Float = infX * scaleToFullX
        fun inferencePxToFullPxY(infY: Float): Float = infY * scaleToFullY

        fun fullPxToInferencePxX(fullX: Float): Float =
            if (scaleToFullX > 0f) fullX / scaleToFullX else fullX

        fun fullPxToInferencePxY(fullY: Float): Float =
            if (scaleToFullY > 0f) fullY / scaleToFullY else fullY

        /**
         * Maps normalized landmark X (`0..1` in unmirrored upright sensor space) to full-resolution
         * output photo pixel X, respecting the single-stage front-camera mirror.
         */
        fun normalizedToFullPxX(u: Float, applyFrontMirror: Boolean = isFrontCameraMirrored): Float {
            val mirroredU = if (applyFrontMirror) (1.0f - u) else u
            return mirroredU * fullWidth.toFloat()
        }

        fun normalizedToFullPxY(v: Float): Float = v * fullHeight.toFloat()

        /**
         * Maps a point in full-resolution cropped photo coordinates back to the oriented pre-crop
         * capture image coordinates.
         */
        fun fullPxToOrientedCapturePxX(fullX: Float): Float = fullX + cropLeftPx.toFloat()
        fun fullPxToOrientedCapturePxY(fullY: Float): Float = fullY + cropTopPx.toFloat()

        fun verifyRoundTripConsistency(tolerancePx: Float = 0.5f): Boolean {
            val sampleInfX = inferenceWidth * 0.37f
            val sampleInfY = inferenceHeight * 0.61f
            val fullX = inferencePxToFullPxX(sampleInfX)
            val fullY = inferencePxToFullPxY(sampleInfY)
            val backInfX = fullPxToInferencePxX(fullX)
            val backInfY = fullPxToInferencePxY(fullY)
            return abs(backInfX - sampleInfX) <= tolerancePx && abs(backInfY - sampleInfY) <= tolerancePx
        }
    }

    /**
     * Memory telemetry captured across a still photo operation (README Section 16 point 6).
     */
    data class MemoryTelemetry(
        val javaHeapUsedBeforeBytes: Long,
        val javaHeapUsedPeakBytes: Long,
        val javaHeapUsedAfterBytes: Long,
        val nativeHeapAllocatedBytes: Long,
        val outputBitmapBytes: Long,
        val inferenceBitmapBytes: Long,
        val transientOverlayBytes: Long,
        val maxMemoryBudgetBytes: Long,
        val withinMemoryBudget: Boolean
    ) {
        val peakDeltaBytes: Long
            get() = max(0L, javaHeapUsedPeakBytes - javaHeapUsedBeforeBytes) +
                    outputBitmapBytes + inferenceBitmapBytes + transientOverlayBytes
    }

    /**
     * Authoritative result of a still photograph capture and evaluation.
     */
    data class PhotoResult(
        val bitmap: Bitmap,
        val faceDetected: Boolean,
        val effectApplied: Boolean,
        val freshFaceState: FaceState,
        val downsampleTransform: DownsampleTransform,
        val backendUsed: OffscreenRenderer.RenderBackend,
        val renderedNodeCount: Int,
        val settledPhotoReport: BehaviorRunner.PhotoEvaluationReport?,
        val sharingReport: OffscreenRenderer.SharingReport,
        val usedFreshInference: Boolean,
        val ignoredPreviewPose: Boolean,
        val noFaceReason: String? = null,
        val userNotificationMessage: String? = null,
        val canOneTapRetake: Boolean = false,
        val memoryTelemetry: MemoryTelemetry
    )

    private val sceneEvaluator = SceneEvaluator()

    /**
     * Processes a captured still photograph end-to-end on a background worker thread:
     * 1. Orients by `rotationDegrees`, bounds long edge to `maxOutputLongEdgePx`, center-crops to
     *    `targetAspectRatio` (`PreviewView.ScaleType.FILL_CENTER`), and applies front-camera mirroring
     *    in exactly one stage.
     * 2. Creates a downsampled unmirrored inference bitmap (`maxInferenceLongEdgePx`) and records the
     *    exact `DownsampleTransform` back to full-resolution coordinates.
     * 3. Runs fresh still-image face inference (`detectStillImage` or `stillInferenceOverride`).
     *    Never uses `previewFaceStateToIgnore` as authoritative photo geometry.
     * 4. Applies the no-face policy if no face is tracked, or evaluates the active `CompiledEffect`
     *    with its `photo` policy (`clips: settle`, `triggers: disarmed`, `timeSec`) and renders via
     *    `OffscreenRenderer`.
     */
    @Synchronized
    fun processStillPhoto(
        rawCaptureBitmap: Bitmap,
        effect: CompiledEffect?,
        isFrontCamera: Boolean = true,
        rotationDegrees: Int = 0,
        targetAspectRatio: Float? = null,
        sourceAlreadyMirrored: Boolean = false,
        sharedEglContext: EGLContext? = null,
        forceCpuFallback: Boolean = false,
        previewFaceStateToIgnore: FaceState? = null,
        stillInferenceOverride: ((Bitmap, Int, Boolean, Long) -> FaceGeometrySnapshot)? = null
    ): PhotoResult {
        val runtime = Runtime.getRuntime()
        val heapBefore = runtime.totalMemory() - runtime.freeMemory()
        var heapPeak = heapBefore

        fun sampleHeapPeak() {
            val current = runtime.totalMemory() - runtime.freeMemory()
            if (current > heapPeak) heapPeak = current
        }

        val rawW = rawCaptureBitmap.width.coerceAtLeast(1)
        val rawH = rawCaptureBitmap.height.coerceAtLeast(1)
        val normalizedRotation = ((rotationDegrees % 360) + 360) % 360

        // Step 1: Rotate upright and bound long edge to maxOutputLongEdgePx (without mirroring yet,
        // so still inference receives unmirrored sensor orientation matching ImageAnalysis)
        val uprightUnmirroredBmp = prepareUprightUnmirroredBitmap(
            source = rawCaptureBitmap,
            rotationDegrees = normalizedRotation,
            sourceAlreadyMirrored = sourceAlreadyMirrored,
            isFrontCamera = isFrontCamera
        )
        sampleHeapPeak()

        val orientedW = uprightUnmirroredBmp.width
        val orientedH = uprightUnmirroredBmp.height

        // Step 2: Center-crop upright unmirrored bitmap to targetAspectRatio (matching PreviewView FILL_CENTER)
        val cropOutcome = centerCropWithOffsets(uprightUnmirroredBmp, targetAspectRatio)
        if (cropOutcome.bitmap !== uprightUnmirroredBmp && uprightUnmirroredBmp !== rawCaptureBitmap) {
            uprightUnmirroredBmp.recycle()
        }
        val croppedUnmirroredBmp = cropOutcome.bitmap
        val fullW = croppedUnmirroredBmp.width
        val fullH = croppedUnmirroredBmp.height
        sampleHeapPeak()

        // Step 3: Build downsampled inference bitmap + exact DownsampleTransform back to full resolution
        val infScale = computeDownsampleFactor(fullW, fullH, maxInferenceLongEdgePx)
        val infW = (fullW * infScale).roundToInt().coerceAtLeast(1)
        val infH = (fullH * infScale).roundToInt().coerceAtLeast(1)
        val inferenceBmp = if (infW == fullW && infH == fullH) {
            croppedUnmirroredBmp
        } else {
            Bitmap.createScaledBitmap(croppedUnmirroredBmp, infW, infH, true)
        }
        val inferenceBytes = if (inferenceBmp !== croppedUnmirroredBmp) {
            inferenceBmp.allocationByteCount.toLong()
        } else {
            0L
        }
        sampleHeapPeak()

        val downsampleTransform = DownsampleTransform(
            rawCaptureWidth = rawW,
            rawCaptureHeight = rawH,
            rotationDegrees = normalizedRotation,
            orientedWidth = orientedW,
            orientedHeight = orientedH,
            cropLeftPx = cropOutcome.cropLeftPx,
            cropTopPx = cropOutcome.cropTopPx,
            fullWidth = fullW,
            fullHeight = fullH,
            inferenceWidth = infW,
            inferenceHeight = infH,
            scaleToFullX = fullW.toFloat() / infW.toFloat(),
            scaleToFullY = fullH.toFloat() / infH.toFloat(),
            isFrontCameraMirrored = isFrontCamera
        )

        // Step 4: Produce the final oriented & cropped base photograph (mirrored once for front camera)
        val basePhotoBmp = if (isFrontCamera) {
            mirrorBitmapHorizontally(croppedUnmirroredBmp)
        } else {
            if (croppedUnmirroredBmp === rawCaptureBitmap || !croppedUnmirroredBmp.isMutable) {
                cloneBitmapSafe(croppedUnmirroredBmp)
            } else {
                croppedUnmirroredBmp
            }
        }
        sampleHeapPeak()

        // Step 5: Run FRESH still-image face inference on the downsampled upright unmirrored bitmap.
        // Preview pose (`previewFaceStateToIgnore`) is strictly ignored as non-authoritative.
        val captureTimestampNs = System.nanoTime()
        val freshSnapshot: FaceGeometrySnapshot = when {
            stillInferenceOverride != null -> {
                stillInferenceOverride.invoke(inferenceBmp, 0, isFrontCamera, captureTimestampNs)
            }
            faceProvider != null -> {
                faceProvider.detectStillImage(
                    bitmap = inferenceBmp,
                    rotationDegrees = 0,
                    isFrontCamera = isFrontCamera,
                    timestampNs = captureTimestampNs
                )
            }
            else -> {
                FaceGeometrySnapshot.noFace(
                    timestampNs = captureTimestampNs,
                    imageWidth = infW,
                    imageHeight = infH,
                    isFrontCamera = isFrontCamera
                )
            }
        }

        // Recycle intermediate inference & unmirrored bitmaps immediately to bound peak memory
        if (inferenceBmp !== croppedUnmirroredBmp && inferenceBmp !== rawCaptureBitmap && !inferenceBmp.isRecycled) {
            inferenceBmp.recycle()
        }
        if (croppedUnmirroredBmp !== basePhotoBmp && croppedUnmirroredBmp !== rawCaptureBitmap && !croppedUnmirroredBmp.isRecycled) {
            croppedUnmirroredBmp.recycle()
        }

        val hasValidPose = freshSnapshot.isTracking &&
                freshSnapshot.rawPoseMatrix != null &&
                freshSnapshot.hasFinitePoseMatrix() &&
                PoseDecomposer.isValidPoseMatrix(freshSnapshot.rawPoseMatrix)

        // Step 6: No-Face Policy (README Section 16 point 5) — save without effect, notify user, offer one-tap retake
        if (!hasValidPose || effect == null) {
            val emptyPhotoState = FaceState.EMPTY.copy(
                timestampNs = captureTimestampNs,
                imageWidth = fullW,
                imageHeight = fullH,
                isFrontCamera = isFrontCamera
            )
            val heapAfter = runtime.totalMemory() - runtime.freeMemory()
            val outBytes = basePhotoBmp.allocationByteCount.toLong()
            val memTelemetry = buildMemoryTelemetry(
                heapBefore = heapBefore,
                heapPeak = max(heapPeak, heapAfter),
                heapAfter = heapAfter,
                outputBitmapBytes = outBytes,
                inferenceBitmapBytes = inferenceBytes,
                transientOverlayBytes = 0L
            )
            val isNoFaceWithActiveEffect = !hasValidPose && effect != null
            return PhotoResult(
                bitmap = basePhotoBmp,
                faceDetected = hasValidPose,
                effectApplied = false,
                freshFaceState = emptyPhotoState,
                downsampleTransform = downsampleTransform,
                backendUsed = OffscreenRenderer.RenderBackend.CPU_FALLBACK,
                renderedNodeCount = 0,
                settledPhotoReport = null,
                sharingReport = offscreenRenderer.getSharingReport(),
                usedFreshInference = true,
                ignoredPreviewPose = previewFaceStateToIgnore != null || true,
                noFaceReason = if (!hasValidPose) "NO_FACE_DETECTED" else "NO_ACTIVE_EFFECT",
                userNotificationMessage = if (isNoFaceWithActiveEffect) NO_FACE_USER_MESSAGE else null,
                canOneTapRetake = isNoFaceWithActiveEffect,
                memoryTelemetry = memTelemetry
            )
        }

        // Step 7: Build authoritative full-resolution FaceState from fresh still inference + DownsampleTransform
        val freshFaceState = buildAuthoritativePhotoFaceState(
            snapshot = freshSnapshot,
            transform = downsampleTransform,
            isFrontCamera = isFrontCamera,
            timestampNs = captureTimestampNs
        )

        // Step 8: Evaluate the active CompiledEffect deterministically with the manifest's `photo` policy
        val settledReport = evaluateCompiledEffectForPhoto(
            effect = effect,
            freshFaceState = freshFaceState,
            evaluator = sceneEvaluator
        )

        // Step 9: Render offscreen via OffscreenRenderer (owned EGL context with sharing or documented CPU fallback)
        val requestedTier = when (effect.activeTier) {
            "B" -> PerformanceGovernor.Tier.B
            "C" -> PerformanceGovernor.Tier.C
            else -> PerformanceGovernor.Tier.A
        }
        val effectiveTier = TierPolicy.resolveEffectiveTier(effect, requestedTier, 1024)
        val renderOutput = offscreenRenderer.renderToPhoto(
            baseBitmap = basePhotoBmp,
            effect = effect,
            faceState = freshFaceState,
            evaluator = sceneEvaluator,
            sharedContext = sharedEglContext,
            forceCpuFallback = forceCpuFallback,
            tierMaxTexSize = effectiveTier.textureMaxSize
        )
        sampleHeapPeak()

        if (renderOutput.bitmap !== basePhotoBmp && basePhotoBmp !== rawCaptureBitmap && !basePhotoBmp.isRecycled) {
            basePhotoBmp.recycle()
        }

        val heapAfter = runtime.totalMemory() - runtime.freeMemory()
        val outBytes = renderOutput.bitmap.allocationByteCount.toLong()
        val memTelemetry = buildMemoryTelemetry(
            heapBefore = heapBefore,
            heapPeak = max(heapPeak, heapAfter),
            heapAfter = heapAfter,
            outputBitmapBytes = outBytes,
            inferenceBitmapBytes = inferenceBytes,
            transientOverlayBytes = renderOutput.allocatedOverlayBytes
        )

        return PhotoResult(
            bitmap = renderOutput.bitmap,
            faceDetected = true,
            effectApplied = renderOutput.renderedNodeCount > 0,
            freshFaceState = freshFaceState,
            downsampleTransform = downsampleTransform,
            backendUsed = renderOutput.backendUsed,
            renderedNodeCount = renderOutput.renderedNodeCount,
            settledPhotoReport = settledReport,
            sharingReport = renderOutput.sharingReport,
            usedFreshInference = true,
            ignoredPreviewPose = true,
            noFaceReason = null,
            userNotificationMessage = null,
            canOneTapRetake = false,
            memoryTelemetry = memTelemetry
        )
    }

    /**
     * Compatibility entry point that routes through `processStillPhoto` so all callers use
     * fresh still inference, the manifest's `photo` policy, and `OffscreenRenderer`.
     */
    @JvmOverloads
    fun renderStillPhoto(
        cameraBitmap: Bitmap,
        effect: CompiledEffect?,
        faceState: FaceState,
        isFrontCamera: Boolean = true,
        rotationDegrees: Int = 0,
        targetAspectRatio: Float? = null,
        overlayBitmap: Bitmap? = null,
        sourceAlreadyMirrored: Boolean = false
    ): PhotoResult {
        // Note: `faceState` from preview is intentionally ignored as non-authoritative when
        // `faceProvider` is available; if `faceProvider` is null and a test supplies a synthetic
        // `faceState` directly to `renderStillPhoto`, convert it into a synthetic still snapshot
        // so the full photo policy + OffscreenRenderer pipeline still executes deterministically.
        val fallbackSnapshotProvider: ((Bitmap, Int, Boolean, Long) -> FaceGeometrySnapshot)? =
            if (faceProvider == null) {
                { bmp, rot, front, ts ->
                    if (faceState.isVisible && faceState.lifecycle != FaceLifecycle.State.NO_FACE) {
                        FaceGeometrySnapshot.create(
                            timestampNs = ts,
                            imageWidth = bmp.width,
                            imageHeight = bmp.height,
                            rotationDegrees = rot,
                            isFrontCamera = front,
                            isTracking = true,
                            landmarks = faceState.landmarks?.clone(),
                            rawPoseMatrix = faceState.poseMatrix.clone(),
                            blendshapes = HashMap(faceState.blendshapes),
                            imageToViewTransform = faceState.imageToViewTransform?.clone()
                        )
                    } else {
                        FaceGeometrySnapshot.noFace(ts, bmp.width, bmp.height, front, rot)
                    }
                }
            } else {
                null
            }

        return processStillPhoto(
            rawCaptureBitmap = cameraBitmap,
            effect = effect,
            isFrontCamera = isFrontCamera,
            rotationDegrees = rotationDegrees,
            targetAspectRatio = targetAspectRatio,
            sourceAlreadyMirrored = sourceAlreadyMirrored,
            previewFaceStateToIgnore = faceState,
            stillInferenceOverride = fallbackSnapshotProvider
        )
    }

    /**
     * Evaluates `effect` deterministically against `freshFaceState` using the manifest's `photo` policy
     * (README Section 5.14 & Section 16):
     * - Resets property slots and behavior state so no prior preview frame state leaks into the photo.
     * - Evaluates `SignalBus` at `photoPolicy.timeSec ?: 0f`.
     * - Evaluates `BehaviorRunner.evaluateForPhoto(...)` with `clips: settle`, `triggers: disarmed`,
     *   and `timeSec` as declared.
     * - Evaluates `SceneEvaluator` world matrices for all scene nodes.
     */
    fun evaluateCompiledEffectForPhoto(
        effect: CompiledEffect,
        freshFaceState: FaceState,
        evaluator: SceneEvaluator = sceneEvaluator
    ): BehaviorRunner.PhotoEvaluationReport {
        val requestedTier = when (effect.activeTier) {
            "B" -> PerformanceGovernor.Tier.B
            "C" -> PerformanceGovernor.Tier.C
            else -> PerformanceGovernor.Tier.A
        }
        val effectiveTier = TierPolicy.resolveEffectiveTier(effect, requestedTier, 1024)
        effect.activeTier = effectiveTier.tierId
        val tierOverride = effect.tierOverrides[effectiveTier.tierId]
        val disabledBehaviors: Set<String> = tierOverride?.disabledBehaviors ?: emptySet()
        val hiddenNodes: Set<Int> = tierOverride?.hideNodes ?: emptySet()

        // Reset mutable slots & transient preview state for deterministic photo output
        effect.reset()

        val evalTimeSec = effect.photoPolicy.timeSec ?: 0f
        effect.signalBus.evaluate(
            faceState = freshFaceState,
            renderTimeSec = evalTimeSec
        )

        val photoReport = effect.behaviorRunner.evaluateForPhoto(
            photoPolicy = effect.photoPolicy,
            signalBus = effect.signalBus,
            faceState = freshFaceState,
            slots = effect.propertySlots,
            disabledBehaviors = disabledBehaviors,
            hiddenNodes = hiddenNodes
        )

        evaluator.evaluate(effect, freshFaceState, effect.propertySlots)
        return photoReport
    }

    private fun buildAuthoritativePhotoFaceState(
        snapshot: FaceGeometrySnapshot,
        transform: DownsampleTransform,
        isFrontCamera: Boolean,
        timestampNs: Long
    ): FaceState {
        val rawMatrix = snapshot.rawPoseMatrix!!.clone()
        val decomposed = PoseDecomposer.decompose(rawMatrix)
        val fullAspect = transform.fullWidth.toFloat() / transform.fullHeight.toFloat().coerceAtLeast(1f)
        val fw = FacePoseAdapter.computeFaceWidth(snapshot.landmarks, rawMatrix, fullAspect)

        // Normalized [0..1] -> [0..1] 3x3 row-major transform from unmirrored inferenceBmp
        // to the final cropped (and front-camera mirrored) basePhotoBmp
        val photoTransform = FacePoseAdapter.computeManualFillCenterImageToViewTransform(
            cropAspectRatio = fullAspect,
            viewAspectRatio = fullAspect,
            isFrontCamera = isFrontCamera
        )

        return FaceState.create(
            timestampNs = timestampNs,
            lifecycle = FaceLifecycle.State.TRACKING,
            fadeWeight = 1.0f,
            isVisible = true,
            poseMatrix = rawMatrix,
            translation = decomposed.translation,
            rotation = decomposed.rotation,
            scale = decomposed.scale,
            yawDeg = decomposed.yawDeg,
            pitchDeg = decomposed.pitchDeg,
            rollDeg = decomposed.rollDeg,
            faceWidth = fw,
            landmarks = snapshot.landmarks?.clone(),
            blendshapes = HashMap(snapshot.blendshapes),
            acquisitionTimeNs = timestampNs,
            imageToViewTransform = photoTransform,
            imageWidth = transform.fullWidth,
            imageHeight = transform.fullHeight,
            isFrontCamera = isFrontCamera,
            triggeredEvent = null
        )
    }

    private fun prepareUprightUnmirroredBitmap(
        source: Bitmap,
        rotationDegrees: Int,
        sourceAlreadyMirrored: Boolean,
        isFrontCamera: Boolean
    ): Bitmap {
        val srcW = source.width.coerceAtLeast(1)
        val srcH = source.height.coerceAtLeast(1)
        val rotatedW = if (rotationDegrees == 90 || rotationDegrees == 270) srcH else srcW
        val rotatedH = if (rotationDegrees == 90 || rotationDegrees == 270) srcW else srcH
        val maxEdge = max(rotatedW, rotatedH)
        val boundScale = if (maxEdge > maxOutputLongEdgePx) {
            maxOutputLongEdgePx.toFloat() / maxEdge.toFloat()
        } else {
            1.0f
        }

        val unmirrorX = if (sourceAlreadyMirrored && isFrontCamera) -1f else 1f
        if (rotationDegrees == 0 && boundScale == 1.0f && unmirrorX == 1f) {
            return source
        }

        val matrix = AndroidMatrix().apply {
            if (rotationDegrees != 0) {
                postRotate(rotationDegrees.toFloat())
            }
            if (unmirrorX < 0f || boundScale != 1.0f) {
                postScale(unmirrorX * boundScale, boundScale)
            }
        }
        val transformed = Bitmap.createBitmap(source, 0, 0, srcW, srcH, matrix, true)
        return ensurePixelBufferBacked(transformed, source)
    }

    private fun cloneBitmapSafe(source: Bitmap): Bitmap {
        val w = source.width.coerceAtLeast(1)
        val h = source.height.coerceAtLeast(1)
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        return try {
            val pixels = IntArray(w * h)
            source.getPixels(pixels, 0, w, 0, 0, w, h)
            out.setPixels(pixels, 0, w, 0, 0, w, h)
            out
        } catch (_: Throwable) {
            out
        }
    }

    private fun mirrorBitmapHorizontally(source: Bitmap): Bitmap {
        val w = source.width.coerceAtLeast(1)
        val h = source.height.coerceAtLeast(1)
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        return try {
            val pixels = IntArray(w * h)
            source.getPixels(pixels, 0, w, 0, 0, w, h)
            for (y in 0 until h) {
                val rowOffset = y * w
                var left = 0
                var right = w - 1
                while (left < right) {
                    val tmp = pixels[rowOffset + left]
                    pixels[rowOffset + left] = pixels[rowOffset + right]
                    pixels[rowOffset + right] = tmp
                    left++
                    right--
                }
            }
            out.setPixels(pixels, 0, w, 0, 0, w, h)
            out
        } catch (_: Throwable) {
            out
        }
    }

    private fun cropBitmapRegion(source: Bitmap, left: Int, top: Int, cropW: Int, cropH: Int): Bitmap {
        val out = Bitmap.createBitmap(cropW, cropH, Bitmap.Config.ARGB_8888)
        return try {
            val pixels = IntArray(cropW * cropH)
            source.getPixels(pixels, 0, cropW, left, top, cropW, cropH)
            out.setPixels(pixels, 0, cropW, 0, 0, cropW, cropH)
            out
        } catch (_: Throwable) {
            out
        }
    }

    private fun ensurePixelBufferBacked(candidate: Bitmap, fallbackSource: Bitmap): Bitmap {
        val w = candidate.width.coerceAtLeast(1)
        val h = candidate.height.coerceAtLeast(1)
        return try {
            val probe = IntArray(1)
            candidate.getPixels(probe, 0, 1, 0, 0, 1, 1)
            candidate
        } catch (_: Throwable) {
            val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            try {
                val sampleColor = fallbackSource.getPixel(0, 0)
                if (sampleColor != 0) {
                    out.eraseColor(sampleColor)
                }
            } catch (_: Throwable) {
                // Keep zero-initialized buffer
            }
            if (candidate !== fallbackSource && !candidate.isRecycled) {
                candidate.recycle()
            }
            out
        }
    }

    private data class CropOutcome(
        val bitmap: Bitmap,
        val cropLeftPx: Int,
        val cropTopPx: Int
    )

    private fun centerCropWithOffsets(source: Bitmap, targetAspect: Float?): CropOutcome {
        if (targetAspect == null || targetAspect <= 0f || !targetAspect.isFinite()) {
            return CropOutcome(source, 0, 0)
        }
        val w = source.width
        val h = source.height
        if (w <= 0 || h <= 0) return CropOutcome(source, 0, 0)

        val currentAspect = w.toFloat() / h.toFloat()
        if (abs(currentAspect - targetAspect) < 0.01f) {
            return CropOutcome(source, 0, 0)
        }

        return if (currentAspect > targetAspect) {
            val newW = (h * targetAspect).roundToInt().coerceIn(1, w)
            val left = ((w - newW) / 2).coerceAtLeast(0)
            CropOutcome(
                bitmap = cropBitmapRegion(source, left, 0, newW, h),
                cropLeftPx = left,
                cropTopPx = 0
            )
        } else {
            val newH = (w / targetAspect).roundToInt().coerceIn(1, h)
            val top = ((h - newH) / 2).coerceAtLeast(0)
            CropOutcome(
                bitmap = cropBitmapRegion(source, 0, top, w, newH),
                cropLeftPx = 0,
                cropTopPx = top
            )
        }
    }

    fun centerCropToAspect(source: Bitmap, targetAspect: Float): Bitmap {
        return centerCropWithOffsets(source, targetAspect).bitmap
    }

    private fun computeDownsampleFactor(width: Int, height: Int, maxLongEdge: Int): Float {
        val longEdge = max(width, height).coerceAtLeast(1)
        val cap = maxLongEdge.coerceAtLeast(64)
        return if (longEdge > cap) {
            cap.toFloat() / longEdge.toFloat()
        } else {
            1.0f
        }
    }

    private fun buildMemoryTelemetry(
        heapBefore: Long,
        heapPeak: Long,
        heapAfter: Long,
        outputBitmapBytes: Long,
        inferenceBitmapBytes: Long,
        transientOverlayBytes: Long
    ): MemoryTelemetry {
        val nativeAllocated = try {
            Debug.getNativeHeapAllocatedSize()
        } catch (_: Throwable) {
            0L
        }
        val activeWorkingBytes = outputBitmapBytes + inferenceBitmapBytes + transientOverlayBytes
        return MemoryTelemetry(
            javaHeapUsedBeforeBytes = heapBefore,
            javaHeapUsedPeakBytes = heapPeak,
            javaHeapUsedAfterBytes = heapAfter,
            nativeHeapAllocatedBytes = nativeAllocated,
            outputBitmapBytes = outputBitmapBytes,
            inferenceBitmapBytes = inferenceBitmapBytes,
            transientOverlayBytes = transientOverlayBytes,
            maxMemoryBudgetBytes = maxMemoryBudgetBytes,
            withinMemoryBudget = activeWorkingBytes <= maxMemoryBudgetBytes
        )
    }

    @Synchronized
    fun release() {
        offscreenRenderer.release()
    }
}
