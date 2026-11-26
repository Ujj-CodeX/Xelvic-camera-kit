package com.example.engine.signal

import com.example.engine.capture.FrameClock
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.perception.MediaPipeFaceGeometryProvider
import java.util.concurrent.atomic.AtomicReference

/**
 * SignalPipeline connects perception to the render thread (README Section 9 & 14).
 *
 * Responsibilities:
 * 1. Decomposes 4x4 matrix into translation, quaternion rotation, and scale.
 * 2. Filters pose quantities separately via independent One-Euro filters (PoseFilter).
 * 3. Filters requested blendshapes independently.
 * 4. Extrapolates filtered pose to render time over a bounded horizon via LatencyCompensator.
 * 5. Updates FaceLifecycle state machine (NO_FACE -> ACQUIRING -> TRACKING -> HOLD -> FADING -> NO_FACE).
 * 6. Publishes immutable FaceState (including `poseTimestampNs`) via AtomicReference without accumulating queues.
 * 7. Provides zero-allocation `consumeLatestPredictedState` for the GL vsync loop.
 */
class SignalPipeline(
    val poseFilter: PoseFilter = PoseFilter(),
    val latencyCompensator: LatencyCompensator = LatencyCompensator(maxHorizonSec = LatencyCompensator.DEFAULT_HORIZON_SEC),
    val lifecycle: FaceLifecycle = FaceLifecycle()
) {
    // Independent One-Euro filters per blendshape channel name
    private val blendshapeFilters = HashMap<String, OneEuroFilter>()

    // Dedicated One-Euro filter and in-frame angular ratio memory for faceWidth (fw / |tz|)
    // so moving the face to the screen border or into profile never causes sudden scale jumps
    private val faceWidthFilter = OneEuroFilter(minCutoff = 1.8f, beta = 4.0f, dCutoff = 2.0f)
    private var lastStableAngularFwRatio: Float = 0f

    // Atomic latest-value handoff to render consumers
    private val latestStateRef = AtomicReference(FaceState.EMPTY)
    private val pendingEventRef = AtomicReference<String?>(null)

    // Runtime toggle for live A/B comparison of pose prediction vs raw filtered pose
    @Volatile
    var isPredictionEnabled: Boolean = true

    // Preallocated buffers used on the signal thread
    private val velScratch = FloatArray(3)
    private val angVelScratch = FloatArray(3)

    // Preallocated mutable output state and scratch buffer confined to the GL thread
    // for zero-allocation execution in `consumeLatestPredictedState`
    private val reusablePredictedState = FaceState()
    private val eulerScratch = FloatArray(3)

    fun getLatestState(): FaceState = latestStateRef.get()

    /**
     * Reads latest FaceState and consumes any pending one-shot lifecycle event (e.g. faceAcquired, faceLost).
     */
    fun consumeLatestState(): FaceState {
        val state = latestStateRef.get()
        val pendingEvt = pendingEventRef.getAndSet(null)
        return if (pendingEvt != null && state.triggeredEvent != pendingEvt) {
            state.copy(triggeredEvent = pendingEvt)
        } else {
            state
        }
    }

    /**
     * Zero-allocation vsync consumer for `XelvicGlSurfaceView.onDrawFrame`:
     * - Consumes any latched one-shot lifecycle event (`faceAcquired` / `faceLost`) from [pendingEventRef]
     *   so events are never dropped across 60/120 Hz vsync ticks.
     * - Reuses a preallocated [reusablePredictedState] and its internal `FloatArray` buffers,
     *   referencing unchanged fields (`landmarks`, `blendshapes`, `imageToViewTransform`) from the
     *   latest state without copying.
     * - Extrapolates rigid pose (`x`/`y` at full gain, `z` at `zTranslationGain`, `rotation` quaternion)
     *   ONLY when [predictionEnabled] is `true` and `lifecycle == FaceLifecycle.State.TRACKING`.
     * - Never extrapolates `scale` or `blendshapes`, and preserves `poseTimestampNs` as the underlying
     *   camera capture timestamp.
     */
    @JvmOverloads
    fun consumeLatestPredictedState(
        renderElapsedNs: Long = FrameClock.elapsedRealtimeNs(),
        predictionEnabled: Boolean = isPredictionEnabled
    ): FaceState {
        val latest = latestStateRef.get()
        val pendingEvt = pendingEventRef.getAndSet(null)

        val out = reusablePredictedState
        out.copyFromForPrediction(latest, pendingEvt)

        if (!predictionEnabled || latest.lifecycle != FaceLifecycle.State.TRACKING || latest.poseTimestampNs <= 0L) {
            return out
        }

        // Predict translation and rotation in-place into out.translation and out.rotation
        latencyCompensator.predict(
            renderTimestampNs = renderElapsedNs,
            outT = out.translation,
            outQ = out.rotation,
            poseTimestampNs = latest.poseTimestampNs
        )

        // Recompose 4x4 poseMatrix in-place using predicted T & Q and unpredicted filtered scale
        PoseDecomposer.recompose(
            t = out.translation,
            q = out.rotation,
            s = out.scale,
            out = out.poseMatrix
        )

        // Keep perspective screen scale (fw / |tz|) invariant when Z translation is extrapolated
        val baseAbsZ = kotlin.math.abs(latest.poseMatrix[14])
        val predAbsZ = kotlin.math.abs(out.poseMatrix[14])
        if (baseAbsZ >= 0.05f && predAbsZ >= 0.05f) {
            val zScaleRatio = (predAbsZ / baseAbsZ).coerceIn(0.85f, 1.15f)
            out.faceWidth = latest.faceWidth * zScaleRatio
        }

        // Update Euler angles in-place without allocating
        PoseDecomposer.quaternionToEulerDeg(out.rotation, eulerScratch)
        out.yawDeg = eulerScratch[0]
        out.pitchDeg = eulerScratch[1]
        out.rollDeg = eulerScratch[2]
        out.timestampNs = renderElapsedNs
        // Note: out.poseTimestampNs remains latest.poseTimestampNs (capture timestamp)
        return out
    }

    fun reset() {
        poseFilter.reset()
        blendshapeFilters.clear()
        faceWidthFilter.reset()
        lastStableAngularFwRatio = 0f
        latencyCompensator.reset()
        lifecycle.reset()
        pendingEventRef.set(null)
        latestStateRef.set(FaceState.EMPTY)
        reusablePredictedState.copyFromForPrediction(FaceState.EMPTY, null)
    }

    /**
     * Process a new snapshot from perception provider. Runs on provider/signal thread.
     */
    @JvmOverloads
    fun processSnapshot(
        snapshot: FaceGeometrySnapshot,
        nowNs: Long = FrameClock.elapsedRealtimeNs()
    ): FaceState {
        val prevLifecycle = lifecycle.currentState

        // Validate tracking and finite pose data (falling back to landmark-calibrated pose if rawPoseMatrix is omitted)
        val candidatePoseMatrix: FloatArray? = when {
            snapshot.rawPoseMatrix != null -> snapshot.rawPoseMatrix
            snapshot.isTracking && snapshot.landmarks != null && snapshot.landmarks.size >= 468 * 3 && snapshot.hasFiniteLandmarks() -> {
                MediaPipeFaceGeometryProvider.calibratePoseMatrix(
                    rawMpMatrix = null,
                    lm = snapshot.landmarks,
                    imageWidth = snapshot.imageWidth.coerceAtLeast(1),
                    imageHeight = snapshot.imageHeight.coerceAtLeast(1),
                    fovYCropDeg = snapshot.projectionParams.fovYDeg
                )
            }
            else -> null
        }
        val hasValidPose = snapshot.isTracking &&
                candidatePoseMatrix != null &&
                (if (candidatePoseMatrix === snapshot.rawPoseMatrix) snapshot.hasFinitePoseMatrix() else candidatePoseMatrix.all { it.isFinite() }) &&
                PoseDecomposer.isValidPoseMatrix(candidatePoseMatrix)

        // 1. Update lifecycle
        val lifecycleTimeNs = if (snapshot.timestampNs > 0L && nowNs < snapshot.timestampNs) {
            snapshot.timestampNs
        } else {
            nowNs
        }
        val triggeredEvent = lifecycle.update(hasValidPose, lifecycleTimeNs)
        if (triggeredEvent != null) {
            pendingEventRef.set(triggeredEvent)
        }
        val newLifecycle = lifecycle.currentState

        if (!hasValidPose) {
            latencyCompensator.reset()
            val prevState = latestStateRef.get()
            val stateReadyNs = FrameClock.elapsedRealtimeNs()
            val resolvedProjParams = if (snapshot.projectionParams !== ProjectionParams.DEFAULT) {
                snapshot.projectionParams
            } else {
                prevState.projectionParams
            }
            val state = FaceState.create(
                timestampNs = nowNs,
                lifecycle = newLifecycle,
                fadeWeight = lifecycle.fadeWeight,
                isVisible = lifecycle.isVisible,
                poseMatrix = prevState.poseMatrix,
                translation = prevState.translation,
                rotation = prevState.rotation,
                scale = prevState.scale,
                yawDeg = prevState.yawDeg,
                pitchDeg = prevState.pitchDeg,
                rollDeg = prevState.rollDeg,
                faceWidth = prevState.faceWidth,
                landmarks = null,
                blendshapes = prevState.blendshapes,
                acquisitionTimeNs = lifecycle.acquisitionTimestampNs,
                imageToViewTransform = snapshot.imageToViewTransform,
                imageWidth = snapshot.imageWidth,
                imageHeight = snapshot.imageHeight,
                isFrontCamera = snapshot.isFrontCamera,
                triggeredEvent = triggeredEvent,
                poseTimestampNs = prevState.poseTimestampNs,
                stateReadyNs = stateReadyNs,
                queueMs = snapshot.queueMs,
                conversionMs = snapshot.conversionMs,
                inferenceMs = snapshot.inferenceMs,
                projectionParams = resolvedProjParams,
                landmarkPoseMatrix = prevState.landmarkPoseMatrix,
                pendingWaitMs = snapshot.pendingWaitMs
            )
            latestStateRef.set(state)
            return state
        }

        val rawMatrix = candidatePoseMatrix!!
        val poseTimestampNs = snapshot.timestampNs

        // 2. Decompose pose matrix into translation, quaternion, and scale
        val decomposed = PoseDecomposer.decompose(rawMatrix)

        // 3. Filter translation, rotation (with sign continuity), and scale independently
        val filteredPose = poseFilter.filter(decomposed, poseTimestampNs)

        // 4. Reset velocity on ACQUIRING -> TRACKING transition (or whenever not already TRACKING)
        val shouldResetVelocity = (prevLifecycle != FaceLifecycle.State.TRACKING) ||
                (newLifecycle != FaceLifecycle.State.TRACKING)

        // 5. Update latency compensator with FILTERED pose and authoritative capture timestamp
        latencyCompensator.update(
            translation = filteredPose.translation,
            rotation = filteredPose.rotation,
            timestampNs = poseTimestampNs,
            scale = filteredPose.scale,
            resetVelocity = shouldResetVelocity
        )
        latencyCompensator.getLinearVelocity(velScratch)
        latencyCompensator.getAngularVelocity(angVelScratch)
        val predErrPosMm = if (latencyCompensator.hasPredictionErrorSample) {
            latencyCompensator.lastPredictionErrorPosMm
        } else 0f
        val predErrRotDeg = if (latencyCompensator.hasPredictionErrorSample) {
            latencyCompensator.lastPredictionErrorRotDeg
        } else 0f

        // 6. Recompose filtered matrix
        val filteredMatrix = PoseDecomposer.recompose(
            filteredPose.translation,
            filteredPose.rotation,
            filteredPose.scale
        )

        // 7. Compute Face Width (FW) using cheek landmarks 234 and 454 and the ANALYSIS-CROP FOV
        val imgAspect = if (snapshot.imageWidth > 0 && snapshot.imageHeight > 0) {
            snapshot.imageWidth.toFloat() / snapshot.imageHeight.toFloat()
        } else {
            snapshot.projectionParams.cropAspectRatio
        }
        val effectiveProjParams = if (snapshot.projectionParams === ProjectionParams.DEFAULT &&
            snapshot.imageWidth > 0 && snapshot.imageHeight > 0
        ) {
            snapshot.projectionParams.copy(cropAspectRatio = imgAspect)
        } else {
            snapshot.projectionParams
        }
        val rawFw = FacePoseAdapter.computeFaceWidth(
            landmarks = snapshot.landmarks,
            matrix = filteredMatrix,
            imageAspectRatio = imgAspect,
            fovYCropDeg = effectiveProjParams.fovYDeg
        )
        if (shouldResetVelocity) {
            faceWidthFilter.reset()
            lastStableAngularFwRatio = 0f
        }
        val absFilteredZ = kotlin.math.abs(filteredMatrix[14])
        val lms = snapshot.landmarks
        val stabilizedFw = if (lms != null && lms.size >= 455 * 3 && absFilteredZ >= 0.05f) {
            val x234 = lms[FacePoseAdapter.LANDMARK_CHEEK_RIGHT * 3]
            val y234 = lms[FacePoseAdapter.LANDMARK_CHEEK_RIGHT * 3 + 1]
            val x454 = lms[FacePoseAdapter.LANDMARK_CHEEK_LEFT * 3]
            val y454 = lms[FacePoseAdapter.LANDMARK_CHEEK_LEFT * 3 + 1]
            val bothCheeksWellInsideFrame = x234 in 0.05f..0.95f && x454 in 0.05f..0.95f &&
                y234 in 0.05f..0.95f && y454 in 0.05f..0.95f
            val currentAngularRatio = rawFw / absFilteredZ
            val boundedFw = if (bothCheeksWellInsideFrame) {
                lastStableAngularFwRatio = if (lastStableAngularFwRatio > 0f) {
                    0.75f * lastStableAngularFwRatio + 0.25f * currentAngularRatio
                } else {
                    currentAngularRatio
                }
                rawFw
            } else if (lastStableAngularFwRatio > 0f) {
                // One cheek is near or past the screen edge: lock angular scale within ±8% of the last stable in-frame ratio
                val minFw = lastStableAngularFwRatio * absFilteredZ * 0.92f
                val maxFw = lastStableAngularFwRatio * absFilteredZ * 1.08f
                rawFw.coerceIn(minFw, maxFw)
            } else {
                rawFw
            }
            faceWidthFilter.filter(boundedFw, poseTimestampNs)
        } else {
            rawFw
        }
        val fw = stabilizedFw

        // 8. Filter requested blendshapes independently
        val smoothedBlendshapes = HashMap<String, Float>()
        for ((name, value) in snapshot.blendshapes) {
            if (value.isFinite()) {
                val filter = blendshapeFilters.getOrPut(name) {
                    OneEuroFilter(minCutoff = 1.5f, beta = 0.02f, dCutoff = 1.0f)
                }
                smoothedBlendshapes[name] = filter.filter(value, poseTimestampNs)
            }
        }

        val stateReadyNs = FrameClock.elapsedRealtimeNs()

        // 9. Construct immutable FaceState with defensive copies, authoritative poseTimestampNs, and projectionParams
        val newState = FaceState.create(
            timestampNs = poseTimestampNs,
            lifecycle = newLifecycle,
            fadeWeight = lifecycle.fadeWeight,
            isVisible = lifecycle.isVisible,
            poseMatrix = filteredMatrix,
            translation = filteredPose.translation,
            rotation = filteredPose.rotation,
            scale = filteredPose.scale,
            yawDeg = filteredPose.yawDeg,
            pitchDeg = filteredPose.pitchDeg,
            rollDeg = filteredPose.rollDeg,
            faceWidth = fw,
            landmarks = snapshot.landmarks,
            blendshapes = smoothedBlendshapes,
            acquisitionTimeNs = lifecycle.acquisitionTimestampNs,
            imageToViewTransform = snapshot.imageToViewTransform,
            imageWidth = snapshot.imageWidth,
            imageHeight = snapshot.imageHeight,
            isFrontCamera = snapshot.isFrontCamera,
            triggeredEvent = triggeredEvent,
            poseTimestampNs = poseTimestampNs,
            stateReadyNs = stateReadyNs,
            queueMs = snapshot.queueMs,
            conversionMs = snapshot.conversionMs,
            inferenceMs = snapshot.inferenceMs,
            velocity = velScratch,
            angularVelocity = angVelScratch,
            predErrPosMm = predErrPosMm,
            predErrRotDeg = predErrRotDeg,
            projectionParams = effectiveProjParams,
            landmarkPoseMatrix = rawMatrix,
            pendingWaitMs = snapshot.pendingWaitMs
        )

        latestStateRef.set(newState)
        return newState
    }

    /**
     * Reads predicted FaceState at target render time (allocating copy for non-GL callers/tests).
     * Extrapolates rigid pose only when `lifecycle == TRACKING`, does NOT extrapolate scale or
     * blendshapes, and preserves `poseTimestampNs` as the capture timestamp of the underlying sensor frame.
     */
    fun getPredictedState(renderTimestampNs: Long = FrameClock.elapsedRealtimeNs()): FaceState {
        val current = latestStateRef.get()
        if (!isPredictionEnabled || current.lifecycle != FaceLifecycle.State.TRACKING || current.poseTimestampNs <= 0L) {
            return current
        }

        val predT = FloatArray(3)
        val predQ = FloatArray(4)
        latencyCompensator.predict(
            renderTimestampNs = renderTimestampNs,
            outT = predT,
            outQ = predQ,
            poseTimestampNs = current.poseTimestampNs
        )
        val euler = PoseDecomposer.quaternionToEulerDeg(predQ)
        val predMatrix = PoseDecomposer.recompose(predT, predQ, current.scale)
        val baseAbsZ = kotlin.math.abs(current.poseMatrix[14])
        val predAbsZ = kotlin.math.abs(predMatrix[14])
        val adjustedFw = if (baseAbsZ >= 0.05f && predAbsZ >= 0.05f) {
            current.faceWidth * (predAbsZ / baseAbsZ).coerceIn(0.85f, 1.15f)
        } else {
            current.faceWidth
        }

        return current.copy(
            timestampNs = renderTimestampNs,
            poseTimestampNs = current.poseTimestampNs,
            poseMatrix = predMatrix,
            translation = predT,
            rotation = predQ,
            scale = current.scale.clone(),
            yawDeg = euler[0],
            pitchDeg = euler[1],
            rollDeg = euler[2],
            faceWidth = adjustedFw
        )
    }
}
