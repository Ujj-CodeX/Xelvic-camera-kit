package com.example.engine.signal

/**
 * FaceState passed from the signal pipeline to the render thread.
 *
 * Published tracker snapshots in `SignalPipeline.latestStateRef` are created via [create] with
 * defensive copies and treated as immutable snapshots.
 *
 * For zero-allocation vsync prediction in `onDrawFrame`, `SignalPipeline.consumeLatestPredictedState`
 * reuses a single GL-thread-confined [FaceState] instance via [copyFromForPrediction], mutating its
 * preallocated `poseMatrix`, `translation`, `rotation`, `scale`, `velocity`, and `angularVelocity`
 * arrays in-place while referencing unchanged `landmarks`, `blendshapes`, and `imageToViewTransform`
 * from the latest published state without copying.
 *
 * @property timestampNs Evaluation/render timestamp in `FrameClock.elapsedRealtimeNs()` domain.
 * @property poseTimestampNs Authoritative camera sensor capture timestamp (aligned to
 *   `SystemClock.elapsedRealtimeNanos()` via `ClockOffsetCalibrator`) of the frame from which
 *   this pose was derived. `ageAtRender = renderTimeNs - poseTimestampNs` measures
 *   capture-timestamp -> render, NOT full glass-to-glass latency.
 */
data class FaceState(
    var timestampNs: Long = 0L,
    var lifecycle: FaceLifecycle.State = FaceLifecycle.State.NO_FACE,
    var fadeWeight: Float = 0f,
    var isVisible: Boolean = false,
    val poseMatrix: FloatArray = FloatArray(16) { if (it % 5 == 0) 1f else 0f },
    val translation: FloatArray = FloatArray(3),
    val rotation: FloatArray = floatArrayOf(0f, 0f, 0f, 1f),
    val scale: FloatArray = floatArrayOf(1f, 1f, 1f),
    var yawDeg: Float = 0f,
    var pitchDeg: Float = 0f,
    var rollDeg: Float = 0f,
    var faceWidth: Float = 0.14f,
    var landmarks: FloatArray? = null, // 478 x 3 normalized coords
    var blendshapes: Map<String, Float> = emptyMap(),
    var acquisitionTimeNs: Long = 0L,
    var imageToViewTransform: FloatArray? = null,
    var imageWidth: Int = 0,
    var imageHeight: Int = 0,
    var isFrontCamera: Boolean = true,
    var triggeredEvent: String? = null,
    var poseTimestampNs: Long = timestampNs,
    var stateReadyNs: Long = 0L,
    var queueMs: Float = 0f,
    var conversionMs: Float = 0f,
    var pendingWaitMs: Float = 0f,
    var inferenceMs: Float = 0f,
    val velocity: FloatArray = FloatArray(3),
    val angularVelocity: FloatArray = FloatArray(3),
    var predErrPosMm: Float = 0f,
    var predErrRotDeg: Float = 0f,
    var projectionParams: ProjectionParams = ProjectionParams.DEFAULT,
    val landmarkPoseMatrix: FloatArray = poseMatrix.clone()
) {
    /**
     * Copies scalar fields and preallocated array contents from [source] into `this` without any
     * heap allocations, and references [source]'s `landmarks`, `blendshapes`, `imageToViewTransform`,
     * and `projectionParams` directly. Used exclusively on the GL thread by `SignalPipeline.consumeLatestPredictedState`.
     */
    fun copyFromForPrediction(source: FaceState, triggeredEventOverride: String?) {
        timestampNs = source.timestampNs
        lifecycle = source.lifecycle
        fadeWeight = source.fadeWeight
        isVisible = source.isVisible
        System.arraycopy(source.poseMatrix, 0, poseMatrix, 0, 16)
        System.arraycopy(source.translation, 0, translation, 0, 3)
        System.arraycopy(source.rotation, 0, rotation, 0, 4)
        System.arraycopy(source.scale, 0, scale, 0, 3)
        yawDeg = source.yawDeg
        pitchDeg = source.pitchDeg
        rollDeg = source.rollDeg
        faceWidth = source.faceWidth
        landmarks = source.landmarks
        blendshapes = source.blendshapes
        acquisitionTimeNs = source.acquisitionTimeNs
        imageToViewTransform = source.imageToViewTransform
        imageWidth = source.imageWidth
        imageHeight = source.imageHeight
        isFrontCamera = source.isFrontCamera
        triggeredEvent = triggeredEventOverride
        poseTimestampNs = source.poseTimestampNs
        stateReadyNs = source.stateReadyNs
        queueMs = source.queueMs
        conversionMs = source.conversionMs
        pendingWaitMs = source.pendingWaitMs
        inferenceMs = source.inferenceMs
        System.arraycopy(source.velocity, 0, velocity, 0, 3)
        System.arraycopy(source.angularVelocity, 0, angularVelocity, 0, 3)
        predErrPosMm = source.predErrPosMm
        predErrRotDeg = source.predErrRotDeg
        projectionParams = source.projectionParams
        System.arraycopy(source.landmarkPoseMatrix, 0, landmarkPoseMatrix, 0, 16)
    }

    companion object {
        val EMPTY = FaceState()

        /**
         * Creates an immutable-by-convention FaceState snapshot with defensive copies of all arrays and maps.
         */
        @JvmOverloads
        fun create(
            timestampNs: Long,
            lifecycle: FaceLifecycle.State,
            fadeWeight: Float,
            isVisible: Boolean,
            poseMatrix: FloatArray,
            translation: FloatArray,
            rotation: FloatArray,
            scale: FloatArray,
            yawDeg: Float,
            pitchDeg: Float,
            rollDeg: Float,
            faceWidth: Float,
            landmarks: FloatArray?,
            blendshapes: Map<String, Float>,
            acquisitionTimeNs: Long,
            imageToViewTransform: FloatArray?,
            imageWidth: Int,
            imageHeight: Int,
            isFrontCamera: Boolean,
            triggeredEvent: String? = null,
            poseTimestampNs: Long = timestampNs,
            stateReadyNs: Long = 0L,
            queueMs: Float = 0f,
            conversionMs: Float = 0f,
            inferenceMs: Float = 0f,
            velocity: FloatArray = FloatArray(3),
            angularVelocity: FloatArray = FloatArray(3),
            predErrPosMm: Float = 0f,
            predErrRotDeg: Float = 0f,
            projectionParams: ProjectionParams = ProjectionParams.DEFAULT,
            landmarkPoseMatrix: FloatArray = poseMatrix,
            pendingWaitMs: Float = 0f
        ): FaceState {
            return FaceState(
                timestampNs = timestampNs,
                lifecycle = lifecycle,
                fadeWeight = fadeWeight,
                isVisible = isVisible,
                poseMatrix = poseMatrix.clone(),
                translation = translation.clone(),
                rotation = rotation.clone(),
                scale = scale.clone(),
                yawDeg = yawDeg,
                pitchDeg = pitchDeg,
                rollDeg = rollDeg,
                faceWidth = faceWidth,
                landmarks = landmarks?.clone(),
                blendshapes = java.util.Collections.unmodifiableMap(java.util.HashMap(blendshapes)),
                acquisitionTimeNs = acquisitionTimeNs,
                imageToViewTransform = imageToViewTransform?.clone(),
                imageWidth = imageWidth,
                imageHeight = imageHeight,
                isFrontCamera = isFrontCamera,
                triggeredEvent = triggeredEvent,
                poseTimestampNs = poseTimestampNs,
                stateReadyNs = stateReadyNs,
                queueMs = queueMs,
                conversionMs = sourceSafeFloat(conversionMs),
                pendingWaitMs = sourceSafeFloat(pendingWaitMs),
                inferenceMs = sourceSafeFloat(inferenceMs),
                velocity = velocity.clone(),
                angularVelocity = angularVelocity.clone(),
                predErrPosMm = predErrPosMm,
                predErrRotDeg = predErrRotDeg,
                projectionParams = projectionParams,
                landmarkPoseMatrix = landmarkPoseMatrix.clone()
            )
        }

        private fun sourceSafeFloat(v: Float): Float = if (v.isFinite()) v else 0f
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as FaceState
        return timestampNs == other.timestampNs &&
                poseTimestampNs == other.poseTimestampNs &&
                lifecycle == other.lifecycle &&
                fadeWeight == other.fadeWeight &&
                isVisible == other.isVisible &&
                faceWidth == other.faceWidth &&
                poseMatrix.contentEquals(other.poseMatrix) &&
                translation.contentEquals(other.translation) &&
                rotation.contentEquals(other.rotation) &&
                scale.contentEquals(other.scale)
    }

    override fun hashCode(): Int {
        var result = timestampNs.hashCode()
        result = 31 * result + poseTimestampNs.hashCode()
        result = 31 * result + lifecycle.hashCode()
        result = 31 * result + fadeWeight.hashCode()
        result = 31 * result + isVisible.hashCode()
        result = 31 * result + faceWidth.hashCode()
        result = 31 * result + poseMatrix.contentHashCode()
        result = 31 * result + translation.contentHashCode()
        result = 31 * result + rotation.contentHashCode()
        result = 31 * result + scale.contentHashCode()
        return result
    }
}
