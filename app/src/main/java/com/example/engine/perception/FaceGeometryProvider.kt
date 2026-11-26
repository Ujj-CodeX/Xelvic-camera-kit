package com.example.engine.perception

import com.example.engine.capture.AnalysisFrame
import com.example.engine.capture.FrameClock

enum class AnalysisResolutionMode {
    LEGACY,
    SMALL
}

enum class ConversionMode {
    FAST,
    OLD
}

data class ProviderConfig(
    val outputPose: Boolean = true,
    val outputBlendshapes: Boolean = true,
    val outputLandmarks: Boolean = true,
    val maxFaces: Int = 1,
    val useGpu: Boolean = false,
    val requestedBlendshapes: Set<String> = emptySet(),
    val analysisLongEdgePx: Int = 320
)

interface FaceGeometryProvider {
    val modelSha256: String
    val activeDelegateLabel: String
        get() = "CPU"
    val statusMessage: String
        get() = ""
    val hasInitError: Boolean
        get() = false
    val isPaused: Boolean
        get() = false
    val lastSwapMs: Float
        get() = 0f
    val skipConversionIfPendingFresh: Boolean
        get() = false
    val framesSkippedBeforeConversionCount: Int
        get() = 0
    val conversionMode: ConversionMode
        get() = ConversionMode.FAST

    fun start(config: ProviderConfig)
    fun stop()
    fun submit(frame: AnalysisFrame)
    fun setListener(listener: (FaceGeometrySnapshot) -> Unit)
    fun setPaused(paused: Boolean) {}
    fun updateConfig(config: ProviderConfig) {}
    fun updateAnalysisSize(longEdgePx: Int) {}
    fun updateTargetCadenceHz(cadenceHz: Int) {}
    fun switchDelegate(useGpu: Boolean) {}
    fun setBlendshapesEnabled(enabled: Boolean) {}
    fun setUseLowThreadPriority(lowPriority: Boolean) {}
    fun setSkipConversionIfPendingFresh(enabled: Boolean) {}
    fun setConversionMode(mode: ConversionMode) {}
    fun setResolutionModeForTelemetry(mode: AnalysisResolutionMode) {}

    /**
     * Runs fresh, synchronous still-image face inference on a background worker (README Section 16).
     * Never returns cached preview pose.
     */
    fun detectStillImage(
        bitmap: android.graphics.Bitmap,
        rotationDegrees: Int = 0,
        isFrontCamera: Boolean = true,
        timestampNs: Long = FrameClock.elapsedRealtimeNs()
    ): FaceGeometrySnapshot {
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
            modelSha256 = modelSha256
        )
    }
}

object ProviderRegistry {
    private val providers = HashMap<String, FaceGeometryProvider>()
    private val capabilities = hashSetOf("provider.face.mediapipe@1")

    fun register(name: String, provider: FaceGeometryProvider) {
        providers[name] = provider
        capabilities.add("provider.face.$name@1")
    }

    fun get(name: String): FaceGeometryProvider? = providers[name]

    fun getRegisteredCapabilities(): Set<String> = capabilities
}
