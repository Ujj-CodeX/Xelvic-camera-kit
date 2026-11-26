package com.example.engine.scene

import com.example.engine.render.PassGraph
import com.example.engine.render.RegionMaskPass

/**
 * Immutable CompiledEffect: All references, params, selectors, and expressions
 * resolved to indexed tables at install time (README Section 2, 5 & 21 Step 3).
 * Zero string lookups and zero JSON parsing during the render frame loop!
 */
data class CompiledEffect(
    val packageId: String,
    val packageName: String,
    val schemaVersion: String,
    val nodes: List<CompiledNode>,
    val materials: Map<String, CompiledMaterial>,
    val assets: Map<String, CompiledAsset>,
    val lighting: CompiledLighting?,
    val signalBus: SignalBus,
    val behaviorRunner: BehaviorRunner,
    val anchorResolver: AnchorResolver,
    val propertySlots: PropertySlots,
    val passGraph: PassGraph = PassGraph.defaultCanonical9StageGraph(),
    val maskDefs: Map<String, RegionMaskPass.MaskExpr> = RegionMaskPass.BUILTIN_MASKS,
    var activeTier: String = "A",
    val minTier: String? = null,
    val tierOverrides: Map<String, TierOverride> = emptyMap(),
    val requiredLandmarkCount: Int = 478,
    val requiresPose: Boolean = true,
    val requiresBlendshapes: Boolean = true,
    val requiresIris: Boolean = false,
    val requestedBlendshapes: Set<String> = emptySet(),
    val photoPolicy: CompiledPhotoPolicy = CompiledPhotoPolicy(),
    val lifecyclePolicy: CompiledLifecyclePolicy = CompiledLifecyclePolicy()
) {
    fun reset() {
        propertySlots.reset()
        behaviorRunner.reset()
    }
}

data class CompiledNode(
    val index: Int,
    val id: String,
    val parentIndex: Int,
    val tags: List<String>,
    val type: String, // "sprite", "model", "faceMesh", "occluder", "post", "overlay", "eyeColor", "group"
    val anchor: String?,
    val basePosition: FloatArray, // [x, y, z] in FW
    val baseRotation: FloatArray, // [qx, qy, qz, qw]
    val baseScale: FloatArray,    // [sx, sy, sz]
    val rotationFollow: String,   // "full", "roll", "eyeLine", "none"
    val materialId: String?,
    val baseVisible: Boolean,
    val baseOpacity: Float,
    val renderOrder: Int,
    // Type-specific fields
    val spriteAssetId: String? = null,
    val spriteWidthFW: Float = 0.5f,
    val spriteHeightFW: Float = 0.5f,
    val spritePivot: FloatArray = floatArrayOf(0.5f, 0.5f),
    val modelAssetId: String? = null,
    val occluderShape: String? = null, // "faceMesh", "ellipsoid", "box"
    val pushBackFW: Float = 0.015f,
    val radiiFW: FloatArray? = null,
    val occluderSegments: IntArray? = null,
    val faceMeshRegion: String? = null,
    val postEffect: String? = null,
    val postParams: Map<String, Float> = emptyMap(),
    val postAssetId: String? = null,
    val overlayFit: String = "cover",
    val overlaySafeArea: Boolean = false,
    val spriteRows: Int = 1,
    val spriteCols: Int = 1,
    val spriteFps: Float = 0f,
    val spriteLoop: Boolean = true,
    val spriteFrameCount: Int = 0,
    val spriteFrameIndex: Int = -1,
    val spriteStartFrame: Int = 0,
    val particleCount: Int = 1,
    val particleSpreadX: Float = 0.35f,
    val particleSpreadY: Float = 0.35f,
    val particleDriftX: Float = 0f,
    val particleDriftY: Float = 0f,
    val particleTwinkleSpeed: Float = 0f,
    val particleScaleMin: Float = 0.65f,
    val particleScaleMax: Float = 1.15f,
    val particleSeed: Int = 0
)

data class CompiledMaterial(
    val id: String,
    val template: String, // "unlit@1", "lambert@1", "pbr@1", "faceTexture@1"
    val mapAssetId: String?,
    val maskAssetId: String?,
    val normalMapAssetId: String? = null,
    val metallicRoughnessMapAssetId: String? = null,
    val occlusionMapAssetId: String? = null,
    val emissiveMapAssetId: String? = null,
    val color: FloatArray = floatArrayOf(1f, 1f, 1f),
    val opacity: Float = 1f,
    val metallic: Float = 0.0f,
    val roughness: Float = 0.45f,
    val normalScale: Float = 1.0f,
    val occlusionStrength: Float = 1.0f,
    val fresnelSheen: Float = 0.15f,
    val emissive: FloatArray = floatArrayOf(0f, 0f, 0f),
    val hasExplicitMetallic: Boolean = false,
    val hasExplicitRoughness: Boolean = false,
    val hasExplicitEmissive: Boolean = false,
    val uvLayout: String = "mediapipe_canonical_468_v1",
    val protectEyes: Boolean = true,
    val protectLips: Boolean = true,
    val protectMouthInterior: Boolean = true,
    val blend: String = "alpha", // "opaque", "alpha", "additive", "multiply", "softLight", "screen"
    val doubleSided: Boolean = false,
    val depthTest: Boolean = true,
    val depthWrite: Boolean = true,
    val cull: Boolean = true
)

data class CompiledAsset(
    val id: String,
    val type: String, // "image", "model", "mask", "lut"
    val localPath: String,
    val sha256: String,
    val bytes: Long
)

data class CompiledLighting(
    val space: String = "camera",
    val ambient: FloatArray = floatArrayOf(0.3f, 0.3f, 0.3f),
    val dirToLight: FloatArray = floatArrayOf(-0.35f, 0.65f, 0.68f),
    val lightColor: FloatArray = floatArrayOf(1f, 1f, 1f),
    val lightIntensity: Float = 0.8f,
    val lightCount: Int = 1,
    val lightDirs: FloatArray = floatArrayOf(
        dirToLight.getOrElse(0) { -0.35f },
        dirToLight.getOrElse(1) { 0.65f },
        dirToLight.getOrElse(2) { 0.68f },
        0f, 0f, 1f
    ),
    val lightColors: FloatArray = floatArrayOf(
        lightColor.getOrElse(0) { 1f },
        lightColor.getOrElse(1) { 1f },
        lightColor.getOrElse(2) { 1f },
        0f, 0f, 0f
    ),
    val lightIntensities: FloatArray = floatArrayOf(lightIntensity, 0f)
) {
    companion object {
        const val MAX_LIGHTS = 2
    }
}

data class TierOverride(
    val tierId: String,
    val showNodes: Set<Int>,
    val hideNodes: Set<Int>,
    val disabledBehaviors: Set<String>,
    val textureMaxSize: Int = 1024,
    val disabledPasses: Set<String> = emptySet(),
    val passScaleOverrides: Map<String, Float> = emptyMap(),
    val paramOverrides: Map<String, Float> = emptyMap()
)

data class CompiledPhotoPolicy(
    val clips: String = "settle", // "settle", "freeze"
    val triggers: String = "disarmed",
    val timeSec: Float? = null,
    val seed: Int? = null
)

data class CompiledLifecyclePolicy(
    val holdMs: Long = 150L,
    val fadeMs: Long = 200L,
    val fadeCurve: String = "cubicOut",
    val afterFade: String = "hideAndReset",
    val cancelFade: Boolean = true,
    val easeInMs: Long = 100L,
    val preserveState: Boolean = true,
    val replayStartOn: Boolean = false
)
