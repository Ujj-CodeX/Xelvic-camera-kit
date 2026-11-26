package com.example.engine.pkg

import com.example.engine.perception.ProviderRegistry
import com.example.engine.render.NodeRendererRegistry
import com.example.engine.render.PostRenderer
import com.example.engine.scene.BehaviorRegistry
import com.example.engine.scene.SignalSourceRegistry

/**
 * CapabilityRegistry: Single source of truth for versioned engine features.
 * Packages declare requires.features (e.g. "node.model@1", "behavior.spring@1").
 * The compiler validates package requirements against this registry (README Section 5.3 & 7).
 */
object CapabilityRegistry {

    private val supportedFeatures = hashSetOf(
        "runtime@3",
        "node.group@1",
        "node.sprite@1",
        "node.model@1",
        "node.faceMesh@1",
        "node.occluder@1",
        "node.post@1",
        "post.colorGrade@1",
        "post.blur@1",
        "post.skinSoften@1",
        "post.bilateral@1",
        "post.softFocus@1",
        "post.warp@1",
        "post.colorMatrix@1",
        "post.vignette@1",
        "post.lut@1",
        "post.grain@1",
        "post.sharpen@1",
        "material.unlit@1",
        "material.lambert@1",
        "material.pbr@1",
        "material.faceTexture@1",
        "material.recolor@1",
        "blend.opaque@1",
        "blend.alpha@1",
        "blend.additive@1",
        "blend.multiply@1",
        "blend.softLight@1",
        "blend.screen@1",
        "node.eyeColor@1",
        "node.overlay@1",
        "sprite.flipbook@1",
        "overlay.flipbook@1",
        "sprite.particles@1",
        "overlay.particles@1",
        "pass.recolor@1",
        "pass.skinSoften@1",
        "pass.bilateral@1",
        "pass.softFocus@1",
        "pass.warp@1",
        "pass.colorMatrix@1",
        "pass.vignette@1",
        "pass.lut@1",
        "pass.grain@1",
        "pass.sharpen@1",
        "pixel.recolor@1",
        "tracker.iris@1",
        "behavior.clip@1",
        "behavior.map@1",
        "behavior.trigger@1",
        "behavior.spring@1",
        "occluder.faceMesh@1",
        "occluder.ellipsoid@1",
        "occluder.box@1",
        "signal.blendshapes@1",
        "signal.pose@1",
        "signal.tap@1",
        "input.tap@1",
        "provider.face.mediapipe@1",
        "model.gltf2@1",
        "anchor.headCenter@1",
        "anchor.eyesCenter@1",
        "pose.rigidHeadAttachment@1",
        "occlusion.faceMesh@1",
        "occlusion.headProxyEllipsoid@1",
        "occlusion.earTempleProxy@1",
        "lifecycle.faceLost@1",
        "render.opaqueModel@1",
        "behavior.boundedFloat@1"
    )

    init {
        // Collect from registries
        supportedFeatures.addAll(NodeRendererRegistry.getCapabilities())
        supportedFeatures.addAll(PostRenderer.getCapabilities())
        supportedFeatures.addAll(BehaviorRegistry.getCapabilities())
        supportedFeatures.addAll(SignalSourceRegistry.getCapabilities())
        supportedFeatures.addAll(ProviderRegistry.getRegisteredCapabilities())
    }

    fun isFeatureSupported(featureNameWithVersion: String): Boolean {
        return supportedFeatures.contains(featureNameWithVersion)
    }

    fun getAllSupportedFeatures(): Set<String> {
        return supportedFeatures
    }

    /**
     * Negotiate package requirements.
     * @return Pair(isCompatible: Boolean, missingFeatures: List<String>)
     */
    fun negotiate(requiredFeatures: List<String>): Pair<Boolean, List<String>> {
        val missing = ArrayList<String>()
        for (feat in requiredFeatures) {
            if (!isFeatureSupported(feat)) {
                missing.add(feat)
            }
        }
        return Pair(missing.isEmpty(), missing)
    }

    /**
     * Negotiate optional features with fallback degradation policies.
     */
    fun negotiateOptional(optionalFeatures: List<String>): Map<String, Boolean> {
        val results = HashMap<String, Boolean>()
        for (feat in optionalFeatures) {
            results[feat] = isFeatureSupported(feat)
        }
        return results
    }
}

