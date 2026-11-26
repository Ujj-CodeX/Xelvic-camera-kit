package com.example.engine.render

/**
 * NodeRendererRegistry: Defines engine-supported primitive node types.
 * Each node type has a versioned feature identifier registered in CapabilityRegistry (README Section 7).
 */
object NodeRendererRegistry {

    private val supportedNodeTypes = hashMapOf(
        "group" to "node.group@1",
        "sprite" to "node.sprite@1",
        "model" to "node.model@1",
        "faceMesh" to "node.faceMesh@1",
        "eyeColor" to "node.eyeColor@1",
        "recolor" to "pixel.recolor@1",
        "overlay" to "node.overlay@1",
        "occluder" to "node.occluder@1",
        "post" to "node.post@1"
    )

    fun isNodeTypeSupported(nodeType: String): Boolean {
        return supportedNodeTypes.containsKey(nodeType)
    }

    fun getCapabilityForNodeType(nodeType: String): String? {
        return supportedNodeTypes[nodeType]
    }

    fun getSupportedNodeTypes(): Set<String> {
        return supportedNodeTypes.keys
    }

    fun getCapabilities(): Set<String> {
        return supportedNodeTypes.values.toSet()
    }
}
