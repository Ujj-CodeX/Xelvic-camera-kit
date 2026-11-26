package com.example.engine.scene

/**
 * BehaviorRegistry: Defines engine-supported primitive behavior types.
 * Each behavior has a versioned feature identifier registered in CapabilityRegistry (README Section 5.11 & 7).
 */
object BehaviorRegistry {

    private val supportedBehaviors = hashMapOf(
        "clip" to "behavior.clip@1",
        "map" to "behavior.map@1",
        "trigger" to "behavior.trigger@1",
        "spring" to "behavior.spring@1"
    )

    fun isBehaviorSupported(behaviorType: String): Boolean {
        return supportedBehaviors.containsKey(behaviorType)
    }

    fun getCapabilityForBehavior(behaviorType: String): String? {
        return supportedBehaviors[behaviorType]
    }

    fun getSupportedBehaviorTypes(): Set<String> {
        return supportedBehaviors.keys
    }

    fun getCapabilities(): Set<String> {
        return supportedBehaviors.values.toSet()
    }
}
