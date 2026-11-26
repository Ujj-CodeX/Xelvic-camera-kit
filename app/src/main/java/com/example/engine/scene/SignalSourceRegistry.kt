package com.example.engine.scene

/**
 * SignalSourceRegistry: Defines engine-provided signal sources (README Section 5.10 & 7).
 * Built-in sources provide smooth sensor, face tracking, time, and touch inputs.
 */
object SignalSourceRegistry {

    private val staticBuiltins = hashSetOf(
        "face.present",
        "face.fade",
        "face.pose.yaw",
        "face.pose.pitch",
        "face.pose.roll",
        "time.render",
        "time.sinceAcquired",
        "input.tap"
    )

    val STANDARD_BLENDSHAPES = hashSetOf(
        "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
        "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
        "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
        "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
        "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
        "jawForward", "jawLeft", "jawOpen", "jawRight",
        "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
        "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
        "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
        "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
        "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
        "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
        "tongueOut"
    )

    fun isSignalSupported(signalName: String): Boolean {
        if (staticBuiltins.contains(signalName)) return true
        if (signalName.startsWith("face.blend.")) {
            val bsName = signalName.substring("face.blend.".length)
            return STANDARD_BLENDSHAPES.contains(bsName) || bsName.isNotEmpty()
        }
        if (signalName.startsWith("sig.")) return true // Derived signal
        if (signalName.startsWith("param.") || signalName.startsWith("$")) return true // Tunable param
        return false
    }

    fun getCapabilities(): Set<String> {
        return hashSetOf(
            "signal.pose@1",
            "signal.blendshapes@1",
            "signal.tap@1",
            "input.tap@1"
        )
    }
}
