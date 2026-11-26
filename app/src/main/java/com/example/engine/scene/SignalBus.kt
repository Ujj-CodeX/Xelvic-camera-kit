package com.example.engine.scene

import com.example.engine.signal.FaceState

/**
 * SignalBus evaluates built-in signals and derived signals in topological order
 * into a fixed preallocated float slot array (README Section 2, 5.10 & 21 Step 3).
 * Zero string lookups and zero allocations during frame evaluation!
 */
class SignalBus(
    val totalSlots: Int,
    val derivedSignals: List<Pair<Int, ExpressionVM.Program>>, // (slotIndex, compiledProgram) in topological dependency order
    val slotIndexMap: Map<String, Int>
) {
    val slotValues = FloatArray(totalSlots)
    private val vm = ExpressionVM()
    private var tapTriggered = false

    fun triggerTap() {
        tapTriggered = true
    }

    fun evaluate(faceState: FaceState, renderTimeSec: Float) {
        // Built-in signals
        setSlot("face.present", if (faceState.isVisible) 1f else 0f)
        setSlot("face.fade", faceState.fadeWeight)
        setSlot("face.pose.yaw", faceState.yawDeg)
        setSlot("face.pose.pitch", faceState.pitchDeg)
        setSlot("face.pose.roll", faceState.rollDeg)
        setSlot("time.render", renderTimeSec)

        val sinceAcquired = if (faceState.acquisitionTimeNs > 0) {
            (faceState.timestampNs - faceState.acquisitionTimeNs).toFloat() / 1_000_000_000f
        } else 0f
        setSlot("time.sinceAcquired", sinceAcquired)

        setSlot("input.tap", if (tapTriggered) 1f else 0f)
        tapTriggered = false

        // Blendshape signals
        for ((name, value) in faceState.blendshapes) {
            setSlot("face.blend.$name", value)
        }

        // Derived signals in strictly topological dependency order
        for ((slotIdx, program) in derivedSignals) {
            val res = vm.execute(program, slotValues)
            slotValues[slotIdx] = res
        }
    }

    fun getSlot(name: String): Float {
        val idx = slotIndexMap[name] ?: return 0f
        return if (idx in slotValues.indices) slotValues[idx] else 0f
    }

    fun setSlot(name: String, value: Float) {
        val idx = slotIndexMap[name] ?: return
        if (idx in slotValues.indices) {
            slotValues[idx] = value
        }
    }
}
