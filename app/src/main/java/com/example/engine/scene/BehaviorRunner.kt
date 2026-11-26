package com.example.engine.scene

import com.example.engine.signal.FaceState
import kotlin.math.*

/**
 * BehaviorRunner executes all 4 behavior primitives in strictly deterministic composition order:
 * Base -> Clips -> Maps -> Springs -> Lifecycle Fade
 *
 * Extended in Flow 2 — Chunk G0 to support universal signal-to-parameter bindings:
 * `NODE_PROPERTY` (`POSITION`, `ROTATION`, `SCALE`, `OPACITY`, `VISIBLE`) as well as
 * `PASS_UNIFORM`, `MATERIAL_PARAM`, and `MASK_WEIGHT` stored in [PropertySlots.dynamicParams].
 */
class BehaviorRunner(
    val clips: List<CompiledClip>,
    val maps: List<CompiledMap>,
    val triggers: List<CompiledTrigger>,
    val springs: List<CompiledSpring>,
    val nodeCount: Int
) {
    var onCustomEvent: ((String) -> Unit)? = null

    data class PropertyWrite(
        val nodeIndex: Int,
        val propertyType: PropertyType,
        val componentIndex: Int = 0,
        val mode: WriteMode,
        val dynamicParamSlotIndex: Int = -1
    )

    enum class PropertyType {
        POSITION,
        ROTATION,
        SCALE,
        OPACITY,
        VISIBLE,
        PASS_UNIFORM,
        MATERIAL_PARAM,
        MASK_WEIGHT
    }

    enum class WriteMode {
        SET,
        MULTIPLY,
        ADD
    }

    class CompiledClip(
        val id: String,
        val targetNodeIndex: Int,
        val animationName: String,
        val loop: Boolean,
        val startOn: String?,
        val speed: Float,
        val holdLast: Boolean,
        var isPlaying: Boolean = false,
        var currentTimeSec: Float = 0f,
        var durationSec: Float = 1.0f
    ) {
        fun reset() {
            isPlaying = false
            currentTimeSec = 0f
        }
    }

    class CompiledMap(
        val id: String,
        val inputSlotIndex: Int,
        val inMin: Float,
        val inMax: Float,
        val outMin: Float,
        val outMax: Float,
        val easing: String,
        val clamp: Boolean,
        val smoothingTau: Float,
        val initial: Float,
        val writes: List<PropertyWrite>,
        var currentValue: Float = initial
    ) {
        fun reset() {
            currentValue = initial
        }
    }

    class TriggerAction(
        val action: String, // "playClip", "stopClip", "setProperty", "toggleVisible", "emit"
        val clipId: String? = null,
        val propertyWrite: PropertyWrite? = null,
        val value: Float = 0f
    )

    class CompiledTrigger(
        val id: String,
        val inputSlotIndex: Int?, // null if event-driven
        val onEvent: String?,     // "faceAcquired", "faceLost", "tap"
        val risingAbove: Float?,
        val fallingBelow: Float?,
        val cooldownMs: Long,
        val armedInitially: Boolean,
        val onFire: List<TriggerAction>,
        var isArmed: Boolean = armedInitially,
        var lastFireTimeMs: Long = -100_000L
    ) {
        fun reset() {
            isArmed = armedInitially
            lastFireTimeMs = -100_000L
        }
    }

    class CompiledSpring(
        val id: String,
        val targetNodeIndex: Int,
        val mode: String, // "rotationInertia", "positionInertia"
        val stiffness: Float,
        val dampingRatio: Float,
        val gain: Float,
        val maxAngleDeg: Float,
        val lowPassTau: Float,
        val substepMaxSec: Float = 0.00833f,
        // Simulation state
        var pos: Float = 0f,
        var vel: Float = 0f,
        var prevPoseAngle: Float = 0f
    ) {
        fun reset() {
            pos = 0f
            vel = 0f
            prevPoseAngle = 0f
        }
    }

    fun reset() {
        for (c in clips) c.reset()
        for (m in maps) m.reset()
        for (t in triggers) t.reset()
        for (s in springs) s.reset()
    }

    /**
     * Step behaviors on each frame.
     * @param dt Elapsed seconds (clamped to 0.001..0.05)
     * @param signalBus Evaluated signal bus
     * @param faceState Current face state
     * @param slots Output property slots to be modified in-place
     * @param disabledBehaviors Behavior IDs disabled by the active quality tier
     * @param hiddenNodes Node indices hidden by the active quality tier (neither drawn nor animated)
     */
    @JvmOverloads
    fun step(
        dt: Float,
        signalBus: SignalBus,
        faceState: FaceState,
        slots: PropertySlots,
        nowMs: Long = System.currentTimeMillis(),
        disabledBehaviors: Set<String> = emptySet(),
        hiddenNodes: Set<Int> = emptySet()
    ) {
        val clampedDt = dt.coerceIn(0.001f, 0.05f)

        // 0. Base: Reset property slots to identity rest state
        slots.reset()

        for (hiddenIdx in hiddenNodes) {
            if (hiddenIdx in 0 until nodeCount) {
                slots.visible[hiddenIdx] = 0f
            }
        }

        // 1. Check startOn events (e.g. faceAcquired) or initial tracking after filter switch
        val event = faceState.triggeredEvent
        if (event != null) {
            for (clip in clips) {
                if (disabledBehaviors.contains(clip.id) || hiddenNodes.contains(clip.targetNodeIndex)) continue
                if (clip.startOn == event) {
                    clip.isPlaying = true
                    clip.currentTimeSec = 0f
                }
            }
        } else if (faceState.isVisible) {
            for (clip in clips) {
                if (disabledBehaviors.contains(clip.id) || hiddenNodes.contains(clip.targetNodeIndex)) continue
                if (clip.startOn == "faceAcquired" && !clip.isPlaying && clip.currentTimeSec == 0f) {
                    clip.isPlaying = true
                }
            }
        }

        // 2. Process Triggers
        processTriggers(clampedDt, signalBus, faceState, slots, nowMs, disabledBehaviors)

        // 3. Step Clips
        stepClips(clampedDt, disabledBehaviors, hiddenNodes)

        // 4. Step Maps
        stepMaps(clampedDt, signalBus, slots, disabledBehaviors, hiddenNodes)

        // 5. Step Springs
        stepSprings(clampedDt, faceState, slots, disabledBehaviors, hiddenNodes)

        // 6. Apply Lifecycle Fade (scale x fadeWeight, opacity x fadeWeight)
        val fade = faceState.fadeWeight
        for (i in 0 until nodeCount) {
            slots.scale[i * 3] *= fade
            slots.scale[i * 3 + 1] *= fade
            slots.scale[i * 3 + 2] *= fade
            slots.opacity[i] *= fade
            if (!faceState.isVisible || hiddenNodes.contains(i)) {
                slots.visible[i] = 0f
            }
        }
    }

    data class PhotoEvaluationReport(
        val clipsPolicyApplied: String,
        val triggersPolicyApplied: String,
        val explicitTimeSecApplied: Float?,
        val settledClipTimesSec: Map<String, Float>,
        val triggersArmedState: Map<String, Boolean>
    )

    /**
     * Evaluates all behaviors deterministically for still-photo capture according to the package's
     * `CompiledPhotoPolicy` (README Section 5.14 & 16):
     * - `clips: "settle"` -> hold-last clips jump to/retain their settled end state (`durationSec`),
     *   looping clips start at `t = 0`, and other clips follow their `t = 0` rest state
     *   (or evaluate at `photoPolicy.timeSec` if explicitly declared).
     * - `triggers: "disarmed"` -> all triggers are disarmed and do not fire during still evaluation.
     * - `maps` -> evaluate deterministically to their settled target value from the fresh photo signals.
     * - `springs` -> settled at zero-velocity equilibrium (`pos = 0`, `vel = 0`).
     */
    @JvmOverloads
    fun evaluateForPhoto(
        photoPolicy: CompiledPhotoPolicy,
        signalBus: SignalBus,
        faceState: FaceState,
        slots: PropertySlots,
        disabledBehaviors: Set<String> = emptySet(),
        hiddenNodes: Set<Int> = emptySet()
    ): PhotoEvaluationReport {
        slots.reset()

        for (hiddenIdx in hiddenNodes) {
            if (hiddenIdx in 0 until nodeCount) {
                slots.visible[hiddenIdx] = 0f
            }
        }

        val explicitTime = photoPolicy.timeSec
        val settledClipTimes = LinkedHashMap<String, Float>()
        val triggersArmed = LinkedHashMap<String, Boolean>()

        // 1. Triggers policy ("disarmed" by default)
        if (photoPolicy.triggers == "disarmed") {
            for (trig in triggers) {
                trig.isArmed = false
                triggersArmed[trig.id] = false
            }
        } else {
            processTriggers(0.016f, signalBus, faceState, slots, 0L, disabledBehaviors)
            for (trig in triggers) {
                triggersArmed[trig.id] = trig.isArmed
            }
        }

        // 2. Clips policy ("settle" by default, with optional explicit timeSec)
        for (clip in clips) {
            if (disabledBehaviors.contains(clip.id) || hiddenNodes.contains(clip.targetNodeIndex)) {
                clip.isPlaying = false
                clip.currentTimeSec = 0f
                settledClipTimes[clip.id] = 0f
                continue
            }
            val dur = clip.durationSec.coerceAtLeast(0.001f)
            if (photoPolicy.clips == "settle") {
                when {
                    clip.holdLast && !clip.loop -> {
                        clip.isPlaying = true
                        clip.currentTimeSec = if (explicitTime != null && explicitTime >= 0f) {
                            explicitTime.coerceIn(0f, dur)
                        } else {
                            dur
                        }
                    }
                    clip.loop -> {
                        clip.isPlaying = true
                        clip.currentTimeSec = if (explicitTime != null && explicitTime >= 0f) {
                            explicitTime.coerceIn(0f, dur)
                        } else {
                            0f
                        }
                    }
                    else -> {
                        if (explicitTime != null && explicitTime >= 0f) {
                            clip.isPlaying = explicitTime <= dur
                            clip.currentTimeSec = explicitTime.coerceIn(0f, dur)
                        } else {
                            // Other non-holdLast, non-looping clips follow their t=0 state
                            clip.isPlaying = false
                            clip.currentTimeSec = 0f
                        }
                    }
                }
            } else {
                val t = (explicitTime ?: 0f).coerceAtLeast(0f)
                clip.isPlaying = true
                clip.currentTimeSec = if (clip.loop) (t % dur) else t.coerceAtMost(dur)
            }
            settledClipTimes[clip.id] = clip.currentTimeSec
        }

        // 3. Maps settle deterministically at their target value for the fresh photo signals
        for (map in maps) {
            if (disabledBehaviors.contains(map.id)) continue
            val rawInput = signalBus.slotValues[map.inputSlotIndex]
            val norm = if (abs(map.inMax - map.inMin) > 1e-6f) {
                ((rawInput - map.inMin) / (map.inMax - map.inMin)).let {
                    if (map.clamp) it.coerceIn(0f, 1f) else it
                }
            } else 0f

            val eased = ease(norm, map.easing)
            val target = map.outMin + (map.outMax - map.outMin) * eased
            map.currentValue = target

            for (write in map.writes) {
                if (write.nodeIndex >= 0 && hiddenNodes.contains(write.nodeIndex)) continue
                applyWrite(write, map.currentValue, slots)
            }
        }

        // 4. Springs settle at rest equilibrium (0 displacement, 0 velocity)
        for (spring in springs) {
            spring.pos = 0f
            spring.vel = 0f
            spring.prevPoseAngle = faceState.rollDeg
        }

        // 5. Apply visibility and fade weight
        val fade = if (faceState.isVisible) faceState.fadeWeight.coerceIn(0f, 1f) else 0f
        for (i in 0 until nodeCount) {
            slots.scale[i * 3] *= fade
            slots.scale[i * 3 + 1] *= fade
            slots.scale[i * 3 + 2] *= fade
            slots.opacity[i] *= fade
            if (!faceState.isVisible || hiddenNodes.contains(i)) {
                slots.visible[i] = 0f
            }
        }

        return PhotoEvaluationReport(
            clipsPolicyApplied = photoPolicy.clips,
            triggersPolicyApplied = photoPolicy.triggers,
            explicitTimeSecApplied = explicitTime,
            settledClipTimesSec = settledClipTimes,
            triggersArmedState = triggersArmed
        )
    }

    private fun processTriggers(
        dt: Float,
        signalBus: SignalBus,
        faceState: FaceState,
        slots: PropertySlots,
        nowMs: Long,
        disabledBehaviors: Set<String>
    ) {
        val triggeredEvent = faceState.triggeredEvent

        for (trig in triggers) {
            if (disabledBehaviors.contains(trig.id)) continue
            var shouldFire = false

            // Event-driven trigger
            if (trig.onEvent != null) {
                if (triggeredEvent != null && triggeredEvent == trig.onEvent) {
                    if (nowMs - trig.lastFireTimeMs >= trig.cooldownMs) {
                        shouldFire = true
                    }
                }
            } else if (trig.inputSlotIndex != null) {
                // Signal threshold trigger with hysteresis
                val value = signalBus.slotValues[trig.inputSlotIndex]
                val rising = trig.risingAbove ?: Float.MAX_VALUE
                val falling = trig.fallingBelow ?: Float.MIN_VALUE

                if (trig.isArmed && value >= rising) {
                    if (nowMs - trig.lastFireTimeMs >= trig.cooldownMs) {
                        shouldFire = true
                        trig.isArmed = false
                        trig.lastFireTimeMs = nowMs
                    }
                } else if (!trig.isArmed && value <= falling) {
                    trig.isArmed = true
                }
            }

            if (shouldFire) {
                trig.lastFireTimeMs = nowMs
                executeActions(trig.onFire, slots)
            }
        }
    }

    private fun executeActions(actions: List<TriggerAction>, slots: PropertySlots) {
        for (action in actions) {
            when (action.action) {
                "playClip" -> {
                    val clip = clips.find { it.id == action.clipId }
                    if (clip != null) {
                        clip.isPlaying = true
                        clip.currentTimeSec = 0f
                    }
                }
                "stopClip" -> {
                    val clip = clips.find { it.id == action.clipId }
                    clip?.isPlaying = false
                }
                "setProperty" -> {
                    val write = action.propertyWrite
                    if (write != null) {
                        applyWrite(write, action.value, slots)
                    }
                }
                "toggleVisible" -> {
                    val write = action.propertyWrite
                    if (write != null && write.nodeIndex in 0 until nodeCount) {
                        val cur = slots.visible[write.nodeIndex]
                        slots.visible[write.nodeIndex] = if (cur > 0.5f) 0f else 1f
                    }
                }
                "emit" -> {
                    val evt = action.clipId ?: "custom"
                    onCustomEvent?.invoke(evt)
                }
            }
        }
    }

    private fun stepClips(dt: Float, disabledBehaviors: Set<String>, hiddenNodes: Set<Int>) {
        for (clip in clips) {
            if (disabledBehaviors.contains(clip.id) || hiddenNodes.contains(clip.targetNodeIndex)) continue
            if (clip.isPlaying) {
                clip.currentTimeSec += dt * clip.speed
                if (clip.currentTimeSec >= clip.durationSec) {
                    if (clip.loop) {
                        clip.currentTimeSec %= clip.durationSec
                    } else if (clip.holdLast) {
                        clip.currentTimeSec = clip.durationSec
                    } else {
                        clip.isPlaying = false
                    }
                }
            }
        }
    }

    private fun stepMaps(
        dt: Float,
        signalBus: SignalBus,
        slots: PropertySlots,
        disabledBehaviors: Set<String>,
        hiddenNodes: Set<Int>
    ) {
        for (map in maps) {
            if (disabledBehaviors.contains(map.id)) continue
            val rawInput = signalBus.slotValues[map.inputSlotIndex]
            val norm = if (abs(map.inMax - map.inMin) > 1e-6f) {
                ((rawInput - map.inMin) / (map.inMax - map.inMin)).let {
                    if (map.clamp) it.coerceIn(0f, 1f) else it
                }
            } else 0f

            val eased = ease(norm, map.easing)
            val target = map.outMin + (map.outMax - map.outMin) * eased

            // Frame-rate independent exponential smoothing: y += (target - y) * (1 - exp(-dt / tau))
            val tau = map.smoothingTau.coerceAtLeast(0.001f)
            val alpha = 1.0f - exp(-dt / tau)
            map.currentValue += (target - map.currentValue) * alpha

            for (write in map.writes) {
                if (write.nodeIndex >= 0 && hiddenNodes.contains(write.nodeIndex)) continue
                applyWrite(write, map.currentValue, slots)
            }
        }
    }

    private fun stepSprings(
        dt: Float,
        faceState: FaceState,
        slots: PropertySlots,
        disabledBehaviors: Set<String>,
        hiddenNodes: Set<Int>
    ) {
        for (spring in springs) {
            if (disabledBehaviors.contains(spring.id) || hiddenNodes.contains(spring.targetNodeIndex)) continue
            val currentAngle = faceState.rollDeg
            val deltaAngle = currentAngle - spring.prevPoseAngle
            spring.prevPoseAngle = currentAngle

            // Target displacement proportional to angular velocity
            val targetDisp = (deltaAngle * spring.gain).coerceIn(-spring.maxAngleDeg, spring.maxAngleDeg)

            // Integrate spring-damper: a = -stiffness * (pos - target) - 2 * dampingRatio * sqrt(stiffness) * vel
            val k = spring.stiffness
            val c = 2.0f * spring.dampingRatio * sqrt(k)

            var remainingDt = dt
            val substep = spring.substepMaxSec
            while (remainingDt > 0f) {
                val step = min(remainingDt, substep)
                val accel = -k * (spring.pos - targetDisp) - c * spring.vel
                spring.vel += accel * step
                spring.pos += spring.vel * step
                remainingDt -= step
            }

            // Apply spring rotation or position to target node
            val node = spring.targetNodeIndex
            if (node in 0 until nodeCount) {
                if (spring.mode == "rotationInertia") {
                    // Small angle quaternion around Z: [0, 0, sin(theta/2), cos(theta/2)]
                    val rad = Math.toRadians(spring.pos.toDouble() * 0.5).toFloat()
                    val sz = sin(rad); val cz = cos(rad)

                    // Multiply onto existing rotation
                    val qx = slots.rotation[node * 4]
                    val qy = slots.rotation[node * 4 + 1]
                    val qz = slots.rotation[node * 4 + 2]
                    val qw = slots.rotation[node * 4 + 3]

                    slots.rotation[node * 4] = qx * cz + qy * sz
                    slots.rotation[node * 4 + 1] = qy * cz - qx * sz
                    slots.rotation[node * 4 + 2] = qz * cz + qw * sz
                    slots.rotation[node * 4 + 3] = qw * cz - qz * sz
                } else {
                    slots.position[node * 3] += spring.pos
                }
            }
        }
    }

    private fun applyWrite(write: PropertyWrite, value: Float, slots: PropertySlots) {
        when (write.propertyType) {
            PropertyType.PASS_UNIFORM,
            PropertyType.MATERIAL_PARAM,
            PropertyType.MASK_WEIGHT -> {
                val dynIdx = write.dynamicParamSlotIndex
                if (dynIdx in 0 until slots.dynamicParamCount) {
                    when (write.mode) {
                        WriteMode.SET -> slots.dynamicParams[dynIdx] = value
                        WriteMode.MULTIPLY -> slots.dynamicParams[dynIdx] *= value
                        WriteMode.ADD -> slots.dynamicParams[dynIdx] += value
                    }
                }
                return
            }
            else -> Unit
        }

        val node = write.nodeIndex
        if (node !in 0 until nodeCount) return

        when (write.propertyType) {
            PropertyType.POSITION -> {
                val idx = node * 3 + write.componentIndex
                when (write.mode) {
                    WriteMode.SET -> slots.position[idx] = value
                    WriteMode.MULTIPLY -> slots.position[idx] *= value
                    WriteMode.ADD -> slots.position[idx] += value
                }
            }
            PropertyType.SCALE -> {
                val idx = node * 3 + write.componentIndex
                when (write.mode) {
                    WriteMode.SET -> slots.scale[idx] = value
                    WriteMode.MULTIPLY -> slots.scale[idx] *= value
                    WriteMode.ADD -> slots.scale[idx] += value
                }
            }
            PropertyType.ROTATION -> {
                val idx = node * 4 + write.componentIndex
                when (write.mode) {
                    WriteMode.SET -> slots.rotation[idx] = value
                    WriteMode.MULTIPLY -> slots.rotation[idx] *= value
                    WriteMode.ADD -> slots.rotation[idx] += value
                }
            }
            PropertyType.OPACITY -> {
                when (write.mode) {
                    WriteMode.SET -> slots.opacity[node] = value
                    WriteMode.MULTIPLY -> slots.opacity[node] *= value
                    WriteMode.ADD -> slots.opacity[node] += value
                }
            }
            PropertyType.VISIBLE -> {
                slots.visible[node] = if (value > 0.5f) 1f else 0f
            }
            else -> Unit
        }
    }

    private fun ease(t: Float, type: String): Float {
        val c = t.coerceIn(0f, 1f)
        return when (type) {
            "smoothstep" -> c * c * (3f - 2f * c)
            "cubicIn" -> c * c * c
            "cubicOut" -> {
                val f = c - 1f
                f * f * f + 1f
            }
            else -> c
        }
    }
}
