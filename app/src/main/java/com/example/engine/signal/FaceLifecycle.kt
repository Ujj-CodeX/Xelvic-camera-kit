package com.example.engine.signal

/**
 * Configuration for FaceLifecycle state transitions (README Section 5.12).
 */
data class LifecycleConfig(
    val holdMs: Long = 200L,
    val fadeMs: Long = 250L,
    val fadeCurve: String = "cubicOut",
    val afterFade: String = "hideAndReset", // "hideAndReset" | "keep"
    val cancelFade: Boolean = true,
    val easeInMs: Long = 100L,
    val preserveState: Boolean = true,
    val replayStartOn: Boolean = false
) {
    companion object {
        val DEFAULT = LifecycleConfig()
    }
}

/**
 * Complete Face Lifecycle State Machine (README Sections 5.12 & 9).
 *
 * States progression:
 * NO_FACE -> ACQUIRING -> TRACKING -> HOLD -> FADING -> NO_FACE
 *
 * Re-acquisition returns to TRACKING per configured policy.
 */
class FaceLifecycle(
    var config: LifecycleConfig = LifecycleConfig.DEFAULT
) {
    // Convenience secondary constructor for direct parameterization
    constructor(
        holdMs: Long = 200L,
        fadeMs: Long = 250L,
        fadeCurve: String = "cubicOut",
        afterFade: String = "hideAndReset",
        easeInMs: Long = 100L,
        cancelFade: Boolean = true,
        preserveState: Boolean = true,
        replayStartOn: Boolean = false
    ) : this(
        LifecycleConfig(
            holdMs = holdMs,
            fadeMs = fadeMs,
            fadeCurve = fadeCurve,
            afterFade = afterFade,
            cancelFade = cancelFade,
            easeInMs = easeInMs,
            preserveState = preserveState,
            replayStartOn = replayStartOn
        )
    )

    enum class State {
        NO_FACE,
        ACQUIRING,
        TRACKING,
        HOLD,
        FADING
    }

    var currentState: State = State.NO_FACE
        private set

    var fadeWeight: Float = 0f // 1.0f = full visibility, 0.0f = invisible
        private set

    var isVisible: Boolean = false
        private set

    var acquisitionTimestampNs: Long = 0L
        private set

    private var faceLostTimestampNs: Long = 0L
    private var stateEnterTimestampNs: Long = 0L
    private var easeInStartNs: Long = 0L

    fun applyConfig(newConfig: LifecycleConfig) {
        config = newConfig
    }

    fun reset() {
        currentState = State.NO_FACE
        fadeWeight = 0f
        isVisible = false
        acquisitionTimestampNs = 0L
        faceLostTimestampNs = 0L
        stateEnterTimestampNs = 0L
        easeInStartNs = 0L
    }

    /**
     * Update state machine on each frame.
     * @param hasFace True if tracker detected a face in latest snapshot
     * @param nowNs Current monotonic timestamp in nanoseconds
     * @return Event triggered this frame, if any ("faceAcquired", "faceLost", "reset", or null)
     */
    fun update(hasFace: Boolean, nowNs: Long): String? {
        var triggeredEvent: String? = null

        if (hasFace) {
            when (currentState) {
                State.NO_FACE -> {
                    currentState = State.ACQUIRING
                    acquisitionTimestampNs = nowNs
                    stateEnterTimestampNs = nowNs
                    easeInStartNs = nowNs
                    fadeWeight = 0.25f
                    isVisible = true
                    triggeredEvent = "faceAcquired"
                }
                State.ACQUIRING -> {
                    val elapsedMs = (nowNs - easeInStartNs) / 1_000_000L
                    if (elapsedMs >= config.easeInMs) {
                        currentState = State.TRACKING
                        fadeWeight = 1.0f
                    } else {
                        val progress = if (config.easeInMs > 0) {
                            (elapsedMs.toFloat() / config.easeInMs.toFloat()).coerceIn(0.25f, 1f)
                        } else 1.0f
                        fadeWeight = progress
                    }
                    isVisible = true
                }
                State.TRACKING -> {
                    fadeWeight = 1.0f
                    isVisible = true
                }
                State.HOLD, State.FADING -> {
                    if (config.cancelFade) {
                        // Re-acquisition cancels fade immediately
                        currentState = State.TRACKING
                        fadeWeight = 1.0f
                        isVisible = true
                    } else {
                        if (currentState == State.HOLD) {
                            currentState = State.TRACKING
                            fadeWeight = 1.0f
                            isVisible = true
                        }
                    }
                }
            }
        } else {
            // No face detected this frame
            when (currentState) {
                State.TRACKING, State.ACQUIRING -> {
                    currentState = State.HOLD
                    faceLostTimestampNs = nowNs
                    stateEnterTimestampNs = nowNs
                    fadeWeight = 1.0f
                    isVisible = true
                }
                State.HOLD -> {
                    val elapsedMs = (nowNs - faceLostTimestampNs) / 1_000_000L
                    if (elapsedMs >= config.holdMs) {
                        currentState = State.FADING
                        stateEnterTimestampNs = faceLostTimestampNs + (config.holdMs * 1_000_000L)
                        val elapsedFadeMs = (nowNs - stateEnterTimestampNs) / 1_000_000L
                        if (elapsedFadeMs >= config.fadeMs) {
                            currentState = State.NO_FACE
                            fadeWeight = 0.0f
                            isVisible = (config.afterFade == "keep")
                            triggeredEvent = "faceLost"
                        } else {
                            val progress = if (config.fadeMs > 0) {
                                (elapsedFadeMs.toFloat() / config.fadeMs.toFloat()).coerceIn(0f, 1f)
                            } else 1.0f
                            fadeWeight = evaluateCurve(1f - progress, config.fadeCurve)
                            isVisible = true
                        }
                    } else {
                        fadeWeight = 1.0f
                        isVisible = true
                    }
                }
                State.FADING -> {
                    val elapsedFadeMs = (nowNs - stateEnterTimestampNs) / 1_000_000L
                    if (elapsedFadeMs >= config.fadeMs) {
                        currentState = State.NO_FACE
                        fadeWeight = 0.0f
                        isVisible = (config.afterFade == "keep")
                        triggeredEvent = "faceLost"
                    } else {
                        val progress = if (config.fadeMs > 0) {
                            (elapsedFadeMs.toFloat() / config.fadeMs.toFloat()).coerceIn(0f, 1f)
                        } else 1.0f
                        fadeWeight = evaluateCurve(1f - progress, config.fadeCurve)
                        isVisible = true
                    }
                }
                State.NO_FACE -> {
                    fadeWeight = 0f
                    isVisible = (config.afterFade == "keep")
                }
            }
        }

        return triggeredEvent
    }

    private fun evaluateCurve(t: Float, curve: String): Float {
        val clamped = t.coerceIn(0f, 1f)
        return when (curve) {
            "cubicOut" -> {
                val f = clamped - 1f
                f * f * f + 1f
            }
            "linear" -> clamped
            "quadOut" -> 1f - (1f - clamped) * (1f - clamped)
            else -> clamped
        }
    }
}
