package com.example.engine.governor

import com.example.engine.scene.CompiledEffect
import com.example.engine.scene.TierOverride

/**
 * TierPolicy: Evaluates package-defined quality tiers (A, B, C), show/hide node sets,
 * disabled behaviors, disabled passes, pass scale overrides, parameter overrides,
 * textureMaxSize, optional minTier constraints, and single-tier packages
 * (README Section 5.13, 13 & 21 Step 6; Flow 2 — Chunk G0).
 */
object TierPolicy {

    data class EffectiveTierConfig(
        val tierId: String,
        val showNodes: Set<Int>,
        val hideNodes: Set<Int>,
        val disabledBehaviors: Set<String>,
        val textureMaxSize: Int,
        val disabledPasses: Set<String> = emptySet(),
        val passScaleOverrides: Map<String, Float> = emptyMap(),
        val paramOverrides: Map<String, Float> = emptyMap()
    )

    private val TIER_ORDER = listOf(
        PerformanceGovernor.Tier.A,
        PerformanceGovernor.Tier.B,
        PerformanceGovernor.Tier.C
    )

    /**
     * Clamps the governor's requested tier against the package's optional `minTier` constraint,
     * and resolves single-tier or multi-tier package overrides.
     */
    fun resolveEffectiveTier(
        effect: CompiledEffect,
        requestedTier: PerformanceGovernor.Tier,
        governorMaxTextureSize: Int = 1024
    ): EffectiveTierConfig {
        val clampedTier = clampToMinTier(requestedTier, effect.minTier)
        val overrides = effect.tierOverrides

        if (overrides.isEmpty()) {
            // Package defines no explicit tier array: valid single-tier default
            val defaultTexCap = when (clampedTier) {
                PerformanceGovernor.Tier.A -> 1024
                PerformanceGovernor.Tier.B -> 512
                PerformanceGovernor.Tier.C -> 256
            }.coerceAtMost(governorMaxTextureSize)

            return EffectiveTierConfig(
                tierId = clampedTier.name,
                showNodes = emptySet(),
                hideNodes = emptySet(),
                disabledBehaviors = emptySet(),
                textureMaxSize = defaultTexCap,
                disabledPasses = emptySet(),
                passScaleOverrides = emptyMap(),
                paramOverrides = emptyMap()
            )
        }

        // Exact match if present
        val directMatch = overrides[clampedTier.name]
        if (directMatch != null) {
            return toConfig(directMatch, governorMaxTextureSize)
        }

        // Single-tier package or missing intermediate tier: fall back to nearest available tier
        val fallbackOverride = when (clampedTier) {
            PerformanceGovernor.Tier.A -> overrides["A"] ?: overrides["B"] ?: overrides["C"] ?: overrides.values.first()
            PerformanceGovernor.Tier.B -> overrides["B"] ?: overrides["A"] ?: overrides["C"] ?: overrides.values.first()
            PerformanceGovernor.Tier.C -> overrides["C"] ?: overrides["B"] ?: overrides["A"] ?: overrides.values.first()
        }
        return toConfig(fallbackOverride, governorMaxTextureSize)
    }

    private fun toConfig(override: TierOverride, governorMaxTextureSize: Int): EffectiveTierConfig {
        return EffectiveTierConfig(
            tierId = override.tierId,
            showNodes = override.showNodes,
            hideNodes = override.hideNodes,
            disabledBehaviors = override.disabledBehaviors,
            textureMaxSize = override.textureMaxSize.coerceAtMost(governorMaxTextureSize),
            disabledPasses = override.disabledPasses,
            passScaleOverrides = override.passScaleOverrides,
            paramOverrides = override.paramOverrides
        )
    }

    /**
     * Enforces optional package `minTier` (e.g. if minTier is "B", Tier.C is clamped to Tier.B).
     */
    fun clampToMinTier(
        requested: PerformanceGovernor.Tier,
        minTierStr: String?
    ): PerformanceGovernor.Tier {
        if (minTierStr.isNullOrBlank()) return requested
        val minTier = when (minTierStr.trim().uppercase()) {
            "A" -> PerformanceGovernor.Tier.A
            "B" -> PerformanceGovernor.Tier.B
            "C" -> PerformanceGovernor.Tier.C
            else -> return requested
        }
        val reqIdx = TIER_ORDER.indexOf(requested)
        val minIdx = TIER_ORDER.indexOf(minTier)
        // Lower index = higher quality tier (A=0, B=1, C=2). Cannot degrade past minIdx.
        return if (reqIdx > minIdx) minTier else requested
    }

    /**
     * Determines whether a scene node is visible under the active tier override.
     * Hidden nodes in a tier are neither drawn nor animated (README Section 5.13).
     */
    fun isNodeVisibleInTier(
        nodeIndex: Int,
        baseVisible: Boolean,
        tierOverride: TierOverride?
    ): Boolean {
        if (tierOverride == null) return baseVisible
        if (tierOverride.hideNodes.contains(nodeIndex)) return false
        if (tierOverride.showNodes.contains(nodeIndex)) return true
        return baseVisible
    }

    /**
     * Determines whether a behavior is enabled under the active tier override.
     * Disabled behaviors are skipped entirely (README Section 5.13).
     */
    fun isBehaviorEnabledInTier(
        behaviorId: String,
        tierOverride: TierOverride?
    ): Boolean {
        if (tierOverride == null) return true
        return !tierOverride.disabledBehaviors.contains(behaviorId)
    }

    /**
     * Determines whether a pass node in `PassGraph` is enabled under the active tier override
     * (`tiers[tier].disablePasses`).
     */
    fun isPassEnabledInTier(
        passId: String,
        tierOverride: TierOverride?
    ): Boolean {
        if (tierOverride == null) return true
        return !tierOverride.disabledPasses.contains(passId)
    }
}
