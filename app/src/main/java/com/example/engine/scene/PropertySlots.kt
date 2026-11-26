package com.example.engine.scene

/**
 * Property slots represent animated values written by behaviors:
 * - Node properties: position (x, y, z), rotation (qx, qy, qz, qw), scale (sx, sy, sz), opacity, visible
 * - Universal parameter binding slots (Flow 2 — Chunk G0):
 *   - `pass:<passId>.<paramName>` (PASS_UNIFORM)
 *   - `mat:<matId>.<paramName>` (MATERIAL_PARAM)
 *   - `mask:<maskId>.<paramName>` (MASK_WEIGHT)
 *   - `post:<nodeId>.<paramName>` (POST_PARAM)
 */
class PropertySlots @JvmOverloads constructor(
    val nodeCount: Int,
    initialDynamicParams: Map<String, Float> = emptyMap()
) {
    // 3 position floats per node
    val position = FloatArray(nodeCount * 3)
    // 4 rotation floats per node (quaternion)
    val rotation = FloatArray(nodeCount * 4) { if (it % 4 == 3) 1f else 0f }
    // 3 scale floats per node
    val scale = FloatArray(nodeCount * 3) { 1f }
    // 1 opacity float per node
    val opacity = FloatArray(nodeCount) { 1f }
    // 1 visible float per node (1f = visible, 0f = hidden)
    val visible = FloatArray(nodeCount) { 1f }

    // Universal dynamic parameter slots (PASS_UNIFORM, MATERIAL_PARAM, MASK_WEIGHT, POST_PARAM)
    private val mutableParamIndexMap = LinkedHashMap<String, Int>(initialDynamicParams.size.coerceAtLeast(8))
    val dynamicParamIndexMap: Map<String, Int>
        get() = mutableParamIndexMap

    var dynamicParams = FloatArray(initialDynamicParams.size.coerceAtLeast(16))
        private set
    private var dynamicParamDefaults = FloatArray(initialDynamicParams.size.coerceAtLeast(16))

    var dynamicParamCount: Int = 0
        private set

    init {
        for ((key, defaultVal) in initialDynamicParams) {
            registerDynamicParam(key, defaultVal)
        }
    }

    /**
     * Registers a named dynamic parameter slot at compile time (or returns existing index)
     * and records its default reset value.
     */
    fun registerDynamicParam(key: String, defaultValue: Float = 0f): Int {
        val normalized = key.trim()
        val existing = mutableParamIndexMap[normalized]
        if (existing != null) {
            return existing
        }
        val idx = dynamicParamCount++
        if (idx >= dynamicParams.size) {
            val newCap = (dynamicParams.size * 2).coerceAtLeast(16)
            dynamicParams = dynamicParams.copyOf(newCap)
            dynamicParamDefaults = dynamicParamDefaults.copyOf(newCap)
        }
        mutableParamIndexMap[normalized] = idx
        dynamicParams[idx] = defaultValue
        dynamicParamDefaults[idx] = defaultValue
        return idx
    }

    fun getDynamicParam(key: String, fallback: Float = 0f): Float {
        val idx = mutableParamIndexMap[key] ?: return fallback
        return dynamicParams[idx]
    }

    fun setDynamicParam(key: String, value: Float) {
        val idx = registerDynamicParam(key, value)
        dynamicParams[idx] = value
    }

    fun reset() {
        for (i in 0 until nodeCount) {
            position[i * 3] = 0f
            position[i * 3 + 1] = 0f
            position[i * 3 + 2] = 0f

            rotation[i * 4] = 0f
            rotation[i * 4 + 1] = 0f
            rotation[i * 4 + 2] = 0f
            rotation[i * 4 + 3] = 1f

            scale[i * 3] = 1f
            scale[i * 3 + 1] = 1f
            scale[i * 3 + 2] = 1f

            opacity[i] = 1f
            visible[i] = 1f
        }
        for (i in 0 until dynamicParamCount) {
            dynamicParams[i] = dynamicParamDefaults[i]
        }
    }
}
