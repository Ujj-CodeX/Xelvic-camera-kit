package com.example.engine.signal

/**
 * PoseFilter encapsulates independent One-Euro adaptive low-pass filtering for:
 * 1. Translation vector (x, y, z)
 * 2. Quaternion rotation (qx, qy, qz, qw) with sign continuity
 * 3. Scale vector (sx, sy, sz)
 *
 * Filtering parameters are independently configurable per quantity (README Section 9).
 * Never applies low-pass filtering directly to raw 4x4 matrix entries.
 */
class PoseFilter @JvmOverloads constructor(
    val translationMinCutoff: Float = 1.35f,
    val translationBeta: Float = 38.0f,
    val translationDCutoff: Float = 2.8f,
    val rotationMinCutoff: Float = 1.5f,
    val rotationBeta: Float = 14.0f,
    val rotationDCutoff: Float = 2.8f,
    val scaleMinCutoff: Float = 0.8f,
    val scaleBeta: Float = 0.05f,
    val scaleDCutoff: Float = 1.5f,
    val translationHighVelocityBeta: Float = 1800.0f,
    val translationHighSpeedThreshold: Float = 0.04f,
    val rotationHighVelocityBeta: Float = 260.0f,
    val rotationHighSpeedThreshold: Float = 0.25f
) {
    val translationFilter = OneEuroFilterVec3(
        minCutoff = translationMinCutoff,
        beta = translationBeta,
        dCutoff = translationDCutoff,
        highVelocityBeta = translationHighVelocityBeta,
        highSpeedThreshold = translationHighSpeedThreshold
    )

    val rotationFilter = OneEuroFilterQuaternion(
        minCutoff = rotationMinCutoff,
        beta = rotationBeta,
        dCutoff = rotationDCutoff,
        highVelocityBeta = rotationHighVelocityBeta,
        highSpeedThreshold = rotationHighSpeedThreshold
    )

    val scaleFilter = OneEuroFilterVec3(
        minCutoff = scaleMinCutoff,
        beta = scaleBeta,
        dCutoff = scaleDCutoff
    )

    fun reset() {
        translationFilter.reset()
        rotationFilter.reset()
        scaleFilter.reset()
    }

    /**
     * Filters a decomposed pose at the given monotonic timestamp.
     */
    fun filter(
        decomposed: PoseDecomposer.DecomposedPose,
        timestampNs: Long
    ): PoseDecomposer.DecomposedPose {
        val filteredT = translationFilter.filter(
            decomposed.translation[0],
            decomposed.translation[1],
            decomposed.translation[2],
            timestampNs
        )

        val filteredQ = rotationFilter.filter(
            decomposed.rotation[0],
            decomposed.rotation[1],
            decomposed.rotation[2],
            decomposed.rotation[3],
            timestampNs
        )

        val filteredS = scaleFilter.filter(
            decomposed.scale[0],
            decomposed.scale[1],
            decomposed.scale[2],
            timestampNs
        )

        val euler = PoseDecomposer.quaternionToEulerDeg(filteredQ)

        return PoseDecomposer.DecomposedPose(
            translation = filteredT,
            rotation = filteredQ,
            scale = filteredS,
            yawDeg = euler[0],
            pitchDeg = euler[1],
            rollDeg = euler[2]
        )
    }
}
