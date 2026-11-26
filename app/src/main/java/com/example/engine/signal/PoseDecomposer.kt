package com.example.engine.signal

import kotlin.math.*

/**
 * Decomposes a 4x4 facial transformation matrix into translation, quaternion rotation, and scale.
 * Matrices from MediaPipe FaceLandmarker are 4x4 column-major.
 *
 * Implements strict Section 9 contract:
 * - Validate incoming matrix entries (must be 16 finite floats).
 * - Extract translation from 4th column.
 * - Extract scale as column-vector norms.
 * - Extract normalized rotation quaternion with unit norm.
 * - Extract Euler angles (yaw, pitch, roll in degrees).
 * - Recompose TRS into 4x4 column-major matrix.
 */
object PoseDecomposer {

    data class DecomposedPose(
        val translation: FloatArray, // [x, y, z]
        val rotation: FloatArray,    // [qx, qy, qz, qw]
        val scale: FloatArray,       // [sx, sy, sz]
        val yawDeg: Float,
        val pitchDeg: Float,
        val rollDeg: Float
    ) {
        companion object {
            fun create(
                translation: FloatArray,
                rotation: FloatArray,
                scale: FloatArray,
                yawDeg: Float = 0f,
                pitchDeg: Float = 0f,
                rollDeg: Float = 0f
            ): DecomposedPose {
                return DecomposedPose(
                    translation = translation.clone(),
                    rotation = rotation.clone(),
                    scale = scale.clone(),
                    yawDeg = yawDeg,
                    pitchDeg = pitchDeg,
                    rollDeg = rollDeg
                )
            }
        }

        fun hasFiniteValues(): Boolean {
            for (v in translation) if (!v.isFinite()) return false
            for (v in rotation) if (!v.isFinite()) return false
            for (v in scale) if (!v.isFinite()) return false
            return yawDeg.isFinite() && pitchDeg.isFinite() && rollDeg.isFinite()
        }

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as DecomposedPose
            return translation.contentEquals(other.translation) &&
                    rotation.contentEquals(other.rotation) &&
                    scale.contentEquals(other.scale)
        }

        override fun hashCode(): Int {
            var result = translation.contentHashCode()
            result = 31 * result + rotation.contentHashCode()
            result = 31 * result + scale.contentHashCode()
            return result
        }
    }

    /**
     * Validates that the matrix has length 16 and all entries are finite.
     */
    fun isValidPoseMatrix(m: FloatArray?): Boolean {
        if (m == null || m.size != 16) return false
        for (v in m) {
            if (!v.isFinite()) return false
        }
        return true
    }

    /**
     * Decomposes 4x4 column-major matrix into translation, quaternion, and scale.
     * Throws IllegalArgumentException if matrix is invalid or non-finite.
     */
    fun decompose(m: FloatArray): DecomposedPose {
        require(isValidPoseMatrix(m)) { "Pose matrix must contain 16 finite float values" }

        // Translation is 4th column: m[12], m[13], m[14]
        val tx = m[12]
        val ty = m[13]
        val tz = m[14]

        // Basis vectors (columns 0, 1, 2)
        var c0x = m[0]; var c0y = m[1]; var c0z = m[2]
        var c1x = m[4]; var c1y = m[5]; var c1z = m[6]
        var c2x = m[8]; var c2y = m[9]; var c2z = m[10]

        // Scale is magnitude of each column vector
        val sx = sqrt(c0x * c0x + c0y * c0y + c0z * c0z).coerceAtLeast(1e-6f)
        val sy = sqrt(c1x * c1x + c1y * c1y + c1z * c1z).coerceAtLeast(1e-6f)
        val sz = sqrt(c2x * c2x + c2y * c2y + c2z * c2z).coerceAtLeast(1e-6f)

        // Normalize columns to get pure rotation matrix
        c0x /= sx; c0y /= sx; c0z /= sx
        c1x /= sy; c1y /= sy; c1z /= sy
        c2x /= sz; c2y /= sz; c2z /= sz

        // Rotation matrix to Quaternion (Shepperd's algorithm)
        val trace = c0x + c1y + c2z
        var qx: Float
        var qy: Float
        var qz: Float
        var qw: Float

        if (trace > 0f) {
            val s = 0.5f / sqrt(trace + 1.0f)
            qw = 0.25f / s
            qx = (c1z - c2y) * s
            qy = (c2x - c0z) * s
            qz = (c0y - c1x) * s
        } else {
            if (c0x > c1y && c0x > c2z) {
                val s = 2.0f * sqrt(1.0f + c0x - c1y - c2z)
                qw = (c1z - c2y) / s
                qx = 0.25f * s
                qy = (c1x + c0y) / s
                qz = (c2x + c0z) / s
            } else if (c1y > c2z) {
                val s = 2.0f * sqrt(1.0f + c1y - c0x - c2z)
                qw = (c2x - c0z) / s
                qx = (c1x + c0y) / s
                qy = 0.25f * s
                qz = (c2y + c1z) / s
            } else {
                val s = 2.0f * sqrt(1.0f + c2z - c0x - c1y)
                qw = (c0y - c1x) / s
                qx = (c2x + c0z) / s
                qy = (c2y + c1z) / s
                qz = 0.25f * s
            }
        }

        // Normalize quaternion
        val qlen = sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if (qlen > 1e-6f) {
            val inv = 1f / qlen
            qx *= inv; qy *= inv; qz *= inv; qw *= inv
        } else {
            qx = 0f; qy = 0f; qz = 0f; qw = 1f
        }

        // Euler angles from rotation matrix (yaw, pitch, roll in degrees)
        // Pitch = asin(-c2y), Yaw = atan2(c2x, c2z), Roll = atan2(c0y, c1y)
        val pitch = asin((-c2y).coerceIn(-1f, 1f)) * (180f / PI.toFloat())
        val yaw = atan2(c2x, c2z) * (180f / PI.toFloat())
        val roll = atan2(c0y, c1y) * (180f / PI.toFloat())

        return DecomposedPose(
            translation = floatArrayOf(tx, ty, tz),
            rotation = floatArrayOf(qx, qy, qz, qw),
            scale = floatArrayOf(sx, sy, sz),
            yawDeg = yaw,
            pitchDeg = pitch,
            rollDeg = roll
        )
    }

    /**
     * Compute Euler angles (yaw, pitch, roll in degrees) directly from a quaternion [qx, qy, qz, qw].
     * Accepts a preallocated [out] array of length >= 3 for zero-allocation rendering.
     */
    @JvmOverloads
    fun quaternionToEulerDeg(q: FloatArray, out: FloatArray = FloatArray(3)): FloatArray {
        val qx = q[0]; val qy = q[1]; val qz = q[2]; val qw = q[3]

        val xx = qx * qx; val yy = qy * qy; val zz = qz * qz
        val xy = qx * qy; val xz = qx * qz; val yz = qy * qz
        val wx = qw * qx; val wy = qw * qy; val wz = qw * qz

        // Column 2
        val c2x = 2f * (xz + wy)
        val c2y = 2f * (yz - wx)
        val c2z = 1f - 2f * (xx + yy)

        // Column 0 & 1 for roll
        val c0y = 2f * (xy + wz)
        val c1y = 1f - 2f * (xx + zz)

        val pitch = asin((-c2y).coerceIn(-1f, 1f)) * (180f / PI.toFloat())
        val yaw = atan2(c2x, c2z) * (180f / PI.toFloat())
        val roll = atan2(c0y, c1y) * (180f / PI.toFloat())

        out[0] = yaw
        out[1] = pitch
        out[2] = roll
        return out
    }

    /**
     * Recompose TRS into 4x4 column-major matrix.
     */
    fun recompose(
        t: FloatArray,
        q: FloatArray,
        s: FloatArray,
        out: FloatArray = FloatArray(16)
    ): FloatArray {
        require(t.size >= 3 && q.size >= 4 && s.size >= 3 && out.size >= 16) {
            "Invalid buffer sizes for TRS recomposition"
        }

        val qx = q[0]; val qy = q[1]; val qz = q[2]; val qw = q[3]
        val sx = s[0]; val sy = s[1]; val sz = s[2]

        val xx = qx * qx; val yy = qy * qy; val zz = qz * qz
        val xy = qx * qy; val xz = qx * qz; val yz = qy * qz
        val wx = qw * qx; val wy = qw * qy; val wz = qw * qz

        // Column 0
        out[0] = (1f - 2f * (yy + zz)) * sx
        out[1] = (2f * (xy + wz)) * sx
        out[2] = (2f * (xz - wy)) * sx
        out[3] = 0f

        // Column 1
        out[4] = (2f * (xy - wz)) * sy
        out[5] = (1f - 2f * (xx + zz)) * sy
        out[6] = (2f * (yz + wx)) * sy
        out[7] = 0f

        // Column 2
        out[8] = (2f * (xz + wy)) * sz
        out[9] = (2f * (yz - wx)) * sz
        out[10] = (1f - 2f * (xx + yy)) * sz
        out[11] = 0f

        // Column 3
        out[12] = t[0]
        out[13] = t[1]
        out[14] = t[2]
        out[15] = 1f

        return out
    }
}
