package com.example.engine.render

import android.opengl.GLES30
import com.example.engine.pkg.UvRegistry
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import com.example.engine.signal.ProjectionParams
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * FaceMeshRenderer: Live vertex buffer updates for the 468-vertex canonical MediaPipe face mesh
 * (README Section 5.7, 5.8, 10 Stage 11, 11.4, 21 Step 8).
 *
 * - Uploads static triangle topology (`CANONICAL_TRIANGLES`, 898 triangles = 2694 `uint16` indices)
 *   and static UV coordinates + vertex region weights (`CANONICAL_UVS` + `VERTEX_PROTECTION_WEIGHTS`)
 *   ONLY ONCE during `init()`.
 * - Updates only dynamic 3D vertex positions (`positionVbo`, 468 * 3 floats) every frame via
 *   `glBufferSubData` using a pre-allocated direct `FloatBuffer` with zero frame-loop allocations.
 * - Shares a single canonical unprojection function ([unprojectLandmarksToCameraSpace]) and rigid
 *   pose-correction helper ([computePoseCorrectedMvp]) with [OccluderPass] so both the face-mesh
 *   renderer and the face-mesh depth occluder produce identical camera-space and clip-space vertices.
 * - Tracks all GL buffers (`vao`, `positionVbo`, `uvVbo`, `ibo`) with `GlResourceOwner`.
 */
class FaceMeshRenderer {

    companion object {
        /**
         * Shared Stage 11 landmark-to-camera-space unprojection (README Section 10 Stage 11):
         *
         * Converts the 468 normalized analysis-crop landmarks `(lx, ly, lz)` from [faceState]
         * into metric 3D camera space (`+X` right, `+Y` up, `-Z` into screen) at the landmark-time
         * pose depth `rawTz = faceState.landmarkPoseMatrix[14]` using the ANALYSIS-CROP vertical FOV
         * (`faceState.projectionParams.fovYDeg`) and crop aspect ratio (`cropAspect`).
         *
         * Explicit Symbol Definitions:
         * - `fovYCropDeg`: Vertical field-of-view (degrees) of the analysis crop in which landmarks
         *   are normalized (`[0..1]`).
         * - `cropAspect`: Width / height aspect ratio of the oriented analysis crop (`cropW / cropH`).
         * - `f = 1 / tan(fovYCropDeg / 2)`: Dimensionless vertical focal factor for the analysis crop.
         * - `absZ`: Clamped positive camera-space depth `|landmarkPoseMatrix[14]|` in meters (`[0.25m, 1.20m]`).
         * - `wCam = 2 * absZ * cropAspect / f`: Metric width (meters) of the analysis frustum at depth `-absZ`.
         * - `hCam = 2 * absZ / f`: Metric height (meters) of the analysis frustum at depth `-absZ`.
         * - `fw`: Metric inter-cheek face width (`|#234 - #454|`) in meters (`>= 0.08m`).
         * - `pushBackFW`: View-ray pushback distance in `FW` units (`0f` for rendered face texture mesh;
         *   `>= 0f` for the depth-only face-mesh occluder). Moves each vertex $v$ radially away from
         *   the camera origin $(0, 0, 0)$ along its view ray:
         *   $$v' = v \cdot \left(1 + \frac{\text{pushBackFW} \cdot \text{fw}}{\|v\|}\right)$$
         *   preserving screen-space $(x_{\text{ndc}}, y_{\text{ndc}})$ while increasing depth.
         *
         * Note: Node offset and scale are ignored/forbidden (identity) for the face mesh and
         * face-mesh occluder so camera-space vertices never drift from tracked facial features.
         * Zero heap allocations are performed on this path.
         */
        @JvmOverloads
        fun unprojectLandmarksToCameraSpace(
            faceState: FaceState,
            viewportAspect: Float = 9f / 16f,
            pushBackFW: Float = 0f,
            outPositions: FloatArray,
            outBuffer: FloatBuffer? = null
        ): Boolean {
            val lm = faceState.landmarks ?: return false
            val count = UvRegistry.VERTEX_COUNT
            if (lm.size < count * 3 || outPositions.size < count * 3) return false

            val viewAspect = viewportAspect.coerceAtLeast(0.01f)
            val projParams = faceState.projectionParams
            val cropAspect = if (projParams !== ProjectionParams.DEFAULT) {
                projParams.cropAspectRatio.coerceAtLeast(0.01f)
            } else {
                viewAspect
            }
            val fovYCropDeg = projParams.fovYDeg
            val f = FacePoseAdapter.focalFactorFromFovY(fovYCropDeg)

            // Landmarks correspond to the tracker-time unfiltered pose (landmarkPoseMatrix)
            val rawTz = faceState.landmarkPoseMatrix[14].let {
                if (abs(it) > 0.05f) it else faceState.poseMatrix[14]
            }
            val absZ = (if (abs(rawTz) > 0.05f) abs(rawTz) else 0.50f).coerceIn(0.25f, 1.20f)

            val wCam = 2.0f * absZ * cropAspect / f
            val hCam = 2.0f * absZ / f
            val fw = faceState.faceWidth.coerceAtLeast(0.08f)
            val pushDistMeters = pushBackFW * fw

            for (i in 0 until count) {
                val baseIdx = i * 3
                val lx = lm[baseIdx]
                val ly = lm[baseIdx + 1]
                val lz = lm[baseIdx + 2]

                var vx = (lx - 0.5f) * wCam
                var vy = -(ly - 0.5f) * hCam
                // MediaPipe landmark Z is scaled relative to image width; negative Z is closer to the camera (+Z in GL)
                var vz = -absZ - (lz * wCam)

                if (pushDistMeters != 0f) {
                    val rayLen = sqrt(vx * vx + vy * vy + vz * vz).coerceAtLeast(1e-6f)
                    val rayScale = 1.0f + (pushDistMeters / rayLen)
                    vx *= rayScale
                    vy *= rayScale
                    vz *= rayScale
                }

                outPositions[baseIdx] = vx
                outPositions[baseIdx + 1] = vy
                outPositions[baseIdx + 2] = vz
            }

            if (outBuffer != null) {
                outBuffer.clear()
                outBuffer.put(outPositions, 0, count * 3)
                (outBuffer as java.nio.Buffer).position(0)
            }
            return true
        }

        /**
         * Unprojects a single landmark `landmarkId` into metric camera space via the shared
         * [FacePoseAdapter.unprojectLandmarkToCameraSpace] implementation.
         */
        @JvmOverloads
        fun unprojectSingleLandmarkToCameraSpace(
            faceState: FaceState,
            landmarkId: Int,
            viewportAspect: Float = 9f / 16f,
            pushBackFW: Float = 0f,
            outCam: FloatArray
        ): Boolean {
            return FacePoseAdapter.unprojectLandmarkToCameraSpace(
                faceState = faceState,
                landmarkId = landmarkId,
                viewportAspect = viewportAspect,
                pushBackFW = pushBackFW,
                outCam = outCam
            )
        }

        /**
         * Pure-Kotlin zero-allocation rigid inverse of a 4x4 column-major rigid pose matrix [m]:
         * $$M = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix} \implies M^{-1} = \begin{bmatrix} R^T & -R^T t \\ 0 & 1 \end{bmatrix}$$
         * Normalizes the columns of $R$ if any numerical scale drift is present so $R^T$ is exact.
         */
        fun invertRigidPoseMatrix(m: FloatArray, outInv: FloatArray) {
            // Column 0 of R
            var r00 = m[0]; var r10 = m[1]; var r20 = m[2]
            val n0 = sqrt(r00 * r00 + r10 * r10 + r20 * r20)
            if (n0 > 1e-6f) { val inv0 = 1f / n0; r00 *= inv0; r10 *= inv0; r20 *= inv0 }

            // Column 1 of R
            var r01 = m[4]; var r11 = m[5]; var r21 = m[6]
            val n1 = sqrt(r01 * r01 + r11 * r11 + r21 * r21)
            if (n1 > 1e-6f) { val inv1 = 1f / n1; r01 *= inv1; r11 *= inv1; r21 *= inv1 }

            // Column 2 of R
            var r02 = m[8]; var r12 = m[9]; var r22 = m[10]
            val n2 = sqrt(r02 * r02 + r12 * r12 + r22 * r22)
            if (n2 > 1e-6f) { val inv2 = 1f / n2; r02 *= inv2; r12 *= inv2; r22 *= inv2 }

            val tx = m[12]
            val ty = m[13]
            val tz = m[14]

            // Transpose 3x3 rotation block into outInv (column-major)
            outInv[0] = r00; outInv[1] = r01; outInv[2] = r02; outInv[3] = 0f
            outInv[4] = r10; outInv[5] = r11; outInv[6] = r12; outInv[7] = 0f
            outInv[8] = r20; outInv[9] = r21; outInv[10] = r22; outInv[11] = 0f

            // -R^T * t
            outInv[12] = -(r00 * tx + r10 * ty + r20 * tz)
            outInv[13] = -(r01 * tx + r11 * ty + r21 * tz)
            outInv[14] = -(r02 * tx + r12 * ty + r22 * tz)
            outInv[15] = 1f
        }

        /**
         * Pure-Kotlin zero-allocation 4x4 column-major matrix multiplication (`out = lhs * rhs`).
         * Safe to call when `out` aliases neither `lhs` nor `rhs`.
         */
        fun multiply4x4(lhs: FloatArray, rhs: FloatArray, out: FloatArray) {
            for (col in 0 until 4) {
                val cBase = col * 4
                val r0 = rhs[cBase]
                val r1 = rhs[cBase + 1]
                val r2 = rhs[cBase + 2]
                val r3 = rhs[cBase + 3]
                out[cBase] = lhs[0] * r0 + lhs[4] * r1 + lhs[8] * r2 + lhs[12] * r3
                out[cBase + 1] = lhs[1] * r0 + lhs[5] * r1 + lhs[9] * r2 + lhs[13] * r3
                out[cBase + 2] = lhs[2] * r0 + lhs[6] * r1 + lhs[10] * r2 + lhs[14] * r3
                out[cBase + 3] = lhs[3] * r0 + lhs[7] * r1 + lhs[11] * r2 + lhs[15] * r3
            }
        }

        /**
         * Computes the rigid pose-delta correction matrix:
         * $$T = M_{\text{pred}} \cdot M_{\text{pose\_at\_landmark\_time}}^{-1}$$
         * with zero heap allocations.
         */
        fun computeRigidPoseCorrection(
            predPoseMatrix: FloatArray,
            landmarkPoseMatrix: FloatArray,
            outCorrectionT: FloatArray,
            scratchInvLandmark: FloatArray
        ) {
            invertRigidPoseMatrix(landmarkPoseMatrix, scratchInvLandmark)
            multiply4x4(predPoseMatrix, scratchInvLandmark, outCorrectionT)
        }

        /**
         * Folds the rigid pose correction $T = M_{\text{pred}} \cdot M_{\text{landmark}}^{-1}$ into
         * the MVP uniform (`outMvp = viewProjectionMatrix * T`) so that unfiltered landmark vertices
         * follow the exact same filtered+predicted rigid pose as 3D props while preserving
         * non-rigid expression deformation.
         */
        fun computePoseCorrectedMvp(
            viewProjectionMatrix: FloatArray,
            faceState: FaceState,
            outMvp: FloatArray,
            scratchInvLandmark: FloatArray,
            scratchRigidDelta: FloatArray
        ) {
            computeRigidPoseCorrection(
                predPoseMatrix = faceState.poseMatrix,
                landmarkPoseMatrix = faceState.landmarkPoseMatrix,
                outCorrectionT = scratchRigidDelta,
                scratchInvLandmark = scratchInvLandmark
            )
            multiply4x4(viewProjectionMatrix, scratchRigidDelta, outMvp)
        }

        /**
         * Transforms 3D camera-space vertices (`count * 3` floats) through `mvpMatrix` (4x4 column-major)
         * into 4D clip-space coordinates `[cx, cy, cz, cw]` (`count * 4` floats).
         */
        @JvmOverloads
        fun computeClipSpacePositions(
            cameraPositions: FloatArray,
            mvpMatrix: FloatArray,
            outClipPositions: FloatArray = FloatArray(UvRegistry.VERTEX_COUNT * 4)
        ): FloatArray {
            val vertCount = minOf(UvRegistry.VERTEX_COUNT, cameraPositions.size / 3, outClipPositions.size / 4)
            val m0 = mvpMatrix[0]; val m4 = mvpMatrix[4]; val m8 = mvpMatrix[8]; val m12 = mvpMatrix[12]
            val m1 = mvpMatrix[1]; val m5 = mvpMatrix[5]; val m9 = mvpMatrix[9]; val m13 = mvpMatrix[13]
            val m2 = mvpMatrix[2]; val m6 = mvpMatrix[6]; val m10 = mvpMatrix[10]; val m14 = mvpMatrix[14]
            val m3 = mvpMatrix[3]; val m7 = mvpMatrix[7]; val m11 = mvpMatrix[11]; val m15 = mvpMatrix[15]

            for (i in 0 until vertCount) {
                val vBase = i * 3
                val x = cameraPositions[vBase]
                val y = cameraPositions[vBase + 1]
                val z = cameraPositions[vBase + 2]

                val cBase = i * 4
                outClipPositions[cBase] = m0 * x + m4 * y + m8 * z + m12
                outClipPositions[cBase + 1] = m1 * x + m5 * y + m9 * z + m13
                outClipPositions[cBase + 2] = m2 * x + m6 * y + m10 * z + m14
                outClipPositions[cBase + 3] = m3 * x + m7 * y + m11 * z + m15
            }
            return outClipPositions
        }

        fun resolveMouthOpenWeight(faceState: FaceState?): Float {
            if (faceState == null) return 0f
            return maxOf(
                (faceState.blendshapes["jawOpen"] ?: 0f).coerceIn(0f, 1f),
                (com.example.engine.pkg.FaceRegions.computeMouthOpeningRatio(faceState.landmarks) * 2.5f).coerceIn(0f, 1f)
            )
        }

        fun resolveEyeBlinkWeight(faceState: FaceState?): Float {
            if (faceState == null) return 0f
            val bL = (faceState.blendshapes["eyeBlinkLeft"] ?: 0f).coerceIn(0f, 1f)
            val bR = (faceState.blendshapes["eyeBlinkRight"] ?: 0f).coerceIn(0f, 1f)
            return ((bL + bR) * 0.5f).coerceIn(0f, 1f)
        }

        @JvmOverloads
        fun blendRgbWithBase(
            baseRgb: FloatArray,
            srcRgb: FloatArray,
            alpha: Float,
            blendMode: BlendMode,
            outRgb: FloatArray = FloatArray(3)
        ): FloatArray {
            return BlendMode.blendRgbWithBase(
                baseR = baseRgb.getOrElse(0) { 0f },
                baseG = baseRgb.getOrElse(1) { 0f },
                baseB = baseRgb.getOrElse(2) { 0f },
                srcR = srcRgb.getOrElse(0) { 0f },
                srcG = srcRgb.getOrElse(1) { 0f },
                srcB = srcRgb.getOrElse(2) { 0f },
                effectiveAlpha = alpha,
                mode = blendMode,
                outRgb = outRgb
            )
        }
    }

    private var vao = 0
    private var positionVbo = 0
    private var uvVbo = 0
    private var ibo = 0

    // Pre-allocated direct buffer for zero-allocation per-frame vertex updates (468 * 3 floats)
    private val positionBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(UvRegistry.VERTEX_COUNT * 3 * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()

    // CPU-side copy of the latest 3D vertex positions in metric camera space (for inspection & unit tests)
    val livePositions = FloatArray(UvRegistry.VERTEX_COUNT * 3)

    // Pre-allocated 16-float matrices for rigid pose correction T = M_pred * inv(M_landmark) and uMVP
    private val scratchInvLandmark = FloatArray(16)
    val lastRigidCorrectionMatrix = FloatArray(16).apply {
        this[0] = 1f; this[5] = 1f; this[10] = 1f; this[15] = 1f
    }
    val lastEffectiveMvp = FloatArray(16).apply {
        this[0] = 1f; this[5] = 1f; this[10] = 1f; this[15] = 1f
    }

    var indexCount = 0
        private set

    // Deterministic telemetry counters verifying static topology/UV reuse vs live vertex updates
    var topologyUploadCount = 0
        private set
    var uvUploadCount = 0
        private set
    var vertexUpdateCount = 0
        private set
    var renderCallCount = 0
        private set

    var isInitialized = false
        private set

    var lastRenderedOpacity: Float = 0f
        private set
    var lastRenderedTextureId: Int = 0
        private set
    var lastRenderedMaskId: Int = 0
        private set
    var lastFrontFaceMode: Int = GLES30.GL_CCW
        private set
    var lastUsedFovYCropDeg: Float = ProjectionParams.DEFAULT_FOV_Y_DEG
        private set
    var lastUsedFovYViewDeg: Float = ProjectionParams.DEFAULT_FOV_Y_DEG
        private set

    @JvmOverloads
    fun init(resourceOwner: GlResourceOwner? = null) {
        val uvs = UvRegistry.CANONICAL_UVS
        val tris = UvRegistry.CANONICAL_TRIANGLES
        val protectionWeights = UvRegistry.VERTEX_PROTECTION_WEIGHTS

        require(uvs.size == UvRegistry.VERTEX_COUNT * 2) {
            "Canonical UV array size mismatch: ${uvs.size}"
        }
        require(tris.size == UvRegistry.INDEX_COUNT) {
            "Canonical triangle index count mismatch: ${tris.size}"
        }

        indexCount = tris.size
        val vertCount = UvRegistry.VERTEX_COUNT

        // Static UV + region-protection buffer: 3 floats per vertex [u, v, protectionWeight]
        val staticUvData = FloatArray(vertCount * 3)
        for (i in 0 until vertCount) {
            staticUvData[i * 3] = uvs[i * 2]
            staticUvData[i * 3 + 1] = uvs[i * 2 + 1]
            staticUvData[i * 3 + 2] = protectionWeights[i]
        }

        // Initial zeroed positions
        livePositions.fill(0f)
        positionBuffer.clear()
        positionBuffer.put(livePositions)
        (positionBuffer as java.nio.Buffer).position(0)

        vao = 0
        positionVbo = 0
        uvVbo = 0
        ibo = 0
        try {
            val vaos = IntArray(1)
            val vbos = IntArray(2)
            val ibos = IntArray(1)
            GLES30.glGenVertexArrays(1, vaos, 0)
            GLES30.glGenBuffers(2, vbos, 0)
            GLES30.glGenBuffers(1, ibos, 0)

            vao = vaos[0]
            positionVbo = vbos[0]
            uvVbo = vbos[1]
            ibo = ibos[0]

            resourceOwner?.trackVao(vao)
            resourceOwner?.trackBuffer(positionVbo)
            resourceOwner?.trackBuffer(uvVbo)
            resourceOwner?.trackBuffer(ibo)

            if (vao != 0) {
                GLES30.glBindVertexArray(vao)

                // 1. Dynamic position VBO (location = 0, vec3)
                GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, positionVbo)
                GLES30.glBufferData(
                    GLES30.GL_ARRAY_BUFFER,
                    vertCount * 3 * 4,
                    positionBuffer,
                    GLES30.GL_DYNAMIC_DRAW
                )
                GLES30.glEnableVertexAttribArray(0)
                GLES30.glVertexAttribPointer(0, 3, GLES30.GL_FLOAT, false, 3 * 4, 0)

                // 2. Static UV + RegionMask VBO (location = 1: vec2 UV, location = 2: float regionMask)
                GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, uvVbo)
                val uvBuf = ByteBuffer
                    .allocateDirect(staticUvData.size * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer()
                uvBuf.put(staticUvData)
                (uvBuf as java.nio.Buffer).position(0)
                GLES30.glBufferData(
                    GLES30.GL_ARRAY_BUFFER,
                    staticUvData.size * 4,
                    uvBuf,
                    GLES30.GL_STATIC_DRAW
                )
                val uvStride = 3 * 4
                GLES30.glEnableVertexAttribArray(1)
                GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, uvStride, 0)
                GLES30.glEnableVertexAttribArray(2)
                GLES30.glVertexAttribPointer(2, 1, GLES30.GL_FLOAT, false, uvStride, 2 * 4)

                // 3. Static Triangle Index Buffer (uint16)
                GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, ibo)
                val sb = ByteBuffer
                    .allocateDirect(tris.size * 2)
                    .order(ByteOrder.nativeOrder())
                    .asShortBuffer()
                sb.put(tris)
                (sb as java.nio.Buffer).position(0)
                GLES30.glBufferData(
                    GLES30.GL_ELEMENT_ARRAY_BUFFER,
                    tris.size * 2,
                    sb,
                    GLES30.GL_STATIC_DRAW
                )

                GLES30.glBindVertexArray(0)
                GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
            }
        } catch (_: Throwable) {
            // Headless JVM unit test environment where GLES30 native context is not active
        }

        topologyUploadCount++
        uvUploadCount++
        isInitialized = true
    }

    /**
     * Updates the 468 dynamic vertex positions from the current [faceState] without re-uploading
     * static triangle indices or UV coordinates, and with zero heap allocations.
     *
     * Delegates to [unprojectLandmarksToCameraSpace] with `pushBackFW = 0f` (identity node transform).
     */
    @JvmOverloads
    fun updateVertices(
        faceState: FaceState,
        viewportAspect: Float = 9f / 16f
    ): Boolean {
        val viewAspect = viewportAspect.coerceAtLeast(0.01f)
        val projParams = faceState.projectionParams
        val cropAspect = if (projParams !== ProjectionParams.DEFAULT) {
            projParams.cropAspectRatio.coerceAtLeast(0.01f)
        } else {
            viewAspect
        }
        val fovYCropDeg = projParams.fovYDeg
        val fovYViewDeg = FacePoseAdapter.computeVisibleViewFovYDeg(
            fovYCropDeg = fovYCropDeg,
            cropAspectRatio = cropAspect,
            viewAspectRatio = viewAspect
        )
        lastUsedFovYCropDeg = fovYCropDeg
        lastUsedFovYViewDeg = fovYViewDeg

        val ok = unprojectLandmarksToCameraSpace(
            faceState = faceState,
            viewportAspect = viewAspect,
            pushBackFW = 0f,
            outPositions = livePositions,
            outBuffer = positionBuffer
        )
        if (!ok) return false

        val count = UvRegistry.VERTEX_COUNT
        if (positionVbo != 0) {
            try {
                GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, positionVbo)
                GLES30.glBufferSubData(
                    GLES30.GL_ARRAY_BUFFER,
                    0,
                    count * 3 * 4,
                    positionBuffer
                )
                GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
            } catch (_: Throwable) {
                // Ignore in headless JVM test
            }
        }

        vertexUpdateCount++
        return true
    }

    /**
     * Reads back the live 3D camera-space position `[x, y, z]` of a specific landmark vertex index (`0..467`).
     */
    fun getLiveVertexPosition(vertexIndex: Int, out: FloatArray = FloatArray(3)): FloatArray {
        require(vertexIndex in 0 until UvRegistry.VERTEX_COUNT) {
            "Vertex index out of range: $vertexIndex"
        }
        val base = vertexIndex * 3
        out[0] = livePositions[base]
        out[1] = livePositions[base + 1]
        out[2] = livePositions[base + 2]
        return out
    }

    enum class BlendMode(val shaderCode: Int, val canonicalName: String) {
        ALPHA(0, "alpha"),
        MULTIPLY(1, "multiply"),
        SOFT_LIGHT(2, "softLight"),
        SCREEN(3, "screen"),
        ADD(4, "add"),
        OPAQUE(5, "opaque");

        val requiresCameraPixels: Boolean
            get() = this == MULTIPLY || this == SOFT_LIGHT || this == SCREEN

        companion object {
            fun fromString(raw: String?): BlendMode {
                return when (raw?.trim()?.lowercase()) {
                    "multiply", "mul" -> MULTIPLY
                    "softlight", "soft_light", "soft-light" -> SOFT_LIGHT
                    "screen" -> SCREEN
                    "add", "additive" -> ADD
                    "opaque", "none" -> OPAQUE
                    else -> ALPHA
                }
            }

            /**
             * Evaluates a single channel `[0, 1]` under [mode] (`ALPHA`, `MULTIPLY`, `SOFT_LIGHT`, `SCREEN`, `ADD`, `OPAQUE`).
             */
            fun applyChannel(base: Float, src: Float, mode: BlendMode): Float {
                val b = base.coerceIn(0f, 1f)
                val s = src.coerceIn(0f, 1f)
                return when (mode) {
                    ALPHA, OPAQUE -> s
                    MULTIPLY -> (b * s).coerceIn(0f, 1f)
                    SOFT_LIGHT -> ((1f - 2f * s) * b * b + 2f * s * b).coerceIn(0f, 1f)
                    SCREEN -> (1f - (1f - b) * (1f - s)).coerceIn(0f, 1f)
                    ADD -> (b + s).coerceIn(0f, 1f)
                }
            }

            /**
             * Blends `srcRgb` onto `baseRgb` with `effectiveAlpha` using [mode] and writes `[r, g, b]` into `outRgb`.
             */
            fun blendRgbWithBase(
                baseR: Float,
                baseG: Float,
                baseB: Float,
                srcR: Float,
                srcG: Float,
                srcB: Float,
                effectiveAlpha: Float,
                mode: BlendMode,
                outRgb: FloatArray = FloatArray(3)
            ): FloatArray {
                val a = effectiveAlpha.coerceIn(0f, 1f)
                val bR = baseR.coerceIn(0f, 1f)
                val bG = baseG.coerceIn(0f, 1f)
                val bB = baseB.coerceIn(0f, 1f)
                val blendedR = applyChannel(bR, srcR, mode)
                val blendedG = applyChannel(bG, srcG, mode)
                val blendedB = applyChannel(bB, srcB, mode)
                outRgb[0] = (bR + (blendedR - bR) * a).coerceIn(0f, 1f)
                outRgb[1] = (bG + (blendedG - bG) * a).coerceIn(0f, 1f)
                outRgb[2] = (bB + (blendedB - bB) * a).coerceIn(0f, 1f)
                return outRgb
            }
        }
    }

    var lastRenderedBlendMode: BlendMode = BlendMode.ALPHA
        private set
    var lastRenderedCullEnabled: Boolean = false
        private set
    var lastRenderedMouthOpen: Float = 0f
        private set
    var lastRenderedEyeBlink: Float = 0f
        private set

    /**
     * Evaluates the effective premultiplied-alpha output color `[r, g, b, a]` for a given UV sample,
     * texture sample, mask sample, opacity, blend mode (`alpha`, `multiply`, `softLight`, `screen`),
     * dynamic mouth/blink state, and region-protection flags, matching `FS_FACE_TEXTURE`.
     */
    @JvmOverloads
    fun evaluateEffectiveFragment(
        u: Float,
        v: Float,
        texRgba: FloatArray,
        isTexPremultiplied: Boolean = true,
        maskValue: Float = 1.0f,
        materialOpacity: Float = 1.0f,
        tintColor: FloatArray = floatArrayOf(1f, 1f, 1f, 1f),
        protectEyes: Boolean = true,
        protectLips: Boolean = true,
        protectMouthInterior: Boolean = true,
        blendMode: String = "alpha",
        cameraBaseRgb: FloatArray? = null,
        mouthOpenWeight: Float = 0f,
        eyeBlinkWeight: Float = 0f
    ): FloatArray {
        var regionMask = UvRegistry.evaluateRegionProtectionMask(
            u = u,
            v = v,
            protectEyes = protectEyes,
            protectLips = protectLips,
            protectMouthInterior = protectMouthInterior
        )
        // Dynamic mouth_inner subtraction when mouth opens so face paint never leaks onto teeth
        if (protectMouthInterior && mouthOpenWeight > 0.02f) {
            val mouthRadiiY = 0.025f + mouthOpenWeight.coerceIn(0f, 1f) * 0.045f
            val dx = (u - 0.5000f) / 0.098f
            val dy = (v - 0.6939f) / mouthRadiiY
            val dist = kotlin.math.sqrt(dx * dx + dy * dy)
            val t = ((dist - 0.85f) / 0.30f).coerceIn(0f, 1f)
            val dynamicMouthCut = t * t * (3f - 2f * t)
            regionMask = minOf(regionMask, dynamicMouthCut)
        }
        // Dynamic eyelid blink attenuation when blinking
        if (protectEyes && eyeBlinkWeight > 0.02f && regionMask < 0.85f) {
            regionMask *= (1f - eyeBlinkWeight.coerceIn(0f, 1f))
        }

        val clampedMask = maskValue.coerceIn(0f, 1f)
        val clampedOpacity = materialOpacity.coerceIn(0f, 1f)
        val tintA = tintColor.getOrElse(3) { 1f }.coerceIn(0f, 1f)
        val effectiveMask = (clampedMask * regionMask * clampedOpacity * tintA).coerceIn(0f, 1f)

        val texA = texRgba.getOrElse(3) { 1f }.coerceIn(0f, 1f)
        val outA = (texA * effectiveMask).coerceIn(0f, 1f)
        val mode = BlendMode.fromString(blendMode)

        if (cameraBaseRgb != null && cameraBaseRgb.size >= 3 && mode.requiresCameraPixels) {
            val safeA = texA.coerceAtLeast(0.0001f)
            val unpremulR = (if (isTexPremultiplied) texRgba[0] / safeA else texRgba[0]).coerceIn(0f, 1f)
            val unpremulG = (if (isTexPremultiplied) texRgba[1] / safeA else texRgba[1]).coerceIn(0f, 1f)
            val unpremulB = (if (isTexPremultiplied) texRgba[2] / safeA else texRgba[2]).coerceIn(0f, 1f)
            val srcR = (unpremulR * tintColor.getOrElse(0) { 1f }).coerceIn(0f, 1f)
            val srcG = (unpremulG * tintColor.getOrElse(1) { 1f }).coerceIn(0f, 1f)
            val srcB = (unpremulB * tintColor.getOrElse(2) { 1f }).coerceIn(0f, 1f)

            val blended = BlendMode.blendRgbWithBase(
                baseR = cameraBaseRgb[0],
                baseG = cameraBaseRgb[1],
                baseB = cameraBaseRgb[2],
                srcR = srcR,
                srcG = srcG,
                srcB = srcB,
                effectiveAlpha = outA,
                mode = mode
            )
            return floatArrayOf(blended[0] * outA, blended[1] * outA, blended[2] * outA, outA)
        }

        val baseR = if (isTexPremultiplied) texRgba[0] else texRgba[0] * texA
        val baseG = if (isTexPremultiplied) texRgba[1] else texRgba[1] * texA
        val baseB = if (isTexPremultiplied) texRgba[2] else texRgba[2] * texA

        val outR = baseR * tintColor.getOrElse(0) { 1f } * effectiveMask
        val outG = baseG * tintColor.getOrElse(1) { 1f } * effectiveMask
        val outB = baseB * tintColor.getOrElse(2) { 1f } * effectiveMask

        return floatArrayOf(outR, outG, outB, outA)
    }

    @JvmOverloads
    fun render(
        viewProjectionMatrix: FloatArray,
        textureId: Int,
        maskId: Int?,
        opacity: Float,
        program: Int,
        tintColor: FloatArray? = null,
        protectEyes: Boolean = true,
        protectLips: Boolean = true,
        protectMouthInterior: Boolean = true,
        texturePremultiplied: Boolean = true,
        frontFacing: Boolean = false,
        faceState: FaceState? = null,
        blendMode: String = "alpha",
        cull: Boolean = false,
        cameraTextureId: Int = 0
    ) {
        val clampedOpacity = opacity.coerceIn(0f, 1f)
        val parsedBlend = BlendMode.fromString(blendMode)
        lastRenderedOpacity = clampedOpacity
        lastRenderedTextureId = textureId
        lastRenderedMaskId = maskId ?: 0
        lastRenderedBlendMode = parsedBlend
        lastRenderedCullEnabled = cull
        val frontFaceMode = FacePoseAdapter.glFrontFaceForCamera(frontFacing)
        lastFrontFaceMode = frontFaceMode
        renderCallCount++

        val mouthOpen = if (faceState != null) {
            maxOf(
                (faceState.blendshapes["jawOpen"] ?: 0f).coerceIn(0f, 1f),
                (com.example.engine.pkg.FaceRegions.computeMouthOpeningRatio(faceState.landmarks) * 2.5f).coerceIn(0f, 1f)
            )
        } else {
            0f
        }
        val eyeBlink = if (faceState != null) {
            val bL = (faceState.blendshapes["eyeBlinkLeft"] ?: 0f).coerceIn(0f, 1f)
            val bR = (faceState.blendshapes["eyeBlinkRight"] ?: 0f).coerceIn(0f, 1f)
            ((bL + bR) * 0.5f).coerceIn(0f, 1f)
        } else {
            0f
        }
        lastRenderedMouthOpen = mouthOpen
        lastRenderedEyeBlink = eyeBlink

        if (faceState != null) {
            computePoseCorrectedMvp(
                viewProjectionMatrix = viewProjectionMatrix,
                faceState = faceState,
                outMvp = lastEffectiveMvp,
                scratchInvLandmark = scratchInvLandmark,
                scratchRigidDelta = lastRigidCorrectionMatrix
            )
        } else {
            System.arraycopy(viewProjectionMatrix, 0, lastEffectiveMvp, 0, 16)
        }

        if (vao == 0 || indexCount == 0 || program == 0 || clampedOpacity <= 0.0001f) return

        GLES30.glFrontFace(frontFaceMode)
        if (cull) {
            GLES30.glEnable(GLES30.GL_CULL_FACE)
            GLES30.glCullFace(GLES30.GL_BACK)
        } else {
            // Default for faceTexture@1: back-face culling disabled so +/-40 deg yaw never culls the far cheek
            GLES30.glDisable(GLES30.GL_CULL_FACE)
        }

        GLES30.glUseProgram(program)
        val uMvpLoc = GLES30.glGetUniformLocation(program, "uMVP")
        val uOpacityLoc = GLES30.glGetUniformLocation(program, "uOpacity")
        val uHasMaskLoc = GLES30.glGetUniformLocation(program, "uHasMask")
        val uHasCamLoc = GLES30.glGetUniformLocation(program, "uHasCameraTexture")
        val uBlendLoc = GLES30.glGetUniformLocation(program, "uBlendMode")
        val uTintLoc = GLES30.glGetUniformLocation(program, "uTintColor")
        val uProtectEyesLoc = GLES30.glGetUniformLocation(program, "uProtectEyes")
        val uProtectLipsLoc = GLES30.glGetUniformLocation(program, "uProtectLips")
        val uProtectMouthLoc = GLES30.glGetUniformLocation(program, "uProtectMouthInterior")
        val uTexPremultLoc = GLES30.glGetUniformLocation(program, "uTexturePremultiplied")
        val uMouthOpenLoc = GLES30.glGetUniformLocation(program, "uMouthOpen")
        val uEyeBlinkLoc = GLES30.glGetUniformLocation(program, "uEyeBlink")

        GLES30.glUniformMatrix4fv(uMvpLoc, 1, false, lastEffectiveMvp, 0)
        GLES30.glUniform1f(uOpacityLoc, clampedOpacity)
        GLES30.glUniform1i(uHasMaskLoc, if (maskId != null && maskId != 0) 1 else 0)
        if (uBlendLoc >= 0) GLES30.glUniform1i(uBlendLoc, parsedBlend.shaderCode)
        if (uMouthOpenLoc >= 0) GLES30.glUniform1f(uMouthOpenLoc, mouthOpen)
        if (uEyeBlinkLoc >= 0) GLES30.glUniform1f(uEyeBlinkLoc, eyeBlink)

        val r = tintColor?.getOrNull(0) ?: 1f
        val g = tintColor?.getOrNull(1) ?: 1f
        val b = tintColor?.getOrNull(2) ?: 1f
        val a = tintColor?.getOrNull(3) ?: 1f
        if (uTintLoc >= 0) GLES30.glUniform4f(uTintLoc, r, g, b, a)
        if (uProtectEyesLoc >= 0) GLES30.glUniform1i(uProtectEyesLoc, if (protectEyes) 1 else 0)
        if (uProtectLipsLoc >= 0) GLES30.glUniform1i(uProtectLipsLoc, if (protectLips) 1 else 0)
        if (uProtectMouthLoc >= 0) GLES30.glUniform1i(uProtectMouthLoc, if (protectMouthInterior) 1 else 0)
        if (uTexPremultLoc >= 0) GLES30.glUniform1i(uTexPremultLoc, if (texturePremultiplied) 1 else 0)

        if (textureId != 0) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
            val uTexLoc = GLES30.glGetUniformLocation(program, "uTexture")
            GLES30.glUniform1i(uTexLoc, 0)
        }

        if (maskId != null && maskId != 0) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, maskId)
            val uMaskLoc = GLES30.glGetUniformLocation(program, "uMask")
            GLES30.glUniform1i(uMaskLoc, 1)
        }

        if (cameraTextureId != 0) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, cameraTextureId)
            val uCamLoc = GLES30.glGetUniformLocation(program, "uCameraTexture")
            if (uCamLoc >= 0) GLES30.glUniform1i(uCamLoc, 2)
            if (uHasCamLoc >= 0) GLES30.glUniform1i(uHasCamLoc, 1)
        } else {
            if (uHasCamLoc >= 0) GLES30.glUniform1i(uHasCamLoc, 0)
        }

        GLES30.glBindVertexArray(vao)
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, indexCount, GLES30.GL_UNSIGNED_SHORT, 0)
        GLES30.glBindVertexArray(0)
        if (cameraTextureId != 0) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE2)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        }
        if (maskId != null && maskId != 0) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        }
        if (textureId != 0) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        }
    }

    @JvmOverloads
    fun release(resourceOwner: GlResourceOwner? = null) {
        if (resourceOwner != null) {
            if (vao != 0) resourceOwner.queueDeleteVao(vao)
            if (positionVbo != 0) resourceOwner.queueDeleteBuffer(positionVbo)
            if (uvVbo != 0) resourceOwner.queueDeleteBuffer(uvVbo)
            if (ibo != 0) resourceOwner.queueDeleteBuffer(ibo)
        } else {
            try {
                if (vao != 0) GLES30.glDeleteVertexArrays(1, intArrayOf(vao), 0)
                if (positionVbo != 0) GLES30.glDeleteBuffers(1, intArrayOf(positionVbo), 0)
                if (uvVbo != 0) GLES30.glDeleteBuffers(1, intArrayOf(uvVbo), 0)
                if (ibo != 0) GLES30.glDeleteBuffers(1, intArrayOf(ibo), 0)
            } catch (_: Throwable) {
                // Ignore in headless JVM test
            }
        }
        vao = 0
        positionVbo = 0
        uvVbo = 0
        ibo = 0
        isInitialized = false
    }
}
