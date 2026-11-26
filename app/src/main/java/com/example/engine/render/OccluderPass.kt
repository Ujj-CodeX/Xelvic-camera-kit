package com.example.engine.render

import android.opengl.GLES30
import com.example.engine.pkg.UvRegistry
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * OccluderPass: Renders depth-only geometry (face mesh pushed back along view rays,
 * head ellipsoid, or box) with color writes disabled (README Section 10 Stage 11, 11.1, 11.3, 13 & 21 Step 4/6).
 *
 * - Shares [FaceMeshRenderer.unprojectLandmarksToCameraSpace] and [FaceMeshRenderer.computePoseCorrectedMvp]
 *   with [FaceMeshRenderer] so the face-mesh occluder builds vertices in metric camera space,
 *   applies the rigid pose correction $T = M_{\text{pred}} \cdot M_{\text{landmark}}^{-1}$ in `uMVP`,
 *   pushes vertices radially along view rays from the camera origin, and ignores node offset/scale.
 * - Disables face culling (`glDisable(GL_CULL_FACE)`) during the face-mesh depth-only occluder draw.
 * - Supports multi-LOD ellipsoid segment counts (16x24, 12x18, 8x12) controlled by PerformanceGovernor.
 */
class OccluderPass {

    private class EllipsoidMesh(
        var vao: Int = 0,
        var vbo: Int = 0,
        var ibo: Int = 0,
        var indexCount: Int = 0
    )

    // Pre-built ellipsoid LODs for governor degradation step 4 (16x24 -> 12x18 -> 8x12)
    private val ellipsoidHigh = EllipsoidMesh() // 16 x 24
    private val ellipsoidMed = EllipsoidMesh()  // 12 x 18
    private val ellipsoidLow = EllipsoidMesh()  // 8 x 12

    private var boxVao = 0
    private var boxVbo = 0
    private var boxIbo = 0
    private var boxIndexCount = 0

    private var faceMeshVao = 0
    private var faceMeshVbo = 0
    private var faceMeshIbo = 0
    private var faceMeshIndexCount = UvRegistry.INDEX_COUNT

    // Pre-allocated direct buffer and CPU-side array for zero-allocation per-frame faceMesh updates
    private val faceMeshBuffer: FloatBuffer = ByteBuffer
        .allocateDirect(UvRegistry.VERTEX_COUNT * 3 * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
    val liveFaceMeshPositions = FloatArray(UvRegistry.VERTEX_COUNT * 3)

    // Pre-allocated 16-float matrices for rigid pose correction T = M_pred * inv(M_landmark) and uMVP
    private val scratchInvLandmark = FloatArray(16)
    val lastRigidCorrectionMatrix = FloatArray(16).apply {
        this[0] = 1f; this[5] = 1f; this[10] = 1f; this[15] = 1f
    }
    val lastEffectiveFaceMeshMvp = FloatArray(16).apply {
        this[0] = 1f; this[5] = 1f; this[10] = 1f; this[15] = 1f
    }

    private val delScratch = IntArray(1)

    var lastFrontFaceMode: Int = GLES30.GL_CCW
        private set

    var lastFaceMeshCullFaceEnabled: Boolean = true
        private set

    var faceMeshVertexUpdateCount: Int = 0
        private set

    fun init(resourceOwner: GlResourceOwner? = null) {
        buildEllipsoidMesh(16, 24, ellipsoidHigh, resourceOwner)
        buildEllipsoidMesh(12, 18, ellipsoidMed, resourceOwner)
        buildEllipsoidMesh(8, 12, ellipsoidLow, resourceOwner)
        buildBox(resourceOwner)
        buildFaceMeshBuffers(resourceOwner)
    }

    private fun buildEllipsoidMesh(
        latSegments: Int,
        lonSegments: Int,
        target: EllipsoidMesh,
        resourceOwner: GlResourceOwner?
    ) {
        val vertList = ArrayList<Float>()
        val idxList = ArrayList<Short>()

        for (i in 0..latSegments) {
            val theta = i * PI.toFloat() / latSegments
            val sinTheta = sin(theta)
            val cosTheta = cos(theta)

            for (j in 0..lonSegments) {
                val phi = j * 2f * PI.toFloat() / lonSegments
                val sinPhi = sin(phi)
                val cosPhi = cos(phi)

                val x = cosPhi * sinTheta
                val y = cosTheta
                val rawZ = sinPhi * sinTheta
                // Keep the cranial proxy volume behind the front facial skin plane (faceMesh occluder
                // handles precise front-surface depth occlusion with pushBackFW) so the ellipsoid
                // occludes behind the temples/cranium during yaw without clipping forehead/glasses props.
                val z = if (rawZ > 0f) (rawZ * 0.25f) - 0.45f else rawZ - 0.45f

                vertList.add(x); vertList.add(y); vertList.add(z)
            }
        }

        for (i in 0 until latSegments) {
            for (j in 0 until lonSegments) {
                val first = (i * (lonSegments + 1) + j).toShort()
                val second = (first + lonSegments + 1).toShort()

                idxList.add(first); idxList.add(second); idxList.add((first + 1).toShort())
                idxList.add(second); idxList.add((second + 1).toShort()); idxList.add((first + 1).toShort())
            }
        }

        target.indexCount = idxList.size
        target.vao = 0; target.vbo = 0; target.ibo = 0
        val vertices = vertList.toFloatArray()
        val indices = idxList.toShortArray()

        try {
            val vaos = IntArray(1); GLES30.glGenVertexArrays(1, vaos, 0)
            val vbos = IntArray(1); GLES30.glGenBuffers(1, vbos, 0)
            val ibos = IntArray(1); GLES30.glGenBuffers(1, ibos, 0)
            target.vao = vaos[0]; target.vbo = vbos[0]; target.ibo = ibos[0]

            resourceOwner?.trackVao(target.vao)
            resourceOwner?.trackBuffer(target.vbo)
            resourceOwner?.trackBuffer(target.ibo)

            GLES30.glBindVertexArray(target.vao)

            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, target.vbo)
            val fb = ByteBuffer.allocateDirect(vertices.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            fb.put(vertices).position(0)
            GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, vertices.size * 4, fb, GLES30.GL_STATIC_DRAW)

            GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, target.ibo)
            val sb = ByteBuffer.allocateDirect(indices.size * 2).order(ByteOrder.nativeOrder()).asShortBuffer()
            sb.put(indices).position(0)
            GLES30.glBufferData(GLES30.GL_ELEMENT_ARRAY_BUFFER, indices.size * 2, sb, GLES30.GL_STATIC_DRAW)

            GLES30.glEnableVertexAttribArray(0)
            GLES30.glVertexAttribPointer(0, 3, GLES30.GL_FLOAT, false, 3 * 4, 0)

            GLES30.glBindVertexArray(0)
        } catch (_: Throwable) {
            // Headless JVM test environment
        }
    }

    private fun buildBox(resourceOwner: GlResourceOwner?) {
        val vertices = floatArrayOf(
            -0.5f, -0.5f, -0.5f,
             0.5f, -0.5f, -0.5f,
             0.5f,  0.5f, -0.5f,
            -0.5f,  0.5f, -0.5f,
            -0.5f, -0.5f,  0.5f,
             0.5f, -0.5f,  0.5f,
             0.5f,  0.5f,  0.5f,
            -0.5f,  0.5f,  0.5f
        )
        val indices = shortArrayOf(
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            2, 3, 7, 2, 7, 6,
            0, 4, 7, 0, 7, 3,
            1, 2, 6, 1, 6, 5
        )
        boxIndexCount = indices.size
        boxVao = 0; boxVbo = 0; boxIbo = 0

        try {
            val vaos = IntArray(1); GLES30.glGenVertexArrays(1, vaos, 0)
            val vbos = IntArray(1); GLES30.glGenBuffers(1, vbos, 0)
            val ibos = IntArray(1); GLES30.glGenBuffers(1, ibos, 0)
            boxVao = vaos[0]; boxVbo = vbos[0]; boxIbo = ibos[0]

            resourceOwner?.trackVao(boxVao)
            resourceOwner?.trackBuffer(boxVbo)
            resourceOwner?.trackBuffer(boxIbo)

            GLES30.glBindVertexArray(boxVao)

            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, boxVbo)
            val fb = ByteBuffer.allocateDirect(vertices.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            fb.put(vertices).position(0)
            GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, vertices.size * 4, fb, GLES30.GL_STATIC_DRAW)

            GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, boxIbo)
            val sb = ByteBuffer.allocateDirect(indices.size * 2).order(ByteOrder.nativeOrder()).asShortBuffer()
            sb.put(indices).position(0)
            GLES30.glBufferData(GLES30.GL_ELEMENT_ARRAY_BUFFER, indices.size * 2, sb, GLES30.GL_STATIC_DRAW)

            GLES30.glEnableVertexAttribArray(0)
            GLES30.glVertexAttribPointer(0, 3, GLES30.GL_FLOAT, false, 3 * 4, 0)

            GLES30.glBindVertexArray(0)
        } catch (_: Throwable) {
            // Headless JVM test environment
        }
    }

    private fun buildFaceMeshBuffers(resourceOwner: GlResourceOwner?) {
        val tris = UvRegistry.CANONICAL_TRIANGLES
        faceMeshIndexCount = tris.size
        val vertCount = UvRegistry.VERTEX_COUNT

        liveFaceMeshPositions.fill(0f)
        faceMeshBuffer.clear()
        faceMeshBuffer.put(liveFaceMeshPositions)
        (faceMeshBuffer as java.nio.Buffer).position(0)
        faceMeshVao = 0; faceMeshVbo = 0; faceMeshIbo = 0

        try {
            val vaos = IntArray(1); GLES30.glGenVertexArrays(1, vaos, 0)
            val vbos = IntArray(1); GLES30.glGenBuffers(1, vbos, 0)
            val ibos = IntArray(1); GLES30.glGenBuffers(1, ibos, 0)
            faceMeshVao = vaos[0]; faceMeshVbo = vbos[0]; faceMeshIbo = ibos[0]

            resourceOwner?.trackVao(faceMeshVao)
            resourceOwner?.trackBuffer(faceMeshVbo)
            resourceOwner?.trackBuffer(faceMeshIbo)

            GLES30.glBindVertexArray(faceMeshVao)

            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, faceMeshVbo)
            GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, vertCount * 3 * 4, faceMeshBuffer, GLES30.GL_DYNAMIC_DRAW)

            GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, faceMeshIbo)
            val sb = ByteBuffer.allocateDirect(tris.size * 2).order(ByteOrder.nativeOrder()).asShortBuffer()
            sb.put(tris).position(0)
            GLES30.glBufferData(GLES30.GL_ELEMENT_ARRAY_BUFFER, tris.size * 2, sb, GLES30.GL_STATIC_DRAW)

            GLES30.glEnableVertexAttribArray(0)
            GLES30.glVertexAttribPointer(0, 3, GLES30.GL_FLOAT, false, 3 * 4, 0)

            GLES30.glBindVertexArray(0)
        } catch (_: Throwable) {
            // Headless JVM test environment
        }
    }

    @JvmOverloads
    fun beginPass(frontFacing: Boolean = false) {
        val frontFaceMode = FacePoseAdapter.glFrontFaceForCamera(frontFacing)
        lastFrontFaceMode = frontFaceMode
        try {
            GLES30.glFrontFace(frontFaceMode)
            GLES30.glEnable(GLES30.GL_DEPTH_TEST)
            GLES30.glDepthMask(true)
            // Disable color writing: depth writes only!
            GLES30.glColorMask(false, false, false, false)
        } catch (_: Throwable) {
        }
    }

    @JvmOverloads
    fun renderEllipsoid(
        mvpMatrix: FloatArray,
        program: Int,
        latSegments: Int = 16
    ) {
        val mesh = when {
            latSegments <= 8 -> ellipsoidLow
            latSegments <= 12 -> ellipsoidMed
            else -> ellipsoidHigh
        }
        if (mesh.vao == 0) return

        GLES30.glUseProgram(program)
        val uMvpLoc = GLES30.glGetUniformLocation(program, "uMVP")
        GLES30.glUniformMatrix4fv(uMvpLoc, 1, false, mvpMatrix, 0)

        GLES30.glBindVertexArray(mesh.vao)
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, mesh.indexCount, GLES30.GL_UNSIGNED_SHORT, 0)
        GLES30.glBindVertexArray(0)
    }

    fun renderBox(mvpMatrix: FloatArray, program: Int) {
        if (boxVao == 0) return

        GLES30.glUseProgram(program)
        val uMvpLoc = GLES30.glGetUniformLocation(program, "uMVP")
        GLES30.glUniformMatrix4fv(uMvpLoc, 1, false, mvpMatrix, 0)

        GLES30.glBindVertexArray(boxVao)
        GLES30.glDrawElements(GLES30.GL_TRIANGLES, boxIndexCount, GLES30.GL_UNSIGNED_SHORT, 0)
        GLES30.glBindVertexArray(0)
    }

    /**
     * Unprojects [faceState] landmarks into metric camera space via [FaceMeshRenderer.unprojectLandmarksToCameraSpace]
     * with radial view-ray pushback (`pushBackFW`) and zero heap allocation.
     * Node offset and scale are ignored/forbidden (identity).
     */
    @JvmOverloads
    fun updateFaceMeshVertices(
        faceState: FaceState,
        viewportAspect: Float = 9f / 16f,
        pushBackFW: Float = 0f
    ): Boolean {
        val ok = FaceMeshRenderer.unprojectLandmarksToCameraSpace(
            faceState = faceState,
            viewportAspect = viewportAspect,
            pushBackFW = pushBackFW.coerceAtLeast(0f),
            outPositions = liveFaceMeshPositions,
            outBuffer = faceMeshBuffer
        )
        if (!ok) return false
        faceMeshVertexUpdateCount++
        return true
    }

    /**
     * Renders the 468-vertex depth-only face-mesh occluder in metric camera space:
     * 1. Unprojects landmarks via [FaceMeshRenderer.unprojectLandmarksToCameraSpace] with view-ray `pushBackFW`.
     * 2. Computes rigid pose correction $T = M_{\text{pred}} \cdot M_{\text{landmark}}^{-1}$ and folds it
     *    into `uMVP = viewProjectionMatrix * T`.
     * 3. Disables face culling (`GLES30.glDisable(GLES30.GL_CULL_FACE)`) during the depth-only draw.
     */
    @JvmOverloads
    fun renderFaceMesh(
        faceState: FaceState,
        viewProjectionMatrix: FloatArray,
        program: Int,
        pushBackFW: Float,
        viewportAspect: Float = 9f / 16f
    ) {
        if (!updateFaceMeshVertices(faceState, viewportAspect, pushBackFW)) return

        FaceMeshRenderer.computePoseCorrectedMvp(
            viewProjectionMatrix = viewProjectionMatrix,
            faceState = faceState,
            outMvp = lastEffectiveFaceMeshMvp,
            scratchInvLandmark = scratchInvLandmark,
            scratchRigidDelta = lastRigidCorrectionMatrix
        )

        // Disable face culling during the face-mesh depth-only occluder pass
        lastFaceMeshCullFaceEnabled = false
        try {
            GLES30.glDisable(GLES30.GL_CULL_FACE)
        } catch (_: Throwable) {
        }

        if (faceMeshVbo != 0 && faceMeshVao != 0 && program != 0) {
            val count = UvRegistry.VERTEX_COUNT
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, faceMeshVbo)
            GLES30.glBufferSubData(GLES30.GL_ARRAY_BUFFER, 0, count * 3 * 4, faceMeshBuffer)
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)

            GLES30.glUseProgram(program)
            val uMvpLoc = GLES30.glGetUniformLocation(program, "uMVP")
            GLES30.glUniformMatrix4fv(uMvpLoc, 1, false, lastEffectiveFaceMeshMvp, 0)

            GLES30.glBindVertexArray(faceMeshVao)
            GLES30.glDrawElements(GLES30.GL_TRIANGLES, faceMeshIndexCount, GLES30.GL_UNSIGNED_SHORT, 0)
            GLES30.glBindVertexArray(0)
        }
    }

    fun endPass() {
        try {
            // Re-enable color writing
            GLES30.glColorMask(true, true, true, true)
        } catch (_: Throwable) {
        }
    }

    private fun releaseEllipsoid(mesh: EllipsoidMesh) {
        if (mesh.vao != 0) { delScratch[0] = mesh.vao; GLES30.glDeleteVertexArrays(1, delScratch, 0) }
        if (mesh.vbo != 0) { delScratch[0] = mesh.vbo; GLES30.glDeleteBuffers(1, delScratch, 0) }
        if (mesh.ibo != 0) { delScratch[0] = mesh.ibo; GLES30.glDeleteBuffers(1, delScratch, 0) }
        mesh.vao = 0; mesh.vbo = 0; mesh.ibo = 0
    }

    fun release() {
        try {
            releaseEllipsoid(ellipsoidHigh)
            releaseEllipsoid(ellipsoidMed)
            releaseEllipsoid(ellipsoidLow)

            if (boxVao != 0) { delScratch[0] = boxVao; GLES30.glDeleteVertexArrays(1, delScratch, 0) }
            if (boxVbo != 0) { delScratch[0] = boxVbo; GLES30.glDeleteBuffers(1, delScratch, 0) }
            if (boxIbo != 0) { delScratch[0] = boxIbo; GLES30.glDeleteBuffers(1, delScratch, 0) }

            if (faceMeshVao != 0) { delScratch[0] = faceMeshVao; GLES30.glDeleteVertexArrays(1, delScratch, 0) }
            if (faceMeshVbo != 0) { delScratch[0] = faceMeshVbo; GLES30.glDeleteBuffers(1, delScratch, 0) }
            if (faceMeshIbo != 0) { delScratch[0] = faceMeshIbo; GLES30.glDeleteBuffers(1, delScratch, 0) }
        } catch (_: Throwable) {
        }
        boxVao = 0; boxVbo = 0; boxIbo = 0
        faceMeshVao = 0; faceMeshVbo = 0; faceMeshIbo = 0
    }
}
