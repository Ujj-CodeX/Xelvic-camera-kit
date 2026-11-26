package com.example.engine.render

import android.opengl.EGL14
import android.opengl.GLES30

/**
 * RtPool (Flow 2 — Chunk G0):
 *
 * Reusable Render Target Pool managing FBO + texture attachments across `FULL` (1.0),
 * `HALF` (0.5), and `QUARTER` (0.25) resolution scales and formats (`RGBA8`, `R8`).
 *
 * Key guarantees:
 * - Zero per-frame heap or GPU allocations after initial warmup: temporary targets leased via
 *   [acquireTemp] are returned to the free pool at [endFrame] and reused on subsequent frames.
 * - Named render targets (`$rt:<name>`, `$mask:<id>`, `$scene`) leased via [acquireNamed] persist
 *   across frames at their requested resolution and scale until [resize] or [releaseAll].
 * - All GL handles (FBOs, textures, depth renderbuffers) are tracked via [GlResourceOwner].
 * - Safe on headless JVM unit tests where no active EGL context is present (assigns deterministic
 *   virtual handles so pool reuse, scale calculations, and memory budgets can be unit-tested).
 */
class RtPool {

    enum class RtFormat(val bytesPerPixel: Int) {
        RGBA8(4),
        R8(1)
    }

    enum class RtScale(val factor: Float) {
        FULL(1.0f),
        HALF(0.5f),
        QUARTER(0.25f);

        companion object {
            fun fromFactor(factor: Float): RtScale {
                return when {
                    factor <= 0.30f -> QUARTER
                    factor <= 0.65f -> HALF
                    else -> FULL
                }
            }
        }
    }

    data class PooledRenderTarget(
        val poolSlotId: Int,
        var name: String,
        val width: Int,
        val height: Int,
        val format: RtFormat,
        val scaleFactor: Float,
        val withDepth: Boolean,
        val fboId: Int,
        val texId: Int,
        val depthRboId: Int,
        val byteSize: Long,
        var inUse: Boolean = true,
        var isNamed: Boolean = false
    )

    data class ResolvedTarget(
        val ref: String,
        val fboId: Int,
        val texId: Int,
        val width: Int,
        val height: Int,
        val format: RtFormat
    )

    private val namedTargets = LinkedHashMap<String, PooledRenderTarget>(16)
    private val tempPool = ArrayList<PooledRenderTarget>(16)

    private var nextVirtualHandle: Int = 1000
    private var nextSlotId: Int = 1

    var allocationCount: Int = 0
        private set
    var cacheHitCount: Int = 0
        private set

    val totalAllocatedTargets: Int
        get() = namedTargets.size + tempPool.size

    val totalAllocatedBytes: Long
        get() {
            var sum = 0L
            for (t in namedTargets.values) sum += t.byteSize
            for (t in tempPool) sum += t.byteSize
            return sum
        }

    val activeNamedTargetNames: Set<String>
        get() = namedTargets.keys

    fun computeScaledDimension(baseDim: Int, scaleFactor: Float): Int {
        val clampedScale = scaleFactor.coerceIn(0.125f, 1.0f)
        return (baseDim.coerceAtLeast(1) * clampedScale).toInt().coerceAtLeast(1)
    }

    /**
     * Acquires or reuses a persistent named render target (e.g. `"$rt:blurTemp"`, `"$mask:skin"`, `"$scene"`).
     */
    fun acquireNamed(
        name: String,
        baseWidth: Int,
        baseHeight: Int,
        format: RtFormat = RtFormat.RGBA8,
        scaleFactor: Float = 1.0f,
        withDepth: Boolean = false,
        resourceOwner: GlResourceOwner? = null
    ): PooledRenderTarget {
        val targetW = computeScaledDimension(baseWidth, scaleFactor)
        val targetH = computeScaledDimension(baseHeight, scaleFactor)
        val normalizedName = normalizeTargetName(name)

        val existing = namedTargets[normalizedName]
        if (existing != null) {
            if (existing.width == targetW &&
                existing.height == targetH &&
                existing.format == format &&
                existing.withDepth == withDepth
            ) {
                existing.inUse = true
                cacheHitCount++
                return existing
            }
            // Dimensions or format changed; destroy old target before reallocating
            deleteTargetGlResources(existing, resourceOwner)
            namedTargets.remove(normalizedName)
        }

        val created = allocateTarget(
            name = normalizedName,
            width = targetW,
            height = targetH,
            format = format,
            scaleFactor = scaleFactor.coerceIn(0.125f, 1.0f),
            withDepth = withDepth,
            isNamed = true,
            resourceOwner = resourceOwner
        )
        namedTargets[normalizedName] = created
        return created
    }

    /**
     * Leases a temporary render target for the current frame. Recycled automatically at [endFrame].
     */
    fun acquireTemp(
        baseWidth: Int,
        baseHeight: Int,
        format: RtFormat = RtFormat.RGBA8,
        scaleFactor: Float = 1.0f,
        withDepth: Boolean = false,
        debugLabel: String = "temp",
        resourceOwner: GlResourceOwner? = null
    ): PooledRenderTarget {
        val targetW = computeScaledDimension(baseWidth, scaleFactor)
        val targetH = computeScaledDimension(baseHeight, scaleFactor)

        for (i in 0 until tempPool.size) {
            val candidate = tempPool[i]
            if (!candidate.inUse &&
                candidate.width == targetW &&
                candidate.height == targetH &&
                candidate.format == format &&
                candidate.withDepth == withDepth
            ) {
                candidate.inUse = true
                candidate.name = debugLabel
                cacheHitCount++
                return candidate
            }
        }

        val created = allocateTarget(
            name = debugLabel,
            width = targetW,
            height = targetH,
            format = format,
            scaleFactor = scaleFactor.coerceIn(0.125f, 1.0f),
            withDepth = withDepth,
            isNamed = false,
            resourceOwner = resourceOwner
        )
        tempPool.add(created)
        return created
    }

    fun getNamedTarget(name: String): PooledRenderTarget? {
        return namedTargets[normalizeTargetName(name)]
    }

    /**
     * Resolves a canonical target reference (`$camera`, `$scene`, `$mask:<id>`, `$rt:<name>`, `$output`).
     */
    fun resolveTargetRef(
        ref: String,
        viewportWidth: Int,
        viewportHeight: Int,
        cameraTexId: Int = 0,
        defaultOutputFboId: Int = 0,
        resourceOwner: GlResourceOwner? = null
    ): ResolvedTarget {
        val trimmed = ref.trim()
        return when {
            trimmed == RenderTargetRef.TARGET_CAMERA -> {
                ResolvedTarget(
                    ref = RenderTargetRef.TARGET_CAMERA,
                    fboId = 0,
                    texId = cameraTexId,
                    width = viewportWidth,
                    height = viewportHeight,
                    format = RtFormat.RGBA8
                )
            }
            trimmed == RenderTargetRef.TARGET_OUTPUT -> {
                ResolvedTarget(
                    ref = RenderTargetRef.TARGET_OUTPUT,
                    fboId = defaultOutputFboId,
                    texId = 0,
                    width = viewportWidth,
                    height = viewportHeight,
                    format = RtFormat.RGBA8
                )
            }
            trimmed == RenderTargetRef.TARGET_SCENE -> {
                val rt = acquireNamed(
                    name = RenderTargetRef.TARGET_SCENE,
                    baseWidth = viewportWidth,
                    baseHeight = viewportHeight,
                    format = RtFormat.RGBA8,
                    scaleFactor = 1.0f,
                    withDepth = true,
                    resourceOwner = resourceOwner
                )
                ResolvedTarget(
                    ref = RenderTargetRef.TARGET_SCENE,
                    fboId = rt.fboId,
                    texId = rt.texId,
                    width = rt.width,
                    height = rt.height,
                    format = rt.format
                )
            }
            trimmed.startsWith(RenderTargetRef.PREFIX_MASK) -> {
                val rt = acquireNamed(
                    name = trimmed,
                    baseWidth = viewportWidth,
                    baseHeight = viewportHeight,
                    format = RtFormat.R8,
                    scaleFactor = 0.5f,
                    withDepth = false,
                    resourceOwner = resourceOwner
                )
                ResolvedTarget(
                    ref = trimmed,
                    fboId = rt.fboId,
                    texId = rt.texId,
                    width = rt.width,
                    height = rt.height,
                    format = rt.format
                )
            }
            trimmed.startsWith(RenderTargetRef.PREFIX_RT) -> {
                val rt = acquireNamed(
                    name = trimmed,
                    baseWidth = viewportWidth,
                    baseHeight = viewportHeight,
                    format = RtFormat.RGBA8,
                    scaleFactor = 1.0f,
                    withDepth = false,
                    resourceOwner = resourceOwner
                )
                ResolvedTarget(
                    ref = trimmed,
                    fboId = rt.fboId,
                    texId = rt.texId,
                    width = rt.width,
                    height = rt.height,
                    format = rt.format
                )
            }
            else -> {
                val rt = acquireNamed(
                    name = "${RenderTargetRef.PREFIX_RT}$trimmed",
                    baseWidth = viewportWidth,
                    baseHeight = viewportHeight,
                    format = RtFormat.RGBA8,
                    scaleFactor = 1.0f,
                    withDepth = false,
                    resourceOwner = resourceOwner
                )
                ResolvedTarget(
                    ref = "${RenderTargetRef.PREFIX_RT}$trimmed",
                    fboId = rt.fboId,
                    texId = rt.texId,
                    width = rt.width,
                    height = rt.height,
                    format = rt.format
                )
            }
        }
    }

    /**
     * Marks all temporary render targets as free at the end of a frame without freeing GL memory.
     */
    fun endFrame() {
        for (i in 0 until tempPool.size) {
            tempPool[i].inUse = false
        }
    }

    /**
     * Releases all pooled FBOs, textures, and depth renderbuffers on context loss or teardown.
     */
    fun releaseAll(resourceOwner: GlResourceOwner? = null) {
        for (t in namedTargets.values) {
            deleteTargetGlResources(t, resourceOwner)
        }
        namedTargets.clear()
        for (t in tempPool) {
            deleteTargetGlResources(t, resourceOwner)
        }
        tempPool.clear()
    }

    private fun normalizeTargetName(name: String): String {
        val trimmed = name.trim()
        return if (trimmed.startsWith("$")) trimmed else "\$rt:$trimmed"
    }

    private fun allocateTarget(
        name: String,
        width: Int,
        height: Int,
        format: RtFormat,
        scaleFactor: Float,
        withDepth: Boolean,
        isNamed: Boolean,
        resourceOwner: GlResourceOwner?
    ): PooledRenderTarget {
        allocationCount++
        val slotId = nextSlotId++
        val depthBytes = if (withDepth) width.toLong() * height.toLong() * 2L else 0L
        val colorBytes = width.toLong() * height.toLong() * format.bytesPerPixel.toLong()
        val totalBytes = colorBytes + depthBytes

        if (hasActiveGlContext()) {
            val fbos = IntArray(1)
            val texs = IntArray(1)
            GLES30.glGenFramebuffers(1, fbos, 0)
            GLES30.glGenTextures(1, texs, 0)
            val fbo = fbos[0]
            val tex = texs[0]
            var rbo = 0
            if (withDepth) {
                val rbos = IntArray(1)
                GLES30.glGenRenderbuffers(1, rbos, 0)
                rbo = rbos[0]
            }

            if (fbo != 0 && tex != 0) {
                resourceOwner?.trackFramebuffer(fbo)
                resourceOwner?.trackTexture(tex)
                if (rbo != 0) {
                    resourceOwner?.trackRenderbuffer(rbo)
                }

                val prevDrawFbo = IntArray(1)
                val prevReadFbo = IntArray(1)
                val prevTex2D = IntArray(1)
                GLES30.glGetIntegerv(GLES30.GL_DRAW_FRAMEBUFFER_BINDING, prevDrawFbo, 0)
                GLES30.glGetIntegerv(GLES30.GL_READ_FRAMEBUFFER_BINDING, prevReadFbo, 0)
                GLES30.glGetIntegerv(GLES30.GL_TEXTURE_BINDING_2D, prevTex2D, 0)

                GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, tex)
                val internalFormat = if (format == RtFormat.R8) GLES30.GL_R8 else GLES30.GL_RGBA8
                val pixelFormat = if (format == RtFormat.R8) GLES30.GL_RED else GLES30.GL_RGBA
                GLES30.glTexImage2D(
                    GLES30.GL_TEXTURE_2D,
                    0,
                    internalFormat,
                    width,
                    height,
                    0,
                    pixelFormat,
                    GLES30.GL_UNSIGNED_BYTE,
                    null
                )
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
                GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
                GLES30.glFramebufferTexture2D(
                    GLES30.GL_FRAMEBUFFER,
                    GLES30.GL_COLOR_ATTACHMENT0,
                    GLES30.GL_TEXTURE_2D,
                    tex,
                    0
                )

                if (rbo != 0) {
                    val prevRbo = IntArray(1)
                    GLES30.glGetIntegerv(GLES30.GL_RENDERBUFFER_BINDING, prevRbo, 0)
                    GLES30.glBindRenderbuffer(GLES30.GL_RENDERBUFFER, rbo)
                    GLES30.glRenderbufferStorage(GLES30.GL_RENDERBUFFER, GLES30.GL_DEPTH_COMPONENT16, width, height)
                    GLES30.glFramebufferRenderbuffer(
                        GLES30.GL_FRAMEBUFFER,
                        GLES30.GL_DEPTH_ATTACHMENT,
                        GLES30.GL_RENDERBUFFER,
                        rbo
                    )
                    GLES30.glBindRenderbuffer(GLES30.GL_RENDERBUFFER, prevRbo[0])
                }

                GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, prevTex2D[0])
                GLES30.glBindFramebuffer(GLES30.GL_DRAW_FRAMEBUFFER, prevDrawFbo[0])
                GLES30.glBindFramebuffer(GLES30.GL_READ_FRAMEBUFFER, prevReadFbo[0])

                return PooledRenderTarget(
                    poolSlotId = slotId,
                    name = name,
                    width = width,
                    height = height,
                    format = format,
                    scaleFactor = scaleFactor,
                    withDepth = withDepth,
                    fboId = fbo,
                    texId = tex,
                    depthRboId = rbo,
                    byteSize = totalBytes,
                    inUse = true,
                    isNamed = isNamed
                )
            }
        }

        // Headless JVM test environment fallback: assign deterministic virtual handles and track in GlResourceOwner
        val vFbo = nextVirtualHandle++
        val vTex = nextVirtualHandle++
        val vRbo = if (withDepth) nextVirtualHandle++ else 0
        resourceOwner?.trackFramebuffer(vFbo)
        resourceOwner?.trackTexture(vTex)
        if (vRbo != 0) {
            resourceOwner?.trackRenderbuffer(vRbo)
        }

        return PooledRenderTarget(
            poolSlotId = slotId,
            name = name,
            width = width,
            height = height,
            format = format,
            scaleFactor = scaleFactor,
            withDepth = withDepth,
            fboId = vFbo,
            texId = vTex,
            depthRboId = vRbo,
            byteSize = totalBytes,
            inUse = true,
            isNamed = isNamed
        )
    }

    private fun deleteTargetGlResources(target: PooledRenderTarget, resourceOwner: GlResourceOwner?) {
        if (resourceOwner != null) {
            if (target.fboId != 0) resourceOwner.queueDeleteFramebuffer(target.fboId)
            if (target.texId != 0) resourceOwner.queueDeleteTexture(target.texId)
            if (target.depthRboId != 0) {
                resourceOwner.untrackRenderbuffer(target.depthRboId)
                if (hasActiveGlContext()) {
                    try {
                        GLES30.glDeleteRenderbuffers(1, intArrayOf(target.depthRboId), 0)
                    } catch (_: Throwable) {
                    }
                }
            }
            if (resourceOwner.isOnGlThread()) {
                resourceOwner.drainPendingDeletions()
            }
        } else if (hasActiveGlContext()) {
            try {
                if (target.fboId != 0) GLES30.glDeleteFramebuffers(1, intArrayOf(target.fboId), 0)
                if (target.texId != 0) GLES30.glDeleteTextures(1, intArrayOf(target.texId), 0)
                if (target.depthRboId != 0) GLES30.glDeleteRenderbuffers(1, intArrayOf(target.depthRboId), 0)
            } catch (_: Throwable) {
            }
        }
    }

    private fun hasActiveGlContext(): Boolean {
        return try {
            val ctx = EGL14.eglGetCurrentContext()
            ctx != null && ctx != EGL14.EGL_NO_CONTEXT
        } catch (_: Throwable) {
            false
        }
    }
}
