package com.example.engine.render

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix as AndroidMatrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.Rect
import android.graphics.RectF
import android.opengl.GLES30
import android.opengl.Matrix
import com.example.engine.scene.CompiledNode
import com.example.engine.signal.FacePoseAdapter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * SpriteRenderer: Draws textured 2D quads in face-local FW space (`node.sprite@1`)
 * and screen-space 2D overlays in Stage 8 `OVERLAY_NODES` (`node.overlay@1`).
 */
class SpriteRenderer {

    data class OverlayPlacement(
        val dstLeftPx: Float,
        val dstTopPx: Float,
        val dstRightPx: Float,
        val dstBottomPx: Float,
        val u0: Float,
        val v0: Float,
        val u1: Float,
        val v1: Float,
        val isMirroredHorizontally: Boolean = false
    )

    companion object {
        const val SAFE_AREA_VERTICAL_INSET_FRACTION = 0.05f
        const val SAFE_AREA_HORIZONTAL_INSET_FRACTION = 0.03f

        /**
         * Computes deterministic screen-space placement and UV window for `node.overlay@1`
         * (`fit = "cover" | "contain"`, `safeArea = true | false`).
         *
         * Single-Mirror Rule (Flow 5 — Chunk Primitives-2):
         * Screen-space overlays (frames, Polaroid borders, date stamps, text badges) are NEVER
         * mirrored horizontally on the front camera (`u0 < u1` always, `isMirroredHorizontally = false`),
         * ensuring text and asymmetric branding remain readable in both preview and captured photos.
         */
        fun computeOverlayPlacement(
            viewportWidth: Int,
            viewportHeight: Int,
            assetWidth: Int,
            assetHeight: Int,
            fit: String = "cover",
            safeArea: Boolean = false
        ): OverlayPlacement {
            val vw = viewportWidth.coerceAtLeast(1).toFloat()
            val vh = viewportHeight.coerceAtLeast(1).toFloat()
            val aw = assetWidth.coerceAtLeast(1).toFloat()
            val ah = assetHeight.coerceAtLeast(1).toFloat()

            val insetX = if (safeArea) vw * SAFE_AREA_HORIZONTAL_INSET_FRACTION else 0f
            val insetY = if (safeArea) vh * SAFE_AREA_VERTICAL_INSET_FRACTION else 0f
            val boxLeft = insetX
            val boxTop = insetY
            val boxRight = (vw - insetX).coerceAtLeast(boxLeft + 1f)
            val boxBottom = (vh - insetY).coerceAtLeast(boxTop + 1f)
            val boxW = boxRight - boxLeft
            val boxH = boxBottom - boxTop

            val boxAspect = boxW / boxH
            val assetAspect = aw / ah

            return if (fit.equals("contain", ignoreCase = true)) {
                val (drawW, drawH) = if (assetAspect > boxAspect) {
                    boxW to (boxW / assetAspect)
                } else {
                    (boxH * assetAspect) to boxH
                }
                val left = boxLeft + (boxW - drawW) * 0.5f
                val top = boxTop + (boxH - drawH) * 0.5f
                OverlayPlacement(
                    dstLeftPx = left,
                    dstTopPx = top,
                    dstRightPx = left + drawW,
                    dstBottomPx = top + drawH,
                    u0 = 0f,
                    v0 = 0f,
                    u1 = 1f,
                    v1 = 1f,
                    isMirroredHorizontally = false
                )
            } else {
                // Default "cover": fills the target box and crops excess UVs symmetrically
                var u0 = 0f
                var v0 = 0f
                var u1 = 1f
                var v1 = 1f
                if (assetAspect > boxAspect) {
                    val visibleFracX = (boxAspect / assetAspect).coerceIn(0f, 1f)
                    val marginU = (1f - visibleFracX) * 0.5f
                    u0 = marginU
                    u1 = 1f - marginU
                } else if (assetAspect < boxAspect) {
                    val visibleFracY = (assetAspect / boxAspect).coerceIn(0f, 1f)
                    val marginV = (1f - visibleFracY) * 0.5f
                    v0 = marginV
                    v1 = 1f - marginV
                }
                OverlayPlacement(
                    dstLeftPx = boxLeft,
                    dstTopPx = boxTop,
                    dstRightPx = boxRight,
                    dstBottomPx = boxBottom,
                    u0 = u0,
                    v0 = v0,
                    u1 = u1,
                    v1 = v1,
                    isMirroredHorizontally = false
                )
            }
        }

        /**
         * Computes the active frame's `(uOffset, vOffset, uScale, vScale)` UV sub-rectangle
         * for a grid sprite-sheet (`rows x cols`) at `renderTimeSec` and `fps`.
         * Supports explicit `frameIndex`, `frameCount` (< rows*cols), `startFrame` phase offset,
         * and optional half-texel border inset when `texWidth`/`texHeight` are provided.
         */
        @JvmOverloads
        fun computeFlipbookUvRect(
            rows: Int,
            cols: Int,
            fps: Float,
            loop: Boolean = true,
            renderTimeSec: Float = 0f,
            outRect4: FloatArray = FloatArray(4),
            frameCount: Int = 0,
            explicitFrameIndex: Int = -1,
            startFrame: Int = 0,
            texWidth: Int = 0,
            texHeight: Int = 0
        ): FloatArray {
            val r = rows.coerceAtLeast(1)
            val c = cols.coerceAtLeast(1)
            val gridCapacity = r * c
            val totalFrames = if (frameCount in 1..gridCapacity) frameCount else gridCapacity
            if (gridCapacity <= 1 || (fps <= 0f && explicitFrameIndex < 0 && startFrame <= 0)) {
                outRect4[0] = 0f
                outRect4[1] = 0f
                outRect4[2] = 1f
                outRect4[3] = 1f
                return outRect4
            }
            val start = startFrame.coerceAtLeast(0)
            val frameIdx = when {
                explicitFrameIndex >= 0 -> {
                    val idx = explicitFrameIndex + start
                    if (loop) (idx % totalFrames) else idx.coerceIn(0, totalFrames - 1)
                }
                fps > 0f -> {
                    val rawFrame = (renderTimeSec.coerceAtLeast(0f) * fps).toInt().coerceAtLeast(0) + start
                    if (loop) (rawFrame % totalFrames) else rawFrame.coerceAtMost(totalFrames - 1)
                }
                else -> {
                    if (loop) (start % totalFrames) else start.coerceIn(0, totalFrames - 1)
                }
            }
            val colIdx = (frameIdx % c).coerceIn(0, c - 1)
            val rowIdx = (frameIdx / c).coerceIn(0, r - 1)
            val uScale = 1f / c.toFloat()
            val vScale = 1f / r.toFloat()
            val insetU = if (c > 1 && texWidth >= c * 4) (0.5f / texWidth.toFloat()).coerceAtMost(uScale * 0.1f) else 0f
            val insetV = if (r > 1 && texHeight >= r * 4) (0.5f / texHeight.toFloat()).coerceAtMost(vScale * 0.1f) else 0f
            outRect4[0] = colIdx * uScale + insetU
            outRect4[1] = rowIdx * vScale + insetV
            outRect4[2] = (uScale - 2f * insetU).coerceAtLeast(1e-5f)
            outRect4[3] = (vScale - 2f * insetV).coerceAtLeast(1e-5f)
            return outRect4
        }

        /**
         * Composes an [OverlayPlacement] UV crop window (`u0..u1, v0..v1` from `fit = "cover"`)
         * inside the active flipbook cell (`fbU0, fbV0, fbUScale, fbVScale`).
         */
        @JvmOverloads
        fun composeOverlayUvRect(
            placement: OverlayPlacement,
            flipbookRect4: FloatArray,
            outRect4: FloatArray = FloatArray(4)
        ): FloatArray {
            val fbU0 = flipbookRect4[0]
            val fbV0 = flipbookRect4[1]
            val fbUScale = flipbookRect4[2]
            val fbVScale = flipbookRect4[3]
            val cropU0 = placement.u0.coerceIn(0f, 1f)
            val cropV0 = placement.v0.coerceIn(0f, 1f)
            val cropUScale = (placement.u1 - placement.u0).coerceIn(0f, 1f)
            val cropVScale = (placement.v1 - placement.v0).coerceIn(0f, 1f)
            outRect4[0] = fbU0 + cropU0 * fbUScale
            outRect4[1] = fbV0 + cropV0 * fbVScale
            outRect4[2] = cropUScale * fbUScale
            outRect4[3] = cropVScale * fbVScale
            return outRect4
        }

        /**
         * Evaluates deterministic zero-allocation particle state `[offsetX, offsetY, scale, twinkleAlpha, frameOffset]`
         * for particle `particleIndex` in `[0, node.particleCount)`.
         */
        @JvmOverloads
        fun evaluateParticleInstance(
            node: CompiledNode,
            particleIndex: Int,
            renderTimeSec: Float,
            outState5: FloatArray = FloatArray(5)
        ): FloatArray {
            val count = node.particleCount.coerceIn(1, 32)
            val t = renderTimeSec.coerceAtLeast(0f)
            if (count <= 1 && particleIndex <= 0) {
                val dx = node.particleDriftX * t
                val dy = node.particleDriftY * t
                val twinkle = if (node.particleTwinkleSpeed > 0f) {
                    val wave = 0.5f + 0.5f * kotlin.math.sin(t * node.particleTwinkleSpeed * (2f * Math.PI.toFloat()))
                    (0.2f + 0.8f * wave).coerceIn(0f, 1f)
                } else {
                    1f
                }
                outState5[0] = dx
                outState5[1] = dy
                outState5[2] = 1f
                outState5[3] = twinkle
                outState5[4] = 0f
                return outState5
            }

            // Deterministic integer hash (zero heap allocations)
            var h = (node.particleSeed xor (particleIndex * 0x45d9f3b) xor (node.index * 7919))
            h = ((h ushr 16) xor h) * 0x45d9f3b
            h = ((h ushr 16) xor h) * 0x45d9f3b
            h = (h ushr 16) xor h
            val h0 = ((h and 0x3FF) / 1023f)
            val h1 = (((h ushr 10) and 0x3FF) / 1023f)
            val h2 = (((h ushr 20) and 0x3FF) / 1023f)
            val h3 = (((h * 31) and 0x3FF) / 1023f)

            val baseX = (h0 * 2f - 1f) * node.particleSpreadX
            val baseY = (h1 * 2f - 1f) * node.particleSpreadY
            val hasDrift = kotlin.math.abs(node.particleDriftX) > 1e-4f || kotlin.math.abs(node.particleDriftY) > 1e-4f
            val lifeT: Float
            val lifeEnvelope: Float
            val offsetX: Float
            val offsetY: Float
            if (hasDrift) {
                val rawPhase = (t * 0.45f + h2) % 1.0f
                lifeT = if (rawPhase < 0f) rawPhase + 1.0f else rawPhase
                offsetX = baseX + node.particleDriftX * lifeT
                offsetY = baseY + node.particleDriftY * lifeT
                lifeEnvelope = kotlin.math.sin(lifeT * Math.PI.toFloat()).coerceIn(0.05f, 1f)
            } else {
                offsetX = baseX
                offsetY = baseY
                lifeEnvelope = 1f
            }

            val baseScale = node.particleScaleMin + h3 * (node.particleScaleMax - node.particleScaleMin)
            val twinkleAlpha: Float
            val finalScale: Float
            if (node.particleTwinkleSpeed > 0f) {
                val phase = h2 * (2f * Math.PI.toFloat())
                val wave = 0.5f + 0.5f * kotlin.math.sin(t * node.particleTwinkleSpeed * (2f * Math.PI.toFloat()) + phase)
                twinkleAlpha = ((0.18f + 0.82f * wave) * lifeEnvelope).coerceIn(0f, 1f)
                finalScale = baseScale * (0.85f + 0.15f * wave)
            } else {
                twinkleAlpha = lifeEnvelope.coerceIn(0f, 1f)
                finalScale = baseScale
            }
            val frameOffset = if (particleIndex == 0) 0f else (particleIndex * 2 + (h0 * 5f).toInt()).toFloat()

            outState5[0] = offsetX
            outState5[1] = offsetY
            outState5[2] = finalScale.coerceAtLeast(0.05f)
            outState5[3] = twinkleAlpha
            outState5[4] = frameOffset
            return outState5
        }

        /**
         * Configures GLES30 blend state for `FS_UNLIT` premultiplied-alpha output.
         */
        fun applyGlBlendMode(blendMode: String?) {
            when (blendMode?.lowercase()) {
                "opaque" -> {
                    GLES30.glDisable(GLES30.GL_BLEND)
                }
                "additive", "add" -> {
                    GLES30.glEnable(GLES30.GL_BLEND)
                    GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE)
                }
                "screen" -> {
                    GLES30.glEnable(GLES30.GL_BLEND)
                    GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE_MINUS_SRC_COLOR)
                }
                "multiply" -> {
                    GLES30.glEnable(GLES30.GL_BLEND)
                    GLES30.glBlendFunc(GLES30.GL_DST_COLOR, GLES30.GL_ONE_MINUS_SRC_ALPHA)
                }
                else -> {
                    GLES30.glEnable(GLES30.GL_BLEND)
                    GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE_MINUS_SRC_ALPHA)
                }
            }
        }

        /**
         * Composites a screen-space `overlay` bitmap onto `targetBitmap` in Stage 8 (`OVERLAY_NODES`),
         * respecting `fit` (`"cover"` | `"contain"`), `safeArea`, `opacity`, `colorTint`, and unmirrored text orientation.
         * Supports animated flipbook sprite sheets (`rows`, `cols`, `fps`, `loop`, `renderTimeSec`, `frameCount`, `explicitFrameIndex`, `startFrame`).
         */
        @JvmOverloads
        fun applyOverlayToBitmap(
            targetBitmap: Bitmap,
            overlayBitmap: Bitmap,
            fit: String = "cover",
            safeArea: Boolean = false,
            opacity: Float = 1f,
            blendMode: String = "alpha",
            @Suppress("UNUSED_PARAMETER") isFrontCamera: Boolean = true,
            rows: Int = 1,
            cols: Int = 1,
            fps: Float = 0f,
            loop: Boolean = true,
            renderTimeSec: Float = 0f,
            frameCount: Int = 0,
            explicitFrameIndex: Int = -1,
            startFrame: Int = 0,
            colorTint: FloatArray? = null,
            offsetNormX: Float = 0f,
            offsetNormY: Float = 0f,
            scaleFactor: Float = 1f
        ): OverlayPlacement {
            val vw = targetBitmap.width.coerceAtLeast(1)
            val vh = targetBitmap.height.coerceAtLeast(1)
            val r = rows.coerceAtLeast(1)
            val c = cols.coerceAtLeast(1)
            val frameW = (overlayBitmap.width / c).coerceAtLeast(1)
            val frameH = (overlayBitmap.height / r).coerceAtLeast(1)
            val aw = overlayBitmap.width.coerceAtLeast(1)
            val ah = overlayBitmap.height.coerceAtLeast(1)
            val placement = computeOverlayPlacement(
                viewportWidth = vw,
                viewportHeight = vh,
                assetWidth = frameW,
                assetHeight = frameH,
                fit = fit,
                safeArea = safeArea
            )
            val effOpacity = opacity.coerceIn(0f, 1f)
            if (effOpacity <= 1e-4f) return placement

            val uvRect = computeFlipbookUvRect(
                rows = r,
                cols = c,
                fps = fps,
                loop = loop,
                renderTimeSec = renderTimeSec,
                frameCount = frameCount,
                explicitFrameIndex = explicitFrameIndex,
                startFrame = startFrame,
                texWidth = aw,
                texHeight = ah
            )
            val fbU0 = uvRect[0]
            val fbV0 = uvRect[1]
            val fbUScale = uvRect[2]
            val fbVScale = uvRect[3]

            val tintR = colorTint?.getOrElse(0) { 1f }?.coerceIn(0f, 1f) ?: 1f
            val tintG = colorTint?.getOrElse(1) { 1f }?.coerceIn(0f, 1f) ?: 1f
            val tintB = colorTint?.getOrElse(2) { 1f }?.coerceIn(0f, 1f) ?: 1f

            val sf = scaleFactor.coerceAtLeast(0.01f)
            val baseCx = (placement.dstLeftPx + placement.dstRightPx) * 0.5f + offsetNormX * vw.toFloat() * 0.5f
            val baseCy = (placement.dstTopPx + placement.dstBottomPx) * 0.5f - offsetNormY * vh.toFloat() * 0.5f
            val halfW = (placement.dstRightPx - placement.dstLeftPx) * 0.5f * sf
            val halfH = (placement.dstBottomPx - placement.dstTopPx) * 0.5f * sf
            val rawX0 = (baseCx - halfW).roundToInt()
            val rawY0 = (baseCy - halfH).roundToInt()
            val rawX1 = (baseCx + halfW).roundToInt()
            val rawY1 = (baseCy + halfH).roundToInt()
            val fullW = (rawX1 - rawX0).coerceAtLeast(1)
            val fullH = (rawY1 - rawY0).coerceAtLeast(1)
            val dstX0 = rawX0.coerceIn(0, vw)
            val dstY0 = rawY0.coerceIn(0, vh)
            val dstX1 = rawX1.coerceIn(dstX0, vw)
            val dstY1 = rawY1.coerceIn(dstY0, vh)
            if (dstX1 <= dstX0 || dstY1 <= dstY0) return placement

            val targetPixels = IntArray(vw * vh)
            val overlayPixels = IntArray(aw * ah)
            targetBitmap.getPixels(targetPixels, 0, vw, 0, 0, vw, vh)
            overlayBitmap.getPixels(overlayPixels, 0, aw, 0, 0, aw, ah)

            val mode = blendMode.lowercase()
            for (dy in dstY0 until dstY1) {
                val ty = ((dy - rawY0) + 0.5f) / fullH.toFloat()
                val localV = (placement.v0 + (placement.v1 - placement.v0) * ty).coerceIn(0f, 1f)
                val v = (fbV0 + localV * fbVScale).coerceIn(0f, 1f)
                val sy = (v * ah.toFloat()).toInt().coerceIn(0, ah - 1)
                val dstRow = dy * vw
                val srcRow = sy * aw
                for (dx in dstX0 until dstX1) {
                    val tx = ((dx - rawX0) + 0.5f) / fullW.toFloat()
                    // Unmirrored horizontal UV sampling (u0 -> u1 left-to-right even on front camera)
                    val localU = (placement.u0 + (placement.u1 - placement.u0) * tx).coerceIn(0f, 1f)
                    val u = (fbU0 + localU * fbUScale).coerceIn(0f, 1f)
                    val sx = (u * aw.toFloat()).toInt().coerceIn(0, aw - 1)

                    val srcPx = overlayPixels[srcRow + sx]
                    val srcA = (((srcPx ushr 24) and 0xFF) / 255f) * effOpacity
                    if (srcA <= 1e-4f) continue

                    val srcR = (((srcPx ushr 16) and 0xFF) / 255f) * tintR
                    val srcG = (((srcPx ushr 8) and 0xFF) / 255f) * tintG
                    val srcB = ((srcPx and 0xFF) / 255f) * tintB

                    val dstPx = targetPixels[dstRow + dx]
                    val dstA = (dstPx ushr 24) and 0xFF
                    val dstR = ((dstPx ushr 16) and 0xFF) / 255f
                    val dstG = ((dstPx ushr 8) and 0xFF) / 255f
                    val dstB = (dstPx and 0xFF) / 255f

                    val (blendR, blendG, blendB) = when (mode) {
                        "screen" -> Triple(
                            1f - (1f - dstR) * (1f - srcR),
                            1f - (1f - dstG) * (1f - srcG),
                            1f - (1f - dstB) * (1f - srcB)
                        )
                        "multiply" -> Triple(dstR * srcR, dstG * srcG, dstB * srcB)
                        "additive", "add" -> Triple(
                            (dstR + srcR).coerceIn(0f, 1f),
                            (dstG + srcG).coerceIn(0f, 1f),
                            (dstB + srcB).coerceIn(0f, 1f)
                        )
                        else -> Triple(srcR, srcG, srcB)
                    }

                    val outR = ((dstR + (blendR - dstR) * srcA).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                    val outG = ((dstG + (blendG - dstG) * srcA).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                    val outB = ((dstB + (blendB - dstB) * srcA).coerceIn(0f, 1f) * 255f).roundToInt().coerceIn(0, 255)
                    val outA = max(dstA, (srcA * 255f).roundToInt().coerceIn(0, 255))
                    targetPixels[dstRow + dx] = (outA shl 24) or (outR shl 16) or (outG shl 8) or outB
                }
            }
            targetBitmap.setPixels(targetPixels, 0, vw, 0, 0, vw, vh)
            return placement
        }
    }

    private var vao = 0
    private var vbo = 0
    private var ibo = 0
    private val overlayMvpScratch = FloatArray(16)
    private val particleMvpScratch = FloatArray(16)
    private val particleStateScratch = FloatArray(5)
    private val flipbookUvScratch = floatArrayOf(0f, 0f, 1f, 1f)
    private val composedOverlayUvScratch = floatArrayOf(0f, 0f, 1f, 1f)
    private val lastColorUniformScratch = floatArrayOf(1f, 1f, 1f, 1f)

    var lastFrontFaceMode: Int = GLES30.GL_CCW
        private set

    var lastOverlayPlacement: OverlayPlacement? = null
        private set

    val lastFlipbookUvRect: FloatArray
        get() = flipbookUvScratch

    val lastComposedOverlayUvRect: FloatArray
        get() = composedOverlayUvScratch

    val lastColorUniform: FloatArray
        get() = lastColorUniformScratch

    var lastRenderedInstanceCount: Int = 0
        private set

    @JvmOverloads
    fun init(resourceOwner: GlResourceOwner? = null) {
        vao = 0
        vbo = 0
        ibo = 0
        // Unit quad [-0.5, 0.5] with UVs
        val vertices = floatArrayOf(
            // x, y, z,   u, v
            -0.5f, -0.5f, 0f, 0f, 1f,
             0.5f, -0.5f, 0f, 1f, 1f,
             0.5f,  0.5f, 0f, 1f, 0f,
            -0.5f,  0.5f, 0f, 0f, 0f
        )
        val indices = shortArrayOf(0, 1, 2, 0, 2, 3)

        try {
            val vaos = IntArray(1); GLES30.glGenVertexArrays(1, vaos, 0)
            val vbos = IntArray(1); GLES30.glGenBuffers(1, vbos, 0)
            val ibos = IntArray(1); GLES30.glGenBuffers(1, ibos, 0)
            vao = vaos[0]; vbo = vbos[0]; ibo = ibos[0]

            resourceOwner?.trackVao(vao)
            resourceOwner?.trackBuffer(vbo)
            resourceOwner?.trackBuffer(ibo)

            if (vao != 0) {
                GLES30.glBindVertexArray(vao)

                GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, vbo)
                val fb = ByteBuffer.allocateDirect(vertices.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
                fb.put(vertices).position(0)
                GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, vertices.size * 4, fb, GLES30.GL_STATIC_DRAW)

                GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, ibo)
                val sb = ByteBuffer.allocateDirect(indices.size * 2).order(ByteOrder.nativeOrder()).asShortBuffer()
                sb.put(indices).position(0)
                GLES30.glBufferData(GLES30.GL_ELEMENT_ARRAY_BUFFER, indices.size * 2, sb, GLES30.GL_STATIC_DRAW)

                val stride = 5 * 4
                GLES30.glEnableVertexAttribArray(0)
                GLES30.glVertexAttribPointer(0, 3, GLES30.GL_FLOAT, false, stride, 0)

                GLES30.glEnableVertexAttribArray(1)
                GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, stride, 3 * 4)

                GLES30.glEnableVertexAttribArray(2)
                GLES30.glVertexAttribPointer(2, 2, GLES30.GL_FLOAT, false, stride, 3 * 4)

                GLES30.glBindVertexArray(0)
            }
        } catch (_: Throwable) {
            // Headless JVM test environment
        }
    }

    @JvmOverloads
    fun render(
        node: CompiledNode,
        mvpMatrix: FloatArray,
        textureId: Int,
        opacity: Float,
        program: Int,
        frontFacing: Boolean = false,
        renderTimeSec: Float = 0f,
        colorTint: FloatArray? = null,
        explicitFrameIndex: Int = node.spriteFrameIndex,
        texWidth: Int = 0,
        texHeight: Int = 0
    ) {
        val frontFaceMode = FacePoseAdapter.glFrontFaceForCamera(frontFacing)
        lastFrontFaceMode = frontFaceMode
        val instanceCount = node.particleCount.coerceIn(1, 32)
        lastRenderedInstanceCount = instanceCount
        val tintR = colorTint?.getOrElse(0) { 1f }?.coerceIn(0f, 1f) ?: 1f
        val tintG = colorTint?.getOrElse(1) { 1f }?.coerceIn(0f, 1f) ?: 1f
        val tintB = colorTint?.getOrElse(2) { 1f }?.coerceIn(0f, 1f) ?: 1f
        val baseOpacity = opacity.coerceIn(0f, 1f)

        computeFlipbookUvRect(
            rows = node.spriteRows,
            cols = node.spriteCols,
            fps = node.spriteFps,
            loop = node.spriteLoop,
            renderTimeSec = renderTimeSec,
            outRect4 = flipbookUvScratch,
            frameCount = node.spriteFrameCount,
            explicitFrameIndex = explicitFrameIndex,
            startFrame = node.spriteStartFrame,
            texWidth = texWidth,
            texHeight = texHeight
        )
        lastColorUniformScratch[0] = tintR
        lastColorUniformScratch[1] = tintG
        lastColorUniformScratch[2] = tintB
        lastColorUniformScratch[3] = baseOpacity

        if (vao == 0) return

        GLES30.glFrontFace(frontFaceMode)
        GLES30.glDisable(GLES30.GL_CULL_FACE)
        GLES30.glUseProgram(program)
        val uMvpLoc = GLES30.glGetUniformLocation(program, "uMVP")
        val uColorLoc = GLES30.glGetUniformLocation(program, "uColor")
        val uHasTexLoc = GLES30.glGetUniformLocation(program, "uHasTexture")
        val uUvRectLoc = GLES30.glGetUniformLocation(program, "uUvRect")

        GLES30.glUniform1i(uHasTexLoc, if (textureId != 0) 1 else 0)
        if (textureId != 0) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
            val uTexLoc = GLES30.glGetUniformLocation(program, "uTexture")
            GLES30.glUniform1i(uTexLoc, 0)
        }

        GLES30.glBindVertexArray(vao)
        for (pIdx in 0 until instanceCount) {
            evaluateParticleInstance(node, pIdx, renderTimeSec, particleStateScratch)
            val pDx = particleStateScratch[0]
            val pDy = particleStateScratch[1]
            val pScale = particleStateScratch[2]
            val pTwinkle = particleStateScratch[3]
            val pFrameOffset = particleStateScratch[4].toInt()
            val instOpacity = (baseOpacity * pTwinkle).coerceIn(0f, 1f)
            if (instOpacity <= 1e-4f) continue

            if (pIdx > 0 || pFrameOffset != 0) {
                computeFlipbookUvRect(
                    rows = node.spriteRows,
                    cols = node.spriteCols,
                    fps = node.spriteFps,
                    loop = node.spriteLoop,
                    renderTimeSec = renderTimeSec,
                    outRect4 = flipbookUvScratch,
                    frameCount = node.spriteFrameCount,
                    explicitFrameIndex = explicitFrameIndex,
                    startFrame = node.spriteStartFrame + pFrameOffset,
                    texWidth = texWidth,
                    texHeight = texHeight
                )
            }
            if (pDx != 0f || pDy != 0f || pScale != 1f) {
                System.arraycopy(mvpMatrix, 0, particleMvpScratch, 0, 16)
                Matrix.translateM(particleMvpScratch, 0, pDx, pDy, 0f)
                Matrix.scaleM(particleMvpScratch, 0, pScale, pScale, 1f)
                GLES30.glUniformMatrix4fv(uMvpLoc, 1, false, particleMvpScratch, 0)
            } else {
                GLES30.glUniformMatrix4fv(uMvpLoc, 1, false, mvpMatrix, 0)
            }

            GLES30.glUniform4f(uColorLoc, tintR, tintG, tintB, instOpacity)
            if (uUvRectLoc >= 0) {
                GLES30.glUniform4f(
                    uUvRectLoc,
                    flipbookUvScratch[0],
                    flipbookUvScratch[1],
                    flipbookUvScratch[2],
                    flipbookUvScratch[3]
                )
            }
            GLES30.glDrawElements(GLES30.GL_TRIANGLES, 6, GLES30.GL_UNSIGNED_SHORT, 0)
        }
        GLES30.glBindVertexArray(0)
        if (textureId != 0) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        }
    }

    /**
     * Renders a screen-space `node.overlay@1` (`OVERLAY_NODES`, Stage 8) after post-processing.
     * Uses unmirrored NDC placement so text/timestamps/frames are never horizontally flipped,
     * and composes `placement.u0..u1, v0..v1` cover-crop UVs with the active flipbook frame cell.
     */
    @JvmOverloads
    fun renderOverlay(
        node: CompiledNode,
        textureId: Int,
        opacity: Float,
        program: Int,
        viewportWidth: Int,
        viewportHeight: Int,
        assetWidth: Int = viewportWidth,
        assetHeight: Int = viewportHeight,
        renderTimeSec: Float = 0f,
        colorTint: FloatArray? = null,
        explicitFrameIndex: Int = node.spriteFrameIndex
    ): OverlayPlacement {
        val r = node.spriteRows.coerceAtLeast(1)
        val c = node.spriteCols.coerceAtLeast(1)
        val placement = computeOverlayPlacement(
            viewportWidth = viewportWidth,
            viewportHeight = viewportHeight,
            assetWidth = (assetWidth / c).coerceAtLeast(1),
            assetHeight = (assetHeight / r).coerceAtLeast(1),
            fit = node.overlayFit,
            safeArea = node.overlaySafeArea
        )
        computeFlipbookUvRect(
            rows = r,
            cols = c,
            fps = node.spriteFps,
            loop = node.spriteLoop,
            renderTimeSec = renderTimeSec,
            outRect4 = flipbookUvScratch,
            frameCount = node.spriteFrameCount,
            explicitFrameIndex = explicitFrameIndex,
            startFrame = node.spriteStartFrame,
            texWidth = assetWidth,
            texHeight = assetHeight
        )
        composeOverlayUvRect(placement, flipbookUvScratch, composedOverlayUvScratch)
        val tintR = colorTint?.getOrElse(0) { 1f }?.coerceIn(0f, 1f) ?: 1f
        val tintG = colorTint?.getOrElse(1) { 1f }?.coerceIn(0f, 1f) ?: 1f
        val tintB = colorTint?.getOrElse(2) { 1f }?.coerceIn(0f, 1f) ?: 1f
        val baseOpacity = opacity.coerceIn(0f, 1f)
        lastColorUniformScratch[0] = tintR
        lastColorUniformScratch[1] = tintG
        lastColorUniformScratch[2] = tintB
        lastColorUniformScratch[3] = baseOpacity
        val instanceCount = node.particleCount.coerceIn(1, 32)
        lastOverlayPlacement = placement
        lastFrontFaceMode = GLES30.GL_CCW
        lastRenderedInstanceCount = instanceCount
        if (vao == 0 || program == 0) return placement

        val vw = viewportWidth.coerceAtLeast(1).toFloat()
        val vh = viewportHeight.coerceAtLeast(1).toFloat()
        val ndcLeft = (placement.dstLeftPx / vw) * 2f - 1f
        val ndcRight = (placement.dstRightPx / vw) * 2f - 1f
        val ndcTop = 1f - (placement.dstTopPx / vh) * 2f
        val ndcBottom = 1f - (placement.dstBottomPx / vh) * 2f

        val centerX = (ndcLeft + ndcRight) * 0.5f
        val centerY = (ndcBottom + ndcTop) * 0.5f
        val scaleX = (ndcRight - ndcLeft)
        val scaleY = (ndcTop - ndcBottom)

        Matrix.setIdentityM(overlayMvpScratch, 0)
        Matrix.translateM(overlayMvpScratch, 0, centerX, centerY, 0f)
        Matrix.scaleM(overlayMvpScratch, 0, scaleX, scaleY, 1f)

        GLES30.glFrontFace(GLES30.GL_CCW)
        GLES30.glDisable(GLES30.GL_CULL_FACE)
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glDepthMask(false)
        GLES30.glUseProgram(program)

        val uMvpLoc = GLES30.glGetUniformLocation(program, "uMVP")
        val uColorLoc = GLES30.glGetUniformLocation(program, "uColor")
        val uHasTexLoc = GLES30.glGetUniformLocation(program, "uHasTexture")
        val uUvRectLoc = GLES30.glGetUniformLocation(program, "uUvRect")

        GLES30.glUniform1i(uHasTexLoc, if (textureId != 0) 1 else 0)
        if (textureId != 0) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, textureId)
            val uTexLoc = GLES30.glGetUniformLocation(program, "uTexture")
            GLES30.glUniform1i(uTexLoc, 0)
        }

        GLES30.glBindVertexArray(vao)
        for (pIdx in 0 until instanceCount) {
            evaluateParticleInstance(node, pIdx, renderTimeSec, particleStateScratch)
            val pDx = particleStateScratch[0]
            val pDy = particleStateScratch[1]
            val pScale = particleStateScratch[2]
            val pTwinkle = particleStateScratch[3]
            val pFrameOffset = particleStateScratch[4].toInt()
            val instOpacity = (baseOpacity * pTwinkle).coerceIn(0f, 1f)
            if (instOpacity <= 1e-4f) continue

            if (pIdx > 0 || pFrameOffset != 0) {
                computeFlipbookUvRect(
                    rows = r,
                    cols = c,
                    fps = node.spriteFps,
                    loop = node.spriteLoop,
                    renderTimeSec = renderTimeSec,
                    outRect4 = flipbookUvScratch,
                    frameCount = node.spriteFrameCount,
                    explicitFrameIndex = explicitFrameIndex,
                    startFrame = node.spriteStartFrame + pFrameOffset,
                    texWidth = assetWidth,
                    texHeight = assetHeight
                )
                composeOverlayUvRect(placement, flipbookUvScratch, composedOverlayUvScratch)
            }
            if (pDx != 0f || pDy != 0f || pScale != 1f) {
                Matrix.setIdentityM(particleMvpScratch, 0)
                Matrix.translateM(particleMvpScratch, 0, centerX + pDx, centerY + pDy, 0f)
                Matrix.scaleM(particleMvpScratch, 0, scaleX * pScale, scaleY * pScale, 1f)
                GLES30.glUniformMatrix4fv(uMvpLoc, 1, false, particleMvpScratch, 0)
            } else {
                GLES30.glUniformMatrix4fv(uMvpLoc, 1, false, overlayMvpScratch, 0)
            }
            GLES30.glUniform4f(uColorLoc, tintR, tintG, tintB, instOpacity)
            if (uUvRectLoc >= 0) {
                GLES30.glUniform4f(
                    uUvRectLoc,
                    composedOverlayUvScratch[0],
                    composedOverlayUvScratch[1],
                    composedOverlayUvScratch[2],
                    composedOverlayUvScratch[3]
                )
            }
            GLES30.glDrawElements(GLES30.GL_TRIANGLES, 6, GLES30.GL_UNSIGNED_SHORT, 0)
        }
        GLES30.glBindVertexArray(0)
        if (textureId != 0) {
            GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)
        }
        return placement
    }

    @JvmOverloads
    fun release(resourceOwner: GlResourceOwner? = null) {
        if (resourceOwner != null) {
            if (vao != 0) resourceOwner.queueDeleteVao(vao)
            if (vbo != 0) resourceOwner.queueDeleteBuffer(vbo)
            if (ibo != 0) resourceOwner.queueDeleteBuffer(ibo)
        } else {
            try {
                if (vao != 0) GLES30.glDeleteVertexArrays(1, intArrayOf(vao), 0)
                if (vbo != 0) GLES30.glDeleteBuffers(1, intArrayOf(vbo), 0)
                if (ibo != 0) GLES30.glDeleteBuffers(1, intArrayOf(ibo), 0)
            } catch (_: Throwable) {
            }
        }
        vao = 0; vbo = 0; ibo = 0
    }
}
