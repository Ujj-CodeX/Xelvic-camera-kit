package com.example.legacy

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import com.example.engine.signal.FaceState

/**
 * Legacy Canvas-based StickerRenderer fallback path for simple 2D overlays.
 */
class StickerRenderer {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
    private val matrix = Matrix()
    private val drawRect = RectF()

    fun drawSticker(
        canvas: Canvas,
        bitmap: Bitmap,
        faceState: FaceState,
        viewWidth: Float,
        viewHeight: Float,
        anchorX: Float = 0f,
        anchorY: Float = 0f,
        scaleFW: Float = 0.5f,
        opacity: Float = 1.0f
    ) {
        if (!faceState.isVisible || opacity < 0.01f) return

        val fwPixels = (faceState.faceWidth * viewWidth).coerceAtLeast(60f)
        val w = scaleFW * fwPixels
        val h = w * (bitmap.height.toFloat() / bitmap.width.toFloat())

        // Projected face position
        val px = (faceState.translation[0] + 0.5f) * viewWidth + anchorX * fwPixels
        val py = (-faceState.translation[1] + 0.5f) * viewHeight + anchorY * fwPixels

        paint.alpha = (opacity * 255).toInt().coerceIn(0, 255)

        canvas.save()
        canvas.translate(px, py)
        canvas.rotate(-faceState.rollDeg)

        drawRect.set(-w * 0.5f, -h * 0.5f, w * 0.5f, h * 0.5f)
        canvas.drawBitmap(bitmap, null, drawRect, paint)
        canvas.restore()
    }
}
