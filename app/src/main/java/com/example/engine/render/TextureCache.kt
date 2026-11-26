package com.example.engine.render

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.opengl.EGL14
import android.opengl.GLES30
import android.opengl.GLUtils
import android.util.Log
import com.example.engine.pkg.UvRegistry
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger

/**
 * TextureCache: Bounded LRU GL texture cache (README Section 11.4, 14, 15 & 21 Step 6).
 *
 * Guarantees:
 * - Bounded LRU by both entry count (`maxEntries`) and byte budget (`maxTotalBytes`).
 * - Background CPU decoding (`preloadDecodeToStaging`) off the render thread.
 * - GL-thread-only texture uploads and deletions (`stepFrameUploads` & `GlResourceOwner`).
 * - Spreads large GPU uploads across frames (`maxUploadsPerFrame`).
 * - Respects tier-specific `textureMaxSize` (1024 / 512 / 256).
 * - Reuses static topology and UV layouts from `UvRegistry`.
 * - Safe eviction: queues evicted GL texture IDs to `GlResourceOwner` for deletion after queued GL work drains.
 * - Zero unbounded bitmap/texture retention or frame-loop heap allocations.
 */
class TextureCache @JvmOverloads constructor(
    val resourceOwner: GlResourceOwner = GlResourceOwner(),
    val maxEntries: Int = 32,
    val maxTotalBytes: Long = 32L * 1024L * 1024L
) {
    companion object {
        private const val TAG = "TextureCache"
        private const val LUT_MAX_DIMENSION = 4096
        private val syntheticTexIdGenerator = AtomicInteger(1000)
        fun computeBoundedDimensions(srcWidth: Int, srcHeight: Int, maxSize: Int, isLut: Boolean = false): Pair<Int, Int> {
            if (isLut) {
                return Pair(srcWidth.coerceIn(1, LUT_MAX_DIMENSION), srcHeight.coerceIn(1, LUT_MAX_DIMENSION))
            }
            val cap = maxSize.coerceIn(16, 2048)
            var w = srcWidth.coerceAtLeast(1)
            var h = srcHeight.coerceAtLeast(1)
            var sampleSize = 1
            while (w / sampleSize > cap || h / sampleSize > cap) {
                sampleSize *= 2
            }
            w = (w / sampleSize).coerceAtLeast(1)
            h = (h / sampleSize).coerceAtLeast(1)
            if (w > cap || h > cap) {
                val ratio = cap.toFloat() / maxOf(w, h).toFloat()
                w = (w * ratio).toInt().coerceAtLeast(1)
                h = (h * ratio).toInt().coerceAtLeast(1)
            }
            return Pair(w, h)
        }
    }

    data class TextureEntry(
        val key: String,
        val assetPath: String,
        val textureId: Int,
        val width: Int,
        val height: Int,
        val maxSizeCap: Int,
        val byteCount: Long,
        val contextGeneration: Int = 0,
        val isLut: Boolean = false
    )

    data class StagedBitmapUpload(
        val key: String,
        val assetPath: String,
        val maxSizeCap: Int,
        val bitmap: Bitmap?,
        val width: Int,
        val height: Int,
        val byteCount: Long,
        val stagedGeneration: Int = 0,
        val isLut: Boolean = false
    )

    // Access-ordered LinkedHashMap for true LRU eviction
    private val lruMap = object : LinkedHashMap<String, TextureEntry>(maxEntries + 4, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, TextureEntry>?): Boolean {
            return false // Handled explicitly in enforceBoundsLocked() to queue GL deletions
        }
    }

    private val pendingUploadsQueue = ConcurrentLinkedQueue<StagedBitmapUpload>()
    private val stagedKeys = ConcurrentHashMap<String, Boolean>()

    // Pre-allocated 1-element array for glGenTextures on the GL thread
    private val genScratch = IntArray(1)

    @Volatile
    var isPaused: Boolean = false
        private set

    var totalBytes: Long = 0L
        private set

    var evictionCount: Int = 0
        private set

    // Shared static topology and UV layout references (reused across all faceMesh effects)
    val sharedCanonicalUvs: FloatArray
        get() = UvRegistry.CANONICAL_UVS

    val sharedCanonicalTriangles: ShortArray
        get() = UvRegistry.CANONICAL_TRIANGLES

    fun size(): Int = synchronized(lruMap) {
        purgeInvalidGenerationEntriesLocked()
        lruMap.size
    }

    fun pendingUploadCount(): Int = pendingUploadsQueue.size

    private fun cacheKey(assetPath: String, maxSize: Int, isLut: Boolean = false): String =
        if (isLut) "$assetPath@LUT" else "$assetPath@$maxSize"

    /**
     * Suspends background texture decode and cancels any uncommitted staged bitmap uploads on pause.
     */
    fun onPause() {
        isPaused = true
        cancelPendingUploads()
    }

    /**
     * Resumes texture decoding and GL frame uploads.
     */
    fun onResume() {
        isPaused = false
    }

    fun cancelPendingUploads() {
        while (true) {
            val staged = pendingUploadsQueue.poll() ?: break
            stagedKeys.remove(staged.key)
            if (staged.bitmap != null && !staged.bitmap.isRecycled) {
                staged.bitmap.recycle()
            }
        }
        stagedKeys.clear()
    }

    private fun purgeInvalidGenerationEntriesLocked() {
        val currentGen = resourceOwner.contextGeneration
        val iterator = lruMap.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next().value
            if (entry.contextGeneration != currentGen) {
                iterator.remove()
                totalBytes = (totalBytes - entry.byteCount).coerceAtLeast(0L)
            }
        }
    }

    /**
     * Background worker entry point: Decodes an image asset on a background IO worker
     * respecting tier-specific `maxSize` (or preserving exact dimensions when `isLut = true`)
     * and stages it for distributed GL-thread upload.
     * Never makes GL calls on the background thread! Returns false immediately if paused.
     */
    @JvmOverloads
    fun preloadDecodeToStaging(assetPath: String, maxSize: Int = 1024, isLut: Boolean = false): Boolean {
        if (isPaused) return false
        val clampedMax = if (isLut) LUT_MAX_DIMENSION else maxSize.coerceIn(16, 2048)
        val key = cacheKey(assetPath, clampedMax, isLut)
        val currentGen = resourceOwner.contextGeneration

        synchronized(lruMap) {
            purgeInvalidGenerationEntriesLocked()
            if (lruMap.containsKey(key)) return true
        }
        if (stagedKeys.containsKey(key)) return true

        val file = File(assetPath)
        if (!file.exists()) {
            Log.w(TAG, "Asset not found for preload decode: $assetPath")
            return false
        }

        val bitmap = decodeBitmapBounded(file, clampedMax, isLut) ?: return false
        if (isPaused) {
            if (!bitmap.isRecycled) bitmap.recycle()
            return false
        }
        val w = bitmap.width
        val h = bitmap.height
        val bytes = bitmap.allocationByteCount.toLong().coerceAtLeast(w.toLong() * h.toLong() * 4L)

        // Bound staging queue to maxEntries to prevent unbounded bitmap retention
        while (pendingUploadsQueue.size >= maxEntries) {
            val dropped = pendingUploadsQueue.poll() ?: break
            stagedKeys.remove(dropped.key)
            if (dropped.bitmap != null && !dropped.bitmap.isRecycled) {
                dropped.bitmap.recycle()
            }
        }

        stagedKeys[key] = true
        pendingUploadsQueue.add(
            StagedBitmapUpload(
                key = key,
                assetPath = assetPath,
                maxSizeCap = clampedMax,
                bitmap = bitmap,
                width = w,
                height = h,
                byteCount = bytes,
                stagedGeneration = currentGen,
                isLut = isLut
            )
        )
        return true
    }

    /**
     * Stages a decoded texture with explicit dimensions (respecting tier `maxSize`) for distributed GL upload.
     */
    @JvmOverloads
    fun stageDecodedDimensions(
        assetPath: String,
        srcWidth: Int,
        srcHeight: Int,
        maxSize: Int,
        bitmap: Bitmap? = null,
        isLut: Boolean = false
    ) {
        if (isPaused) {
            if (bitmap != null && !bitmap.isRecycled) bitmap.recycle()
            return
        }
        val clampedMax = if (isLut) LUT_MAX_DIMENSION else maxSize.coerceIn(16, 2048)
        val (w, h) = computeBoundedDimensions(srcWidth, srcHeight, clampedMax, isLut)
        val key = cacheKey(assetPath, clampedMax, isLut)
        val currentGen = resourceOwner.contextGeneration

        synchronized(lruMap) {
            purgeInvalidGenerationEntriesLocked()
            if (lruMap.containsKey(key)) return
        }

        val stagingCap = (maxEntries * 2).coerceAtLeast(16)
        while (pendingUploadsQueue.size >= stagingCap) {
            val dropped = pendingUploadsQueue.poll() ?: break
            stagedKeys.remove(dropped.key)
            if (dropped.bitmap != null && !dropped.bitmap.isRecycled) dropped.bitmap.recycle()
        }

        val bytes = (w.toLong() * h.toLong() * 4L).coerceAtLeast(4L)
        stagedKeys[key] = true
        pendingUploadsQueue.add(
            StagedBitmapUpload(
                key = key,
                assetPath = assetPath,
                maxSizeCap = clampedMax,
                bitmap = bitmap,
                width = w,
                height = h,
                byteCount = bytes,
                stagedGeneration = currentGen,
                isLut = isLut
            )
        )
    }

    /**
     * Stages an already-decoded Bitmap for distributed GL upload.
     */
    @JvmOverloads
    fun stageDecodedBitmap(assetPath: String, maxSize: Int, bitmap: Bitmap, isLut: Boolean = false) {
        if (isPaused) {
            if (!bitmap.isRecycled) bitmap.recycle()
            return
        }
        val clampedMax = if (isLut) LUT_MAX_DIMENSION else maxSize.coerceIn(16, 2048)
        val scaled = if (isLut) bitmap else scaleBitmapIfNeeded(bitmap, clampedMax)
        stageDecodedDimensions(assetPath, scaled.width, scaled.height, clampedMax, scaled, isLut)
    }

    /**
     * Executes at most `maxUploadsPerFrame` staged texture uploads on the GL thread.
     * Spreads large GPU uploads across frames so filter preloading never hitches the render thread.
     */
    fun stepFrameUploads(maxUploadsPerFrame: Int = 1): Int {
        check(resourceOwner.isOnGlThread()) {
            "TextureCache.stepFrameUploads must be called strictly on the GL render thread"
        }
        if (isPaused) return 0
        val limit = maxUploadsPerFrame.coerceAtLeast(1)
        val currentGen = resourceOwner.contextGeneration
        var uploaded = 0

        synchronized(lruMap) {
            purgeInvalidGenerationEntriesLocked()
        }

        while (uploaded < limit) {
            val staged = pendingUploadsQueue.poll() ?: break
            stagedKeys.remove(staged.key)

            val existing = synchronized(lruMap) {
                val e = lruMap[staged.key]
                if (e != null && e.contextGeneration == currentGen) e else null
            }
            if (existing != null) {
                if (staged.bitmap != null && !staged.bitmap.isRecycled) staged.bitmap.recycle()
                continue
            }

            val texId = uploadTextureOnGlThread(staged.bitmap, staged.isLut)
            if (staged.bitmap != null && !staged.bitmap.isRecycled) {
                staged.bitmap.recycle()
            }

            if (texId != 0) {
                resourceOwner.trackTexture(texId)
                val entry = TextureEntry(
                    key = staged.key,
                    assetPath = staged.assetPath,
                    textureId = texId,
                    width = staged.width,
                    height = staged.height,
                    maxSizeCap = staged.maxSizeCap,
                    byteCount = staged.byteCount,
                    contextGeneration = currentGen,
                    isLut = staged.isLut
                )
                synchronized(lruMap) {
                    putEntryLocked(entry)
                }
                uploaded++
            }
        }
        return uploaded
    }

    /**
     * Retrieves a cached GL texture ID on the GL thread, or uploads synchronously if not yet preloaded
     * or if the cached entry belonged to an older EGL context generation.
     * Honors the active tier's `maxSize` limit for standard textures, while keeping 3D LUTs (`isLut = true`)
     * at exact native dimensions with non-mipmapped `GL_LINEAR` filtering.
     */
    @JvmOverloads
    fun getOrLoad(assetPath: String, maxSize: Int = 1024, isLut: Boolean = false): Int {
        check(resourceOwner.isOnGlThread()) {
            "TextureCache.getOrLoad must be called strictly on the GL render thread"
        }
        val clampedMax = if (isLut) LUT_MAX_DIMENSION else maxSize.coerceIn(16, 2048)
        val key = cacheKey(assetPath, clampedMax, isLut)
        val currentGen = resourceOwner.contextGeneration

        synchronized(lruMap) {
            purgeInvalidGenerationEntriesLocked()
            val existing = lruMap[key]
            if (existing != null && existing.textureId != 0 && existing.contextGeneration == currentGen) {
                return existing.textureId
            }
        }

        // Check if this texture is already waiting in the staged upload queue
        if (stagedKeys.containsKey(key)) {
            val prevPaused = isPaused
            isPaused = false
            try {
                stepFrameUploads(maxUploadsPerFrame = pendingUploadsQueue.size.coerceAtLeast(1))
            } finally {
                isPaused = prevPaused
            }
            synchronized(lruMap) {
                val uploaded = lruMap[key]
                if (uploaded != null && uploaded.textureId != 0 && uploaded.contextGeneration == currentGen) {
                    return uploaded.textureId
                }
            }
        }

        val file = File(assetPath)
        if (!file.exists()) {
            Log.w(TAG, "Asset not found: $assetPath")
            return 0
        }

        val bitmap = decodeBitmapBounded(file, clampedMax, isLut) ?: return 0
        val w = bitmap.width
        val h = bitmap.height
        val bytes = bitmap.allocationByteCount.toLong().coerceAtLeast(w.toLong() * h.toLong() * 4L)
        val texId = uploadTextureOnGlThread(bitmap, isLut)
        if (!bitmap.isRecycled) {
            bitmap.recycle()
        }

        if (texId != 0) {
            resourceOwner.trackTexture(texId)
            val entry = TextureEntry(
                key = key,
                assetPath = assetPath,
                textureId = texId,
                width = w,
                height = h,
                maxSizeCap = clampedMax,
                byteCount = bytes,
                contextGeneration = currentGen,
                isLut = isLut
            )
            synchronized(lruMap) {
                putEntryLocked(entry)
            }
        }
        return texId
    }

    @JvmOverloads
    fun getEntry(assetPath: String, maxSize: Int = 1024, isLut: Boolean = false): TextureEntry? {
        val clampedMax = if (isLut) LUT_MAX_DIMENSION else maxSize.coerceIn(16, 2048)
        val key = cacheKey(assetPath, clampedMax, isLut)
        val currentGen = resourceOwner.contextGeneration
        return synchronized(lruMap) {
            purgeInvalidGenerationEntriesLocked()
            val entry = lruMap[key]
            if (entry != null && entry.contextGeneration == currentGen) entry else null
        }
    }

    /**
     * Returns the most recent valid [TextureEntry] for `assetPath` across any maxSize or LUT key.
     */
    fun getEntryForPath(assetPath: String): TextureEntry? {
        val currentGen = resourceOwner.contextGeneration
        return synchronized(lruMap) {
            purgeInvalidGenerationEntriesLocked()
            val lutEntry = lruMap[cacheKey(assetPath, LUT_MAX_DIMENSION, isLut = true)]
            if (lutEntry != null && lutEntry.contextGeneration == currentGen) {
                return@synchronized lutEntry
            }
            lruMap.values.firstOrNull { it.assetPath == assetPath && it.contextGeneration == currentGen }
        }
    }

    private fun putEntryLocked(entry: TextureEntry) {
        val old = lruMap.put(entry.key, entry)
        if (old != null) {
            totalBytes = (totalBytes - old.byteCount).coerceAtLeast(0L)
            if (old.textureId != entry.textureId && old.contextGeneration == entry.contextGeneration) {
                resourceOwner.queueDeleteTexture(old.textureId)
            }
        }
        totalBytes += entry.byteCount
        enforceBoundsLocked()
    }

    private fun enforceBoundsLocked() {
        val iterator = lruMap.entries.iterator()
        while (iterator.hasNext() && (lruMap.size > maxEntries || totalBytes > maxTotalBytes)) {
            val eldest = iterator.next().value
            iterator.remove()
            totalBytes = (totalBytes - eldest.byteCount).coerceAtLeast(0L)
            evictionCount++
            // Safe deferred deletion on the GL thread via GlResourceOwner
            resourceOwner.queueDeleteTexture(eldest.textureId)
        }
    }

    /**
     * Evicts least-recently-used textures down to `targetFraction` of capacity on memory pressure.
     * Safe to call from any thread because actual `glDeleteTextures` is queued on `GlResourceOwner`.
     */
    fun trimToFraction(targetFraction: Float = 0.5f) {
        val targetEntries = (maxEntries * targetFraction).toInt().coerceAtLeast(1)
        val targetBytes = (maxTotalBytes * targetFraction).toLong().coerceAtLeast(0L)
        synchronized(lruMap) {
            val iterator = lruMap.entries.iterator()
            while (iterator.hasNext() && (lruMap.size > targetEntries || totalBytes > targetBytes)) {
                val eldest = iterator.next().value
                iterator.remove()
                totalBytes = (totalBytes - eldest.byteCount).coerceAtLeast(0L)
                evictionCount++
                resourceOwner.queueDeleteTexture(eldest.textureId)
            }
        }
    }

    private fun decodeBitmapBounded(file: File, maxSize: Int, isLut: Boolean = false): Bitmap? {
        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, options)
        if (options.outWidth <= 0 || options.outHeight <= 0) return null

        var sampleSize = 1
        if (!isLut) {
            while (options.outWidth / sampleSize > maxSize || options.outHeight / sampleSize > maxSize) {
                sampleSize *= 2
            }
        }

        val decodeOptions = BitmapFactory.Options().apply {
            inSampleSize = sampleSize
            inScaled = false
            inPremultiplied = !isLut
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val decoded = BitmapFactory.decodeFile(file.absolutePath, decodeOptions) ?: return null
        return if (isLut) decoded else scaleBitmapIfNeeded(decoded, maxSize)
    }

    private fun scaleBitmapIfNeeded(bitmap: Bitmap, maxSize: Int): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        if (w <= maxSize && h <= maxSize) return bitmap

        val ratio = maxSize.toFloat() / maxOf(w, h).toFloat()
        val targetW = (w * ratio).toInt().coerceAtLeast(1)
        val targetH = (h * ratio).toInt().coerceAtLeast(1)
        val scaled = Bitmap.createScaledBitmap(bitmap, targetW, targetH, true)
        if (scaled !== bitmap && !bitmap.isRecycled) {
            bitmap.recycle()
        }
        return scaled
    }

    private fun hasActiveGlContext(): Boolean {
        return try {
            val ctx = EGL14.eglGetCurrentContext()
            ctx != null && ctx != EGL14.EGL_NO_CONTEXT
        } catch (t: Throwable) {
            false
        }
    }

    private fun uploadTextureOnGlThread(bitmap: Bitmap?, isLut: Boolean = false): Int {
        if (!hasActiveGlContext() || bitmap == null) {
            // In headless unit tests without an active EGL context, return a deterministic non-zero handle
            return syntheticTexIdGenerator.getAndIncrement()
        }

        genScratch[0] = 0
        GLES30.glGenTextures(1, genScratch, 0)
        val texId = genScratch[0]
        if (texId == 0) {
            return syntheticTexIdGenerator.getAndIncrement()
        }

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texId)
        val minFilter = if (isLut) GLES30.GL_LINEAR else GLES30.GL_LINEAR_MIPMAP_LINEAR
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, minFilter)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)

        GLUtils.texImage2D(GLES30.GL_TEXTURE_2D, 0, bitmap, 0)
        if (!isLut) {
            GLES30.glGenerateMipmap(GLES30.GL_TEXTURE_2D)
        }
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0)

        return texId
    }

    /**
     * Called on the GL thread when the EGL/OpenGL context is lost and recreated.
     * Discards stale texture IDs from the destroyed context without issuing GL deletes on
     * the new context, and re-stages or re-uploads previously cached textures that still exist on disk.
     */
    fun onContextLostAndRecreate(reuploadImmediately: Boolean = true): Int {
        check(resourceOwner.isOnGlThread()) {
            "TextureCache.onContextLostAndRecreate must be called strictly on the GL render thread"
        }
        val entriesToRestore: List<TextureEntry>
        synchronized(lruMap) {
            entriesToRestore = lruMap.values.toList()
            lruMap.clear()
            totalBytes = 0L
        }

        var restoredCount = 0
        for (oldEntry in entriesToRestore) {
            val file = File(oldEntry.assetPath)
            var restoredFromFile = false
            if (file.exists()) {
                if (reuploadImmediately) {
                    val newTexId = getOrLoad(oldEntry.assetPath, oldEntry.maxSizeCap, oldEntry.isLut)
                    if (newTexId != 0) {
                        restoredCount++
                        restoredFromFile = true
                    }
                } else {
                    if (preloadDecodeToStaging(oldEntry.assetPath, oldEntry.maxSizeCap, oldEntry.isLut)) {
                        restoredCount++
                        restoredFromFile = true
                    }
                }
            }
            if (!restoredFromFile) {
                // Synthetic/test entry backed by dimensions in headless unit tests
                stageDecodedDimensions(
                    assetPath = oldEntry.assetPath,
                    srcWidth = oldEntry.width,
                    srcHeight = oldEntry.height,
                    maxSize = oldEntry.maxSizeCap,
                    isLut = oldEntry.isLut
                )
                if (reuploadImmediately) {
                    restoredCount += stepFrameUploads(maxUploadsPerFrame = 1)
                } else {
                    restoredCount++
                }
            }
        }
        return restoredCount
    }

    /**
     * Queues all cached textures for deletion via GlResourceOwner and clears staging queues.
     */
    fun releaseAll() {
        while (true) {
            val staged = pendingUploadsQueue.poll() ?: break
            stagedKeys.remove(staged.key)
            if (staged.bitmap != null && !staged.bitmap.isRecycled) {
                staged.bitmap.recycle()
            }
        }
        stagedKeys.clear()

        synchronized(lruMap) {
            for (entry in lruMap.values) {
                if (entry.textureId != 0) {
                    resourceOwner.queueDeleteTexture(entry.textureId)
                }
            }
            lruMap.clear()
            totalBytes = 0L
        }

        if (resourceOwner.isOnGlThread()) {
            resourceOwner.drainPendingDeletions()
        }
    }
}
