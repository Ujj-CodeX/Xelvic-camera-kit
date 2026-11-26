package com.example.engine.capture

import android.graphics.SurfaceTexture
import android.opengl.Matrix
import android.os.Handler
import android.os.Looper
import android.util.Size
import android.view.Surface
import androidx.camera.core.Preview
import androidx.core.content.ContextCompat
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs

/**
 * FrameRing interface (Chunk B0):
 * Stores camera frame slots with their hardware sensor timestamps (`SurfaceTexture.getTimestamp()`)
 * and single-owner texture transform matrices.
 *
 * - Default `delayDepth` is `0` (returns the most recently recorded camera frame immediately).
 * - Future chunks (e.g., tracker-to-camera frame synchronization) can configure `delayDepth > 0`
 *   without changing `CameraInputSurface` or `SingleSceneCompositionSurface`.
 */
interface FrameRing {
    val delayDepth: Int
    val capacity: Int
    val latestSensorTimestampNs: Long
    val latestAlignedElapsedRealtimeNs: Long
    val recordedFrameCount: Long

    data class FrameSlot(
        var textureId: Int = 0,
        var sensorTimestampNs: Long = 0L,
        var alignedElapsedRealtimeNs: Long = 0L,
        val texMatrix: FloatArray = FloatArray(16).apply { Matrix.setIdentityM(this, 0) },
        var isMirroredH: Boolean = true,
        var isExternalOes: Boolean = true
    )

    /**
     * Records a new camera frame and its sensor timestamp into the ring buffer with zero heap allocation.
     */
    fun recordFrame(
        textureId: Int,
        sensorTimestampNs: Long,
        alignedElapsedRealtimeNs: Long,
        texMatrix: FloatArray,
        isMirroredH: Boolean = true,
        isExternalOes: Boolean = true
    ): FrameSlot

    /**
     * Reads the camera frame slot at [delayDepth] (0 = latest frame).
     */
    fun peekFrame(depth: Int = delayDepth, outTexMatrix: FloatArray? = null): FrameSlot?

    /**
     * Finds the recorded camera frame slot whose `sensorTimestampNs` (or `alignedElapsedRealtimeNs`)
     * is closest to [targetTimestampNs], enabling optional tracker-to-camera timestamp synchronization.
     */
    fun findFrameClosestToTimestampNs(
        targetTimestampNs: Long,
        useAlignedClock: Boolean = false,
        outTexMatrix: FloatArray? = null
    ): FrameSlot? = peekFrame(delayDepth, outTexMatrix)

    fun clear()
}

/**
 * Zero-allocation ring buffer implementation of [FrameRing] (default `delayDepth = 0`).
 */
class DefaultFrameRing @JvmOverloads constructor(
    override var delayDepth: Int = 0,
    override val capacity: Int = 8
) : FrameRing {

    init {
        require(capacity >= 1) { "FrameRing capacity must be >= 1" }
        require(delayDepth in 0 until capacity) { "delayDepth ($delayDepth) must be in 0 until $capacity" }
    }

    private val slots = Array(capacity) { FrameRing.FrameSlot() }
    private var writeIndex = 0
    private var count = 0

    @Volatile
    override var latestSensorTimestampNs: Long = 0L
        private set

    @Volatile
    override var latestAlignedElapsedRealtimeNs: Long = 0L
        private set

    private val frameCounter = AtomicLong(0L)
    override val recordedFrameCount: Long
        get() = frameCounter.get()

    @Synchronized
    override fun recordFrame(
        textureId: Int,
        sensorTimestampNs: Long,
        alignedElapsedRealtimeNs: Long,
        texMatrix: FloatArray,
        isMirroredH: Boolean,
        isExternalOes: Boolean
    ): FrameRing.FrameSlot {
        val slot = slots[writeIndex]
        slot.textureId = textureId
        slot.sensorTimestampNs = sensorTimestampNs
        slot.alignedElapsedRealtimeNs = alignedElapsedRealtimeNs
        if (texMatrix.size >= 16) {
            System.arraycopy(texMatrix, 0, slot.texMatrix, 0, 16)
        } else {
            Matrix.setIdentityM(slot.texMatrix, 0)
        }
        slot.isMirroredH = isMirroredH
        slot.isExternalOes = isExternalOes

        writeIndex = (writeIndex + 1) % capacity
        if (count < capacity) {
            count++
        }
        latestSensorTimestampNs = sensorTimestampNs
        latestAlignedElapsedRealtimeNs = alignedElapsedRealtimeNs
        frameCounter.incrementAndGet()
        return slot
    }

    @Synchronized
    override fun peekFrame(depth: Int, outTexMatrix: FloatArray?): FrameRing.FrameSlot? {
        if (count == 0) return null
        val clampedDepth = depth.coerceIn(0, count - 1)
        val idx = (writeIndex - 1 - clampedDepth + capacity * 2) % capacity
        val slot = slots[idx]
        if (outTexMatrix != null && outTexMatrix.size >= 16) {
            System.arraycopy(slot.texMatrix, 0, outTexMatrix, 0, 16)
        }
        return slot
    }

    @Synchronized
    override fun findFrameClosestToTimestampNs(
        targetTimestampNs: Long,
        useAlignedClock: Boolean,
        outTexMatrix: FloatArray?
    ): FrameRing.FrameSlot? {
        if (count == 0) return null
        var bestSlot: FrameRing.FrameSlot? = null
        var bestDiff = Long.MAX_VALUE
        for (d in 0 until count) {
            val idx = (writeIndex - 1 - d + capacity * 2) % capacity
            val candidate = slots[idx]
            val ts = if (useAlignedClock) candidate.alignedElapsedRealtimeNs else candidate.sensorTimestampNs
            val diff = abs(ts - targetTimestampNs)
            if (diff < bestDiff) {
                bestDiff = diff
                bestSlot = candidate
            }
        }
        if (bestSlot != null && outTexMatrix != null && outTexMatrix.size >= 16) {
            System.arraycopy(bestSlot.texMatrix, 0, outTexMatrix, 0, 16)
        }
        return bestSlot
    }

    @Synchronized
    override fun clear() {
        writeIndex = 0
        count = 0
        latestSensorTimestampNs = 0L
        latestAlignedElapsedRealtimeNs = 0L
    }
}

/**
 * CameraInputSurface (Chunk B0):
 * Modeled after a clean single-owner camera input stage:
 * - `attachToGlContext(textureId)`
 * - `readFrame()`
 * - `subscribe(onFrameAvailable)`
 * - `options` (`cropCenter`, `mirrorH`, `mirrorV`, `pausable`)
 *
 * Enforces the **Single-Owner Input Transform Rule**:
 * Mirror (`mirrorH`, `mirrorV`), aspect-ratio center crop (`cropCenter`), and sensor/display
 * rotation are applied **only** in this input stage so downstream shaders (`SingleSceneCompositionSurface`,
 * `FaceMeshRenderer`, `PostRenderer`, `OffscreenRenderer`) never double-mirror or re-crop.
 */
interface CameraInputSurface {

    data class InputOptions(
        val cropCenter: Boolean = true,
        val mirrorH: Boolean = true,
        val mirrorV: Boolean = false,
        val pausable: Boolean = true,
        val rotationDegrees: Int = 0,
        val sourceWidth: Int = 1080,
        val sourceHeight: Int = 1920,
        val viewportWidth: Int = 1080,
        val viewportHeight: Int = 1920
    )

    val options: InputOptions
    val frameRing: FrameRing
    val isAttachedToGl: Boolean
    val boundTextureId: Int
    val isPaused: Boolean
    val hasValidFrame: Boolean
    val lastSensorTimestampNs: Long

    /**
     * Attaches this camera input to the active GL context with [textureId] (`GL_TEXTURE_EXTERNAL_OES`).
     */
    fun attachToGlContext(textureId: Int)

    /**
     * Detaches from the current GL context during context loss / `onStop` without destroying the
     * underlying CameraX `SurfaceProvider` connection when possible.
     */
    fun detachFromGlContext()

    /**
     * Updates the GL texture image to the latest camera frame on the GL thread, records its
     * sensor timestamp in [frameRing], and writes the single-owner 4x4 UV transform matrix
     * (including `mirrorH`, `mirrorV`, `cropCenter`, and `rotationDegrees`) into [outTexMatrix].
     */
    fun readFrame(outTexMatrix: FloatArray? = null): FrameRing.FrameSlot?

    /**
     * Updates the GL texture image on the GL thread and returns the recorded camera frame slot
     * closest to [targetTimestampNs] when [targetTimestampNs] > 0L (or falls back to
     * `frameRing.peekFrame(frameRing.delayDepth)`).
     */
    fun readFrame(
        outTexMatrix: FloatArray?,
        targetTimestampNs: Long,
        useAlignedClock: Boolean = true
    ): FrameRing.FrameSlot? = readFrame(outTexMatrix)

    /**
     * Subscribes a callback invoked whenever a new camera frame arrives.
     */
    fun subscribe(onFrameAvailable: () -> Unit)

    /**
     * Updates input options (`cropCenter`, `mirrorH`, `mirrorV`, `pausable`, `rotationDegrees`, dimensions).
     */
    fun updateOptions(newOptions: InputOptions)

    /**
     * Pauses or resumes frame consumption when [InputOptions.pausable] is enabled.
     */
    fun setPaused(paused: Boolean)

    /**
     * Releases all underlying `SurfaceTexture` and `Surface` resources.
     */
    fun release()

    companion object {
        /**
         * Computes the single-owner 4x4 UV texture transform matrix combining:
         * 1. Horizontal & vertical mirror (`mirrorH`, `mirrorV`) about UV center `(0.5, 0.5)`.
         * 2. `FILL_CENTER` aspect-ratio crop (`cropCenter`) matching `sourceWidth/sourceHeight`
         *    (after `rotationDegrees`) to `viewportWidth/viewportHeight`.
         * 3. Additional rotation (`rotationDegrees` when not already baked into `rawSurfaceTransform16`).
         * 4. The `SurfaceTexture.getTransformMatrix` 4x4 matrix (`rawSurfaceTransform16`).
         *
         * Because mirroring, cropping, and rotation are folded into `outMatrix16` here,
         * `SingleSceneCompositionSurface` sets `uMirrorX = 0` (single owner).
         */
        fun computeSingleOwnerInputMatrix(
            options: InputOptions,
            rawSurfaceTransform16: FloatArray? = null,
            applyAdditionalRotation: Boolean = false,
            outMatrix16: FloatArray = FloatArray(16)
        ): FloatArray {
            val isIdentityRawSt = rawSurfaceTransform16 == null ||
                rawSurfaceTransform16.size < 16 ||
                (abs(rawSurfaceTransform16[0] - 1f) < 1e-4f &&
                    abs(rawSurfaceTransform16[5] - 1f) < 1e-4f &&
                    abs(rawSurfaceTransform16[1]) < 1e-4f &&
                    abs(rawSurfaceTransform16[4]) < 1e-4f)

            val hasRotation90Or270InRawSt = !isIdentityRawSt &&
                rawSurfaceTransform16 != null &&
                abs(rawSurfaceTransform16[1]) > abs(rawSurfaceTransform16[0])

            val rotNorm = ((options.rotationDegrees % 360) + 360) % 360
            val swapDims = hasRotation90Or270InRawSt ||
                rotNorm == 90 ||
                rotNorm == 270 ||
                (!isIdentityRawSt && options.sourceWidth > options.sourceHeight && options.viewportWidth < options.viewportHeight)

            val orientedSrcW = if (swapDims) {
                options.sourceHeight.coerceAtLeast(1)
            } else {
                options.sourceWidth.coerceAtLeast(1)
            }
            val orientedSrcH = if (swapDims) {
                options.sourceWidth.coerceAtLeast(1)
            } else {
                options.sourceHeight.coerceAtLeast(1)
            }

            val srcAspect = orientedSrcW.toFloat() / orientedSrcH.toFloat()
            val viewW = options.viewportWidth.coerceAtLeast(1)
            val viewH = options.viewportHeight.coerceAtLeast(1)
            val viewAspect = viewW.toFloat() / viewH.toFloat()

            var scaleU = 1f
            var scaleV = 1f
            if (options.cropCenter && abs(srcAspect - viewAspect) > 1e-4f) {
                if (srcAspect > viewAspect) {
                    // Source is wider than viewport: crop horizontal edges (scale U < 1)
                    scaleU = (viewAspect / srcAspect).coerceIn(0.05f, 1f)
                } else {
                    // Source is taller than viewport: crop vertical edges (scale V < 1)
                    scaleV = (srcAspect / viewAspect).coerceIn(0.05f, 1f)
                }
            }

            // A standard Android SurfaceTexture transform has a vertical buffer flip (det2x2 < 0 when unmirrored,
            // det2x2 > 0 when HAL already applied horizontal mirror for the front camera).
            val shouldMirrorU = if (!isIdentityRawSt && rawSurfaceTransform16 != null) {
                val det2x2 = rawSurfaceTransform16[0] * rawSurfaceTransform16[5] -
                    rawSurfaceTransform16[1] * rawSurfaceTransform16[4]
                val alreadyMirroredHInRawSt = det2x2 > 0f
                options.mirrorH != alreadyMirroredHInRawSt
            } else {
                options.mirrorH
            }

            val signU = if (shouldMirrorU) -1f else 1f
            val signV = if (options.mirrorV) -1f else 1f

            // Build UV-center transform T(0.5, 0.5) * R(rot) * S(signU * scaleU, signV * scaleV) * T(-0.5, -0.5)
            val uvAdjust = FloatArray(16)
            val scratch = FloatArray(16)
            Matrix.setIdentityM(uvAdjust, 0)
            Matrix.translateM(uvAdjust, 0, 0.5f, 0.5f, 0f)
            val needsExtraRot = (applyAdditionalRotation && rotNorm != 0) ||
                (!isIdentityRawSt && !hasRotation90Or270InRawSt && (rotNorm == 90 || rotNorm == 270))
            if (needsExtraRot) {
                val hasVerticalFlipInRawSt = !isIdentityRawSt &&
                    rawSurfaceTransform16 != null &&
                    rawSurfaceTransform16[5] < 0f
                val angleDeg = if (hasVerticalFlipInRawSt) rotNorm.toFloat() else -rotNorm.toFloat()
                Matrix.rotateM(uvAdjust, 0, angleDeg, 0f, 0f, 1f)
            }
            Matrix.scaleM(uvAdjust, 0, signU * scaleU, signV * scaleV, 1f)
            Matrix.translateM(uvAdjust, 0, -0.5f, -0.5f, 0f)

            if (rawSurfaceTransform16 != null && rawSurfaceTransform16.size >= 16) {
                Matrix.multiplyMM(scratch, 0, rawSurfaceTransform16, 0, uvAdjust, 0)
                System.arraycopy(scratch, 0, outMatrix16, 0, 16)
            } else {
                System.arraycopy(uvAdjust, 0, outMatrix16, 0, 16)
            }
            return outMatrix16
        }
    }
}

/**
 * Default implementation of [CameraInputSurface] backed by an Android [SurfaceTexture] for
 * CameraX `Preview.SurfaceProvider` in Path B (`SINGLE_SCENE_GL`), with deterministic
 * synthetic frame support for unit testing.
 */
class SurfaceTextureCameraInputSurface @JvmOverloads constructor(
    initialOptions: CameraInputSurface.InputOptions = CameraInputSurface.InputOptions(),
    override val frameRing: FrameRing = DefaultFrameRing(delayDepth = 0),
    var clockOffsetCalibrator: ClockOffsetCalibrator = ClockOffsetCalibrator()
) : CameraInputSurface {

    @Volatile
    var syncFrameRingToTrackerTimestamp: Boolean = false

    @Volatile
    override var options: CameraInputSurface.InputOptions = initialOptions
        private set

    private val attachedFlag = AtomicBoolean(false)
    private val stAttachedToGl = AtomicBoolean(false)
    override val isAttachedToGl: Boolean
        get() = attachedFlag.get() && (surfaceTexture == null || stAttachedToGl.get())

    private val textureIdRef = AtomicInteger(0)
    override val boundTextureId: Int
        get() = textureIdRef.get()

    private val pausedFlag = AtomicBoolean(false)
    override val isPaused: Boolean
        get() = pausedFlag.get()

    private val pendingAvailableFrames = AtomicInteger(0)
    private val listeners = CopyOnWriteArrayList<() -> Unit>()

    private var surfaceTexture: SurfaceTexture? = null
    private var activeSurface: Surface? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    private val rawStMatrix = FloatArray(16).apply { Matrix.setIdentityM(this, 0) }
    private val computedTexMatrix = FloatArray(16).apply { Matrix.setIdentityM(this, 0) }

    override val hasValidFrame: Boolean
        get() = boundTextureId != 0 && frameRing.recordedFrameCount > 0L

    override val lastSensorTimestampNs: Long
        get() = frameRing.latestSensorTimestampNs

    @Synchronized
    override fun attachToGlContext(textureId: Int) {
        if (textureId == 0) return
        val prevTexId = textureIdRef.getAndSet(textureId)
        val st = surfaceTexture
        if (st != null) {
            try {
                if (stAttachedToGl.get() && prevTexId != textureId) {
                    st.detachFromGLContext()
                    stAttachedToGl.set(false)
                }
                if (!stAttachedToGl.get()) {
                    st.attachToGLContext(textureId)
                    stAttachedToGl.set(true)
                }
                attachedFlag.set(true)
            } catch (_: Throwable) {
                // On JVM unit tests or when SurfaceTexture was created directly with textureId
                stAttachedToGl.set(true)
                attachedFlag.set(true)
            }
        } else {
            attachedFlag.set(true)
        }
    }

    @Synchronized
    override fun detachFromGlContext() {
        val st = surfaceTexture
        if (st != null && stAttachedToGl.get()) {
            try {
                st.detachFromGLContext()
            } catch (_: Throwable) {
            }
        }
        stAttachedToGl.set(false)
        attachedFlag.set(false)
        textureIdRef.set(0)
    }

    @Synchronized
    override fun readFrame(outTexMatrix: FloatArray?): FrameRing.FrameSlot? {
        return readFrame(outTexMatrix, targetTimestampNs = 0L, useAlignedClock = true)
    }

    @Synchronized
    override fun readFrame(
        outTexMatrix: FloatArray?,
        targetTimestampNs: Long,
        useAlignedClock: Boolean
    ): FrameRing.FrameSlot? {
        if (options.pausable && pausedFlag.get()) {
            return selectFrameFromRingLocked(outTexMatrix, targetTimestampNs, useAlignedClock)
        }

        val texId = textureIdRef.get()
        if (texId == 0) return null

        val st = surfaceTexture
        if (st != null && !stAttachedToGl.get()) {
            try {
                st.attachToGLContext(texId)
                stAttachedToGl.set(true)
                attachedFlag.set(true)
            } catch (_: Throwable) {
                stAttachedToGl.set(true)
                attachedFlag.set(true)
            }
        }

        val hadPending = pendingAvailableFrames.getAndSet(0) > 0
        if (st != null && stAttachedToGl.get() && hadPending) {
            try {
                st.updateTexImage()
                st.getTransformMatrix(rawStMatrix)
                val rawTs = st.timestamp
                val nowElapsedNs = FrameClock.elapsedRealtimeNs()
                val sensorTsNs = if (rawTs > 0L) rawTs else nowElapsedNs
                val alignedElapsedNs = if (rawTs > 0L) {
                    clockOffsetCalibrator.toElapsedRealtimeNs(rawTs, nowElapsedNs)
                } else {
                    nowElapsedNs
                }
                val isIdentityRawSt = abs(rawStMatrix[0] - 1f) < 1e-4f &&
                    abs(rawStMatrix[5] - 1f) < 1e-4f &&
                    abs(rawStMatrix[1]) < 1e-4f &&
                    abs(rawStMatrix[4]) < 1e-4f
                val hasRotInRawSt = !isIdentityRawSt && abs(rawStMatrix[1]) > abs(rawStMatrix[0])
                CameraInputSurface.computeSingleOwnerInputMatrix(
                    options = options,
                    rawSurfaceTransform16 = rawStMatrix,
                    applyAdditionalRotation = !hasRotInRawSt,
                    outMatrix16 = computedTexMatrix
                )
                frameRing.recordFrame(
                    textureId = texId,
                    sensorTimestampNs = sensorTsNs,
                    alignedElapsedRealtimeNs = alignedElapsedNs,
                    texMatrix = computedTexMatrix,
                    isMirroredH = options.mirrorH,
                    isExternalOes = true
                )
            } catch (_: Throwable) {
                // Headless JVM test environment fallback
            }
        }

        return selectFrameFromRingLocked(outTexMatrix, targetTimestampNs, useAlignedClock)
    }

    private fun selectFrameFromRingLocked(
        outTexMatrix: FloatArray?,
        targetTimestampNs: Long,
        useAlignedClock: Boolean
    ): FrameRing.FrameSlot? {
        if ((syncFrameRingToTrackerTimestamp || frameRing.delayDepth > 0) && targetTimestampNs > 0L) {
            val matched = frameRing.findFrameClosestToTimestampNs(
                targetTimestampNs = targetTimestampNs,
                useAlignedClock = useAlignedClock,
                outTexMatrix = outTexMatrix
            )
            if (matched != null) {
                return matched
            }
        }
        return frameRing.peekFrame(frameRing.delayDepth, outTexMatrix)
    }

    /**
     * Pushes a synthetic or test camera frame directly through the single-owner transform stage
     * and into [frameRing], notifying subscribers unless paused.
     */
    @Synchronized
    fun pushSyntheticFrame(
        sensorTimestampNs: Long,
        rawSurfaceTransform16: FloatArray? = null,
        alignedElapsedRealtimeNs: Long = FrameClock.elapsedRealtimeNs(),
        isExternalOes: Boolean = true
    ): FrameRing.FrameSlot? {
        if (options.pausable && pausedFlag.get()) {
            return null
        }
        val texId = textureIdRef.get()
        if (texId == 0) return null

        CameraInputSurface.computeSingleOwnerInputMatrix(
            options = options,
            rawSurfaceTransform16 = rawSurfaceTransform16,
            applyAdditionalRotation = false,
            outMatrix16 = computedTexMatrix
        )
        val slot = frameRing.recordFrame(
            textureId = texId,
            sensorTimestampNs = sensorTimestampNs,
            alignedElapsedRealtimeNs = alignedElapsedRealtimeNs,
            texMatrix = computedTexMatrix,
            isMirroredH = options.mirrorH,
            isExternalOes = isExternalOes
        )
        notifySubscribers()
        return slot
    }

    override fun subscribe(onFrameAvailable: () -> Unit) {
        listeners.add(onFrameAvailable)
    }

    private fun notifySubscribers() {
        if (options.pausable && pausedFlag.get()) return
        for (listener in listeners) {
            try {
                listener.invoke()
            } catch (_: Throwable) {
            }
        }
    }

    @Synchronized
    override fun updateOptions(newOptions: CameraInputSurface.InputOptions) {
        val changed = this.options != newOptions
        this.options = newOptions
        if (changed && surfaceTexture != null && frameRing.recordedFrameCount > 0L) {
            val texId = textureIdRef.get()
            val latestTs = frameRing.latestSensorTimestampNs
            if (texId != 0 && latestTs > 0L) {
                val isIdentityRawSt = abs(rawStMatrix[0] - 1f) < 1e-4f &&
                    abs(rawStMatrix[5] - 1f) < 1e-4f &&
                    abs(rawStMatrix[1]) < 1e-4f &&
                    abs(rawStMatrix[4]) < 1e-4f
                val hasRotInRawSt = !isIdentityRawSt && abs(rawStMatrix[1]) > abs(rawStMatrix[0])
                CameraInputSurface.computeSingleOwnerInputMatrix(
                    options = newOptions,
                    rawSurfaceTransform16 = rawStMatrix,
                    applyAdditionalRotation = !hasRotInRawSt,
                    outMatrix16 = computedTexMatrix
                )
                frameRing.recordFrame(
                    textureId = texId,
                    sensorTimestampNs = latestTs,
                    alignedElapsedRealtimeNs = FrameClock.elapsedRealtimeNs(),
                    texMatrix = computedTexMatrix,
                    isMirroredH = newOptions.mirrorH,
                    isExternalOes = true
                )
            }
        }
    }

    override fun setPaused(paused: Boolean) {
        if (options.pausable) {
            pausedFlag.set(paused)
        }
    }

    /**
     * Creates a CameraX [Preview.SurfaceProvider] that supplies an OpenGL [SurfaceTexture]-backed
     * [Surface] to CameraX `Preview` when operating in Path B (`SINGLE_SCENE_GL`).
     */
    fun createSurfaceProvider(executor: Executor): Preview.SurfaceProvider {
        return Preview.SurfaceProvider { request ->
            val resolution: Size = request.resolution
            request.setTransformationInfoListener(executor) { transformationInfo ->
                updateOptions(
                    options.copy(
                        rotationDegrees = transformationInfo.rotationDegrees
                    )
                )
            }
            synchronized(this) {
                options = options.copy(
                    sourceWidth = resolution.width.coerceAtLeast(1),
                    sourceHeight = resolution.height.coerceAtLeast(1)
                )
                releaseSurfaceOnlyLocked()

                // Always create SurfaceTexture in detached mode on the Main Thread so it is attached
                // exclusively on the GLThread in readFrame() / attachToGlContext().
                val st = SurfaceTexture(false).also {
                    stAttachedToGl.set(false)
                }
                st.setDefaultBufferSize(
                    resolution.width.coerceAtLeast(1),
                    resolution.height.coerceAtLeast(1)
                )
                st.setOnFrameAvailableListener({
                    if (!(options.pausable && pausedFlag.get())) {
                        pendingAvailableFrames.incrementAndGet()
                        notifySubscribers()
                    }
                }, mainHandler)

                val surface = Surface(st)
                surfaceTexture = st
                activeSurface = surface

                request.provideSurface(surface, executor) {
                    synchronized(this) {
                        if (activeSurface === surface) {
                            releaseSurfaceOnlyLocked()
                        } else {
                            try {
                                surface.release()
                            } catch (_: Throwable) {
                            }
                            try {
                                st.release()
                            } catch (_: Throwable) {
                            }
                        }
                    }
                }
            }
        }
    }

    private fun releaseSurfaceOnlyLocked() {
        try {
            activeSurface?.release()
        } catch (_: Throwable) {
        }
        activeSurface = null
        try {
            surfaceTexture?.setOnFrameAvailableListener(null)
            surfaceTexture?.release()
        } catch (_: Throwable) {
        }
        surfaceTexture = null
        stAttachedToGl.set(false)
        pendingAvailableFrames.set(0)
        frameRing.clear()
        Matrix.setIdentityM(rawStMatrix, 0)
        Matrix.setIdentityM(computedTexMatrix, 0)
    }

    @Synchronized
    override fun release() {
        releaseSurfaceOnlyLocked()
        attachedFlag.set(false)
        textureIdRef.set(0)
        pendingAvailableFrames.set(0)
        frameRing.clear()
        listeners.clear()
    }
}
