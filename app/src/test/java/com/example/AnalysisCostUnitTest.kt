package com.example

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.PixelFormat
import android.graphics.Rect
import android.util.Size
import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import androidx.test.core.app.ApplicationProvider
import com.example.camera.CameraController
import com.example.engine.capture.AnalysisFrame
import com.example.engine.capture.CameraSession
import com.example.engine.capture.ClockOffsetCalibrator
import com.example.engine.capture.FrameClock
import com.example.engine.debug.LatencyProbe
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.perception.AnalysisResolutionMode
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.perception.MediaPipeFaceGeometryProvider
import com.example.engine.perception.ProviderConfig
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import com.example.engine.signal.SignalPipeline
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import org.junit.After
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.lang.reflect.Proxy
import java.nio.ByteBuffer
import java.util.Optional

/**
 * Unit tests for Chunk: analysis cost:
 * 1. Async ping-pong bitmap safety: at most one inference in flight, alternating `slots[0]` and
 *    `slots[1]`, never overwriting an in-flight slot before its callback completes, and zero
 *    steady-state bitmap allocations.
 * 2. 16:9 aspect ratio preservation (`selectSmall16By9Resolution` picks smallest 16:9 >= 320x180)
 *    and preservation of defaults (`SMALL`, `YUV_420_888`, `CPU`).
 * 3. Direct `ByteBuffer` `YUV_420_888` and `RGBA_8888` known-value pixel tests honoring padded
 *    `rowStride`, `pixelStride == 2` and `pixelStride == 1`, and non-zero `cropRect`.
 * 4. Stage-2 Governor downscale (`320 / 256 / 224`) + rotation with `downscalePaint.isFilterBitmap == true`.
 * 5. Safe delegate switch (old landmarker closed only after in-flight inference finishes, plus
 *    `"GPU->CPU fallback"` status when GPU init fails) and `ClockOffsetCalibrator` + `LatencyProbe`
 *    reset on analysis rebind.
 * 6. Full `conversionMs` timing (through `MPImage` ready) and 5-second GC-count delta (`gc: +N`)
 *    on `LatencyProbe.cachedHudLine`.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class AnalysisCostUnitTest {

    @After
    fun tearDown() {
        FrameClock.customElapsedRealtimeSource = null
    }

    @Test
    fun testKnownValueYuv420AndRgbaConversionWithPaddedRowStridePixelStride2AndCropRect() {
        // Test 8x6 source image with a 4x4 cropRect = Rect(2, 1, 6, 5), padded rowStrideY = 16,
        // and UV plane with padded rowStrideUV = 12 and pixelStrideUV = 2 (NV12/NV21 semi-planar style)
        val width = 8
        val height = 6
        val cropRect = Rect(2, 1, 6, 5) // 4x4 cropped region
        val rowStrideY = 16
        val pixelStrideY = 1
        val rowStrideUV = 12
        val pixelStrideUV = 2

        val yBuf = ByteBuffer.allocateDirect(rowStrideY * height)
        val uBuf = ByteBuffer.allocateDirect(rowStrideUV * (height / 2))
        val vBuf = ByteBuffer.allocateDirect(rowStrideUV * (height / 2))

        // Fill poison values everywhere first so any wrong stride or crop offset fails immediately
        for (i in 0 until yBuf.capacity()) yBuf.put(i, 16.toByte())
        for (i in 0 until uBuf.capacity()) uBuf.put(i, 128.toByte())
        for (i in 0 until vBuf.capacity()) vBuf.put(i, 128.toByte())

        // Populate known Y, U, V values inside the crop region (srcX in 2..5, srcY in 1..4)
        for (srcY in 0 until height) {
            for (srcX in 0 until width) {
                val yVal = (40 + srcY * 20 + srcX * 10).coerceIn(16, 235)
                yBuf.put(srcY * rowStrideY + srcX * pixelStrideY, yVal.toByte())
            }
        }
        for (uvRow in 0 until height / 2) {
            for (uvCol in 0 until width / 2) {
                val uVal = (90 + uvRow * 25 + uvCol * 15).coerceIn(16, 240)
                val vVal = (150 + uvRow * 20 - uvCol * 10).coerceIn(16, 240)
                uBuf.put(uvRow * rowStrideUV + uvCol * pixelStrideUV, uVal.toByte())
                vBuf.put(uvRow * rowStrideUV + uvCol * pixelStrideUV, vVal.toByte())
            }
        }

        val yuvProxy = createMockYuvImageProxy(
            width = width,
            height = height,
            cropRect = cropRect,
            yBuf = yBuf, rowStrideY = rowStrideY, pixelStrideY = pixelStrideY,
            uBuf = uBuf, rowStrideU = rowStrideUV, pixelStrideU = pixelStrideUV,
            vBuf = vBuf, rowStrideV = rowStrideUV, pixelStrideV = pixelStrideUV
        )

        val outArgb = IntArray(cropRect.width() * cropRect.height())
        MediaPipeFaceGeometryProvider.convertYuv420ToArgbPixels(yuvProxy, cropRect, outArgb)

        // Verify every pixel in the 4x4 cropped output matches the exact analytical BT.601 formula
        for (r in 0 until cropRect.height()) {
            for (c in 0 until cropRect.width()) {
                val srcX = cropRect.left + c
                val srcY = cropRect.top + r
                val uvCol = srcX shr 1
                val uvRow = srcY shr 1

                val yVal = (40 + srcY * 20 + srcX * 10).coerceIn(16, 235)
                val uVal = (90 + uvRow * 25 + uvCol * 15).coerceIn(16, 240) - 128
                val vVal = (150 + uvRow * 20 - uvCol * 10).coerceIn(16, 240) - 128

                val y1192 = 1192 * (yVal - 16).coerceAtLeast(0)
                val expR = ((y1192 + 1634 * vVal) shr 10).coerceIn(0, 255)
                val expG = ((y1192 - 833 * vVal - 400 * uVal) shr 10).coerceIn(0, 255)
                val expB = ((y1192 + 2066 * uVal) shr 10).coerceIn(0, 255)
                val expectedColor = (0xFF shl 24) or (expR shl 16) or (expG shl 8) or expB

                val actualColor = outArgb[r * cropRect.width() + c]
                assertEquals(
                    "Mismatch at cropped pixel ($c, $r) from source ($srcX, $srcY)",
                    expectedColor,
                    actualColor
                )
            }
        }

        // Also verify RGBA_8888 conversion with padded rowStride and cropRect
        val rgbaRowStride = width * 4 + 12 // 12 bytes of row padding
        val rgbaBuf = ByteBuffer.allocateDirect(rgbaRowStride * height)
        for (srcY in 0 until height) {
            for (srcX in 0 until width) {
                val off = srcY * rgbaRowStride + srcX * 4
                rgbaBuf.put(off, (10 + srcX * 20).toByte())     // R
                rgbaBuf.put(off + 1, (30 + srcY * 30).toByte()) // G
                rgbaBuf.put(off + 2, (200 - srcX * 15).toByte())// B
                rgbaBuf.put(off + 3, 0xFF.toByte())             // A
            }
        }
        val rgbaProxy = createMockRgbaImageProxy(width, height, cropRect, rgbaBuf, rgbaRowStride)
        val dstBmp = android.graphics.Bitmap.createBitmap(
            cropRect.width(),
            cropRect.height(),
            android.graphics.Bitmap.Config.ARGB_8888
        )
        val scratch = IntArray(cropRect.width() * cropRect.height())
        MediaPipeFaceGeometryProvider.convertRgba8888ToBitmap(rgbaProxy, cropRect, dstBmp, scratch)

        for (r in 0 until cropRect.height()) {
            for (c in 0 until cropRect.width()) {
                val srcX = cropRect.left + c
                val srcY = cropRect.top + r
                val expColor = (0xFF shl 24) or
                    ((10 + srcX * 20) shl 16) or
                    ((30 + srcY * 30) shl 8) or
                    (200 - srcX * 15)
                assertEquals(expColor, dstBmp.getPixel(c, r))
            }
        }
    }

    @Test
    fun testAsyncPingPongBitmapSafetyAndZeroSteadyStateAllocations() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val provider = MediaPipeFaceGeometryProvider(context)

        assertTrue(
            "Stage-2 downscale Paint must have isFilterBitmap = true",
            provider.downscalePaint.isFilterBitmap
        )

        var pendingCallback: (() -> Unit)? = null
        val emptyResult = createMockFaceLandmarkerResult()

        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    // Hold callback in flight until test explicitly triggers it
                    pendingCallback = {
                        resultListener(emptyResult, mpImage)
                    }
                }
                override fun close() {
                    isClosed = true
                }
            }
        }

        provider.start(ProviderConfig(analysisLongEdgePx = 320))
        provider.awaitAnalysisIdle()

        // 1. Submit Frame 1 (640x360, rotated 90 deg -> Stage 2 downscales to 180x320)
        val proxy1 = createSimpleYuvProxy(640, 360, yFill = 80)
        provider.submit(
            AnalysisFrame(
                imageProxy = proxy1,
                timestampNs = 1_000_000_000L,
                rotationDegrees = 90,
                isFrontCamera = true,
                submittedAtNs = 998_000_000L
            )
        )
        provider.awaitAnalysisIdle()

        // Frame 1 is now in-flight in slot 0; slots 1 and 2 are untouched.
        // With CONV:FAST default (Chunk R4), no intermediate sourceBitmap is allocated (sourceBitmap == null)
        // and only 1 targetBitmap (180x320) is allocated per slot.
        assertEquals("Frame 1 must occupy slot 0", 0, provider.inFlightSlotIndex)
        assertTrue("Slot 0 must be marked inUseByInference while callback is pending", provider.slots[0].inUseByInference.get())
        assertFalse("Slot 1 must not be in use while Frame 1 is in flight", provider.slots[1].inUseByInference.get())
        assertNull("CONV:FAST default must not allocate an intermediate sourceBitmap", provider.slots[0].sourceBitmap)
        val slot0TargetBmp = provider.slots[0].targetBitmap
        assertNotNull(slot0TargetBmp)
        assertEquals(180, slot0TargetBmp!!.width)
        assertEquals(320, slot0TargetBmp.height)
        val pixelDuringFlight = slot0TargetBmp.getPixel(10, 10)

        // 2. While Frame 1 is STILL in flight, submit Frame 2 and Frame 3 with different pixel values.
        // Slot 0 (in flight) must NEVER be overwritten; Frame 2 converts into Slot 1 (STAGED) and Frame 3
        // converts into free Slot 2 (replacing Slot 1 in stagedPreparedFrame).
        val proxy2 = createSimpleYuvProxy(640, 360, yFill = 200)
        val proxy3 = createSimpleYuvProxy(640, 360, yFill = 220)
        provider.submit(AnalysisFrame(imageProxy = proxy2, timestampNs = 1_066_000_000L, rotationDegrees = 90, isFrontCamera = true))
        provider.submit(AnalysisFrame(imageProxy = proxy3, timestampNs = 1_133_000_000L, rotationDegrees = 90, isFrontCamera = true))
        provider.awaitAnalysisIdle()

        assertEquals(
            "In-flight slot 0 targetBitmap pixel must NOT be mutated while Frame 1 is in flight",
            pixelDuringFlight,
            slot0TargetBmp.getPixel(10, 10)
        )
        assertNull("CONV:FAST default must not allocate sourceBitmap in Slot 1", provider.slots[1].sourceBitmap)
        assertNull("CONV:FAST default must not allocate sourceBitmap in Slot 2", provider.slots[2].sourceBitmap)
        assertNotNull("Free Slot 1 converts Frame 2 in overlap while Slot 0 is in flight", provider.slots[1].targetBitmap)
        assertNotNull("Free Slot 2 converts Frame 3 in overlap while Slot 0 is in flight and Slot 1 is staged", provider.slots[2].targetBitmap)
        assertEquals(1, provider.pendingReplacedCount)

        // 3. Complete Frame 1's async callback -> releases slot 0 and dispatches Frame 3 on dispatchExecutor into slot 2!
        val cb1 = pendingCallback
        assertNotNull(cb1)
        cb1!!.invoke()
        provider.awaitAnalysisIdle()

        assertEquals("Frame 3 must dispatch from slot 2", 2, provider.inFlightSlotIndex)
        assertFalse("Slot 0 must be released after Frame 1 callback finishes", provider.slots[0].inUseByInference.get())
        assertTrue("Slot 2 must be marked in-use for Frame 3", provider.slots[2].inUseByInference.get())
        val slot1TargetBmp = provider.slots[1].targetBitmap
        val slot2TargetBmp = provider.slots[2].targetBitmap
        assertNotNull(slot1TargetBmp)
        assertNotNull(slot2TargetBmp)
        assertNotSame("Slot 0 and Slot 1 must own distinct target Bitmaps", slot0TargetBmp, slot1TargetBmp)
        assertNotSame("Slot 1 and Slot 2 must own distinct target Bitmaps", slot1TargetBmp, slot2TargetBmp)

        // Complete Frame 3 callback
        pendingCallback!!.invoke()
        provider.awaitAnalysisIdle()
        assertEquals(-1, provider.inFlightSlotIndex)

        // Record bitmap allocation counts after warm-up of slots 0, 1, and 2 (1 targetBitmap each in CONV:FAST)
        val slot0AllocsAfterWarmup = provider.slots[0].bitmapAllocationCount
        val slot1AllocsAfterWarmup = provider.slots[1].bitmapAllocationCount
        val slot2AllocsAfterWarmup = provider.slots[2].bitmapAllocationCount
        assertEquals(1, slot0AllocsAfterWarmup)
        assertEquals(1, slot1AllocsAfterWarmup)
        assertEquals(1, slot2AllocsAfterWarmup)

        // 4. Run 10 more steady-state frames: must cycle through slots 0, 1, 2 and allocate ZERO new Bitmaps!
        for (i in 0 until 10) {
            val p = createSimpleYuvProxy(640, 360, yFill = 100 + i)
            provider.submit(
                AnalysisFrame(
                    imageProxy = p,
                    timestampNs = 2_000_000_000L + i * 66_000_000L,
                    rotationDegrees = 90,
                    isFrontCamera = true
                )
            )
            provider.awaitAnalysisIdle()
            assertEquals(i % 3, provider.lastUsedSlotIndex)
            pendingCallback!!.invoke()
            provider.awaitAnalysisIdle()
        }

        assertEquals(
            "Zero new Bitmaps may be allocated in Slot 0 during steady state",
            slot0AllocsAfterWarmup,
            provider.slots[0].bitmapAllocationCount
        )
        assertEquals(
            "Zero new Bitmaps may be allocated in Slot 1 during steady state",
            slot1AllocsAfterWarmup,
            provider.slots[1].bitmapAllocationCount
        )
        assertEquals(
            "Zero new Bitmaps may be allocated in Slot 2 during steady state",
            slot2AllocsAfterWarmup,
            provider.slots[2].bitmapAllocationCount
        )
        assertNull(provider.slots[0].sourceBitmap)
        assertSame(slot0TargetBmp, provider.slots[0].targetBitmap)
        assertNull(provider.slots[1].sourceBitmap)
        assertSame(slot1TargetBmp, provider.slots[1].targetBitmap)
    }

    @Test
    fun testSmall16By9ResolutionSelectorPreservesAspectRatioAndGovernorStage2Downscale() {
        // 1. Verify CameraController.selectSmall16By9Resolution keeps 16:9 and picks smallest >= 320x180
        val candidates = listOf(
            Size(1920, 1080), // 16:9 large
            Size(1280, 720),  // 16:9 HD
            Size(640, 480),   // 4:3 VGA (must not beat 16:9)
            Size(640, 360),   // 16:9 nHD
            Size(320, 240),   // 4:3 QVGA (must not beat 16:9)
            Size(384, 216),   // 16:9 small
            Size(256, 144)    // 16:9 below 320x180 threshold
        )
        val ordered = CameraController.selectSmall16By9Resolution(candidates)
        assertEquals(
            "Smallest 16:9 resolution >= 320x180 (384x216) must be selected first",
            Size(384, 216),
            ordered[0]
        )
        assertEquals(
            "Second choice must be next smallest 16:9 >= 320x180 (640x360)",
            Size(640, 360),
            ordered[1]
        )

        // 2. Verify Stage-2 Governor downscale (320 -> 256 -> 224) on 16:9 frames
        val context = ApplicationProvider.getApplicationContext<Context>()
        val provider = MediaPipeFaceGeometryProvider(context)
        val slot = MediaPipeFaceGeometryProvider.FrameBufferSlot(0)

        provider.updateAnalysisSize(PerformanceGovernor.ANALYSIS_SIZE_HIGH_PX) // 320
        val bmp320 = provider.convertAndTransformIntoSlot(
            createSimpleYuvProxy(640, 360, 128),
            rotationDegrees = 90,
            slot = slot
        )!!
        assertEquals(180, bmp320.width)
        assertEquals(320, bmp320.height)

        provider.updateAnalysisSize(PerformanceGovernor.ANALYSIS_SIZE_MED_PX) // 256
        val bmp256 = provider.convertAndTransformIntoSlot(
            createSimpleYuvProxy(640, 360, 128),
            rotationDegrees = 90,
            slot = slot
        )!!
        assertEquals(144, bmp256.width)
        assertEquals(256, bmp256.height)

        provider.updateAnalysisSize(PerformanceGovernor.ANALYSIS_SIZE_LOW_PX) // 224
        val bmp224 = provider.convertAndTransformIntoSlot(
            createSimpleYuvProxy(640, 360, 128),
            rotationDegrees = 90,
            slot = slot
        )!!
        assertEquals(126, bmp224.width)
        assertEquals(224, bmp224.height)
    }

    @Test
    fun testDelegateSwitchDeferredCloseGpuFallbackAndRebindResetsCalibratorAndProbe() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val provider = MediaPipeFaceGeometryProvider(context)
        val gov = PerformanceGovernor(context = null, isLowEndDevice = false)
        val pipeline = SignalPipeline()
        val calibrator = ClockOffsetCalibrator(warmupFrames = 5)
        val probe = LatencyProbe()
        val session = CameraSession(provider, pipeline, gov, calibrator, probe)

        var inFlightCb: (() -> Unit)? = null
        val createdHandles = ArrayList<MediaPipeFaceGeometryProvider.LandmarkerHandle>()
        var failGpuCreation = false
        val emptyResult = createMockFaceLandmarkerResult()

        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            if (delegate == Delegate.GPU && failGpuCreation) {
                throw RuntimeException("Simulated GPU delegate initialization failure")
            }
            val handle = object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    inFlightCb = {
                        resultListener(emptyResult, mpImage)
                    }
                }
                override fun close() {
                    isClosed = true
                }
            }
            createdHandles.add(handle)
            handle
        }

        // 1. Default startup is CPU delegate
        session.startProvider(ProviderConfig(useGpu = false))
        provider.awaitAnalysisIdle()
        assertEquals("CPU", provider.activeDelegateLabel)
        assertEquals("CPU", probe.delegateLabel)
        assertEquals(1, createdHandles.size)
        val cpuHandle1 = createdHandles[0]

        // 2. Dispatch a frame so cpuHandle1 has an inference in flight
        provider.submit(
            AnalysisFrame(
                imageProxy = createSimpleYuvProxy(320, 180, 100),
                timestampNs = 1_000_000_000L,
                rotationDegrees = 0,
                isFrontCamera = true
            )
        )
        provider.awaitAnalysisIdle()
        assertNotNull(inFlightCb)

        // 3. Switch to GPU while cpuHandle1's inference is still in flight:
        // New GPU handle is created on analysisExecutor, and cpuHandle1 must NOT be closed until inFlightCb finishes!
        session.setUseGpuDelegate(true)
        provider.awaitAnalysisIdle()
        assertEquals("GPU", provider.activeDelegateLabel)
        assertEquals("GPU", probe.delegateLabel)
        assertEquals(2, createdHandles.size)
        val gpuHandle = createdHandles[1]
        assertFalse(
            "Old landmarker must NOT be closed while its inference is still in flight",
            cpuHandle1.isClosed
        )
        assertFalse(gpuHandle.isClosed)

        // 4. Finish the in-flight inference -> cpuHandle1 is now retired and closed on reaperExecutor!
        inFlightCb!!.invoke()
        provider.awaitAnalysisIdle()
        provider.awaitReaperIdle()
        assertTrue(
            "Old landmarker must be closed once its in-flight inference callback completes",
            cpuHandle1.isClosed
        )
        assertFalse("Active GPU landmarker must remain open", gpuHandle.isClosed)

        // 5. Simulate GPU initialization failure -> must fall back to CPU and report "GPU->CPU fallback"
        failGpuCreation = true
        session.setUseGpuDelegate(true)
        provider.awaitAnalysisIdle()
        provider.awaitReaperIdle()
        assertEquals("GPU->CPU fallback", provider.activeDelegateLabel)
        assertEquals("GPU->CPU fallback", probe.delegateLabel)
        assertTrue("Previous handle must be closed immediately when no inference is in flight", gpuHandle.isClosed)

        // 6. Verify onAnalysisRebound resets ClockOffsetCalibrator and LatencyProbe rolling windows
        calibrator.configureSource(ClockOffsetCalibrator.TIMESTAMP_SOURCE_UNKNOWN)
        for (i in 0 until 5) {
            calibrator.toElapsedRealtimeNs(100_000_000L + i * 10_000_000L, 200_000_000L + i * 10_000_000L)
        }
        assertTrue(calibrator.isCalibrated)
        val postWarmupRenderNs = FrameClock.elapsedRealtimeNs() + 10_000_000_000L
        probe.recordRenderFrame(
            renderTimestampNs = postWarmupRenderNs,
            faceState = FaceState(
                timestampNs = postWarmupRenderNs,
                poseTimestampNs = postWarmupRenderNs - 30_000_000L,
                lifecycle = FaceLifecycle.State.TRACKING,
                isVisible = true
            )
        )
        assertEquals(1, probe.ageSampleCount())
        assertEquals(1, probe.csvRowCount())

        session.onAnalysisRebound(
            formatLabel = "RGBA_8888",
            resolutionModeLabel = AnalysisResolutionMode.SMALL.name
        )
        assertFalse("ClockOffsetCalibrator must reset on analysis rebind", calibrator.isCalibrated)
        assertEquals("LatencyProbe age window must reset on analysis rebind", 0, probe.ageSampleCount())
        assertEquals("LatencyProbe CSV ring must reset on analysis rebind", 0, probe.csvRowCount())
        assertEquals("RGBA_8888", probe.analysisFormatLabel)
        assertEquals("SMALL", probe.resolutionModeLabel)
    }

    @Test
    fun testConversionMsFullTimingAndGcCountDeltaOnCachedHudLine() {
        var fakeClockNs = 10_000_000_000L
        FrameClock.customElapsedRealtimeSource = { fakeClockNs }

        var simulatedGcCount = 4L
        val probe = LatencyProbe(
            windowDurationNs = 5_000_000_000L,
            hudRefreshIntervalNs = 500_000_000L,
            gcCountProvider = { simulatedGcCount }
        )

        val context = ApplicationProvider.getApplicationContext<Context>()
        val provider = MediaPipeFaceGeometryProvider(context)
        val emptyResult = createMockFaceLandmarkerResult()

        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    // At detectAsync entry, conversion (unpack + rotate/downscale + MPImage build) took 3.5 ms
                    fakeClockNs += 11_500_000L // 11.5 ms inference
                    resultListener(emptyResult, mpImage)
                }
                override fun close() {
                    isClosed = true
                }
            }
        }
        provider.start(ProviderConfig(analysisLongEdgePx = 320))
        provider.awaitAnalysisIdle()

        var capturedSnapshot: FaceGeometrySnapshot? = null
        provider.setListener { snap ->
            capturedSnapshot = snap
            probe.recordTrackerSnapshot(snap)
        }

        val proxy = createSimpleYuvProxy(640, 360, 120)
        provider.submit(
            AnalysisFrame(
                imageProxy = proxy,
                timestampNs = fakeClockNs,
                rotationDegrees = 90,
                isFrontCamera = true,
                submittedAtNs = fakeClockNs - 2_000_000L
            )
        )
        provider.awaitAnalysisIdle()

        assertNotNull(capturedSnapshot)
        assertEquals(2.0f, capturedSnapshot!!.queueMs, 0.01f)
        assertTrue("conversionMs must be finite and >= 0", capturedSnapshot!!.conversionMs >= 0f)
        assertEquals(11.5f, capturedSnapshot!!.inferenceMs, 0.1f)

        // First HUD refresh at t = 10.0s with GC count = 4 -> delta = 0
        probe.refreshMetrics(fakeClockNs)
        assertEquals(0, probe.gcCountDelta5s)
        assertTrue(probe.cachedHudLine.contains("gc: +0"))

        // Advance 2.0s within the 5s window and simulate +3 GC events (total 7) -> delta = +3
        fakeClockNs += 2_000_000_000L
        simulatedGcCount = 7L
        probe.refreshMetrics(fakeClockNs)
        assertEquals(3, probe.gcCountDelta5s)
        assertTrue(
            "Cached HUD line must display 5s GC-count delta (gc: +3)",
            probe.cachedHudLine.contains("gc: +3")
        )

        // Advance > 5.0s past the t=10s baseline so the initial sample (4) is evicted and baseline is 7
        fakeClockNs += 4_000_000_000L
        simulatedGcCount = 7L
        probe.refreshMetrics(fakeClockNs)
        assertEquals(
            "GC delta must drop back to 0 once older GC events leave the 5s rolling window",
            0,
            probe.gcCountDelta5s
        )
        assertTrue(probe.cachedHudLine.contains("gc: +0"))
    }

    private fun createMockFaceLandmarkerResult(): FaceLandmarkerResult? = null

    private fun createSimpleYuvProxy(width: Int, height: Int, yFill: Int): ImageProxy {
        val yBuf = ByteBuffer.allocateDirect(width * height)
        val uBuf = ByteBuffer.allocateDirect((width / 2) * (height / 2))
        val vBuf = ByteBuffer.allocateDirect((width / 2) * (height / 2))
        for (i in 0 until yBuf.capacity()) yBuf.put(i, yFill.toByte())
        for (i in 0 until uBuf.capacity()) uBuf.put(i, 128.toByte())
        for (i in 0 until vBuf.capacity()) vBuf.put(i, 128.toByte())
        return createMockYuvImageProxy(
            width = width,
            height = height,
            cropRect = Rect(0, 0, width, height),
            yBuf = yBuf, rowStrideY = width, pixelStrideY = 1,
            uBuf = uBuf, rowStrideU = width / 2, pixelStrideU = 1,
            vBuf = vBuf, rowStrideV = width / 2, pixelStrideV = 1
        )
    }

    private fun createMockPlaneProxy(
        buffer: ByteBuffer,
        rowStride: Int,
        pixelStride: Int
    ): ImageProxy.PlaneProxy {
        return Proxy.newProxyInstance(
            ImageProxy.PlaneProxy::class.java.classLoader,
            arrayOf(ImageProxy.PlaneProxy::class.java)
        ) { _, method, _ ->
            when (method.name) {
                "getBuffer" -> buffer
                "getRowStride" -> rowStride
                "getPixelStride" -> pixelStride
                else -> 0
            }
        } as ImageProxy.PlaneProxy
    }

    private fun createMockYuvImageProxy(
        width: Int,
        height: Int,
        cropRect: Rect,
        yBuf: ByteBuffer, rowStrideY: Int, pixelStrideY: Int,
        uBuf: ByteBuffer, rowStrideU: Int, pixelStrideU: Int,
        vBuf: ByteBuffer, rowStrideV: Int, pixelStrideV: Int
    ): ImageProxy {
        val planes = arrayOf(
            createMockPlaneProxy(yBuf, rowStrideY, pixelStrideY),
            createMockPlaneProxy(uBuf, rowStrideU, pixelStrideU),
            createMockPlaneProxy(vBuf, rowStrideV, pixelStrideV)
        )
        val imageInfo = Proxy.newProxyInstance(
            ImageInfo::class.java.classLoader,
            arrayOf(ImageInfo::class.java)
        ) { _, method, _ ->
            when (method.name) {
                "getTimestamp" -> 1_000_000_000L
                "getRotationDegrees" -> 0
                else -> null
            }
        } as ImageInfo

        return Proxy.newProxyInstance(
            ImageProxy::class.java.classLoader,
            arrayOf(ImageProxy::class.java)
        ) { _, method, _ ->
            when (method.name) {
                "getWidth" -> width
                "getHeight" -> height
                "getFormat" -> ImageFormat.YUV_420_888
                "getCropRect" -> cropRect
                "getPlanes" -> planes
                "getImageInfo" -> imageInfo
                "close" -> Unit
                else -> null
            }
        } as ImageProxy
    }

    private fun createMockRgbaImageProxy(
        width: Int,
        height: Int,
        cropRect: Rect,
        rgbaBuf: ByteBuffer,
        rowStride: Int
    ): ImageProxy {
        val planes = arrayOf(createMockPlaneProxy(rgbaBuf, rowStride, 4))
        return Proxy.newProxyInstance(
            ImageProxy::class.java.classLoader,
            arrayOf(ImageProxy::class.java)
        ) { _, method, _ ->
            when (method.name) {
                "getWidth" -> width
                "getHeight" -> height
                "getFormat" -> PixelFormat.RGBA_8888
                "getCropRect" -> cropRect
                "getPlanes" -> planes
                "close" -> Unit
                else -> null
            }
        } as ImageProxy
    }
}
