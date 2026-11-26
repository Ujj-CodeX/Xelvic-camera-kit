package com.example

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Rect
import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import androidx.camera.core.impl.TagBundle
import androidx.camera.core.impl.utils.ExifData
import com.example.engine.capture.CameraSession
import com.example.engine.capture.FrameClock
import com.example.engine.debug.DebugOverlay
import com.example.engine.debug.LatencyProbe
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.perception.AnalysisResolutionMode
import com.example.engine.perception.ConversionMode
import com.example.engine.perception.MediaPipeFaceGeometryProvider
import com.example.engine.perception.ProviderConfig
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import com.example.engine.signal.SignalPipeline
import com.example.ui.DebugHudLayoutCalculator
import com.example.ui.DebugHudSectionState
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import java.lang.reflect.Proxy
import java.nio.ByteBuffer
import java.util.Locale
import java.util.concurrent.Executor
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.abs
import kotlin.math.max

/**
 * Chunk R4 Unit Tests: Fast YUV Conversion (`CONV:FAST` vs `CONV:OLD`), Per-Stage Conversion
 * Timings, Source/Crop/Rotation Per-Mode Telemetry, and Zero-Allocation Steady State.
 */
class FastYuvConversionR4UnitTest {

    @After
    fun tearDown() {
        FrameClock.customElapsedRealtimeSource = null
    }

    private fun createMockContext(): Context {
        return object : android.content.ContextWrapper(null) {
            override fun getAssets(): AssetManager? = null
        }
    }

    private fun createRect(leftVal: Int, topVal: Int, rightVal: Int, bottomVal: Int): Rect {
        return Rect().apply {
            left = leftVal
            top = topVal
            right = rightVal
            bottom = bottomVal
        }
    }

    private class SyntheticYuvImageProxy(
        private val imgW: Int,
        private val imgH: Int,
        private val rotDeg: Int,
        private val timestampNs: Long,
        private val crop: Rect,
        yBytes: ByteArray,
        private val yRowStrideVal: Int,
        uBytes: ByteArray,
        vBytes: ByteArray,
        private val uvRowStrideVal: Int,
        private val uvPixelStrideVal: Int
    ) : ImageProxy {
        val closeCount = AtomicInteger(0)
        private val yBuf: ByteBuffer = ByteBuffer.wrap(yBytes)
        private val uBuf: ByteBuffer = ByteBuffer.wrap(uBytes)
        private val vBuf: ByteBuffer = ByteBuffer.wrap(vBytes)

        private val planeArray = arrayOf(
            object : ImageProxy.PlaneProxy {
                override fun getBuffer(): ByteBuffer = yBuf
                override fun getRowStride(): Int = yRowStrideVal
                override fun getPixelStride(): Int = 1
            },
            object : ImageProxy.PlaneProxy {
                override fun getBuffer(): ByteBuffer = uBuf
                override fun getRowStride(): Int = uvRowStrideVal
                override fun getPixelStride(): Int = uvPixelStrideVal
            },
            object : ImageProxy.PlaneProxy {
                override fun getBuffer(): ByteBuffer = vBuf
                override fun getRowStride(): Int = uvRowStrideVal
                override fun getPixelStride(): Int = uvPixelStrideVal
            }
        )

        override fun close() {
            closeCount.incrementAndGet()
        }

        override fun getCropRect(): Rect = crop
        override fun setCropRect(rect: Rect?) {}
        override fun getFormat(): Int = 35 // ImageFormat.YUV_420_888
        override fun getHeight(): Int = imgH
        override fun getWidth(): Int = imgW
        override fun getPlanes(): Array<ImageProxy.PlaneProxy> = planeArray
        override fun getImageInfo(): ImageInfo = object : ImageInfo {
            override fun getTagBundle(): TagBundle = TagBundle.emptyBundle()
            override fun getTimestamp(): Long = timestampNs
            override fun getRotationDegrees(): Int = rotDeg
            override fun populateExifData(exifBuilder: ExifData.Builder) {}
        }

        @androidx.camera.core.ExperimentalGetImage
        override fun getImage(): android.media.Image? = null
    }

    private fun buildSyntheticYuvImageProxy(
        width: Int,
        height: Int,
        rotationDegrees: Int,
        uvPixelStride: Int,
        yRowPadding: Int,
        uvRowPadding: Int,
        cropRect: Rect,
        timestampNs: Long = 1_000_000_000L
    ): SyntheticYuvImageProxy {
        val yRowStride = width + yRowPadding
        val chromaW = (width + 1) / 2
        val chromaH = (height + 1) / 2
        val minUvRowBytes = if (chromaW > 0) (chromaW - 1) * uvPixelStride + 1 else 0
        val uvRowStride = minUvRowBytes + uvRowPadding

        val yBytes = ByteArray(yRowStride * height) { 0x55.toByte() }
        val uBytes = ByteArray(uvRowStride * chromaH) { 0x77.toByte() }
        val vBytes = ByteArray(uvRowStride * chromaH) { 0x33.toByte() }

        // Populate deterministic, spatially varying luma and chroma gradients so any index,
        // rotation, stride, or crop mismatch immediately produces large pixel differences.
        for (y in 0 until height) {
            val yRowOff = y * yRowStride
            for (x in 0 until width) {
                val luma = (16 + ((x * 197 + y * 131 + (x xor y) * 17) % 219)).coerceIn(0, 255)
                yBytes[yRowOff + x] = luma.toByte()
            }
        }
        for (cy in 0 until chromaH) {
            val uvRowOff = cy * uvRowStride
            for (cx in 0 until chromaW) {
                val uVal = (32 + ((cx * 73 + cy * 151) % 192)).coerceIn(0, 255)
                val vVal = (32 + ((cx * 163 + cy * 89) % 192)).coerceIn(0, 255)
                val idx = uvRowOff + cx * uvPixelStride
                uBytes[idx] = uVal.toByte()
                vBytes[idx] = vVal.toByte()
            }
        }

        return SyntheticYuvImageProxy(
            imgW = width,
            imgH = height,
            rotDeg = rotationDegrees,
            timestampNs = timestampNs,
            crop = cropRect,
            yBytes = yBytes,
            yRowStrideVal = yRowStride,
            uBytes = uBytes,
            vBytes = vBytes,
            uvRowStrideVal = uvRowStride,
            uvPixelStrideVal = uvPixelStride
        )
    }

    @Test
    fun testFastYuvPixelEquivalenceAgainstReferenceConverterAcrossRotationsStridesPaddingAndCrop() {
        val provider = MediaPipeFaceGeometryProvider(
            context = createMockContext(),
            dispatchExecutorOverride = Executor { r -> r.run() },
            reaperExecutorOverride = Executor { r -> r.run() }
        )

        val rotations = intArrayOf(0, 90, 180, 270)
        val pixelStrides = intArrayOf(1, 2)

        data class TestCase(
            val width: Int,
            val height: Int,
            val cropRect: Rect,
            val targetLongEdge: Int,
            val yPad: Int,
            val uvPad: Int
        )

        val cases = listOf(
            // Case A: 1:1 (no downscale), padded rowStride, full crop
            TestCase(width = 64, height = 36, cropRect = createRect(0, 0, 64, 36), targetLongEdge = 64, yPad = 16, uvPad = 8),
            // Case B: Downscaled (96x54 -> longEdge 48), padded rowStride, full crop
            TestCase(width = 96, height = 54, cropRect = createRect(0, 0, 96, 54), targetLongEdge = 48, yPad = 32, uvPad = 12),
            // Case C: Non-zero cropRect (even offsets) with padded rowStride and downscaling
            TestCase(width = 128, height = 72, cropRect = createRect(12, 8, 108, 62), targetLongEdge = 48, yPad = 24, uvPad = 10),
            // Case D: Non-zero cropRect (odd offsets) with padded rowStride and 1:1 scale
            TestCase(width = 80, height = 50, cropRect = createRect(5, 3, 69, 39), targetLongEdge = 64, yPad = 19, uvPad = 7)
        )

        for (case in cases) {
            provider.updateAnalysisSize(case.targetLongEdge)
            for (rot in rotations) {
                for (uvPixelStride in pixelStrides) {
                    val slotFast = provider.slots[0]
                    val slotOld = provider.slots[1]

                    val proxyFast = buildSyntheticYuvImageProxy(
                        width = case.width,
                        height = case.height,
                        rotationDegrees = rot,
                        uvPixelStride = uvPixelStride,
                        yRowPadding = case.yPad,
                        uvRowPadding = case.uvPad,
                        cropRect = case.cropRect
                    )
                    val proxyOld = buildSyntheticYuvImageProxy(
                        width = case.width,
                        height = case.height,
                        rotationDegrees = rot,
                        uvPixelStride = uvPixelStride,
                        yRowPadding = case.yPad,
                        uvRowPadding = case.uvPad,
                        cropRect = case.cropRect
                    )

                    provider.setConversionMode(ConversionMode.FAST)
                    provider.convertAndTransformIntoSlot(proxyFast, rot, slotFast)
                    val dstWFast: Int = slotFast.targetWidth
                    val dstHFast: Int = slotFast.targetHeight
                    val totalPixels: Int = dstWFast * dstHFast
                    val fastPixels: IntArray = slotFast.targetArgbScratch.copyOf(totalPixels)

                    provider.setConversionMode(ConversionMode.OLD)
                    provider.convertAndTransformIntoSlot(proxyOld, rot, slotOld)
                    val dstWOld: Int = slotOld.targetWidth
                    val dstHOld: Int = slotOld.targetHeight
                    val refPixels: IntArray = slotOld.referenceTargetArgb.copyOf(dstWOld * dstHOld)

                    assertEquals("Target width must match for rot=$rot stride=$uvPixelStride", dstWOld, dstWFast)
                    assertEquals("Target height must match for rot=$rot stride=$uvPixelStride", dstHOld, dstHFast)
                    assertTrue("Target pixel count must be positive", totalPixels > 0)

                    var maxChannelDiff = 0
                    for (i in 0 until totalPixels) {
                        val pFast: Int = fastPixels[i]
                        val pRef: Int = refPixels[i]
                        val aFast = (pFast ushr 24) and 0xFF
                        val rFast = (pFast ushr 16) and 0xFF
                        val gFast = (pFast ushr 8) and 0xFF
                        val bFast = pFast and 0xFF

                        val aRef = (pRef ushr 24) and 0xFF
                        val rRef = (pRef ushr 16) and 0xFF
                        val gRef = (pRef ushr 8) and 0xFF
                        val bRef = pRef and 0xFF

                        val aDiff = abs(aFast - aRef)
                        val rDiff = abs(rFast - rRef)
                        val gDiff = abs(gFast - gRef)
                        val bDiff = abs(bFast - bRef)
                        maxChannelDiff = max(maxChannelDiff, maxOf(aDiff, rDiff, gDiff, bDiff))
                    }

                    assertTrue(
                        "Pixel equivalence failed for rot=$rot, uvPixelStride=$uvPixelStride, crop=${case.cropRect}: maxChannelDiff=$maxChannelDiff (expected <= 1)",
                        maxChannelDiff <= 1
                    )
                }
            }
        }
        provider.stop()
    }

    @Test
    fun testFastYuvZeroAllocationAndTableReuseInSteadyStateAndBenchmarkTiming() {
        var fakeClockNs = 10_000_000_000L
        FrameClock.customElapsedRealtimeSource = {
            fakeClockNs += 50_000L
            fakeClockNs
        }

        val provider = MediaPipeFaceGeometryProvider(
            context = createMockContext(),
            dispatchExecutorOverride = Executor { r -> r.run() },
            reaperExecutorOverride = Executor { r -> r.run() }
        )
        provider.updateAnalysisSize(320)
        provider.setConversionMode(ConversionMode.FAST)

        val slot = provider.slots[0]
        val cropRect = createRect(0, 0, 640, 360)

        // 1. Warm-up conversion on 640x360 -> 180x320 (rot 270, pixelStride 2, padded rowStride 672)
        val warmupProxy = buildSyntheticYuvImageProxy(
            width = 640,
            height = 360,
            rotationDegrees = 270,
            uvPixelStride = 2,
            yRowPadding = 32,
            uvRowPadding = 16,
            cropRect = cropRect
        )
        provider.convertAndTransformIntoSlot(warmupProxy, 270, slot)

        // In CONV:FAST, sourceBitmap is never allocated; only targetBitmap (180x320) is allocated once!
        assertEquals("CONV:FAST must allocate only 1 Bitmap (targetBitmap) in slot", 1, slot.bitmapAllocationCount)
        assertEquals("CONV:FAST must never allocate intermediate sourceBitmap", 0, slot.sourceWidth)
        val initialBitmapAllocCount = slot.bitmapAllocationCount
        val initialByteAllocCount = slot.byteBufferAllocationCount
        val initialIntAllocCount = slot.intBufferAllocationCount
        val initialTableRebuildCount = slot.fastTableRebuildCount

        val yPlaneRef = slot.yPlaneBytes
        val uPlaneRef = slot.uPlaneBytes
        val vPlaneRef = slot.vPlaneBytes
        val targetArgbRef = slot.targetArgbScratch
        val dxYTableRef = slot.dxYOffsetTable
        val dyYTableRef = slot.dyYOffsetTable

        // 2. Run 50 steady-state frames and verify zero allocations & zero lookup-table rebuilds
        for (i in 0 until 50) {
            val proxy = buildSyntheticYuvImageProxy(
                width = 640,
                height = 360,
                rotationDegrees = 270,
                uvPixelStride = 2,
                yRowPadding = 32,
                uvRowPadding = 16,
                cropRect = cropRect,
                timestampNs = 1_000_000_000L + i * 33_333_333L
            )
            provider.convertAndTransformIntoSlot(proxy, 270, slot)
        }

        assertEquals("Zero Bitmap allocations in steady state", initialBitmapAllocCount, slot.bitmapAllocationCount)
        assertEquals("Zero ByteArray allocations in steady state", initialByteAllocCount, slot.byteBufferAllocationCount)
        assertEquals("Zero IntArray allocations in steady state", initialIntAllocCount, slot.intBufferAllocationCount)
        assertEquals("Zero lookup table rebuilds when dimensions/rotation/strides unchanged", initialTableRebuildCount, slot.fastTableRebuildCount)
        assertSame("yPlaneBytes array reused", yPlaneRef, slot.yPlaneBytes)
        assertSame("uPlaneBytes array reused", uPlaneRef, slot.uPlaneBytes)
        assertSame("vPlaneBytes array reused", vPlaneRef, slot.vPlaneBytes)
        assertSame("targetArgbScratch array reused", targetArgbRef, slot.targetArgbScratch)
        assertSame("dxYOffsetTable reused", dxYTableRef, slot.dxYOffsetTable)
        assertSame("dyYOffsetTable reused", dyYTableRef, slot.dyYOffsetTable)

        // Changing rotation (270 -> 90) rebuilds lookup tables once without reallocating plane buffers
        val rot90Proxy = buildSyntheticYuvImageProxy(
            width = 640,
            height = 360,
            rotationDegrees = 90,
            uvPixelStride = 2,
            yRowPadding = 32,
            uvRowPadding = 16,
            cropRect = cropRect
        )
        provider.convertAndTransformIntoSlot(rot90Proxy, 90, slot)
        assertEquals("Changing rotation must rebuild lookup tables once", initialTableRebuildCount + 1, slot.fastTableRebuildCount)
        assertEquals("Plane byte buffers must still not be reallocated", initialByteAllocCount, slot.byteBufferAllocationCount)

        // 3. Benchmark real wall-clock JVM timing: CONV:FAST vs CONV:OLD on 640x360 -> 180x320 (rot 270)
        FrameClock.customElapsedRealtimeSource = null
        val benchProxyFast = buildSyntheticYuvImageProxy(
            width = 640,
            height = 360,
            rotationDegrees = 270,
            uvPixelStride = 2,
            yRowPadding = 32,
            uvRowPadding = 16,
            cropRect = cropRect
        )
        val benchProxyOld = buildSyntheticYuvImageProxy(
            width = 640,
            height = 360,
            rotationDegrees = 270,
            uvPixelStride = 2,
            yRowPadding = 32,
            uvRowPadding = 16,
            cropRect = cropRect
        )
        val benchSlotFast = provider.slots[0]
        val benchSlotOld = provider.slots[2]

        provider.setConversionMode(ConversionMode.FAST)
        repeat(20) {
            provider.convertAndTransformIntoSlot(benchProxyFast, 270, benchSlotFast)
        }
        val iterations = 100
        val fastStartNs = System.nanoTime()
        repeat(iterations) {
            provider.convertAndTransformIntoSlot(benchProxyFast, 270, benchSlotFast)
        }
        val fastAvgMs = (System.nanoTime() - fastStartNs) / (iterations * 1_000_000.0)

        provider.setConversionMode(ConversionMode.OLD)
        repeat(20) {
            provider.convertAndTransformIntoSlot(benchProxyOld, 270, benchSlotOld)
        }
        val oldStartNs = System.nanoTime()
        repeat(iterations) {
            provider.convertAndTransformIntoSlot(benchProxyOld, 270, benchSlotOld)
        }
        val oldAvgMs = (System.nanoTime() - oldStartNs) / (iterations * 1_000_000.0)

        println(
            String.format(
                Locale.US,
                "R4_BENCHMARK_640x360_TO_180x320: CONV:FAST=%.3f ms/frame, CONV:OLD=%.3f ms/frame (speedup=%.2fx)",
                fastAvgMs,
                oldAvgMs,
                if (fastAvgMs > 0.0) oldAvgMs / fastAvgMs else 0.0
            )
        )
        assertTrue(
            "CONV:FAST ($fastAvgMs ms) must be faster than CONV:OLD ($oldAvgMs ms) on 640x360 -> 180x320",
            fastAvgMs < oldAvgMs
        )
        provider.stop()
    }

    @Test
    fun testPerStageConversionTimingsModeSourceTelemetryHudAndCsvAndConvModeToggle() {
        var fakeClockNs = 5_000_000_000L
        FrameClock.customElapsedRealtimeSource = {
            fakeClockNs += 200_000L // +0.2 ms per timestamp query so every stage has non-zero duration
            fakeClockNs
        }

        val probe = LatencyProbe()
        val governor = PerformanceGovernor(context = null, isLowEndDevice = false)
        val signalPipeline = SignalPipeline()

        var activeHandle: MediaPipeFaceGeometryProvider.LandmarkerHandle? = null
        val provider = MediaPipeFaceGeometryProvider(
            context = createMockContext(),
            dispatchExecutorOverride = Executor { r -> r.run() },
            reaperExecutorOverride = Executor { r -> r.run() }
        )
        provider.customLandmarkerFactory = { _, delegate, resultListener, _ ->
            object : MediaPipeFaceGeometryProvider.LandmarkerHandle {
                override val delegate: Delegate = delegate
                override var isClosed: Boolean = false
                override fun detectAsync(mpImage: MPImage, timestampMs: Long) {
                    fakeClockNs += 8_000_000L
                    resultListener(null as FaceLandmarkerResult?, mpImage)
                }

                override fun close() {
                    isClosed = true
                }
            }.also { activeHandle = it }
        }

        val session = CameraSession(
            provider = provider,
            signalPipeline = signalPipeline,
            governor = governor,
            latencyProbe = probe
        )
        val overlay = DebugOverlay(
            latencyProbe = probe,
            governor = governor,
            faceProvider = provider,
            cameraSession = session
        )

        // Default conversion mode is FAST
        assertEquals(ConversionMode.FAST, provider.conversionMode)
        assertEquals(ConversionMode.FAST, session.conversionMode)
        assertEquals("FAST", probe.conversionModeLabel)

        session.startProvider(ProviderConfig(analysisLongEdgePx = 320))
        assertNotNull(activeHandle)

        // Advance past any initial warm-up
        fakeClockNs += 6_000_000_000L

        // 1. Submit a frame in SMALL mode (640x360 @ 270°)
        provider.setResolutionModeForTelemetry(AnalysisResolutionMode.SMALL)
        probe.resolutionModeLabel = "SMALL"
        val smallProxy = buildSyntheticYuvImageProxy(
            width = 640,
            height = 360,
            rotationDegrees = 270,
            uvPixelStride = 2,
            yRowPadding = 16,
            uvRowPadding = 8,
            cropRect = createRect(0, 0, 640, 360),
            timestampNs = fakeClockNs
        )
        session.submitAnalysisFrame(
            imageProxy = smallProxy,
            timestampNs = fakeClockNs,
            rotationDegrees = 270,
            isFrontCamera = true,
            arrivalElapsedRealtimeNs = fakeClockNs
        )
        assertEquals("ImageProxy must be closed once", 1, smallProxy.closeCount.get())

        // 2. Record a frame in LEGACY mode (1280x720 @ 90°) so both SMALL and LEGACY source sizes are populated
        provider.setResolutionModeForTelemetry(AnalysisResolutionMode.LEGACY)
        fakeClockNs += 40_000_000L
        val legacyProxy = buildSyntheticYuvImageProxy(
            width = 1280,
            height = 720,
            rotationDegrees = 90,
            uvPixelStride = 2,
            yRowPadding = 0,
            uvRowPadding = 0,
            cropRect = createRect(0, 0, 1280, 720),
            timestampNs = fakeClockNs
        )
        session.submitAnalysisFrame(
            imageProxy = legacyProxy,
            timestampNs = fakeClockNs,
            rotationDegrees = 90,
            isFrontCamera = false,
            arrivalElapsedRealtimeNs = fakeClockNs
        )
        assertEquals("Legacy ImageProxy must be closed once", 1, legacyProxy.closeCount.get())

        assertEquals("640x360@270°", provider.formatModeSourceSummary(AnalysisResolutionMode.SMALL))
        assertEquals("1280x720@90°", provider.formatModeSourceSummary(AnalysisResolutionMode.LEGACY))

        // Verify per-stage rolling metrics in LatencyProbe
        val fs = FaceState(
            timestampNs = fakeClockNs,
            lifecycle = FaceLifecycle.State.TRACKING,
            isVisible = true,
            imageWidth = 180,
            imageHeight = 320
        )
        probe.recordRenderFrame(renderTimestampNs = fakeClockNs, faceState = fs)
        fakeClockNs += LatencyProbe.DEFAULT_HUD_REFRESH_INTERVAL_NS
        overlay.refreshCachedHudStringsIfNeeded(faceState = fs, fps = 60f, tier = "A", force = true)

        assertTrue("rollingUnpackMs must be > 0", probe.rollingUnpackMs > 0f)
        assertTrue("rollingYuvToRgbMs must be > 0", probe.rollingYuvToRgbMs > 0f)
        assertTrue("rollingBitmapCopyMs must be > 0", probe.rollingBitmapCopyMs > 0f)
        assertEquals("In CONV:FAST, rotateScaleMs is fused into yuvToRgbMs (0.0ms separate stage)", 0f, probe.rollingRotateScaleMs, 0.001f)
        assertTrue("rollingMpImageBuildMs must be > 0", probe.rollingMpImageBuildMs > 0f)

        assertTrue(
            "HUD cachedConvBreakdownLine must show Conv[FAST] and per-stage breakdown, was: ${overlay.cachedConvBreakdownLine}",
            overlay.cachedConvBreakdownLine.startsWith("Conv[FAST]: tot ") &&
                overlay.cachedConvBreakdownLine.contains("unpk ") &&
                overlay.cachedConvBreakdownLine.contains("yuv ") &&
                overlay.cachedConvBreakdownLine.contains("px ") &&
                overlay.cachedConvBreakdownLine.contains("rot ") &&
                overlay.cachedConvBreakdownLine.contains("mp ")
        )
        assertTrue(
            "HUD cachedFovLine must report source size, crop, rotation, and per-mode [S:... L:...] summary, was: ${overlay.cachedFovLine}",
            overlay.cachedFovLine.contains("src:1280x720(crop 1280x720)@90°") &&
                overlay.cachedFovLine.contains("[S:640x360@270° L:1280x720@90°]")
        )

        // Verify CSV columns 23..31
        val csvLines = probe.dumpCsv().trim().lines()
        val headerCols = csvLines.first().split(",")
        assertEquals(32, headerCols.size)
        assertEquals("convMode", headerCols[23])
        assertEquals("unpackMs", headerCols[24])
        assertEquals("yuvToRgbMs", headerCols[25])
        assertEquals("bitmapCopyMs", headerCols[26])
        assertEquals("rotateScaleMs", headerCols[27])
        assertEquals("mpImageBuildMs", headerCols[28])
        assertEquals("srcWidth", headerCols[29])
        assertEquals("srcHeight", headerCols[30])
        assertEquals("srcRotationDeg", headerCols[31])

        val rowCols = csvLines.last().split(",")
        assertEquals(32, rowCols.size)
        assertEquals("FAST", rowCols[23])
        assertEquals("1280", rowCols[29])
        assertEquals("720", rowCols[30])
        assertEquals("90", rowCols[31])

        // 3. Toggle to CONV:OLD at runtime -> verifies warm-up window and Conv[OLD] label
        session.setConversionMode(ConversionMode.OLD)
        assertEquals(ConversionMode.OLD, session.conversionMode)
        assertEquals(ConversionMode.OLD, provider.conversionMode)
        assertEquals("OLD", probe.conversionModeLabel)
        assertTrue("Switching conversion mode must start a warm-up window", probe.isInWarmup(fakeClockNs))

        overlay.refreshCachedHudStringsIfNeeded(faceState = fs, fps = 60f, tier = "A", force = true)
        assertTrue(
            "HUD cachedConvBreakdownLine must update to Conv[OLD], was: ${overlay.cachedConvBreakdownLine}",
            overlay.cachedConvBreakdownLine.startsWith("Conv[OLD]:")
        )

        // 4. Verify 8 debug toggle pills (including CONV) wrap cleanly on 360x800 dp without overlap
        val layout360 = DebugHudLayoutCalculator.measureScreenLayout(
            widthDp = 360f,
            heightDp = 800f,
            density = 2.0f,
            fontScale = 1.0f,
            isDebugEnabled = true,
            sectionState = DebugHudSectionState(
                isLatencyExpanded = true,
                isProjectionExpanded = true,
                isPoseExpanded = true
            )
        )
        assertFalse(layout360.intersects(layout360.topBarBounds, layout360.debugToggleRowBounds))
        assertFalse(layout360.intersects(layout360.debugToggleRowBounds, layout360.hudCardBounds))
        assertFalse(layout360.intersects(layout360.hudCardBounds, layout360.bottomControlsBounds))

        provider.stop()
    }
}
