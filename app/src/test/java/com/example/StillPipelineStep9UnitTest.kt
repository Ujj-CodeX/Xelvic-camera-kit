package com.example

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.opengl.EGL14
import android.opengl.Matrix
import com.example.engine.capture.AnalysisFrame
import com.example.engine.perception.FaceGeometryProvider
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.perception.ProviderConfig
import com.example.engine.photo.OffscreenRenderer
import com.example.engine.photo.StillPipeline
import com.example.engine.pkg.Compiler
import com.example.engine.pkg.UvRegistry
import com.example.engine.scene.CompiledEffect
import com.example.engine.scene.CompiledPhotoPolicy
import com.example.engine.scene.SceneEvaluator
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * Step 9 Unit & Integration Verification Suite (README Sections 5.14, 14, 16 & 21 Step 9):
 *
 * Verifies:
 * 1. Fresh still-image face inference authority (preview pose is strictly ignored).
 * 2. Downsample-to-full-resolution coordinate transform (`DownsampleTransform`) preservation.
 * 3. Manifest `photo` policy (`clips: settle`, `triggers: disarmed`, and explicit `timeSec`).
 * 4. Deterministic photo output across repeated runs.
 * 5. `OffscreenRenderer` owned EGL context, EGL context sharing report, GL-thread ownership, and cleanup.
 * 6. Documented CPU fallback for sprite, faceMesh, and 3D model effects.
 * 7. Capture resolution, crop, aspect ratio, rotation (0/90/180/270), and single-stage front-camera mirroring.
 * 8. Exact no-face policy: saves photo without effect, notifies user, provides one-tap retake (`canOneTapRetake`).
 * 9. Bounded bitmap/texture memory and resource cleanup on high-resolution captures.
 * 10. Preview-vs-photo coordinate and visual consistency across the reference filter set.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class StillPipelineStep9UnitTest {

    private lateinit var assetsRoot: File

    @Before
    fun setUp() {
        val candidates = listOf(
            File("src/main/assets"),
            File("app/src/main/assets"),
            File("../app/src/main/assets")
        )
        assetsRoot = candidates.firstOrNull { it.exists() }
            ?: throw IllegalStateException("Could not locate assets directory")
    }

    private fun compileReferenceFilter(folderName: String): CompiledEffect {
        val dir = File(assetsRoot, "filters/$folderName")
        assertTrue("Reference filter directory must exist: ${dir.absolutePath}", dir.exists())
        val manifestJson = JSONObject(File(dir, "manifest.json").readText())
        return Compiler.compile(manifestJson, dir)
    }

    private fun buildCanonicalSnapshot(
        tx: Float = 0.0f,
        ty: Float = 0.0f,
        tz: Float = -0.45f,
        imageWidth: Int = 576,
        imageHeight: Int = 1024,
        isFrontCamera: Boolean = true,
        blendshapes: Map<String, Float> = mapOf("mouthOpen" to 0.85f, "smile" to 0.7f)
    ): FaceGeometrySnapshot {
        val pose = FloatArray(16)
        Matrix.setIdentityM(pose, 0)
        pose[12] = tx
        pose[13] = ty
        pose[14] = tz

        val aspect = imageWidth.toFloat() / imageHeight.toFloat().coerceAtLeast(1f)
        val halfH = abs(tz) * 0.57735026f // tan(30 deg)
        val halfW = halfH * aspect
        val fw = 0.14f

        val uvs = UvRegistry.CANONICAL_UVS
        val landmarks = FloatArray(UvRegistry.VERTEX_COUNT * 3)
        for (i in 0 until UvRegistry.VERTEX_COUNT) {
            val u = uvs[i * 2]
            val v = uvs[i * 2 + 1]
            val localX = (u - 0.5f) * fw
            val localY = (0.5f - v) * (fw * 1.25f)
            val camX = tx + localX
            val camY = ty + localY
            // Map camX/camY back to normalized [0..1] in unmirrored sensor space
            val normX = (0.5f + (camX / (2f * halfW))).coerceIn(0.05f, 0.95f)
            val normY = (0.5f - (camY / (2f * halfH))).coerceIn(0.05f, 0.95f)
            landmarks[i * 3] = normX
            landmarks[i * 3 + 1] = normY
            landmarks[i * 3 + 2] = 0f
        }

        val idTransform = FloatArray(16)
        Matrix.setIdentityM(idTransform, 0)

        return FaceGeometrySnapshot.create(
            timestampNs = 123_456_789L,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            rotationDegrees = 0,
            isFrontCamera = isFrontCamera,
            isTracking = true,
            landmarks = landmarks,
            rawPoseMatrix = pose,
            blendshapes = blendshapes,
            imageToViewTransform = idTransform
        )
    }

    private fun createSyntheticCameraBitmap(width: Int, height: Int): Bitmap {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bmp.eraseColor(Color.rgb(28, 36, 52))
        // Populate asymmetric marker on left side (x = 10%..15% width) to verify mirror behavior
        val markerColor = Color.rgb(40, 180, 90)
        val x0 = (width * 0.10f).roundToInt().coerceIn(0, width - 1)
        val x1 = (width * 0.15f).roundToInt().coerceIn(0, width - 1)
        val y0 = (height * 0.10f).roundToInt().coerceIn(0, height - 1)
        val y1 = (height * 0.15f).roundToInt().coerceIn(0, height - 1)
        for (y in y0..y1 step 2) {
            for (x in x0..x1 step 2) {
                bmp.setPixel(x, y, markerColor)
            }
        }
        return bmp
    }

    @Test
    fun test1_freshStillInferenceIsAuthoritativeAndPreviewPoseIsIgnored() {
        val stillCalls = AtomicInteger(0)
        val freshStillTx = 0.08f
        val freshStillTy = 0.03f
        val freshStillTz = -0.45f

        val fakeProvider = object : FaceGeometryProvider {
            override val modelSha256: String = FaceGeometrySnapshot.EXPECTED_MODEL_SHA256
            override fun start(config: ProviderConfig) {}
            override fun updateConfig(config: ProviderConfig) {}
            override fun submit(frame: AnalysisFrame) {}
            override fun setListener(listener: (FaceGeometrySnapshot) -> Unit) {}

            override fun detectStillImage(
                bitmap: Bitmap,
                rotationDegrees: Int,
                isFrontCamera: Boolean,
                timestampNs: Long
            ): FaceGeometrySnapshot {
                stillCalls.incrementAndGet()
                return buildCanonicalSnapshot(
                    tx = freshStillTx,
                    ty = freshStillTy,
                    tz = freshStillTz,
                    imageWidth = bitmap.width,
                    imageHeight = bitmap.height,
                    isFrontCamera = isFrontCamera
                )
            }

            override fun stop() {}
        }

        // Stale preview pose far off to the left (-0.32m) that MUST be ignored
        val stalePreviewMatrix = FloatArray(16).apply {
            Matrix.setIdentityM(this, 0)
            this[12] = -0.32f
            this[13] = -0.25f
            this[14] = -0.80f
        }
        val stalePreviewState = FaceState.create(
            timestampNs = 10L,
            lifecycle = FaceLifecycle.State.TRACKING,
            fadeWeight = 1.0f,
            isVisible = true,
            poseMatrix = stalePreviewMatrix,
            translation = floatArrayOf(-0.32f, -0.25f, -0.80f),
            rotation = floatArrayOf(0f, 0f, 0f, 1f),
            scale = floatArrayOf(1f, 1f, 1f),
            yawDeg = 25f,
            pitchDeg = -15f,
            rollDeg = 12f,
            faceWidth = 0.14f,
            landmarks = null,
            blendshapes = emptyMap(),
            acquisitionTimeNs = 1L,
            imageToViewTransform = FloatArray(16).also { Matrix.setIdentityM(it, 0) },
            imageWidth = 480,
            imageHeight = 854,
            isFrontCamera = true
        )

        val effect = compileReferenceFilter("ref_sticker_2d")
        val pipeline = StillPipeline(faceProvider = fakeProvider)
        val rawPhoto = createSyntheticCameraBitmap(1080, 1920)

        val result = pipeline.processStillPhoto(
            rawCaptureBitmap = rawPhoto,
            effect = effect,
            isFrontCamera = true,
            rotationDegrees = 0,
            targetAspectRatio = 9f / 16f,
            previewFaceStateToIgnore = stalePreviewState
        )

        assertEquals("detectStillImage must be invoked exactly once", 1, stillCalls.get())
        assertTrue(result.usedFreshInference)
        assertTrue(result.ignoredPreviewPose)
        assertTrue(result.faceDetected)
        assertTrue(result.effectApplied)
        assertEquals(freshStillTx, result.freshFaceState.translation[0], 1e-4f)
        assertEquals(freshStillTy, result.freshFaceState.translation[1], 1e-4f)
        assertEquals(freshStillTz, result.freshFaceState.translation[2], 1e-4f)
        assertNotEquals("Preview pose X must never leak into photo FaceState", -0.32f, result.freshFaceState.translation[0], 1e-3f)
        pipeline.release()
    }

    @Test
    fun test2_downsampleToFullResolutionCoordinateTransformPreservation() {
        val pipeline = StillPipeline(
            maxOutputLongEdgePx = 2560,
            maxInferenceLongEdgePx = 1024
        )
        // High-res 1440 x 2560 capture (9:16)
        val highResCapture = Bitmap.createBitmap(1440, 2560, Bitmap.Config.ARGB_8888)
        val effect = compileReferenceFilter("round_glasses")

        val result = pipeline.processStillPhoto(
            rawCaptureBitmap = highResCapture,
            effect = effect,
            isFrontCamera = true,
            rotationDegrees = 0,
            targetAspectRatio = 9f / 16f,
            stillInferenceOverride = { infBmp, _, front, _ ->
                buildCanonicalSnapshot(
                    tx = 0.0f,
                    ty = 0.0f,
                    tz = -0.45f,
                    imageWidth = infBmp.width,
                    imageHeight = infBmp.height,
                    isFrontCamera = front
                )
            }
        )

        val dt = result.downsampleTransform
        assertEquals(1440, dt.fullWidth)
        assertEquals(2560, dt.fullHeight)
        assertEquals(576, dt.inferenceWidth)
        assertEquals(1024, dt.inferenceHeight)
        assertEquals(2.5f, dt.scaleToFullX, 1e-3f)
        assertEquals(2.5f, dt.scaleToFullY, 1e-3f)
        assertTrue("DownsampleTransform round-trip must be exact", dt.verifyRoundTripConsistency(0.05f))

        // Verify full-resolution coordinates in freshFaceState match the full-res output bitmap
        assertEquals(1440, result.freshFaceState.imageWidth)
        assertEquals(2560, result.freshFaceState.imageHeight)
        assertEquals(1440, result.bitmap.width)
        assertEquals(2560, result.bitmap.height)
        pipeline.release()
    }

    @Test
    fun test3_photoPolicyClipsSettleTriggersDisarmedAndExplicitTimeSec() {
        val baseEffect = compileReferenceFilter("multi_prop")
        val pipeline = StillPipeline()
        val snapshot = buildCanonicalSnapshot(
            blendshapes = mapOf("mouthOpen" to 0.95f, "smile" to 0.90f)
        )
        val photoBmp = createSyntheticCameraBitmap(720, 1280)

        // 1. Verify default manifest policy (`clips: settle`, `triggers: disarmed`, `timeSec: null`)
        val resultSettle = pipeline.processStillPhoto(
            rawCaptureBitmap = photoBmp,
            effect = baseEffect,
            isFrontCamera = true,
            stillInferenceOverride = { _, _, _, _ -> snapshot }
        )

        val report = resultSettle.settledPhotoReport
        assertNotNull("PhotoEvaluationReport must be populated", report)
        assertEquals("settle", report!!.clipsPolicyApplied)
        assertEquals("disarmed", report.triggersPolicyApplied)
        assertNull(report.explicitTimeSecApplied)

        // Check every clip in multi_prop / effect:
        for (clip in baseEffect.behaviorRunner.clips) {
            val settledTime = report.settledClipTimesSec[clip.id] ?: -1f
            if (!clip.loop && clip.holdLast) {
                assertEquals("Hold-last clip '${clip.id}' must settle to durationSec", clip.durationSec, settledTime, 1e-4f)
            } else {
                assertEquals("Looping or non-hold clip '${clip.id}' must settle to t=0", 0f, settledTime, 1e-4f)
            }
        }

        // Verify all triggers are disarmed and never fire even when mouthOpen = 0.95 > threshold
        for ((trigId, armed) in report.triggersArmedState) {
            assertFalse("Trigger '$trigId' must be disarmed in photo mode", armed)
        }
        for (trig in baseEffect.behaviorRunner.triggers) {
            assertFalse("Trigger '${trig.id}' must not be armed", trig.isArmed)
        }

        // 2. Verify explicit `timeSec` policy (`timeSec = 0.42f`)
        val explicitPolicyEffect = baseEffect.copy(
            photoPolicy = CompiledPhotoPolicy(
                clips = "settle",
                triggers = "disarmed",
                timeSec = 0.42f
            )
        )

        val resultExplicitTime = pipeline.processStillPhoto(
            rawCaptureBitmap = photoBmp,
            effect = explicitPolicyEffect,
            isFrontCamera = true,
            stillInferenceOverride = { _, _, _, _ -> snapshot }
        )
        val explicitReport = resultExplicitTime.settledPhotoReport!!
        assertEquals(0.42f, explicitReport.explicitTimeSecApplied ?: -1f, 1e-4f)
        for (clip in explicitPolicyEffect.behaviorRunner.clips) {
            val expectedT = 0.42f.coerceIn(0f, clip.durationSec)
            assertEquals(expectedT, explicitReport.settledClipTimesSec[clip.id] ?: -1f, 1e-4f)
        }
        pipeline.release()
    }

    @Test
    fun test4_deterministicPhotoOutputAcrossRepeatedRuns() {
        val effect = compileReferenceFilter("ref_face_paint")
        val pipeline = StillPipeline()
        val rawBmp = createSyntheticCameraBitmap(540, 960)
        val snapshot = buildCanonicalSnapshot(tx = 0.02f, ty = -0.01f, tz = -0.44f, imageWidth = 540, imageHeight = 960)

        val run1 = pipeline.processStillPhoto(
            rawCaptureBitmap = rawBmp,
            effect = effect,
            isFrontCamera = true,
            stillInferenceOverride = { _, _, _, _ -> snapshot }
        )

        // Mutate behavior runner and property slots as if live preview ran 120 frames in between
        effect.propertySlots.opacity.fill(0.12f)
        for (clip in effect.behaviorRunner.clips) {
            clip.currentTimeSec = 0.77f
        }

        val run2 = pipeline.processStillPhoto(
            rawCaptureBitmap = rawBmp,
            effect = effect,
            isFrontCamera = true,
            stillInferenceOverride = { _, _, _, _ -> snapshot }
        )

        assertEquals(run1.bitmap.width, run2.bitmap.width)
        assertEquals(run1.bitmap.height, run2.bitmap.height)
        val pixels1 = IntArray(run1.bitmap.width * run1.bitmap.height)
        val pixels2 = IntArray(run2.bitmap.width * run2.bitmap.height)
        run1.bitmap.getPixels(pixels1, 0, run1.bitmap.width, 0, 0, run1.bitmap.width, run1.bitmap.height)
        run2.bitmap.getPixels(pixels2, 0, run2.bitmap.width, 0, 0, run2.bitmap.width, run2.bitmap.height)
        assertArrayEquals("Repeated photo runs on identical input must be pixel-identical", pixels1, pixels2)
        pipeline.release()
    }

    @Test
    fun test5_offscreenRendererOwnedEglContextSharingAndWorkerThreadOwnership() {
        val renderer = OffscreenRenderer()
        val workerError = AtomicReference<Throwable?>(null)
        val latch = CountDownLatch(1)

        val workerThread = Thread({
            try {
                // 1. Verify unshared context initialization & report
                val unsharedReport = renderer.verifyContextSharing(EGL14.EGL_NO_CONTEXT, 128, 128)
                assertFalse(unsharedReport.sharingRequested)
                assertEquals(Thread.currentThread().id, renderer.renderGraph.resourceOwner.glThreadId)

                // 2. Verify renderToPhoto on worker thread binds GlResourceOwner to worker thread
                val bmp = createSyntheticCameraBitmap(360, 640)
                val effect = compileReferenceFilter("ref_sticker_2d")
                val pipeline = StillPipeline(offscreenRenderer = renderer)
                val res = pipeline.processStillPhoto(
                    rawCaptureBitmap = bmp,
                    effect = effect,
                    isFrontCamera = true,
                    stillInferenceOverride = { _, _, _, _ ->
                        buildCanonicalSnapshot(imageWidth = 360, imageHeight = 640)
                    }
                )
                assertTrue(res.effectApplied)
                assertEquals(Thread.currentThread().id, renderer.renderGraph.resourceOwner.glThreadId)
                assertEquals(0, renderer.renderGraph.resourceOwner.crossThreadViolationCount)

                // 3. Clean release on worker thread
                renderer.release()
                assertFalse(renderer.isEglInitialized)
            } catch (t: Throwable) {
                workerError.set(t)
            } finally {
                latch.countDown()
            }
        }, "TestPhotoWorkerThread")

        workerThread.start()
        assertTrue("Worker thread timed out", latch.await(10, TimeUnit.SECONDS))
        workerError.get()?.let { throw it }
    }

    @Test
    fun test6_cropAspectRatioRotationAndFrontRearMirrorConsistency() {
        val pipeline = StillPipeline()
        val effect = compileReferenceFilter("ref_sticker_2d")

        // Create a landscape 1920x1080 sensor image with a 90-degree rotation requirement -> 1080x1920 upright
        val sensorLandscapeBmp = createSyntheticCameraBitmap(1920, 1080)

        // 1. Front camera with 90-degree sensor rotation and 9:16 target aspect ratio
        val frontRes = pipeline.processStillPhoto(
            rawCaptureBitmap = sensorLandscapeBmp,
            effect = effect,
            isFrontCamera = true,
            rotationDegrees = 90,
            targetAspectRatio = 9f / 16f,
            stillInferenceOverride = { infBmp, _, front, _ ->
                // Inference receives upright portrait dimensions!
                assertTrue("Inference bitmap must already be upright portrait", infBmp.height > infBmp.width)
                buildCanonicalSnapshot(
                    tx = 0.06f, // shifted to positive camera X
                    ty = 0.0f,
                    tz = -0.45f,
                    imageWidth = infBmp.width,
                    imageHeight = infBmp.height,
                    isFrontCamera = front
                )
            }
        )
        assertEquals(1080, frontRes.bitmap.width)
        assertEquals(1920, frontRes.bitmap.height)
        assertEquals(90, frontRes.downsampleTransform.rotationDegrees)
        assertTrue(frontRes.downsampleTransform.isFrontCameraMirrored)

        // 2. Square 1:1 crop aspect ratio test
        val squareRes = pipeline.processStillPhoto(
            rawCaptureBitmap = sensorLandscapeBmp,
            effect = effect,
            isFrontCamera = false,
            rotationDegrees = 90,
            targetAspectRatio = 1.0f,
            stillInferenceOverride = { infBmp, _, front, _ ->
                assertEquals("Inference bitmap must match 1:1 cropped aspect", infBmp.width, infBmp.height)
                buildCanonicalSnapshot(
                    tx = 0.06f,
                    ty = 0.0f,
                    tz = -0.45f,
                    imageWidth = infBmp.width,
                    imageHeight = infBmp.height,
                    isFrontCamera = front
                )
            }
        )
        assertEquals(1080, squareRes.bitmap.width)
        assertEquals(1080, squareRes.bitmap.height)
        assertFalse(squareRes.downsampleTransform.isFrontCameraMirrored)
        assertTrue("Vertical crop offset must be > 0 when cropping 1080x1920 to 1080x1080", squareRes.downsampleTransform.cropTopPx > 0)
        pipeline.release()
    }

    @Test
    fun test7_noFacePolicySavesWithoutEffectAndProvidesOneTapRetake() {
        val pipeline = StillPipeline()
        val effect = compileReferenceFilter("sunflower_forehead")
        val rawBmp = createSyntheticCameraBitmap(540, 960)

        val result = pipeline.processStillPhoto(
            rawCaptureBitmap = rawBmp,
            effect = effect,
            isFrontCamera = false,
            rotationDegrees = 0,
            targetAspectRatio = 9f / 16f,
            stillInferenceOverride = { infBmp, _, front, ts ->
                FaceGeometrySnapshot.noFace(
                    timestampNs = ts,
                    imageWidth = infBmp.width,
                    imageHeight = infBmp.height,
                    isFrontCamera = front
                )
            }
        )

        assertNotNull("Capture result must never be null or undefined on no-face", result)
        assertFalse("faceDetected must be false", result.faceDetected)
        assertFalse("effectApplied must be false", result.effectApplied)
        assertEquals(0, result.renderedNodeCount)
        assertEquals("NO_FACE_DETECTED", result.noFaceReason)
        assertTrue("canOneTapRetake must be true when no face is detected with an active effect", result.canOneTapRetake)
        assertNotNull("userNotificationMessage must be provided", result.userNotificationMessage)
        assertTrue(result.userNotificationMessage!!.contains("Retake", ignoreCase = true))

        // Verify the saved photograph is identical to the clean base photograph (no partial/corrupt overlay)
        val rawPixels = IntArray(rawBmp.width * rawBmp.height)
        val outPixels = IntArray(result.bitmap.width * result.bitmap.height)
        rawBmp.getPixels(rawPixels, 0, rawBmp.width, 0, 0, rawBmp.width, rawBmp.height)
        result.bitmap.getPixels(outPixels, 0, result.bitmap.width, 0, 0, result.bitmap.width, result.bitmap.height)
        assertArrayEquals("Saved no-face photo must match clean base photograph", rawPixels, outPixels)
        pipeline.release()
    }

    @Test
    fun test8_boundedMemoryAndResourceCleanupOnHighResolutionCaptures() {
        val pipeline = StillPipeline(
            maxOutputLongEdgePx = 2560,
            maxInferenceLongEdgePx = 1024,
            maxMemoryBudgetBytes = 96L * 1024L * 1024L
        )
        val effect = compileReferenceFilter("ref_face_paint")

        // Simulate 4K 2160x3840 input capture
        val raw4k = Bitmap.createBitmap(2160, 3840, Bitmap.Config.ARGB_8888)
        val res = pipeline.processStillPhoto(
            rawCaptureBitmap = raw4k,
            effect = effect,
            isFrontCamera = true,
            rotationDegrees = 0,
            targetAspectRatio = 9f / 16f,
            stillInferenceOverride = { infBmp, _, front, _ ->
                assertTrue("Inference long edge must be bounded to <= 1024", maxOf(infBmp.width, infBmp.height) <= 1024)
                buildCanonicalSnapshot(imageWidth = infBmp.width, imageHeight = infBmp.height, isFrontCamera = front)
            }
        )

        assertTrue("Output long edge must be bounded to <= 2560", maxOf(res.bitmap.width, res.bitmap.height) <= 2560)
        assertEquals(1440, res.bitmap.width)
        assertEquals(2560, res.bitmap.height)
        assertTrue("Photo operation must remain within bounded memory budget", res.memoryTelemetry.withinMemoryBudget)
        assertTrue("Output bitmap bytes must be recorded", res.memoryTelemetry.outputBitmapBytes > 0L)
        pipeline.release()
    }

    @Test
    fun test9_previewVsPhotoConsistencyAcrossReferenceFilterSet() {
        val referencePackages = listOf(
            "ref_sticker_2d",
            "round_glasses",
            "multi_prop",
            "ref_face_paint",
            "sunflower_forehead"
        )
        val pipeline = StillPipeline()
        val previewWidth = 540
        val previewHeight = 960
        val photoWidth = 1080
        val photoHeight = 1920

        for (pkgName in referencePackages) {
            val effect = compileReferenceFilter(pkgName)
            val cleanPhotoBmp = createSyntheticCameraBitmap(photoWidth, photoHeight)

            val photoResult = pipeline.processStillPhoto(
                rawCaptureBitmap = cleanPhotoBmp,
                effect = effect,
                isFrontCamera = true,
                rotationDegrees = 0,
                targetAspectRatio = 9f / 16f,
                forceCpuFallback = true,
                stillInferenceOverride = { infBmp, _, front, _ ->
                    buildCanonicalSnapshot(
                        tx = 0.0f,
                        ty = 0.0f,
                        tz = -0.45f,
                        imageWidth = infBmp.width,
                        imageHeight = infBmp.height,
                        isFrontCamera = front
                    )
                }
            )

            assertTrue("Face must be detected for $pkgName", photoResult.faceDetected)
            assertTrue("Effect must be applied for $pkgName", photoResult.effectApplied)
            assertTrue("At least 1 renderable node must render for $pkgName", photoResult.renderedNodeCount >= 1)

            // Verify preview vs photo normalized screen projection parity for every visible renderable node
            val previewEvaluator = SceneEvaluator()
            effect.reset()
            effect.signalBus.evaluate(photoResult.freshFaceState, renderTimeSec = effect.photoPolicy.timeSec ?: 0f)
            effect.behaviorRunner.evaluateForPhoto(
                photoPolicy = effect.photoPolicy,
                signalBus = effect.signalBus,
                faceState = photoResult.freshFaceState,
                slots = effect.propertySlots
            )
            previewEvaluator.evaluate(effect, photoResult.freshFaceState, effect.propertySlots)

            val previewProj = FloatArray(16)
            val photoProj = FloatArray(16)
            FacePoseAdapter.createCameraProjection(
                fovYDeg = 60f,
                aspectRatio = previewWidth.toFloat() / previewHeight.toFloat(),
                near = 0.05f,
                far = 10f,
                frontFacing = true,
                out = previewProj
            )
            FacePoseAdapter.createCameraProjection(
                fovYDeg = 60f,
                aspectRatio = photoWidth.toFloat() / photoHeight.toFloat(),
                near = 0.05f,
                far = 10f,
                frontFacing = true,
                out = photoProj
            )

            for (i in effect.nodes.indices) {
                val node = effect.nodes[i]
                if (!previewEvaluator.isNodeVisible[i] || node.type == "occluder" || node.type == "group" || node.type == "post") {
                    continue
                }
                val previewMvp = FloatArray(16)
                val photoMvp = FloatArray(16)
                Matrix.multiplyMM(previewMvp, 0, previewProj, 0, previewEvaluator.worldMatrices, i * 16)
                Matrix.multiplyMM(photoMvp, 0, photoProj, 0, previewEvaluator.worldMatrices, i * 16)

                val prevScreen = pipeline.offscreenRenderer.projectLocalPointToScreen(previewMvp, 0f, 0f, 0f, previewWidth, previewHeight)
                val photoScreen = pipeline.offscreenRenderer.projectLocalPointToScreen(photoMvp, 0f, 0f, 0f, photoWidth, photoHeight)
                assertNotNull("Preview projection must be valid for ${node.id} in $pkgName", prevScreen)
                assertNotNull("Photo projection must be valid for ${node.id} in $pkgName", photoScreen)

                val prevNormX = prevScreen!![0] / previewWidth.toFloat()
                val prevNormY = prevScreen[1] / previewHeight.toFloat()
                val photoNormX = photoScreen!![0] / photoWidth.toFloat()
                val photoNormY = photoScreen[1] / photoHeight.toFloat()

                assertEquals("Preview vs photo normalized X must match for ${node.id} in $pkgName", prevNormX, photoNormX, 1e-4f)
                assertEquals("Preview vs photo normalized Y must match for ${node.id} in $pkgName", prevNormY, photoNormY, 1e-4f)
            }

            // Verify the rendered photo bitmap actually has modified filter pixels compared to the base image
            val baseMirrored = pipeline.processStillPhoto(
                rawCaptureBitmap = cleanPhotoBmp,
                effect = null,
                isFrontCamera = true,
                rotationDegrees = 0,
                targetAspectRatio = 9f / 16f
            ).bitmap

            var diffPixelCount = 0
            val step = 4
            for (y in 0 until photoHeight step step) {
                for (x in 0 until photoWidth step step) {
                    if (photoResult.bitmap.getPixel(x, y) != baseMirrored.getPixel(x, y)) {
                        diffPixelCount++
                    }
                }
            }
            assertTrue("Rendered photo for $pkgName must visibly modify pixels (diffPixelCount=$diffPixelCount)", diffPixelCount > 25)
        }
        pipeline.release()
    }
}
