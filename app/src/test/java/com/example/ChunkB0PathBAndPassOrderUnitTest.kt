package com.example

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.opengl.Matrix
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.test.core.app.ApplicationProvider
import com.example.camera.CameraController
import com.example.engine.capture.CameraInputSurface
import com.example.engine.capture.DefaultFrameRing
import com.example.engine.capture.SurfaceTextureCameraInputSurface
import com.example.engine.debug.DebugOverlay
import com.example.engine.debug.LatencyProbe
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.photo.OffscreenRenderer
import com.example.engine.pkg.Compiler
import com.example.engine.pkg.PackageValidator
import com.example.engine.pkg.UvRegistry
import com.example.engine.render.CompositionMode
import com.example.engine.render.RenderGraph
import com.example.engine.render.SingleSceneCompositionSurface
import com.example.engine.render.XelvicGlSurfaceView
import com.example.engine.scene.SceneEvaluator
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FaceState
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import kotlin.math.abs

/**
 * Flow 1 — Chunk B0 Verification Suite:
 * 1. Pass order fix:
 *    `CAMERA_FRAME -> CAMERA_PIXEL_PASSES -> CLEAR_DEPTH -> OCCLUDERS -> OPAQUE_NODES -> TRANSPARENT_NODES -> POST_NODES -> OVERLAY_NODES -> FINAL_OUTPUT`.
 *    Camera-pixel effects (`faceMesh`, `eyeColor`) execute strictly before 3D occluders and 3D props.
 * 2. `CameraInputSurface` & `FrameRing` (delay depth 0):
 *    `attachToGlContext(textureId)`, `readFrame()`, `subscribe(onFrameAvailable)`, `pausable`,
 *    sensor timestamp recording, and single-owner mirror/crop/rotation (`uMirrorX == 0` downstream).
 * 3. Photo path order fix:
 *    Base photo drawn first, then camera-pixel passes before 3D props, then `post` applied
 *    exactly once (`lastPostApplicationCount == 1`; no CPU double application).
 * 4. `PATH:A` default & HUD telemetry:
 *    `PATH:A` (`TRANSPARENT_OVERLAY`) is the default startup mode; HUD displays `PATH:A`/`PATH:B`,
 *    `cam texture: OK/missing`, and `preview: <fps>fps`.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ChunkB0PathBAndPassOrderUnitTest {

    private fun buildSyntheticFaceState(
        tx: Float = 0.0f,
        ty: Float = 0.0f,
        tz: Float = -0.45f,
        imageWidth: Int = 256,
        imageHeight: Int = 256,
        isFrontCamera: Boolean = true
    ): FaceState {
        val pose = FloatArray(16)
        Matrix.setIdentityM(pose, 0)
        pose[12] = tx
        pose[13] = ty
        pose[14] = tz

        val aspect = imageWidth.toFloat() / imageHeight.toFloat().coerceAtLeast(1f)
        val halfH = abs(tz) * 0.57735026f
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
            val normX = (0.5f + (camX / (2f * halfW))).coerceIn(0.05f, 0.95f)
            val normY = (0.5f - (camY / (2f * halfH))).coerceIn(0.05f, 0.95f)
            landmarks[i * 3] = normX
            landmarks[i * 3 + 1] = normY
            landmarks[i * 3 + 2] = 0f
        }

        val idTransform = FloatArray(16)
        Matrix.setIdentityM(idTransform, 0)

        val snapshot = FaceGeometrySnapshot.create(
            timestampNs = 16_666_666L,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            rotationDegrees = 0,
            isFrontCamera = isFrontCamera,
            isTracking = true,
            landmarks = landmarks,
            rawPoseMatrix = pose,
            blendshapes = mapOf("mouthOpen" to 0.2f, "smile" to 0.4f),
            imageToViewTransform = idTransform
        )

        return FaceState.create(
            timestampNs = snapshot.timestampNs,
            lifecycle = FaceLifecycle.State.TRACKING,
            fadeWeight = 1.0f,
            isVisible = true,
            poseMatrix = pose,
            translation = floatArrayOf(tx, ty, tz),
            rotation = floatArrayOf(0f, 0f, 0f, 1f),
            scale = floatArrayOf(1f, 1f, 1f),
            yawDeg = 0f,
            pitchDeg = 0f,
            rollDeg = 0f,
            faceWidth = fw,
            landmarks = snapshot.landmarks,
            blendshapes = snapshot.blendshapes,
            acquisitionTimeNs = snapshot.timestampNs,
            imageToViewTransform = snapshot.imageToViewTransform,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            isFrontCamera = isFrontCamera
        )
    }

    private fun resolveFilterDir(packageId: String): File {
        val candidates = listOf(
            File("app/src/main/assets/filters/$packageId"),
            File("src/main/assets/filters/$packageId"),
            File("../app/src/main/assets/filters/$packageId")
        )
        return candidates.firstOrNull { it.exists() && File(it, "manifest.json").exists() }
            ?: File(".")
    }

    // -------------------------------------------------------------------------
    // 1. Pass Order Fix: camera-pixel passes strictly before occluders & 3D props
    // -------------------------------------------------------------------------

    @Test
    fun testCameraPixelPassesExecuteBeforeOccludersOpaquePropsAlphaLensesAndPost() {
        val manifestJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "test_chunk_b0_pass_order",
              "name": "Chunk B0 Pass Order Verification",
              "anchors": {
                "nose": { "type": "landmark", "index": 1 }
              },
              "materials": {
                "mat_paint": {
                  "template": "faceTexture@1",
                  "params": {
                    "map": "paint_texture",
                    "uvLayout": "mediapipe_canonical_468_v1",
                    "opacity": 0.9
                  },
                  "blend": "alpha",
                  "depthTest": true,
                  "depthWrite": false
                },
                "mat_opaque_prop": {
                  "template": "lambert@1",
                  "blend": "opaque",
                  "depthTest": true,
                  "depthWrite": true
                },
                "mat_alpha_lens": {
                  "template": "unlit@1",
                  "blend": "alpha",
                  "depthTest": true,
                  "depthWrite": false
                }
              },
              "nodes": [
                {
                  "id": "alpha_lens_node",
                  "type": "sprite",
                  "anchor": "nose",
                  "material": "mat_alpha_lens",
                  "renderOrder": 20
                },
                {
                  "id": "face_paint_node",
                  "type": "faceMesh",
                  "material": "mat_paint",
                  "region": "face_oval",
                  "renderOrder": 999
                },
                {
                  "id": "post_grade_node",
                  "type": "post",
                  "effect": "colorGrade",
                  "params": { "warmth": 0.2 },
                  "renderOrder": 100
                },
                {
                  "id": "opaque_prop_node",
                  "type": "sprite",
                  "anchor": "nose",
                  "material": "mat_opaque_prop",
                  "renderOrder": 5
                },
                {
                  "id": "head_occluder_node",
                  "type": "occluder",
                  "shape": "ellipsoid",
                  "renderOrder": 1
                }
              ]
            }
            """.trimIndent()
        )

        val validation = PackageValidator.validateManifest(manifestJson)
        assertTrue("Manifest must be valid: ${validation.errorReason}", validation.isValid)
        val compiled = Compiler.compile(manifestJson, resolveFilterDir("ref_face_paint"))

        val faceState = buildSyntheticFaceState()
        val evaluator = SceneEvaluator(maxNodes = 64)
        compiled.signalBus.evaluate(faceState, 0.25f)
        compiled.behaviorRunner.step(0.016f, compiled.signalBus, faceState, compiled.propertySlots, 250L)
        evaluator.evaluate(compiled, faceState, compiled.propertySlots)

        val renderGraph = RenderGraph()
        renderGraph.render(compiled, faceState, evaluator, 1080, 1920)

        assertEquals(
            listOf(
                RenderGraph.RenderPass.CAMERA_FRAME,
                RenderGraph.RenderPass.CAMERA_PIXEL_PASSES,
                RenderGraph.RenderPass.CLEAR_DEPTH,
                RenderGraph.RenderPass.OCCLUDERS,
                RenderGraph.RenderPass.OPAQUE_NODES,
                RenderGraph.RenderPass.TRANSPARENT_NODES,
                RenderGraph.RenderPass.POST_NODES,
                RenderGraph.RenderPass.OVERLAY_NODES,
                RenderGraph.RenderPass.FINAL_OUTPUT
            ),
            renderGraph.lastPassOrder
        )

        // Even though face_paint_node has renderOrder=999, it is a camera-pixel pass (faceMesh)
        // and MUST execute in CAMERA_PIXEL_PASSES before head_occluder_node, opaque_prop_node,
        // alpha_lens_node, and post_grade_node so face paint never draws over 3D props.
        assertEquals(1, renderGraph.lastRenderedCameraPixelCount)
        assertEquals(1, renderGraph.lastRenderedOccluderCount)
        assertEquals(1, renderGraph.lastRenderedOpaqueCount)
        assertEquals(1, renderGraph.lastRenderedTransparentCount)
        assertEquals(1, renderGraph.lastRenderedPostCount)
        assertEquals(
            listOf(
                "face_paint_node",
                "head_occluder_node",
                "opaque_prop_node",
                "alpha_lens_node",
                "post_grade_node"
            ),
            renderGraph.lastRenderedNodeIds
        )
    }

    // -------------------------------------------------------------------------
    // 2. CameraInputSurface & FrameRing (Delay Depth 0) + Single-Owner Mirror
    // -------------------------------------------------------------------------

    @Test
    fun testFrameRingDelayDepthZeroAndSensorTimestampRecording() {
        val ring = DefaultFrameRing(delayDepth = 0, capacity = 4)
        assertEquals(0, ring.delayDepth)
        assertEquals(0L, ring.recordedFrameCount)
        assertEquals(0L, ring.latestSensorTimestampNs)

        val idMat = FloatArray(16).apply { Matrix.setIdentityM(this, 0) }
        ring.recordFrame(
            textureId = 42,
            sensorTimestampNs = 100_000_000L,
            alignedElapsedRealtimeNs = 100_500_000L,
            texMatrix = idMat,
            isMirroredH = true,
            isExternalOes = true
        )
        ring.recordFrame(
            textureId = 42,
            sensorTimestampNs = 133_333_333L,
            alignedElapsedRealtimeNs = 133_833_333L,
            texMatrix = idMat,
            isMirroredH = true,
            isExternalOes = true
        )

        assertEquals(2L, ring.recordedFrameCount)
        assertEquals(133_333_333L, ring.latestSensorTimestampNs)

        // With delayDepth = 0, peekFrame returns the latest frame
        val display0 = ring.peekFrame()
        assertNotNull(display0)
        assertEquals(133_333_333L, display0!!.sensorTimestampNs)

        // When delayDepth is adjusted to 1, peekFrame returns the previous frame
        ring.delayDepth = 1
        val display1 = ring.peekFrame()
        assertNotNull(display1)
        assertEquals(100_000_000L, display1!!.sensorTimestampNs)
    }

    @Test
    fun testCameraInputSurfaceSingleOwnerMirrorCropAndPausableLifecycle() {
        val inputSurface = SurfaceTextureCameraInputSurface(
            initialOptions = CameraInputSurface.InputOptions(
                cropCenter = true,
                mirrorH = true,
                mirrorV = false,
                pausable = true,
                rotationDegrees = 0,
                sourceWidth = 1080,
                sourceHeight = 1920,
                viewportWidth = 1080,
                viewportHeight = 1920
            ),
            frameRing = DefaultFrameRing(delayDepth = 0)
        )

        var notifiedCount = 0
        var lastNotifiedSensorTs = 0L
        inputSurface.subscribe {
            notifiedCount++
            lastNotifiedSensorTs = inputSurface.lastSensorTimestampNs
        }

        inputSurface.attachToGlContext(textureId = 19)
        assertTrue(inputSurface.isAttachedToGl)
        assertEquals(19, inputSurface.boundTextureId)
        assertFalse(inputSurface.hasValidFrame)

        // Push frame 1 while unpaused
        val entry1 = inputSurface.pushSyntheticFrame(
            sensorTimestampNs = 500_000_000L,
            alignedElapsedRealtimeNs = 500_100_000L
        )
        assertNotNull(entry1)
        assertTrue(inputSurface.hasValidFrame)
        assertEquals(1, notifiedCount)
        assertEquals(500_000_000L, lastNotifiedSensorTs)
        assertEquals(500_000_000L, inputSurface.lastSensorTimestampNs)

        // Pause input surface (pausable = true): frame 2 must be ignored while paused
        inputSurface.setPaused(true)
        assertTrue(inputSurface.isPaused)
        val ignoredEntry = inputSurface.pushSyntheticFrame(
            sensorTimestampNs = 533_333_333L,
            alignedElapsedRealtimeNs = 533_433_333L
        )
        assertEquals(null, ignoredEntry)
        assertEquals(1, notifiedCount)
        assertEquals(500_000_000L, inputSurface.lastSensorTimestampNs)

        // Resume input surface: frame 3 is recorded
        inputSurface.setPaused(false)
        val entry3 = inputSurface.pushSyntheticFrame(
            sensorTimestampNs = 566_666_666L,
            alignedElapsedRealtimeNs = 566_766_666L
        )
        assertNotNull(entry3)
        assertEquals(2, notifiedCount)
        assertEquals(566_666_666L, inputSurface.lastSensorTimestampNs)

        // Bind to SingleSceneCompositionSurface and verify single-owner mirror policy:
        // CameraInputSurface applies mirrorH in texMatrix (scaleX < 0), while SingleSceneCompositionSurface
        // sets uMirrorX = 0 so the front camera is never double-mirrored.
        val singleScene = SingleSceneCompositionSurface()
        singleScene.onSurfaceCreated(1080, 1920)
        singleScene.onCameraConfigChanged(isFrontCamera = true, rotationDegrees = 0)
        singleScene.bindCameraInputSurface(inputSurface)

        val faceState = buildSyntheticFaceState(isFrontCamera = true)
        singleScene.renderFrame(faceState, dtSec = 0.016f, renderTimeSec = 0.1f, nowMs = 100L)

        assertTrue(singleScene.isCameraTextureOk)
        assertEquals(566_666_666L, singleScene.latestCameraSensorTimestampNs)
        assertEquals(
            "When CameraInputSurface is bound, uMirrorX in shader must be 0 (single owner of mirrorH)",
            0,
            singleScene.lastShaderMirrorXUniform
        )
        val appliedTexMat = singleScene.getCurrentTexMatrix()
        assertTrue(
            "CameraInputSurface texMatrix must own horizontal mirror (scaleX < 0)",
            appliedTexMat[0] < 0f
        )

        inputSurface.release()
    }

    // -------------------------------------------------------------------------
    // 3. Photo Path Order Fix: base photo first, camera-pixel before 3D props, post once
    // -------------------------------------------------------------------------

    @Test
    fun testPhotoPathOrderAndSinglePostExecutionWithoutDoubleApplication() {
        val manifestJson = JSONObject(
            """
            {
              "schema": "xfilter/3",
              "id": "test_photo_order_and_single_post",
              "name": "Photo Order & Single Post Test",
              "anchors": {
                "nose": { "type": "landmark", "index": 1 }
              },
              "assets": [
                {
                  "id": "paint_texture",
                  "type": "texture",
                  "path": "assets/paint.png"
                }
              ],
              "materials": {
                "mat_paint": {
                  "template": "faceTexture@1",
                  "params": {
                    "map": "paint_texture",
                    "uvLayout": "mediapipe_canonical_468_v1",
                    "color": [0.1, 0.8, 0.2, 1.0],
                    "opacity": 0.85
                  },
                  "blend": "alpha",
                  "depthTest": true,
                  "depthWrite": false
                },
                "mat_prop": {
                  "template": "unlit@1",
                  "params": {
                    "color": [0.95, 0.15, 0.15, 1.0]
                  },
                  "blend": "opaque",
                  "depthTest": true,
                  "depthWrite": true
                }
              },
              "nodes": [
                {
                  "id": "late_face_paint",
                  "type": "faceMesh",
                  "material": "mat_paint",
                  "region": "face_oval",
                  "renderOrder": 500
                },
                {
                  "id": "early_3d_prop",
                  "type": "sprite",
                  "anchor": "nose",
                  "scale": [0.9, 0.9, 0.9],
                  "material": "mat_prop",
                  "renderOrder": 1
                },
                {
                  "id": "single_warm_post",
                  "type": "post",
                  "effect": "colorGrade",
                  "params": {
                    "warmth": 0.35,
                    "contrast": 1.05,
                    "saturation": 1.0,
                    "brightness": 0.0,
                    "intensity": 1.0
                  },
                  "renderOrder": 100
                }
              ]
            }
            """.trimIndent()
        )

        val validation = PackageValidator.validateManifest(manifestJson)
        assertTrue("Manifest must be valid: ${validation.errorReason}", validation.isValid)
        val effect = Compiler.compile(manifestJson, resolveFilterDir("ref_face_paint"))

        val basePhoto = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        basePhoto.eraseColor(Color.rgb(120, 120, 120))

        val faceState = buildSyntheticFaceState(imageWidth = 128, imageHeight = 128)
        val evaluator = SceneEvaluator(maxNodes = 64)
        effect.signalBus.evaluate(faceState, 0.5f)
        effect.behaviorRunner.step(0.016f, effect.signalBus, faceState, effect.propertySlots, 500L)
        evaluator.evaluate(effect, faceState, effect.propertySlots)

        val offscreen = OffscreenRenderer()
        val renderResult = offscreen.renderCpuFallback(
            baseBitmap = basePhoto,
            effect = effect,
            faceState = faceState,
            evaluator = evaluator,
            tierMaxTexSize = 1024
        )

        // 1. Verify post was applied EXACTLY ONCE (no CPU double application)
        assertEquals(
            "Photo path must apply post nodes exactly once",
            1,
            offscreen.lastPostApplicationCount
        )
        assertEquals(
            "CPU post application count must be 1",
            1,
            offscreen.cpuPostApplicationCount
        )

        // 2. Verify stage order: BASE_PHOTO -> CAMERA_PIXEL_PASSES -> OPAQUE_NODES -> TRANSPARENT_NODES -> POST_ONCE -> OVERLAY_NODES
        assertEquals(
            listOf(
                "BASE_PHOTO",
                "CAMERA_PIXEL_PASSES",
                "OPAQUE_NODES",
                "TRANSPARENT_NODES",
                "POST_ONCE",
                "OVERLAY_NODES"
            ),
            offscreen.lastPhotoStageOrder
        )

        // 3. Verify late_face_paint (camera-pixel pass) was rendered BEFORE early_3d_prop (opaque 3D prop)
        assertEquals(
            listOf("late_face_paint", "early_3d_prop", "single_warm_post"),
            offscreen.lastRenderedNodeIds
        )

        // 4. Verify the red 3D prop at the nose center (64, 64) is on top of the green face paint
        //    (red channel dominates green channel at nose center because 3D prop drew after face paint)
        val noseCenterPixel = renderResult.bitmap.getPixel(64, 64)
        assertTrue(
            "3D prop must draw on top of face paint at nose center (R=${Color.red(noseCenterPixel)} > G=${Color.green(noseCenterPixel)})",
            Color.red(noseCenterPixel) > Color.green(noseCenterPixel)
        )

        // 5. Verify background pixel (8, 8) matches a single application of colorGrade (not double-applied)
        val expectedSingleGradeBmp = basePhoto.copy(Bitmap.Config.ARGB_8888, true)
        val postRenderer = com.example.engine.render.PostRenderer()
        postRenderer.applyPostNodesToBitmap(expectedSingleGradeBmp, effect, evaluator.isNodeVisible)
        assertEquals(
            "Background pixel must match exactly 1 post application (not double-graded)",
            expectedSingleGradeBmp.getPixel(8, 8),
            renderResult.bitmap.getPixel(8, 8)
        )
    }

    // -------------------------------------------------------------------------
    // 4. PATH:A Default & HUD Telemetry (cam texture OK/missing + preview fps)
    // -------------------------------------------------------------------------

    @Test
    fun testPathADefaultStartupAndHudCameraTextureAndPreviewFpsTelemetry() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val lifecycleOwner = object : LifecycleOwner {
            private val reg = LifecycleRegistry(this).apply {
                currentState = Lifecycle.State.RESUMED
            }
            override val lifecycle: Lifecycle get() = reg
        }

        val controller = CameraController(context, lifecycleOwner)
        val glView = XelvicGlSurfaceView(context)

        // Verify PATH:A (TRANSPARENT_OVERLAY) is the default on both CameraController and XelvicGlSurfaceView
        assertEquals(CompositionMode.TRANSPARENT_OVERLAY, controller.pathMode)
        assertEquals(CompositionMode.TRANSPARENT_OVERLAY, glView.requestedPathMode)
        assertEquals(CompositionMode.TRANSPARENT_OVERLAY, glView.compositionSurface.mode)
        assertEquals("A", controller.cameraSession.latencyProbe.pathModeLabel)
        assertFalse(controller.cameraSession.latencyProbe.cameraTextureOk)

        val probe = LatencyProbe()
        val overlay = DebugOverlay(latencyProbe = probe)
        val faceState = buildSyntheticFaceState()

        // Default PATH:A with no Path B frame yet -> "PATH:A" and "cam texture: missing"
        overlay.refreshCachedHudStringsIfNeeded(faceState = faceState, fps = 58.5f, tier = "A", force = true)
        assertTrue(overlay.cachedCameraLine.contains("PATH:A"))
        assertTrue(overlay.cachedCameraLine.contains("cam texture: missing"))
        assertTrue(overlay.cachedCountersLine.contains("PATH:A"))
        assertTrue(overlay.cachedCountersLine.contains("cam texture: missing"))
        assertTrue(overlay.cachedCountersLine.contains("preview: 58.5fps"))

        // Switch to PATH:B and push a camera frame into CameraController.cameraInputSurface
        controller.setPathMode(CompositionMode.SINGLE_SCENE_GL, previewView = null, glView = glView)
        assertEquals(CompositionMode.SINGLE_SCENE_GL, controller.pathMode)
        assertEquals(CompositionMode.SINGLE_SCENE_GL, glView.requestedPathMode)
        assertEquals(CompositionMode.SINGLE_SCENE_GL, glView.compositionSurface.mode)

        controller.cameraInputSurface.attachToGlContext(textureId = 11)
        controller.cameraInputSurface.pushSyntheticFrame(
            sensorTimestampNs = 777_000_000L,
            alignedElapsedRealtimeNs = 777_100_000L
        )
        assertTrue(controller.cameraSession.latencyProbe.cameraTextureOk)
        assertEquals(777_000_000L, controller.cameraSession.latencyProbe.latestCameraSensorTimestampNs)

        val controllerOverlay = DebugOverlay(
            latencyProbe = controller.cameraSession.latencyProbe,
            governor = controller.governor,
            faceProvider = controller.faceProvider,
            cameraSession = controller.cameraSession
        )
        controllerOverlay.refreshCachedHudStringsIfNeeded(faceState = faceState, fps = 60.0f, tier = "A", force = true)
        assertTrue(controllerOverlay.cachedCameraLine.contains("PATH:B"))
        assertTrue(controllerOverlay.cachedCameraLine.contains("cam texture: OK"))
        assertTrue(controllerOverlay.cachedCountersLine.contains("PATH:B"))
        assertTrue(controllerOverlay.cachedCountersLine.contains("cam texture: OK"))
        assertTrue(controllerOverlay.cachedCountersLine.contains("preview: 60.0fps"))

        // Restore PATH:A default
        controller.setPathMode(CompositionMode.TRANSPARENT_OVERLAY, previewView = null, glView = glView)
        assertEquals(CompositionMode.TRANSPARENT_OVERLAY, controller.pathMode)
        assertEquals(CompositionMode.TRANSPARENT_OVERLAY, glView.requestedPathMode)
        assertEquals(CompositionMode.TRANSPARENT_OVERLAY, glView.compositionSurface.mode)

        controller.stop()
    }
}
