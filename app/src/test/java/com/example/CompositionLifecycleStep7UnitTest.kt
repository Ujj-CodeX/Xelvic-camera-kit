package com.example

import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import com.example.engine.capture.AnalysisFrame
import com.example.engine.capture.CameraSession
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.perception.FaceGeometryProvider
import com.example.engine.perception.FaceGeometrySnapshot
import com.example.engine.perception.ProviderConfig
import com.example.engine.pkg.Compiler
import com.example.engine.render.*
import com.example.engine.scene.CompiledEffect
import com.example.engine.signal.FaceLifecycle
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import com.example.engine.signal.SignalPipeline
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import java.io.File
import java.util.concurrent.atomic.AtomicInteger

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class CompositionLifecycleStep7UnitTest {

    private class FakeProvider : FaceGeometryProvider {
        override val modelSha256: String = "mock_sha_step7"
        var startCount = 0
        var stopCount = 0
        var configUpdateCount = 0
        var lastConfig = ProviderConfig()
        private var listener: ((FaceGeometrySnapshot) -> Unit)? = null

        override fun start(config: ProviderConfig) {
            startCount++
            lastConfig = config
        }

        override fun updateConfig(config: ProviderConfig) {
            configUpdateCount++
            lastConfig = config
        }

        override fun updateAnalysisSize(longEdgePx: Int) {
            lastConfig = lastConfig.copy(analysisLongEdgePx = longEdgePx)
        }

        override fun submit(frame: AnalysisFrame) {
            frame.imageProxy?.close()
        }

        override fun setListener(listener: (FaceGeometrySnapshot) -> Unit) {
            this.listener = listener
        }

        fun emit(snapshot: FaceGeometrySnapshot) {
            listener?.invoke(snapshot)
        }

        override fun stop() {
            stopCount++
        }
    }

    private fun compileTestEffect(pkgDir: File, includePostNode: Boolean = false): CompiledEffect {
        pkgDir.mkdirs()
        val propFile = File(pkgDir, "prop.png")
        val bmp = android.graphics.Bitmap.createBitmap(4, 4, android.graphics.Bitmap.Config.ARGB_8888)
        propFile.outputStream().use { out ->
            bmp.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, out)
        }
        bmp.recycle()
        val postNodeJson = if (includePostNode) {
            """,
            {
              "id": "warm_grade",
              "type": "post",
              "effect": "colorGrade",
              "params": { "warmth": 0.2 },
              "renderOrder": 100
            }"""
        } else ""

        val manifestJson = """
            {
              "schema": "xfilter/3",
              "id": "step7_test_pkg",
              "name": "Step 7 Test Effect",
              "version": "1.0.0",
              "requires": {
                "runtime": 3,
                "features": ["node.sprite@1", "node.occluder@1", "material.unlit@1", "blend.alpha@1", "behavior.map@1"],
                "faces": 1
              },
              "assets": [
                { "id": "tex_prop", "type": "image", "path": "prop.png" }
              ],
              "materials": {
                "mat_prop": {
                  "template": "unlit@1",
                  "params": { "map": "tex_prop" },
                  "blend": "alpha"
                }
              },
              "nodes": [
                {
                  "id": "head_occ",
                  "type": "occluder",
                  "anchor": "head_center",
                  "shape": "ellipsoid",
                  "segments": [16, 24],
                  "scale": [0.6, 0.75, 0.65]
                },
                {
                  "id": "prop_sprite",
                  "type": "sprite",
                  "anchor": "forehead",
                  "material": "mat_prop",
                  "asset": "tex_prop",
                  "size": { "widthFW": 0.5, "heightFW": 0.5 },
                  "transform": {
                    "position": [0.0, 0.25, 0.1],
                    "rotationFollow": "full"
                  },
                  "renderOrder": 10
                }$postNodeJson
              ],
              "signals": {
                "smile_sig": "face.blend.mouthSmileLeft"
              },
              "behaviors": [
                {
                  "id": "smile_scale",
                  "type": "map",
                  "input": "sig.smile_sig",
                  "inputRange": [0.0, 1.0],
                  "outputRange": [1.0, 1.5],
                  "writes": [
                    { "path": "node:prop_sprite.scale", "mode": "multiply" }
                  ]
                }
              ],
              "tiers": {
                "A": { "textureMaxSize": 1024 },
                "B": { "textureMaxSize": 512 },
                "C": { "textureMaxSize": 256, "disable": ["smile_scale"] }
              }
            }
        """.trimIndent()

        return Compiler.compile(JSONObject(manifestJson), pkgDir)
    }

    private fun createTrackingFaceState(isFrontCamera: Boolean = true): FaceState {
        val identityPose = floatArrayOf(
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0.02f, -0.01f, -0.45f, 1f
        )
        return FaceState.create(
            timestampNs = 100_000_000L,
            lifecycle = FaceLifecycle.State.TRACKING,
            fadeWeight = 1.0f,
            isVisible = true,
            poseMatrix = identityPose,
            translation = floatArrayOf(0.02f, -0.01f, -0.45f),
            rotation = floatArrayOf(0f, 0f, 0f, 1f),
            scale = floatArrayOf(1f, 1f, 1f),
            yawDeg = 5f,
            pitchDeg = -2f,
            rollDeg = 1f,
            faceWidth = 0.14f,
            landmarks = FloatArray(478 * 3),
            blendshapes = mapOf("mouthSmileLeft" to 0.6f, "jawOpen" to 0.2f),
            acquisitionTimeNs = 50_000_000L,
            imageToViewTransform = null,
            imageWidth = 720,
            imageHeight = 1280,
            isFrontCamera = isFrontCamera
        )
    }

    @Test
    fun testCompositionSurfaceSelectionAndIdenticalSemantics() {
        val tempDir1 = File(System.getProperty("java.io.tmpdir"), "xelvic_step7_sel1_${System.nanoTime()}")
        val tempDir2 = File(System.getProperty("java.io.tmpdir"), "xelvic_step7_sel2_${System.nanoTime()}")
        try {
            val standardEffect1 = compileTestEffect(tempDir1, includePostNode = false)
            val standardEffect2 = compileTestEffect(tempDir2, includePostNode = false)
            val postEffect = compileTestEffect(tempDir1, includePostNode = true)

            val harness = OffsetTestHarness()
            // 1. Default unmeasured state must preserve TRANSPARENT_OVERLAY (no unconditional migration)
            assertEquals(OffsetMeasurementStatus.NOT_MEASURED, harness.lastDecision.status)
            assertEquals(
                CompositionMode.TRANSPARENT_OVERLAY,
                OffsetTestHarness.selectCompositionMode(standardEffect1, requiresVideoRecording = false, offsetDecision = harness.lastDecision)
            )

            // 2. Post-processing or video recording selects SINGLE_SCENE_GL per README Section 11.5
            assertEquals(
                CompositionMode.SINGLE_SCENE_GL,
                OffsetTestHarness.selectCompositionMode(postEffect, requiresVideoRecording = false, offsetDecision = harness.lastDecision)
            )
            assertEquals(
                CompositionMode.SINGLE_SCENE_GL,
                OffsetTestHarness.selectCompositionMode(standardEffect1, requiresVideoRecording = true, offsetDecision = harness.lastDecision)
            )

            // 3. Verify identical scene evaluation & rendering semantics across both CompositionSurface implementations
            val transparentSurface = TransparentOverlayCompositionSurface()
            val singleSceneSurface = SingleSceneCompositionSurface()

            transparentSurface.activeEffect = standardEffect1
            singleSceneSurface.activeEffect = standardEffect2

            transparentSurface.onSurfaceCreated(1080, 1920)
            singleSceneSurface.onSurfaceCreated(1080, 1920)

            val faceState = createTrackingFaceState(isFrontCamera = true)
            transparentSurface.renderFrame(faceState, dtSec = 0.016f, renderTimeSec = 1.0f, nowMs = 1000L)
            singleSceneSurface.renderFrame(faceState, dtSec = 0.016f, renderTimeSec = 1.0f, nowMs = 1000L)

            // Property slots and world matrices must match 1:1 between TRANSPARENT_OVERLAY and SINGLE_SCENE_GL
            assertArrayEquals(
                standardEffect1.propertySlots.scale,
                standardEffect2.propertySlots.scale,
                1e-5f
            )
            assertArrayEquals(
                standardEffect1.propertySlots.opacity,
                standardEffect2.propertySlots.opacity,
                1e-5f
            )
            assertArrayEquals(
                transparentSurface.sceneEvaluator.worldMatrices,
                singleSceneSurface.sceneEvaluator.worldMatrices,
                1e-5f
            )
        } finally {
            tempDir1.deleteRecursively()
            tempDir2.deleteRecursively()
        }
    }

    @Test
    fun testOffsetTestHarnessMarkerProjectionAndDecisionEvaluation() {
        val harness = OffsetTestHarness()
        val faceState = createTrackingFaceState(isFrontCamera = true)

        // Verify unmeasured baseline does not invent results
        val initialDecision = harness.evaluate(detachmentSpatialThresholdNorm = 0.03f, detachmentLagThresholdMs = 25f)
        assertEquals(OffsetMeasurementStatus.NOT_MEASURED, initialDecision.status)
        assertNull(initialDecision.meanLagMs)
        assertEquals(CompositionMode.TRANSPARENT_OVERLAY, initialDecision.recommendedMode)

        // Project face marker onto normalized viewport coordinates
        val markerNorm = harness.projectFaceMarkerNormalized(faceState, viewportAspect = 9f / 16f)
        assertEquals(2, markerNorm.size)
        assertTrue(markerNorm[0] in 0f..1f)
        assertTrue(markerNorm[1] in 0f..1f)

        // Record aligned samples and verify MEASURED_ALIGNED keeps TRANSPARENT_OVERLAY
        harness.recordSample(
            OffsetTestHarness.OffsetSample(
                frameTimestampNs = 100_000_000L,
                overlayRenderTimestampNs = 108_000_000L, // 8 ms lag
                previewMarkerXNorm = 0.50f,
                previewMarkerYNorm = 0.50f,
                overlayMarkerXNorm = 0.505f,
                overlayMarkerYNorm = 0.504f,
                headSpeedNormalizedPerSec = 1.2f
            )
        )
        val alignedDecision = harness.evaluate(detachmentSpatialThresholdNorm = 0.03f, detachmentLagThresholdMs = 25f)
        assertEquals(OffsetMeasurementStatus.MEASURED_ALIGNED, alignedDecision.status)
        assertFalse(alignedDecision.visibleDetachmentDetected)
        assertEquals(CompositionMode.TRANSPARENT_OVERLAY, alignedDecision.recommendedMode)

        // Record detached sample during fast head motion and verify MEASURED_DETACHED recommends SINGLE_SCENE_GL
        harness.recordSample(
            OffsetTestHarness.OffsetSample(
                frameTimestampNs = 200_000_000L,
                overlayRenderTimestampNs = 245_000_000L, // 45 ms lag
                previewMarkerXNorm = 0.30f,
                previewMarkerYNorm = 0.50f,
                overlayMarkerXNorm = 0.38f, // 0.08 spatial detachment
                overlayMarkerYNorm = 0.50f,
                headSpeedNormalizedPerSec = 2.5f
            )
        )
        val detachedDecision = harness.evaluate(detachmentSpatialThresholdNorm = 0.03f, detachmentLagThresholdMs = 25f)
        assertEquals(OffsetMeasurementStatus.MEASURED_DETACHED, detachedDecision.status)
        assertTrue(detachedDecision.visibleDetachmentDetected)
        assertEquals(CompositionMode.SINGLE_SCENE_GL, detachedDecision.recommendedMode)
        assertEquals(
            CompositionMode.SINGLE_SCENE_GL,
            OffsetTestHarness.selectCompositionMode(effect = null, requiresVideoRecording = false, offsetDecision = detachedDecision)
        )
    }

    @Test
    fun testCameraRotationAndLensSwitchTransformConsistency() {
        // Verify all 4 display rotations normalize cleanly and preserve single-stage front-camera mirroring
        val rotations = listOf(0, 90, 180, 270, 360, -90)
        val expectedNormalized = listOf(0, 90, 180, 270, 0, 270)

        for (i in rotations.indices) {
            val norm = FacePoseAdapter.normalizeRotationDegrees(rotations[i])
            assertEquals(expectedNormalized[i], norm)

            val frontCfg = FacePoseAdapter.deriveCameraTransformConfig(
                rawViewportWidth = 1080,
                rawViewportHeight = 1920,
                rotationDegrees = rotations[i],
                isFrontCamera = true
            )
            val rearCfg = FacePoseAdapter.deriveCameraTransformConfig(
                rawViewportWidth = 1080,
                rawViewportHeight = 1920,
                rotationDegrees = rotations[i],
                isFrontCamera = false
            )

            // Front camera mirrors X exactly once (negative X scale in projection matrix[0]), rear camera is positive
            assertTrue("Front camera projection must mirror X once", frontCfg.projectionMatrix[0] < 0f)
            assertTrue("Rear camera projection must not mirror X", rearCfg.projectionMatrix[0] > 0f)
            assertEquals(kotlin.math.abs(frontCfg.projectionMatrix[0]), rearCfg.projectionMatrix[0], 1e-6f)
            // Negative X in face-local space remains subject-left under both lenses
            assertTrue(FacePoseAdapter.isSubjectLeft(-0.25f))
            assertTrue(FacePoseAdapter.isSubjectRight(0.25f))
        }
    }

    @Test
    fun testPauseResumeSurfaceRecreationAndContextLossGpuRestoration() {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "xelvic_step7_ctx_${System.nanoTime()}")
        try {
            val effect = compileTestEffect(tempDir)
            val surface = TransparentOverlayCompositionSurface()
            surface.activeEffect = effect

            // 1. Initial surface creation & texture upload via renderFrame
            surface.onSurfaceCreated(1080, 1920)
            assertTrue(surface.isSurfaceReady)
            assertTrue(surface.areGlResourcesAllocated)
            assertEquals(1, surface.glCreateCount)
            assertEquals(0, surface.contextLossRecoveryCount)

            // Duplicate onSurfaceCreated on the same context generation must be a no-op
            surface.onSurfaceCreated(1080, 1920)
            assertEquals(1, surface.glCreateCount)
            assertEquals(0, surface.contextLossRecoveryCount)

            val faceState = createTrackingFaceState(isFrontCamera = true)
            surface.renderFrame(faceState, dtSec = 0.016f, renderTimeSec = 0.5f, nowMs = 500L)

            val texCache = surface.renderGraph.textures
            assertEquals("Active effect texture should be uploaded to TextureCache", 1, texCache.size())

            // 2. Short Pause (ON_PAUSE) -> frame rendering is suspended without GL teardown
            surface.onPause()
            assertTrue(surface.isPaused)
            assertTrue("Short pause must NOT release GL resources", surface.areGlResourcesAllocated)
            assertEquals(0, surface.glReleaseCount)
            assertSame("Active CompiledEffect must be preserved across pause", effect, surface.activeEffect)

            // 3. Resume from short pause (ON_RESUME) -> no GL recreate needed
            surface.onResume()
            assertFalse(surface.isPaused)
            assertEquals(1, surface.glCreateCount)
            assertSame("Active CompiledEffect must be preserved across resume", effect, surface.activeEffect)

            // 4. Long Stop (ON_STOP) -> releases GL resources once; restoring on resume + renderFrame creates once
            surface.onStop()
            assertFalse("ON_STOP must release GL resources", surface.areGlResourcesAllocated)
            assertEquals(1, surface.glReleaseCount)
            surface.onStart()
            surface.onResume()
            // Not restored eagerly in onResume; restored once on next renderFrame (or onSurfaceCreated)
            assertFalse(surface.areGlResourcesAllocated)
            surface.renderFrame(faceState, dtSec = 0.016f, renderTimeSec = 0.55f, nowMs = 550L)
            assertTrue(surface.areGlResourcesAllocated)
            assertEquals(1, surface.contextLossRecoveryCount)
            assertEquals(2, surface.glCreateCount)

            // 5. Surface destruction & recreation (simulating background -> foreground surface recreation)
            surface.onSurfaceDestroyed()
            assertFalse(surface.isSurfaceReady)
            surface.onSurfaceCreated(1080, 1920)
            assertTrue(surface.isSurfaceReady)
            assertEquals(2, surface.contextLossRecoveryCount)
            assertSame(effect, surface.activeEffect)

            // 6. Explicit EGL context loss and full GPU resource restoration
            val prevGen = surface.renderGraph.resourceOwner.contextGeneration
            surface.onContextLostAndRecreated()
            assertEquals(3, surface.contextLossRecoveryCount)
            assertEquals(prevGen + 1, surface.renderGraph.resourceOwner.contextGeneration)

            // Verify textures are restored and subsequent frame renders cleanly without crash
            assertEquals("Cached textures must be restored on new GL context", 1, texCache.size())
            surface.renderFrame(faceState, dtSec = 0.016f, renderTimeSec = 0.6f, nowMs = 600L)
            assertTrue(surface.sceneEvaluator.isNodeVisible[1])
        } finally {
            tempDir.deleteRecursively()
        }
    }

    @Test
    fun testCameraSessionLifecycleLensSwitchAndSingleCloseOwnership() {
        val context = RuntimeEnvironment.getApplication()
        val provider = FakeProvider()
        val pipeline = SignalPipeline()
        val governor = PerformanceGovernor(context)
        val session = CameraSession(provider, pipeline, governor)
        val tempDir = File(System.getProperty("java.io.tmpdir"), "xelvic_step7_cam_${System.nanoTime()}")

        try {
            val effect = compileTestEffect(tempDir)
            session.startProvider(ProviderConfig(outputPose = true, outputBlendshapes = true))
            session.applyEffectRequirements(effect)
            assertEquals(1, provider.startCount)
            assertEquals(1, session.trackerStartCount)

            // Create a fake ImageProxy that counts close() invocations
            val closeCounter = AtomicInteger(0)
            val proxy = java.lang.reflect.Proxy.newProxyInstance(
                ImageProxy::class.java.classLoader,
                arrayOf(ImageProxy::class.java)
            ) { _, method, _ ->
                when (method.name) {
                    "close" -> { closeCounter.incrementAndGet(); null }
                    "getWidth" -> 720
                    "getHeight" -> 1280
                    "getFormat" -> 35
                    "getImageInfo" -> java.lang.reflect.Proxy.newProxyInstance(
                        ImageInfo::class.java.classLoader,
                        arrayOf(ImageInfo::class.java)
                    ) { _, m, _ ->
                        when (m.name) {
                            "getTimestamp" -> 100_000_000L
                            "getRotationDegrees" -> 270
                            else -> null
                        }
                    }
                    else -> null
                }
            } as ImageProxy

            // 1. Short pause -> submitted analysis frame must be rejected and closed exactly once
            session.onPause()
            assertTrue(session.isPaused)
            val acceptedWhilePaused = session.submitAnalysisFrame(
                imageProxy = proxy,
                timestampNs = 100_000_000L,
                rotationDegrees = 270,
                isFrontCamera = true
            )
            assertFalse(acceptedWhilePaused)
            assertEquals("Paused frame must still close ImageProxy exactly once", 1, closeCounter.get())

            // 2. Resume after short pause -> frame is accepted, tracker was NOT restarted (trackerStartCount == 1)
            session.onResume(nowElapsedMs = 5_000L)
            assertFalse(session.isPaused)
            val acceptedAfterResume = session.submitAnalysisFrame(
                imageProxy = proxy,
                timestampNs = 200_000_000L,
                rotationDegrees = 270,
                isFrontCamera = true
            )
            assertTrue(acceptedAfterResume)
            assertEquals("Processed frame must close ImageProxy exactly once", 2, closeCounter.get())
            assertEquals("Tracker must not restart across short pause/resume", 1, provider.startCount)
            assertEquals(1, session.trackerStartCount)

            // 3. Long ON_STOP (> 45 s) -> closes provider and lazily restarts on resume (trackerStartCount becomes 2)
            session.onStop(nowElapsedMs = 10_000L)
            assertTrue(session.checkLongStopTimeout(nowElapsedMs = 10_000L + 46_000L))
            assertEquals(1, provider.stopCount)
            assertFalse(session.isProviderStarted)
            session.onStart(nowElapsedMs = 56_000L)
            session.onResume(nowElapsedMs = 56_000L)
            assertTrue(session.isProviderStarted)
            assertEquals("Tracker must lazily restart after >45s ON_STOP", 2, session.trackerStartCount)

            // 4. Lens switch (front -> rear) and display rotation change
            session.onCameraLensChanged(isFront = false, rotationDegrees = 90)
            assertFalse(session.isFrontCamera)
            assertEquals(90, session.displayRotationDegrees)
            assertEquals("Tracker must not restart on lens switch", 2, provider.startCount)
        } finally {
            tempDir.deleteRecursively()
        }
    }

    @Test
    fun testLivePreviewStaysOnTransparentOverlayAndSkipsPostWhenCameraSceneUnwired() {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "xelvic_step7_live_post_${System.nanoTime()}")
        try {
            val postEffect = compileTestEffect(tempDir, includePostNode = true)
            val harness = OffsetTestHarness()

            // 1. When Path B camera scene input is NOT wired (hasCameraSceneInput = false),
            // selectCompositionMode must stay on TRANSPARENT_OVERLAY even for effects with post nodes.
            assertEquals(
                CompositionMode.TRANSPARENT_OVERLAY,
                OffsetTestHarness.selectCompositionMode(
                    effect = postEffect,
                    requiresVideoRecording = false,
                    offsetDecision = harness.lastDecision,
                    hasCameraSceneInput = false
                )
            )
            // When camera scene input IS available (hasCameraSceneInput = true), it selects SINGLE_SCENE_GL.
            assertEquals(
                CompositionMode.SINGLE_SCENE_GL,
                OffsetTestHarness.selectCompositionMode(
                    effect = postEffect,
                    requiresVideoRecording = false,
                    offsetDecision = harness.lastDecision,
                    hasCameraSceneInput = true
                )
            )

            // 2. XelvicGlSurfaceView with cameraTextureId == 0 must stay on TransparentOverlayCompositionSurface
            // when a post-node filter is selected, preventing opaque black clears over PreviewView.
            val context = RuntimeEnvironment.getApplication()
            val glView = XelvicGlSurfaceView(context)
            assertFalse(glView.hasLiveCameraSceneInput)

            glView.activeEffect = postEffect
            assertEquals(CompositionMode.TRANSPARENT_OVERLAY, glView.compositionSurface.mode)
            assertTrue(glView.isPostSkippedInLivePreview(postEffect))

            // Even an explicit setCompositionMode(SINGLE_SCENE_GL) stays on TRANSPARENT_OVERLAY while cameraTextureId == 0
            glView.setCompositionMode(CompositionMode.SINGLE_SCENE_GL)
            assertEquals(CompositionMode.TRANSPARENT_OVERLAY, glView.compositionSurface.mode)

            // 3. Rendering postEffect on TransparentOverlayCompositionSurface skips post nodes in live mode
            // while non-post nodes remain visible and evaluated.
            val transparentSurface = glView.compositionSurface
            transparentSurface.onSurfaceCreated(1080, 1920)
            val faceState = createTrackingFaceState(isFrontCamera = true)
            transparentSurface.renderFrame(faceState, dtSec = 0.016f, renderTimeSec = 0.5f, nowMs = 500L)

            assertEquals("Post nodes must be skipped in TRANSPARENT_OVERLAY live mode", 0, glView.renderGraph.lastRenderedPostCount)
            val spriteIdx = postEffect.nodes.indexOfFirst { it.id == "prop_sprite" }
            val postIdx = postEffect.nodes.indexOfFirst { it.id == "warm_grade" }
            assertTrue("Non-post node must remain visible in live overlay mode", transparentSurface.sceneEvaluator.isNodeVisible[spriteIdx])
            assertFalse("Post node must be marked non-visible in live overlay mode", transparentSurface.sceneEvaluator.isNodeVisible[postIdx])

            // 4. Once a live camera texture is bound (Path B wired), XelvicGlSurfaceView transitions to SINGLE_SCENE_GL
            glView.bindCameraTexture(textureId = 42, isExternalOes = true)
            assertTrue(glView.hasLiveCameraSceneInput)
            assertEquals(CompositionMode.SINGLE_SCENE_GL, glView.compositionSurface.mode)
            assertFalse(glView.isPostSkippedInLivePreview(postEffect))
        } finally {
            tempDir.deleteRecursively()
        }
    }
}
