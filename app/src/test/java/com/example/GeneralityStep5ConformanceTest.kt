package com.example

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.example.engine.pkg.Compiler
import com.example.engine.pkg.PackageValidator
import com.example.engine.scene.AnchorResolver
import com.example.engine.scene.PropertySlots
import com.example.engine.scene.SceneEvaluator
import com.example.engine.signal.FaceState
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class GeneralityStep5ConformanceTest {

    private lateinit var context: Context

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
    }

    private fun resolveFilterDir(filterName: String): File {
        val paths = listOf(
            File("app/src/main/assets/filters", filterName),
            File("src/main/assets/filters", filterName)
        )
        val dir = paths.firstOrNull { it.exists() && it.isDirectory }
        assertNotNull("Filter directory for '$filterName' must exist", dir)
        return dir!!
    }

    @Test
    fun testGlassesPackageValidationAndCompilation() {
        val glassesDir = resolveFilterDir("round_glasses")
        val manifestFile = File(glassesDir, "manifest.json")
        assertTrue("Glasses manifest must exist", manifestFile.exists())

        val json = JSONObject(manifestFile.readText())
        val validation = PackageValidator.validateManifest(json)
        assertTrue("round_glasses manifest must be valid: ${validation.errorReason}", validation.isValid)

        val compiled = Compiler.compile(json, glassesDir)
        assertNotNull("Compiled effect must not be null", compiled)
        assertEquals("round_glasses", compiled.packageId)
        assertEquals("Round Glasses", compiled.packageName)

        // 2 nodes: head occluder and glasses model
        assertEquals(2, compiled.nodes.size)
        val occluder = compiled.nodes.find { it.type == "occluder" }
        assertNotNull("Head occluder node must exist", occluder)
        assertEquals("head_center", occluder!!.anchor)
        assertEquals("ellipsoid", occluder.occluderShape)

        val glassesNode = compiled.nodes.find { it.type == "model" }
        assertNotNull("Glasses model node must exist", glassesNode)
        assertEquals("eye_center", glassesNode!!.anchor)
    }

    @Test
    fun testGlassesTwoAnchorResolution() {
        val glassesDir = resolveFilterDir("round_glasses")
        val json = JSONObject(File(glassesDir, "manifest.json").readText())
        val compiled = Compiler.compile(json, glassesDir)

        val resolver = compiled.anchorResolver
        val faceState = FaceState(
            isVisible = true,
            faceWidth = 0.2f
        )

        val outHead = FloatArray(3)
        resolver.resolveAnchor("head_center", faceState, outHead)
        assertEquals(0f, outHead[0], 0.001f)
        assertEquals(0.20f, outHead[1], 0.001f)
        assertEquals(-0.12f, outHead[2], 0.001f)

        val outEyes = FloatArray(3)
        resolver.resolveAnchor("eye_center", faceState, outEyes)
        assertEquals(0f, outEyes[0], 0.001f)
        assertEquals(0.15f, outEyes[1], 0.001f)
        assertEquals(0.12f, outEyes[2], 0.001f)
    }

    @Test
    fun testGlassesAlphaTransparencyAndOcclusionConfiguration() {
        val glassesDir = resolveFilterDir("round_glasses")
        val json = JSONObject(File(glassesDir, "manifest.json").readText())
        val compiled = Compiler.compile(json, glassesDir)

        // Verify frame and lens materials in compiled effect
        val mats = compiled.materials
        assertTrue("Must contain frame material", mats.containsKey("frame"))
        assertTrue("Must contain lens material", mats.containsKey("lens"))

        val frameMat = mats["frame"]!!
        val lensMat = mats["lens"]!!

        assertEquals("lambert@1", frameMat.template)
        assertEquals("opaque", frameMat.blend)
        assertTrue(frameMat.depthWrite)

        assertEquals("unlit@1", lensMat.template)
        assertEquals("alpha", lensMat.blend)
        assertFalse("Alpha-blended lens must not write depth", lensMat.depthWrite)

        // Head occluder dimensions
        val occluder = compiled.nodes.first { it.type == "occluder" }
        assertNotNull(occluder.radiiFW)
        assertEquals(0.56f, occluder.radiiFW!![0], 0.001f)
        assertEquals(0.66f, occluder.radiiFW!![1], 0.001f)
        assertEquals(0.68f, occluder.radiiFW!![2], 0.001f)
    }

    @Test
    fun testGlassesAbsenceOfUnnecessaryBehaviors() {
        val glassesDir = resolveFilterDir("round_glasses")
        val json = JSONObject(File(glassesDir, "manifest.json").readText())
        val compiled = Compiler.compile(json, glassesDir)

        val runner = compiled.behaviorRunner
        assertTrue("Glasses must have no clips", runner.clips.isEmpty())
        assertTrue("Glasses must have no maps", runner.maps.isEmpty())
        assertTrue("Glasses must have no triggers", runner.triggers.isEmpty())
        assertTrue("Glasses must have no springs", runner.springs.isEmpty())

        // Execution step does zero unnecessary work
        val slots = compiled.propertySlots
        val faceState = FaceState(isVisible = true)
        runner.step(0.016f, compiled.signalBus, faceState, slots)
        assertEquals(1f, slots.visible[0], 0.001f)
        assertEquals(1f, slots.visible[1], 0.001f)
    }

    @Test
    fun testMultiPropValidationAndMultipleNodesResolution() {
        val propDir = resolveFilterDir("multi_prop")
        val manifestFile = File(propDir, "manifest.json")
        val json = JSONObject(manifestFile.readText())

        val validation = PackageValidator.validateManifest(json)
        assertTrue("multi_prop manifest must be valid: ${validation.errorReason}", validation.isValid)

        val compiled = Compiler.compile(json, propDir)
        assertNotNull(compiled)
        assertEquals("multi_prop", compiled.packageId)

        // 3 independent nodes: hat (model), mustache (sprite), monocle (sprite)
        assertEquals(3, compiled.nodes.size)
        val hat = compiled.nodes.find { it.id == "prop_top_hat" }
        val mustache = compiled.nodes.find { it.id == "prop_mustache" }
        val monocle = compiled.nodes.find { it.id == "prop_monocle" }

        assertNotNull(hat)
        assertNotNull(mustache)
        assertNotNull(monocle)

        assertEquals("model", hat!!.type)
        assertEquals("forehead", hat.anchor)

        assertEquals("sprite", mustache!!.type)
        assertEquals("mouth_center", mustache.anchor)

        assertEquals("sprite", monocle!!.type)
        assertEquals("brow_custom", monocle.anchor)
    }

    @Test
    fun testMultiPropCustomAnchorResolution() {
        val propDir = resolveFilterDir("multi_prop")
        val json = JSONObject(File(propDir, "manifest.json").readText())
        val compiled = Compiler.compile(json, propDir)

        val landmarks = FloatArray(478 * 3)
        // Set landmark 70 and 105 with known coordinates
        landmarks[70 * 3] = 0.45f
        landmarks[70 * 3 + 1] = 0.35f
        landmarks[70 * 3 + 2] = 0.01f

        landmarks[105 * 3] = 0.55f
        landmarks[105 * 3 + 1] = 0.35f
        landmarks[105 * 3 + 2] = 0.01f

        val faceState = FaceState(
            isVisible = true,
            landmarks = landmarks,
            faceWidth = 0.2f
        )

        val out = FloatArray(3)
        compiled.anchorResolver.resolveAnchor("brow_custom", faceState, out)
        assertFalse("Custom anchor X must be finite", out[0].isNaN())
        assertFalse("Custom anchor Y must be finite", out[1].isNaN())
        assertFalse("Custom anchor Z must be finite", out[2].isNaN())
    }

    @Test
    fun testMultiPropTapTriggerAndCustomEventEmission() {
        val propDir = resolveFilterDir("multi_prop")
        val json = JSONObject(File(propDir, "manifest.json").readText())
        val compiled = Compiler.compile(json, propDir)

        val runner = compiled.behaviorRunner
        val slots = compiled.propertySlots

        var emittedEvent: String? = null
        runner.onCustomEvent = { eventName ->
            emittedEvent = eventName
        }

        // Monocle is node index 2, initially visible = 1.0
        val monocleIndex = compiled.nodes.indexOfFirst { it.id == "prop_monocle" }
        assertEquals(2, monocleIndex)
        assertEquals(1f, slots.visible[monocleIndex], 0.001f)

        // Fire tap trigger
        val tapFaceState = FaceState(isVisible = true, triggeredEvent = "tap")
        runner.step(0.016f, compiled.signalBus, tapFaceState, slots)

        // Event should be emitted
        assertEquals("custom_multi_prop_tapped", emittedEvent)

        // Visibility should be toggled off (1.0 -> 0.0)
        assertEquals(0f, slots.visible[monocleIndex], 0.001f)
    }

    @Test
    fun testGoldenVectorsAndReferenceSignals() {
        // Test predictable transform hierarchy and matrix outputs against golden vectors
        val evaluator = SceneEvaluator(maxNodes = 4)
        val glassesDir = resolveFilterDir("round_glasses")
        val json = JSONObject(File(glassesDir, "manifest.json").readText())
        val compiled = Compiler.compile(json, glassesDir)

        val faceState = FaceState(
            isVisible = true,
            faceWidth = 0.25f,
            poseMatrix = floatArrayOf(
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f,
                0f, 0f, 1f, 0f,
                0f, 0f, -0.5f, 1f
            )
        )

        evaluator.evaluate(compiled, faceState, compiled.propertySlots)

        assertTrue("Head occluder must be visible", evaluator.isNodeVisible[0])
        assertTrue("Glasses model must be visible", evaluator.isNodeVisible[1])

        val worldMatrices = evaluator.worldMatrices
        assertNotNull(worldMatrices)
        assertTrue("World matrices buffer must accommodate nodes", worldMatrices.size >= 16 * compiled.nodes.size)
        assertEquals(0f, worldMatrices[12], 0.001f)
        assertEquals(0.20f * 0.25f, worldMatrices[13], 0.001f)
        assertEquals(-0.5f + (-0.12f * 0.25f), worldMatrices[14], 0.001f)
    }

    @Test
    fun testEngineZeroSourceDiffAndCIHardcodingRejection() {
        // Scan engine directory
        val engineDirs = listOf(
            File("app/src/main/java/com/example/engine"),
            File("src/main/java/com/example/engine")
        )
        val engineDir = engineDirs.firstOrNull { it.exists() && it.isDirectory }
        assertNotNull("Engine directory must exist", engineDir)

        val forbiddenTokens = listOf(
            "sunflower_forehead",
            "periwinkle_forehead",
            "round_glasses",
            "ref_glasses",
            "multi_prop",
            "ref_multi_prop",
            "star_cheek",
            "ref_sticker_2d",
            "ref_face_paint",
            "sunflower_pivot",
            "periwinkle_pivot",
            "prop_top_hat",
            "prop_mustache",
            "prop_monocle",
            "glasses_frame",
            "heart_mood_filter",
            "heart_crown",
            "furry_glasses_filter",
            "furry_glasses",
            "furry_plush_pink_srgb.png"
        )

        var violations = 0
        engineDir!!.walkTopDown().forEach { file ->
            if (file.isFile && (file.name.endsWith(".kt") || file.name.endsWith(".java"))) {
                val content = file.readText()
                for (token in forbiddenTokens) {
                    val regex = Regex("\\b${Regex.escape(token)}\\b", RegexOption.IGNORE_CASE)
                    if (regex.containsMatchIn(content)) {
                        violations++
                        System.err.println("Violation in ${file.name}: found token '$token'")
                    }
                }
            }
        }

        assertEquals("Zero hardcoded filter/node/asset strings in engine sources", 0, violations)
    }

    @Test
    fun testPeriwinkleForeheadPackageValidationAndCompilation() {
        val periwinkleDir = resolveFilterDir("periwinkle_forehead")
        val manifestFile = File(periwinkleDir, "manifest.json")
        assertTrue("Periwinkle manifest must exist", manifestFile.exists())

        val json = JSONObject(manifestFile.readText())
        val validation = PackageValidator.validateManifest(json)
        assertTrue("periwinkle_forehead manifest must be valid: ${validation.errorReason}", validation.isValid)

        val glbAsset = json.getJSONArray("assets").getJSONObject(0)
        val glbFile = File(periwinkleDir, glbAsset.getString("path"))
        val assetVal = PackageValidator.validateAssetFile(
            glbFile,
            glbAsset.getString("sha256"),
            glbAsset.getLong("bytes")
        )
        assertTrue("periwinkle_forehead GLB must pass hash/byte/structure validation: ${assetVal.errorReason}", assetVal.isValid)

        val compiled = Compiler.compile(json, periwinkleDir)
        assertEquals("periwinkle_forehead", compiled.packageId)
        assertEquals("Periwinkle Forehead", compiled.packageName)

        // Verify model node and occluders
        val modelNode = compiled.nodes.first { it.type == "model" }
        assertEquals("forehead_left_hairline", modelNode.anchor)
        assertEquals(0.0f, modelNode.basePosition[0], 1e-4f)
        assertEquals(0.02f, modelNode.basePosition[1], 1e-4f)
        assertEquals(0.02f, modelNode.basePosition[2], 1e-4f)

        // Verify driver target 1.0 + 0.06*u -> [1.0, 1.06]
        val smileMap = compiled.behaviorRunner.maps.first { it.id == "soft_petal_smile" }
        assertEquals(1.0f, smileMap.outMin, 1e-4f)
        assertEquals(1.06f, smileMap.outMax, 1e-4f)

        // Verify spring parameters (stiffness=70, dampingRatio=0.7, gain=0.18, maxAngleDeg=4.0)
        val spring = compiled.behaviorRunner.springs.first { it.id == "subtle_head_inertia" }
        assertEquals(70.0f, spring.stiffness, 1e-4f)
        assertEquals(0.7f, spring.dampingRatio, 1e-4f)
        assertEquals(0.18f, spring.gain, 1e-4f)
        assertEquals(4.0f, spring.maxAngleDeg, 1e-4f)
    }

    @Test
    fun testHeartMoodFilterPackageValidationAndCompilation() {
        val heartDir = resolveFilterDir("heart_mood_filter")
        val manifestFile = File(heartDir, "manifest.json")
        assertTrue("Heart Mood Filter manifest must exist", manifestFile.exists())

        val json = JSONObject(manifestFile.readText())
        val validation = PackageValidator.validateManifest(json)
        assertTrue("heart_mood_filter manifest must be valid: ${validation.errorReason}", validation.isValid)

        val assetsArr = json.getJSONArray("assets")
        assertEquals(5, assetsArr.length())
        for (i in 0 until assetsArr.length()) {
            val a = assetsArr.getJSONObject(i)
            val assetFile = File(heartDir, a.getString("path"))
            val assetVal = PackageValidator.validateAssetFile(
                assetFile,
                a.getString("sha256"),
                a.getLong("bytes")
            )
            assertTrue("Asset ${a.getString("path")} must pass hash/byte/structure validation: ${assetVal.errorReason}", assetVal.isValid)
        }

        val compiled = Compiler.compile(json, heartDir)
        assertEquals("heart_mood_filter", compiled.packageId)
        assertEquals("Heart Mood Filter", compiled.packageName)

        // Verify occluders (headProxyEllipsoid radii [0.62, 0.82, 0.35] + faceMesh pushBackFW=0.015)
        val ellipsoidOcc = compiled.nodes.first { it.type == "occluder" && it.occluderShape == "ellipsoid" }
        assertEquals("head_center", ellipsoidOcc.anchor)
        assertEquals(0.62f, ellipsoidOcc.radiiFW!![0], 1e-4f)
        assertEquals(0.82f, ellipsoidOcc.radiiFW!![1], 1e-4f)
        assertEquals(0.35f, ellipsoidOcc.radiiFW!![2], 1e-4f)

        val faceMeshOcc = compiled.nodes.first { it.type == "occluder" && it.occluderShape == "faceMesh" }
        assertEquals(0.015f, faceMeshOcc.pushBackFW, 1e-4f)

        // Verify 3 LOD model nodes anchored to head_center at offsetFW [0.0, 0.56, 0.18]
        val modelNodes = compiled.nodes.filter { it.type == "model" }
        assertEquals(3, modelNodes.size)
        for (mNode in modelNodes) {
            assertEquals("head_center", mNode.anchor)
            assertEquals(0.0f, mNode.basePosition[0], 1e-4f)
            assertEquals(0.56f, mNode.basePosition[1], 1e-4f)
            assertEquals(0.18f, mNode.basePosition[2], 1e-4f)
            assertEquals(1.0f, mNode.baseScale[0], 1e-4f)
        }

        // Verify Tier A/B/C LOD visibility overrides
        val lodAIdx = compiled.nodes.indexOfFirst { it.id == "heart_crown_lod_a" }
        val lodBIdx = compiled.nodes.indexOfFirst { it.id == "heart_crown_lod_b" }
        val lodCIdx = compiled.nodes.indexOfFirst { it.id == "heart_crown_lod_c" }
        assertTrue(compiled.tierOverrides["A"]!!.showNodes.contains(lodAIdx))
        assertTrue(compiled.tierOverrides["A"]!!.hideNodes.containsAll(listOf(lodBIdx, lodCIdx)))
        assertTrue(compiled.tierOverrides["B"]!!.showNodes.contains(lodBIdx))
        assertTrue(compiled.tierOverrides["B"]!!.hideNodes.containsAll(listOf(lodAIdx, lodCIdx)))
        assertTrue(compiled.tierOverrides["C"]!!.showNodes.contains(lodCIdx))
        assertTrue(compiled.tierOverrides["C"]!!.hideNodes.containsAll(listOf(lodAIdx, lodBIdx)))

        // Verify gentleFloat map behavior ([-0.006, 0.006] FW on positionY) and spring inertia (maxAngleDeg=2.2)
        val floatMap = compiled.behaviorRunner.maps.first { it.id == "heart_crown_gentle_float" }
        assertEquals(-0.006f, floatMap.outMin, 1e-4f)
        assertEquals(0.006f, floatMap.outMax, 1e-4f)
        val spring = compiled.behaviorRunner.springs.first()
        assertEquals(2.2f, spring.maxAngleDeg, 1e-4f)

        // Verify lifecycle policy (holdMs=160, fadeMs=220)
        assertEquals(160L, compiled.lifecyclePolicy.holdMs)
        assertEquals(220L, compiled.lifecyclePolicy.fadeMs)
    }

    @Test
    fun testFurryGlassesFilterPackageValidationAndCompilation() {
        val furryDir = resolveFilterDir("furry_glasses_filter")
        val manifestFile = File(furryDir, "manifest.json")
        assertTrue("Furry Glasses manifest must exist", manifestFile.exists())

        val json = JSONObject(manifestFile.readText())
        val validation = PackageValidator.validateManifest(json)
        assertTrue("furry_glasses_filter manifest must be valid: ${validation.errorReason}", validation.isValid)

        val assetsArr = json.getJSONArray("assets")
        assertEquals(5, assetsArr.length())
        for (i in 0 until assetsArr.length()) {
            val a = assetsArr.getJSONObject(i)
            val assetFile = File(furryDir, a.getString("path"))
            val assetVal = PackageValidator.validateAssetFile(
                assetFile,
                a.getString("sha256"),
                a.getLong("bytes")
            )
            assertTrue("Asset ${a.getString("path")} must pass hash/byte/structure validation: ${assetVal.errorReason}", assetVal.isValid)
        }

        val compiled = Compiler.compile(json, furryDir)
        assertEquals("furry_glasses_filter", compiled.packageId)
        assertEquals("Furry Glasses", compiled.packageName)

        // Verify all 3 occluders: headProxyEllipsoid (radii [0.58, 0.78, 0.36], centerOffset [0, -0.12, 0]),
        // faceMesh (pushBackFW = 0.012), and earTempleProxy (box occluder)
        val ellipsoidOcc = compiled.nodes.first { it.type == "occluder" && it.occluderShape == "ellipsoid" }
        assertEquals("eyes_center", ellipsoidOcc.anchor)
        assertEquals(0.58f, ellipsoidOcc.radiiFW!![0], 1e-4f)
        assertEquals(0.78f, ellipsoidOcc.radiiFW!![1], 1e-4f)
        assertEquals(0.36f, ellipsoidOcc.radiiFW!![2], 1e-4f)
        assertEquals(0.0f, ellipsoidOcc.basePosition[0], 1e-4f)
        assertEquals(-0.12f, ellipsoidOcc.basePosition[1], 1e-4f)
        assertEquals(0.0f, ellipsoidOcc.basePosition[2], 1e-4f)

        val faceMeshOcc = compiled.nodes.first { it.type == "occluder" && it.occluderShape == "faceMesh" }
        assertEquals(0.012f, faceMeshOcc.pushBackFW, 1e-4f)

        val templeBoxOcc = compiled.nodes.first { it.type == "occluder" && it.occluderShape == "box" }
        assertEquals("eyes_center", templeBoxOcc.anchor)
        assertTrue("Temple box width must stay medial to temple arms (< 0.86 FW)", templeBoxOcc.radiiFW!![0] in 0.80f..0.86f)
        assertTrue("Temple box depth must cover rear temple span", templeBoxOcc.radiiFW!![2] >= 0.36f)

        // Verify 3 LOD model nodes anchored to eyes_center at offsetFW [0.0, 0.0, 0.39] with scale 1.0 (fit.to == ipd)
        val modelNodes = compiled.nodes.filter { it.type == "model" }
        assertEquals(3, modelNodes.size)
        for (mNode in modelNodes) {
            assertEquals("eyes_center", mNode.anchor)
            assertEquals("full", mNode.rotationFollow)
            assertEquals(0.0f, mNode.basePosition[0], 1e-4f)
            assertEquals(0.0f, mNode.basePosition[1], 1e-4f)
            assertEquals(0.39f, mNode.basePosition[2], 1e-4f)
            assertEquals(1.0f, mNode.baseScale[0], 1e-4f)
        }

        // Verify Tier A/B/C LOD visibility overrides
        val lodAIdx = compiled.nodes.indexOfFirst { it.id == "furry_glasses_lod_a" }
        val lodBIdx = compiled.nodes.indexOfFirst { it.id == "furry_glasses_lod_b" }
        val lodCIdx = compiled.nodes.indexOfFirst { it.id == "furry_glasses_lod_c" }
        assertTrue(compiled.tierOverrides["A"]!!.showNodes.contains(lodAIdx))
        assertTrue(compiled.tierOverrides["A"]!!.hideNodes.containsAll(listOf(lodBIdx, lodCIdx)))
        assertTrue(compiled.tierOverrides["B"]!!.showNodes.contains(lodBIdx))
        assertTrue(compiled.tierOverrides["B"]!!.hideNodes.containsAll(listOf(lodAIdx, lodCIdx)))
        assertTrue(compiled.tierOverrides["C"]!!.showNodes.contains(lodCIdx))
        assertTrue(compiled.tierOverrides["C"]!!.hideNodes.containsAll(listOf(lodAIdx, lodBIdx)))

        // Static rigid eyewear must have zero expression/float behaviors
        assertTrue(compiled.behaviorRunner.maps.isEmpty())
        assertTrue(compiled.behaviorRunner.springs.isEmpty())
        assertTrue(compiled.behaviorRunner.clips.isEmpty())

        // Verify lifecycle policy (holdMs=150, fadeMs=220)
        assertEquals(150L, compiled.lifecyclePolicy.holdMs)
        assertEquals(220L, compiled.lifecyclePolicy.fadeMs)
    }

    @Test
    fun testFurryGlassesEyeAlignmentAndYawRotationStability() {
        val furryDir = resolveFilterDir("furry_glasses_filter")
        val json = JSONObject(File(furryDir, "manifest.json").readText())
        val compiled = Compiler.compile(json, furryDir)
        val evaluator = SceneEvaluator(maxNodes = 16)

        // Evaluate across multiple face widths and yaw angles (-40, -30, 0, +30, +40 deg)
        for (fw in listOf(0.14f, 0.16f, 0.20f)) {
            for (yawDeg in listOf(-40f, -30f, 0f, 30f, 40f)) {
                val rad = Math.toRadians(yawDeg.toDouble())
                val c = kotlin.math.cos(rad).toFloat()
                val s = kotlin.math.sin(rad).toFloat()
                val poseMatrix = floatArrayOf(
                    c, 0f, -s, 0f,
                    0f, 1f, 0f, 0f,
                    s, 0f, c, 0f,
                    0f, 0f, -0.48f, 1f
                )
                val faceState = FaceState(
                    isVisible = true,
                    faceWidth = fw,
                    poseMatrix = poseMatrix,
                    landmarkPoseMatrix = poseMatrix,
                    rollDeg = 0f
                )

                evaluator.evaluate(compiled, faceState, compiled.propertySlots)

                val lodAIdx = compiled.nodes.indexOfFirst { it.id == "furry_glasses_lod_a" }
                val templeOccIdx = compiled.nodes.indexOfFirst { it.occluderShape == "box" }
                assertTrue("LOD A must be visible in Tier A", evaluator.isNodeVisible[lodAIdx])
                assertTrue("Temple box occluder must be visible", evaluator.isNodeVisible[templeOccIdx])

                val modelOff = lodAIdx * 16
                val templeOff = templeOccIdx * 16

                // Model world matrix must scale uniformly by fw and stay positioned in front of temple occluder at 0 deg
                val modelScaleX = kotlin.math.hypot(
                    evaluator.worldMatrices[modelOff].toDouble(),
                    evaluator.worldMatrices[modelOff + 2].toDouble()
                ).toFloat()
                assertEquals("Model scale must match faceWidth", fw, modelScaleX, 1e-4f)

                if (yawDeg == 0f) {
                    val modelWorldZ = evaluator.worldMatrices[modelOff + 14]
                    val templeWorldZ = evaluator.worldMatrices[templeOff + 14]
                    assertTrue(
                        "Front eyewear pivot Z ($modelWorldZ) must be closer to camera than medial temple occluder center ($templeWorldZ)",
                        modelWorldZ > templeWorldZ
                    )
                }
            }
        }
    }
}
