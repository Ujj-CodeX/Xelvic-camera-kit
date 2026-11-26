package com.example

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.example.engine.perception.ProviderRegistry
import com.example.engine.pkg.*
import com.example.engine.render.NodeRendererRegistry
import com.example.engine.scene.*
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class PackagePipelineStep3UnitTest {

    private lateinit var context: Context
    private lateinit var installer: PackageInstaller

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        installer = PackageInstaller(context)
    }

    @Test
    fun testStrictSchemaValidationAndFailClosed() {
        // Valid xfilter/3 manifest
        val validManifest = JSONObject().apply {
            put("schema", "xfilter/3")
            put("id", "test_filter")
            put("name", "Test Filter")
            put("version", "1.0.0")
            put("nodes", JSONArray())
        }
        val res1 = PackageValidator.validateManifest(validManifest)
        assertTrue("Valid manifest must pass", res1.isValid)

        // Unknown top-level field must fail closed in xfilter/3
        val invalidManifest = JSONObject(validManifest.toString()).apply {
            put("arbitrary_unknown_field", "malicious_payload")
        }
        val res2 = PackageValidator.validateManifest(invalidManifest)
        assertFalse("Manifest with unknown fields must fail closed", res2.isValid)
        assertTrue(res2.errorReason?.contains("arbitrary_unknown_field") == true)
    }

    @Test
    fun testAssetHashesByteCountsAndSignatures() {
        val tempFile = File.createTempFile("test_asset", ".png")
        tempFile.deleteOnExit()

        // Valid PNG header: 89 50 4E 47 0D 0A 1A 0A
        val validPng = byteArrayOf(
            0x89.toByte(), 0x50.toByte(), 0x4E.toByte(), 0x47.toByte(),
            0x0D.toByte(), 0x0A.toByte(), 0x1A.toByte(), 0x0A.toByte(),
            0x00, 0x00, 0x00, 0x00
        )
        tempFile.writeBytes(validPng)

        val actualSha = PackageValidator.sha256(tempFile)
        val validRes = PackageValidator.validateAssetFile(tempFile, actualSha, tempFile.length())
        assertTrue("Valid asset signature and hash must pass", validRes.isValid)

        // Hash mismatch
        val badHashRes = PackageValidator.validateAssetFile(tempFile, "bad_sha_12345", tempFile.length())
        assertFalse("Hash mismatch must fail", badHashRes.isValid)

        // Corrupted signature
        tempFile.writeBytes(byteArrayOf(0x00, 0x01, 0x02, 0x03, 0x04))
        val badSigRes = PackageValidator.validateAssetFile(tempFile, null, null)
        assertFalse("Corrupted signature must fail", badSigRes.isValid)
    }

    @Test
    fun testCapabilityNegotiationAndFallbacks() {
        // Supported capabilities
        val supported = listOf("node.model@1", "material.lambert@1", "behavior.spring@1")
        val (ok, missing) = CapabilityRegistry.negotiate(supported)
        assertTrue("All registered features must be compatible", ok)
        assertTrue(missing.isEmpty())

        // Unsupported future capability
        val unsupported = listOf("node.model@1", "behavior.quantum_teleport@99")
        val (ok2, missing2) = CapabilityRegistry.negotiate(unsupported)
        assertFalse("Unsupported features must not be compatible", ok2)
        assertEquals(1, missing2.size)
        assertEquals("behavior.quantum_teleport@99", missing2[0])

        // Fallback negotiation
        val optional = listOf("node.post@1", "experimental.feature@1")
        val optResults = CapabilityRegistry.negotiateOptional(optional)
        assertTrue(optResults["node.post@1"] == true)
        assertTrue(optResults["experimental.feature@1"] == false)
    }

    @Test
    fun testParametersAnchorsMaterialsAndNodeReferences() {
        val manifest = JSONObject().apply {
            put("schema", "xfilter/3")
            put("id", "param_test")
            put("name", "Param Test")

            val params = JSONObject().apply {
                val p1 = JSONObject().apply {
                    put("type", "vec3")
                    val def = JSONArray(); def.put(0.0); def.put(0.5); def.put(-0.1)
                    put("default", def)
                }
                put("myOffset", p1)
            }
            put("params", params)

            val anchors = JSONObject().apply {
                val custom = JSONObject().apply {
                    val lms = JSONArray().apply {
                        val lm = JSONObject().apply {
                            put("id", 105)
                            put("w", 1.0)
                        }
                        put(lm)
                    }
                    put("landmarks", lms)
                }
                put("brow_custom", custom)
            }
            put("anchors", anchors)

            val nodes = JSONArray().apply {
                val n0 = JSONObject().apply {
                    put("id", "root_node")
                    put("type", "group")
                    put("anchor", "brow_custom")
                    val transform = JSONObject().apply {
                        put("position", "\$myOffset")
                    }
                    put("transform", transform)
                }
                put(n0)
            }
            put("nodes", nodes)
        }

        val compiled = Compiler.compile(manifest, File(context.filesDir, "test_dir"))
        assertNotNull(compiled)
        assertEquals(1, compiled.nodes.size)
        val node0 = compiled.nodes[0]
        assertEquals("brow_custom", node0.anchor)
        assertEquals(0f, node0.basePosition[0], 0.001f)
        assertEquals(0.5f, node0.basePosition[1], 0.001f)
        assertEquals(-0.1f, node0.basePosition[2], 0.001f)
    }

    @Test
    fun testGlbStructureAndUvValidation() {
        // Pinned UV layout contract
        assertEquals(468, UvRegistry.VERTEX_COUNT)
        assertEquals(468 * 2, UvRegistry.CANONICAL_UVS.size)
        assertTrue(UvRegistry.CANONICAL_TRIANGLES.isNotEmpty())
        assertNotNull(UvRegistry.UV_LAYOUT_SHA256)

        // Test GLB header validation
        val glbFile = File("app/src/main/assets/filters/sunflower_forehead/assets/sunflower_forehead.glb")
        if (glbFile.exists()) {
            val glbRes = PackageValidator.validateGlbStructure(glbFile)
            assertTrue("Standalone GLB must pass structural validation: ${glbRes.errorReason}", glbRes.isValid)
        }
    }

    @Test
    fun testExpressionGrammarVmCyclesAndStackLimits() {
        val vm = ExpressionVM()
        val slots = floatArrayOf(0.5f, 2.0f, 4.0f) // slot 0, slot 1, slot 2

        // 1. Basic operators and allow-listed functions
        val prog1 = vm.compile("clamp(a + 0.5, 0.0, 1.0)") { 0 }
        val res1 = vm.execute(prog1, slots)
        assertEquals(1.0f, res1, 0.001f)

        // 2. Unary minus
        val prog2 = vm.compile("-a * 2") { 0 }
        val res2 = vm.execute(prog2, slots)
        assertEquals(-1.0f, res2, 0.001f)

        // 3. pow and sqrt
        val prog3 = vm.compile("pow(sqrt(c), b)") { if (it == "c") 2 else 1 }
        val res3 = vm.execute(prog3, slots)
        assertEquals(4.0f, res3, 0.001f) // sqrt(4)^2 = 4

        // 4. NaN / Infinity safety
        val progDivZero = vm.compile("a / 0") { 0 }
        val resDivZero = vm.execute(progDivZero, slots)
        assertEquals(0f, resDivZero, 0.0001f)

        // 5. Cycle detection: A -> B and B -> A must throw exception
        val cyclicSignals = mapOf(
            "signalA" to "signalB + 1",
            "signalB" to "signalA * 2"
        )
        try {
            vm.getTopologicalOrder(cyclicSignals)
            fail("Cyclic dependency must throw exception")
        } catch (e: IllegalStateException) {
            assertTrue(e.message?.contains("Cycle detected") == true)
        }

        // 6. Acyclic dependency order: C depends on B, B depends on A
        val acyclicSignals = mapOf(
            "sigC" to "sigB + 1",
            "sigB" to "sigA * 2",
            "sigA" to "1.0"
        )
        val order = vm.getTopologicalOrder(acyclicSignals)
        assertEquals(3, order.size)
        assertTrue(order.indexOf("sigA") < order.indexOf("sigB"))
        assertTrue(order.indexOf("sigB") < order.indexOf("sigC"))
    }

    @Test
    fun testSelectorsAndPropertyPaths() {
        val manifest = JSONObject().apply {
            put("schema", "xfilter/3")
            put("id", "selector_test")
            put("name", "Selector Test")

            val nodes = JSONArray().apply {
                val n0 = JSONObject().apply {
                    put("id", "petals_node")
                    put("type", "model")
                    val tags = JSONArray(); tags.put("flower"); tags.put("petals")
                    put("tags", tags)
                }
                put(n0)
            }
            put("nodes", nodes)

            val behaviors = JSONArray().apply {
                val map0 = JSONObject().apply {
                    put("id", "scale_map")
                    put("type", "map")
                    put("input", "face.present")
                    val inRange = JSONArray(); inRange.put(0.0); inRange.put(1.0)
                    put("inputRange", inRange)
                    val outRange = JSONArray(); outRange.put(0.0); outRange.put(1.0)
                    put("outputRange", outRange)
                    val writes = JSONArray().apply {
                        val w0 = JSONObject().apply {
                            put("path", "tag:petals.scale")
                            put("mode", "multiply")
                        }
                        put(w0)
                    }
                    put("writes", writes)
                }
                put(map0)
            }
            put("behaviors", behaviors)
        }

        val compiled = Compiler.compile(manifest, File(context.filesDir, "test_dir"))
        assertNotNull(compiled)
        assertEquals(1, compiled.behaviorRunner.maps.size)
        val compiledMap = compiled.behaviorRunner.maps[0]
        assertEquals("scale_map", compiledMap.id)
        assertTrue("Property write must target resolved node index 0", compiledMap.writes.any { it.nodeIndex == 0 })
    }

    @Test
    fun testAllFourBehaviorDefinitions() {
        val manifest = JSONObject().apply {
            put("schema", "xfilter/3")
            put("id", "all_behaviors")
            put("name", "All Behaviors")

            val nodes = JSONArray().apply {
                val n0 = JSONObject().apply {
                    put("id", "model_node")
                    put("type", "model")
                }
                put(n0)
            }
            put("nodes", nodes)

            val behaviors = JSONArray().apply {
                // 1. Clip
                put(JSONObject().apply {
                    put("id", "anim_clip")
                    put("type", "clip")
                    put("target", "node:model_node")
                    put("animation", "idle")
                    put("loop", true)
                })
                // 2. Map
                put(JSONObject().apply {
                    put("id", "smile_map")
                    put("type", "map")
                    put("input", "face.present")
                    val inR = JSONArray(); inR.put(0.0); inR.put(1.0); put("inputRange", inR)
                    val outR = JSONArray(); outR.put(1.0); outR.put(2.0); put("outputRange", outR)
                })
                // 3. Trigger
                put(JSONObject().apply {
                    put("id", "jaw_trigger")
                    put("type", "trigger")
                    put("input", "face.blend.jawOpen")
                    put("risingAbove", 0.5)
                    put("fallingBelow", 0.3)
                    val onFire = JSONArray().apply {
                        put(JSONObject().apply {
                            put("action", "playClip")
                            put("clip", "anim_clip")
                        })
                    }
                    put("onFire", onFire)
                })
                // 4. Spring
                put(JSONObject().apply {
                    put("id", "inertia_spring")
                    put("type", "spring")
                    put("target", "node:model_node")
                    put("mode", "rotationInertia")
                    put("stiffness", 90.0)
                    put("dampingRatio", 0.3)
                })
            }
            put("behaviors", behaviors)
        }

        val compiled = Compiler.compile(manifest, File(context.filesDir, "test_dir"))
        assertNotNull(compiled)
        assertEquals(1, compiled.behaviorRunner.clips.size)
        assertEquals(1, compiled.behaviorRunner.maps.size)
        assertEquals(1, compiled.behaviorRunner.triggers.size)
        assertEquals(1, compiled.behaviorRunner.springs.size)
        assertTrue("Requires blendshapes must be true due to jawOpen trigger", compiled.requiresBlendshapes)
        assertTrue(compiled.requestedBlendshapes.contains("jawOpen"))
    }

    @Test
    fun testLegacyAdaptersV1V2AndFace3dToSameCompiledEffect() {
        val testDir = File(context.filesDir, "test_dir")

        // 1. Schema v1
        val v1 = JSONObject().apply {
            put("schema", "v1")
            put("id", "v1_prop")
            put("name", "V1 Sticker")
            val layers = JSONArray().apply {
                put(JSONObject().apply {
                    put("asset", "sticker.png")
                    put("anchor", "FOREHEAD")
                    put("scale", 0.6)
                })
            }
            put("layers", layers)
        }
        val compiledV1 = Compiler.compile(v1, testDir)
        assertNotNull(compiledV1)
        assertEquals("v1_prop", compiledV1.packageId)
        assertEquals("sprite", compiledV1.nodes[0].type)
        assertEquals("forehead", compiledV1.nodes[0].anchor)

        // 2. Schema v2
        val v2 = JSONObject().apply {
            put("schema", "v2")
            put("id", "v2_paint")
            put("name", "V2 Face Paint")
            put("type", "FACE_MESH_TEXTURE")
            put("texture", "paint.png")
        }
        val compiledV2 = Compiler.compile(v2, testDir)
        assertNotNull(compiledV2)
        assertEquals("v2_paint", compiledV2.packageId)
        assertEquals("faceMesh", compiledV2.nodes[0].type)

        // 3. xfilter-face-3d/1 (Sunflower)
        val face3dManifestFile = File("app/src/main/assets/filters/sunflower_forehead/manifest.json")
        if (face3dManifestFile.exists()) {
            val face3dJson = JSONObject(face3dManifestFile.readText())
            val compiledFace3d = Compiler.compile(face3dJson, face3dManifestFile.parentFile!!)
            assertNotNull(compiledFace3d)
            assertEquals("sunflower_forehead", compiledFace3d.packageId)
            assertTrue("Requires pose must be true", compiledFace3d.requiresPose)
            assertTrue("Requires blendshapes must be true", compiledFace3d.requiresBlendshapes)
            assertTrue("Should request jawOpen blendshape", compiledFace3d.requestedBlendshapes.contains("jawOpen"))
            assertTrue("Should request mouthSmileLeft blendshape", compiledFace3d.requestedBlendshapes.contains("mouthSmileLeft"))
            assertTrue("Should request mouthSmileRight blendshape", compiledFace3d.requestedBlendshapes.contains("mouthSmileRight"))
            assertEquals(150L, compiledFace3d.lifecyclePolicy.holdMs)
            assertEquals(200L, compiledFace3d.lifecyclePolicy.fadeMs)
        }
    }

    @Test
    fun testTransactionalInstallationRollbackAndRevalidation() {
        // Create an invalid zip (missing manifest)
        val badZipStream = ByteArrayOutputStream().apply {
            ZipOutputStream(this).use { zos ->
                zos.putNextEntry(ZipEntry("arbitrary_file.txt"))
                zos.write("hello".toByteArray())
                zos.closeEntry()
            }
        }.toByteArray()

        val installRes = installer.installFromStream(badZipStream.inputStream(), "bad_package")
        assertTrue("Installation must fail for missing manifest", installRes is PackageInstaller.InstallResult.Failure)

        val targetDir = File(context.filesDir, "filters/bad_package")
        assertFalse("Target directory must not be created on failed installation", targetDir.exists())

        // Revalidation of non-existent directory must fail closed
        val reval = installer.revalidatePersistedPackage(targetDir)
        assertFalse("Non-existent package revalidation must fail", reval.isValid)
    }
}
