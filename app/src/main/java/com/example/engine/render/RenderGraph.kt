package com.example.engine.render

import android.opengl.GLES30
import android.opengl.Matrix
import android.util.Log
import com.example.engine.governor.PerformanceGovernor
import com.example.engine.governor.TierPolicy
import com.example.engine.scene.*
import com.example.engine.signal.FacePoseAdapter
import com.example.engine.signal.FaceState
import java.io.File

/**
 * RenderGraph coordinates the data-driven PassGraph architecture (Flow 1 — Chunk B0 & Flow 2 — Chunk G0):
 * Default 9-stage canonical PassGraph:
 * 1. Camera frame (`CAMERA_FRAME`, no depth)
 * 2. Camera-pixel passes (`CAMERA_PIXEL_PASSES`: `faceMesh` paint, `eyeColor`) — before 3D occluders/models
 * 3. Depth clear (`CLEAR_DEPTH`)
 * 4. Occluder nodes (`OCCLUDERS`: depth-only `faceMesh`, `ellipsoid`, `box`)
 * 5. Opaque nodes (`OPAQUE_NODES`: depth test/write enabled)
 * 6. Transparent nodes (`TRANSPARENT_NODES`: alpha lenses & props sorted by renderOrder, then depth)
 * 7. Post nodes (`POST_NODES`: sorted by renderOrder)
 * 8. Screen-space overlay nodes (`OVERLAY_NODES`: sorted by renderOrder)
 * 9. Final output (`FINAL_OUTPUT`: present to Surface / Photo FBO)
 *
 * Preallocates scratch buffers for zero frame-loop allocations and integrates with
 * GlResourceOwner, RtPool, PassGraph, TextureCache, TierPolicy, and PerformanceGovernor.
 */
class RenderGraph {

    companion object {
        private const val TAG = "RenderGraph"
        private const val MAX_SCENE_NODES = 128
    }

    val resourceOwner = GlResourceOwner()
    val rtPool = RtPool()
    val materials = MaterialRegistry()
    val textures = TextureCache(resourceOwner = resourceOwner)
    val sprites = SpriteRenderer()
    val faceMesh = FaceMeshRenderer()
    val regionMask = RegionMaskPass()
    val recolor = RecolorPass()
    val occluder = OccluderPass()
    val glbLoader = GlbLoader()
    val modelRenderer = ModelRenderer()
    val post = PostRenderer()

    private val loadedGlbModels = HashMap<String, GlbLoader.GlbModel>()
    private val loadedGlbGenerations = HashMap<String, Int>()
    private val viewProjectionMatrix = FloatArray(16)
    private val nodeMvp = FloatArray(16)
    private val scratchNodeMat = FloatArray(16)

    @Volatile
    var areGlResourcesAllocated: Boolean = false
        private set

    @Volatile
    var allocatedContextGeneration: Int = -1
        private set

    enum class RenderPass {
        CAMERA_FRAME,
        CAMERA_PIXEL_PASSES,
        CLEAR_DEPTH,
        OCCLUDERS,
        OPAQUE_NODES,
        TRANSPARENT_NODES,
        POST_NODES,
        OVERLAY_NODES,
        FINAL_OUTPUT
    }

    // Pre-allocated node index arrays for zero-allocation pass sorting
    private val cameraPixelIndices = IntArray(MAX_SCENE_NODES)
    private val opaqueIndices = IntArray(MAX_SCENE_NODES)
    private val transparentIndices = IntArray(MAX_SCENE_NODES)
    private val postIndices = IntArray(MAX_SCENE_NODES)
    private val overlayIndices = IntArray(MAX_SCENE_NODES)

    private val passOrderScratch = ArrayList<RenderPass>(12)
    val lastPassOrder: List<RenderPass>
        get() = passOrderScratch

    private val executedPassNodeIdsScratch = ArrayList<String>(16)
    val lastExecutedPassNodeIds: List<String>
        get() = executedPassNodeIdsScratch

    private val skippedPassNodeIdsScratch = ArrayList<String>(8)
    val lastSkippedPassNodeIds: List<String>
        get() = skippedPassNodeIdsScratch

    private val executedPassRecordsScratch = ArrayList<PassExecutionRecord>(16)
    val lastExecutedPassRecords: List<PassExecutionRecord>
        get() = executedPassRecordsScratch

    private val renderedNodeIdsScratch = ArrayList<String>(MAX_SCENE_NODES)
    val lastRenderedNodeIds: List<String>
        get() = renderedNodeIdsScratch

    var lastFrameContext: DeterministicFrameContext = DeterministicFrameContext.DEFAULT
        private set

    var lastRenderedCameraPixelCount: Int = 0
        private set
    var lastRenderedOpaqueCount: Int = 0
        private set
    var lastRenderedTransparentCount: Int = 0
        private set
    var lastRenderedOccluderCount: Int = 0
        private set
    var lastRenderedPostCount: Int = 0
        private set
    var lastRenderedOverlayCount: Int = 0
        private set

    var viewportWidth: Int = 1080
        private set
    var viewportHeight: Int = 1920
        private set
    var lastUsedFovYViewDeg: Float = 60.0f
        private set

    fun init() {
        resourceOwner.bindGlThread()
        val currentGen = resourceOwner.contextGeneration
        if (areGlResourcesAllocated && allocatedContextGeneration == currentGen) {
            return
        }
        materials.init()
        resourceOwner.trackProgram(materials.programUnlit)
        resourceOwner.trackProgram(materials.programLambert)
        resourceOwner.trackProgram(materials.programPbr)
        resourceOwner.trackProgram(materials.programFaceTexture)
        resourceOwner.trackProgram(materials.programOccluder)
        sprites.init(resourceOwner)
        faceMesh.init(resourceOwner)
        regionMask.init(resourceOwner)
        recolor.init(resourceOwner)
        occluder.init(resourceOwner)
        post.init(resourceOwner)
        warmShaders()
        areGlResourcesAllocated = true
        allocatedContextGeneration = currentGen
        Log.i(TAG, "RenderGraph initialized GLES 3.0 multi-pass pipeline (generation=$currentGen)")
    }

    fun resize(width: Int, height: Int) {
        viewportWidth = width.coerceAtLeast(1)
        viewportHeight = height.coerceAtLeast(1)
        GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
    }

    /**
     * Restores all GPU resources (shader programs, static VBOs/VAOs, cached textures, and
     * active effect GLB models/textures) on the GL thread after an EGL/OpenGL context loss
     * or surface recreation (README Section 11.4, 14 & 21 Step 7).
     */
    fun onContextRecreated(activeEffect: CompiledEffect? = null, tierMaxTexSize: Int = 1024) {
        resourceOwner.bindGlThread()
        // 1. Discard dead-context handles without issuing invalid glDelete* calls on the new context
        rtPool.releaseAll(null)
        resourceOwner.onContextLost()
        val newGen = resourceOwner.contextGeneration
        loadedGlbModels.clear()
        loadedGlbGenerations.clear()

        // 2. Re-compile and link shader programs and static renderer buffers on the new context
        materials.init()
        resourceOwner.trackProgram(materials.programUnlit)
        resourceOwner.trackProgram(materials.programLambert)
        resourceOwner.trackProgram(materials.programPbr)
        resourceOwner.trackProgram(materials.programFaceTexture)
        resourceOwner.trackProgram(materials.programOccluder)
        sprites.init(resourceOwner)
        faceMesh.init(resourceOwner)
        regionMask.init(resourceOwner)
        recolor.init(resourceOwner)
        occluder.init(resourceOwner)
        post.init(resourceOwner)
        warmShaders()

        // 3. Re-upload cached textures from disk / staged entries
        textures.onContextLostAndRecreate(reuploadImmediately = true)

        // 4. Restore active effect GLB models and textures so rendering resumes immediately
        if (activeEffect != null) {
            restoreEffectResourcesOnGlThread(activeEffect, tierMaxTexSize)
        }
        areGlResourcesAllocated = true
        allocatedContextGeneration = newGen
        Log.i(TAG, "RenderGraph restored all GPU resources after context recreation (generation=$newGen)")
    }

    /**
     * Ensures all textures and GLB models referenced by `effect` are loaded on the current GL context.
     */
    fun restoreEffectResourcesOnGlThread(effect: CompiledEffect, tierMaxTexSize: Int = 1024) {
        resourceOwner.bindGlThread()
        for (asset in effect.assets.values) {
            val file = File(asset.localPath)
            when (asset.type) {
                "texture", "mask", "image", "lut" -> {
                    if (file.exists()) {
                        val isLut = PostRenderer.isAssetUsedAsLut(effect, asset.id)
                        textures.getOrLoad(asset.localPath, tierMaxTexSize, isLut = isLut)
                    }
                }
                "model", "glb" -> {
                    preloadModelOnGlThread(asset.id, file)
                }
            }
        }
    }

    private fun purgeStaleGlbModels() {
        val currentGen = resourceOwner.contextGeneration
        val iter = loadedGlbGenerations.entries.iterator()
        while (iter.hasNext()) {
            val entry = iter.next()
            if (entry.value != currentGen) {
                loadedGlbModels.remove(entry.key)
                iter.remove()
            }
        }
    }

    fun getLoadedModelCount(): Int {
        purgeStaleGlbModels()
        return loadedGlbModels.size
    }

    fun isModelLoaded(assetId: String): Boolean {
        purgeStaleGlbModels()
        return loadedGlbModels.containsKey(assetId) &&
            loadedGlbGenerations[assetId] == resourceOwner.contextGeneration
    }

    /**
     * Warms all engine shader programs on the GL thread so a filter's first frame never hitches.
     */
    fun warmShaders() {
        if (materials.programUnlit != 0) GLES30.glUseProgram(materials.programUnlit)
        if (materials.programLambert != 0) GLES30.glUseProgram(materials.programLambert)
        if (materials.programPbr != 0) GLES30.glUseProgram(materials.programPbr)
        if (materials.programFaceTexture != 0) GLES30.glUseProgram(materials.programFaceTexture)
        if (materials.programOccluder != 0) GLES30.glUseProgram(materials.programOccluder)
        recolor.warmShaders()
        post.warmShaders()
        GLES30.glUseProgram(0)
    }

    /**
     * Preloads a GLB model on the GL thread as part of distributed PackagePreloader uploads.
     * Any model loaded under an older contextGeneration is treated as invalid and rebuilt on demand.
     */
    fun preloadModelOnGlThread(assetId: String, file: File) {
        purgeStaleGlbModels()
        val currentGen = resourceOwner.contextGeneration
        if (loadedGlbModels.containsKey(assetId) && loadedGlbGenerations[assetId] == currentGen) return
        if (file.exists()) {
            val loaded = glbLoader.load(file)
            if (loaded != null) {
                trackGlbModelResources(loaded)
                loadedGlbModels[assetId] = loaded
                loadedGlbGenerations[assetId] = currentGen
            }
        }
    }

    private fun trackGlbModelResources(model: GlbLoader.GlbModel) {
        for (m in model.meshes) {
            resourceOwner.trackVao(m.vao)
            resourceOwner.trackBuffer(m.vbo)
            resourceOwner.trackBuffer(m.ibo)
        }
        for (t in model.textures) {
            resourceOwner.trackTexture(t)
        }
    }

    fun release() {
        resourceOwner.bindGlThread()
        if (!areGlResourcesAllocated && loadedGlbModels.isEmpty() && textures.size() == 0 && rtPool.totalAllocatedTargets == 0) {
            return
        }
        rtPool.releaseAll(resourceOwner)
        materials.release()
        textures.releaseAll()
        sprites.release(resourceOwner)
        faceMesh.release(resourceOwner)
        regionMask.release(resourceOwner)
        recolor.release(resourceOwner)
        occluder.release()
        post.release(resourceOwner)
        for (m in loadedGlbModels.values) m.release()
        loadedGlbModels.clear()
        loadedGlbGenerations.clear()
        resourceOwner.releaseAll()
        areGlResourcesAllocated = false
        allocatedContextGeneration = -1
    }

    /**
     * Executes `effect.passGraph` (which defaults to the 9-stage canonical PassGraph when no custom
     * `"passes"` array is declared in the manifest):
     * 1. Camera frame (`CAMERA_FRAME`)
     * 2. Camera-pixel passes (`CAMERA_PIXEL_PASSES`: `faceMesh` paint, `eyeColor`) — before 3D occluders/models
     * 3. Clear depth (`CLEAR_DEPTH`)
     * 4. Occluders (`OCCLUDERS`: depth-only `faceMesh`, `ellipsoid`, `box`)
     * 5. Opaque 3D/2D nodes (`OPAQUE_NODES`)
     * 6. Alpha lenses / transparent 3D/2D props (`TRANSPARENT_NODES`, sorted by renderOrder, then depth)
     * 7. Post-processing nodes (`POST_NODES`, sorted by renderOrder)
     * 8. Screen-space overlay nodes (`OVERLAY_NODES`, sorted by renderOrder)
     * 9. Final output (`FINAL_OUTPUT`)
     */
    @JvmOverloads
    fun render(
        effect: CompiledEffect,
        faceState: FaceState,
        evaluator: SceneEvaluator,
        viewportWidth: Int = this.viewportWidth,
        viewportHeight: Int = this.viewportHeight,
        governor: PerformanceGovernor? = null,
        frameContext: DeterministicFrameContext = DeterministicFrameContext.DEFAULT
    ) {
        resourceOwner.bindGlThread()
        this.viewportWidth = viewportWidth.coerceAtLeast(1)
        this.viewportHeight = viewportHeight.coerceAtLeast(1)
        this.lastFrameContext = frameContext
        // Drain pending deletions and step distributed texture uploads on the GL thread
        resourceOwner.drainPendingDeletions()
        textures.stepFrameUploads(maxUploadsPerFrame = 1)

        passOrderScratch.clear()
        executedPassNodeIdsScratch.clear()
        skippedPassNodeIdsScratch.clear()
        executedPassRecordsScratch.clear()
        renderedNodeIdsScratch.clear()
        lastRenderedCameraPixelCount = 0
        lastRenderedOccluderCount = 0
        lastRenderedOpaqueCount = 0
        lastRenderedTransparentCount = 0
        lastRenderedPostCount = 0
        lastRenderedOverlayCount = 0
        post.clearExecutedPasses()
        regionMask.clearEvaluatedMasks()
        recolor.clearExecutedRecords()

        val isFaceTracked = faceState.isVisible && faceState.fadeWeight >= 0.001f
        val nodeLimit = effect.nodes.size.coerceAtMost(MAX_SCENE_NODES)
        var hasVisibleGlobalPostOrOverlay = false
        if (!isFaceTracked) {
            for (i in 0 until nodeLimit) {
                val n = effect.nodes[i]
                if (i < evaluator.isNodeVisible.size &&
                    evaluator.isNodeVisible[i] &&
                    (n.type == "post" || n.type == "overlay")
                ) {
                    hasVisibleGlobalPostOrOverlay = true
                    break
                }
            }
        }
        if (!isFaceTracked && !hasVisibleGlobalPostOrOverlay) {
            return
        }

        GLES30.glViewport(0, 0, this.viewportWidth, this.viewportHeight)

        // Compute Camera Perspective Projection Matrix using VISIBLE-VIEW FOV after FILL_CENTER
        val aspect = this.viewportWidth.toFloat() / this.viewportHeight.toFloat().coerceAtLeast(1f)
        val fovYViewDeg = faceState.projectionParams.visibleViewFovYDeg(aspect)
        lastUsedFovYViewDeg = fovYViewDeg
        FacePoseAdapter.createCameraProjection(
            fovYDeg = fovYViewDeg,
            aspectRatio = aspect,
            near = faceState.projectionParams.near,
            far = faceState.projectionParams.far,
            frontFacing = faceState.isFrontCamera,
            out = viewProjectionMatrix
        )

        val nodes = effect.nodes
        val requestedTier = governor?.currentTier ?: when (effect.activeTier) {
            "B" -> PerformanceGovernor.Tier.B
            "C" -> PerformanceGovernor.Tier.C
            else -> PerformanceGovernor.Tier.A
        }
        val govMaxTex = governor?.maxTextureSize ?: 1024
        val effectiveTier = TierPolicy.resolveEffectiveTier(effect, requestedTier, govMaxTex)
        effect.activeTier = effectiveTier.tierId
        val tierOverride = effect.tierOverrides[effectiveTier.tierId]
        val tierTexMaxSize = effectiveTier.textureMaxSize
        val govLatSegments = governor?.occluderLatSegments ?: 16
        val nodeCount = nodes.size.coerceAtMost(MAX_SCENE_NODES)

        // Partition visible renderable nodes into camera-pixel, opaque, transparent (alpha lenses/props), post, and overlay arrays
        var cameraPixelCount = 0
        var opaqueCount = 0
        var transparentCount = 0
        var postCount = 0
        var overlayCount = 0
        for (i in 0 until nodeCount) {
            val node = nodes[i]
            if (!evaluator.isNodeVisible[i]) continue
            if (!TierPolicy.isNodeVisibleInTier(i, node.baseVisible, tierOverride)) continue

            when (node.type) {
                "occluder", "group" -> Unit
                "faceMesh", "eyeColor", "recolor" -> {
                    if (isFaceTracked) {
                        cameraPixelIndices[cameraPixelCount++] = i
                    }
                }
                "post" -> {
                    postIndices[postCount++] = i
                }
                "overlay" -> {
                    overlayIndices[overlayCount++] = i
                }
                else -> {
                    if (isFaceTracked) {
                        val mat = ModelRenderer.resolvePrimaryModelMaterial(node, effect.materials)
                        if (mat?.blend == "opaque") {
                            opaqueIndices[opaqueCount++] = i
                        } else {
                            transparentIndices[transparentCount++] = i
                        }
                    }
                }
            }
        }

        val passGraph = effect.passGraph
        val passNodes = passGraph.passes
        var canonicalCameraPixelNodesRendered = false
        for (pIdx in 0 until passNodes.size) {
            val passNode = passNodes[pIdx]
            if (!passGraph.isPassEnabledInTier(passNode, effectiveTier)) {
                skippedPassNodeIdsScratch.add(passNode.id)
                continue
            }

            val startNs = System.nanoTime()
            val effectiveScale = passGraph.resolveEffectiveScale(passNode, effectiveTier)
            val resolvedParams = passGraph.resolvePassParams(passNode, effect.propertySlots, effectiveTier)

            // Ensure any named render target ($rt:<name> or $mask:<id>) referenced by custom passes is leased from RtPool
            if (passGraph.isCustomManifestGraph) {
                for (readRef in passNode.parsedReads) {
                    if (readRef.kind == RenderTargetRef.TargetKind.MASK) {
                        val maskExpr = effect.maskDefs[readRef.identifier]
                            ?: RegionMaskPass.BUILTIN_MASKS[readRef.identifier]
                            ?: RegionMaskPass.MaskExpr.LandmarkLoop("face_oval")
                        regionMask.evaluateMaskToRtPool(
                            maskId = readRef.identifier,
                            expr = maskExpr,
                            faceState = faceState,
                            viewportWidth = this.viewportWidth,
                            viewportHeight = this.viewportHeight,
                            rtPool = rtPool,
                            propertySlots = effect.propertySlots,
                            scaleFactor = RtPool.RtScale.HALF.factor,
                            resourceOwner = resourceOwner
                        )
                    }
                }
                val writeRef = passNode.parsedWrites
                when (writeRef.kind) {
                    RenderTargetRef.TargetKind.NAMED_RT -> {
                        rtPool.acquireNamed(
                            name = writeRef.raw,
                            baseWidth = this.viewportWidth,
                            baseHeight = this.viewportHeight,
                            format = RtPool.RtFormat.RGBA8,
                            scaleFactor = effectiveScale,
                            withDepth = false,
                            resourceOwner = resourceOwner
                        )
                    }
                    RenderTargetRef.TargetKind.MASK -> {
                        val maskExpr = effect.maskDefs[writeRef.identifier]
                            ?: RegionMaskPass.BUILTIN_MASKS[writeRef.identifier]
                            ?: RegionMaskPass.MaskExpr.LandmarkLoop("face_oval")
                        regionMask.evaluateMaskToRtPool(
                            maskId = writeRef.identifier,
                            expr = maskExpr,
                            faceState = faceState,
                            viewportWidth = this.viewportWidth,
                            viewportHeight = this.viewportHeight,
                            rtPool = rtPool,
                            propertySlots = effect.propertySlots,
                            scaleFactor = effectiveScale,
                            resourceOwner = resourceOwner
                        )
                    }
                    else -> Unit
                }
            }

            when (passNode.stageKind) {
                PassStageKind.CAMERA_FRAME -> {
                    passOrderScratch.add(RenderPass.CAMERA_FRAME)
                }
                PassStageKind.CAMERA_PIXEL_PASSES -> {
                    passOrderScratch.add(RenderPass.CAMERA_PIXEL_PASSES)
                    var passRenderedCameraPixelCount = 0
                    val isRecolorPassTemplate = passNode.templateId == "recolor@1" ||
                        passNode.templateId == "pass.recolor@1" ||
                        passNode.templateId == "pixel.recolor@1" ||
                        passNode.templateId == "eyeColor@1"
                    val isStandaloneRecolorPass = passGraph.isCustomManifestGraph &&
                        isFaceTracked &&
                        isRecolorPassTemplate &&
                        nodes.none { it.id == passNode.id }
                    var matchedCameraPixelNodeIdx = -1
                    if (passGraph.isCustomManifestGraph && cameraPixelCount > 0) {
                        for (k in 0 until cameraPixelCount) {
                            val idx = cameraPixelIndices[k]
                            if (nodes[idx].id == passNode.id) {
                                matchedCameraPixelNodeIdx = idx
                                break
                            }
                        }
                    }

                    val shouldRenderCanonicalCameraPixelNodes = if (!passGraph.isCustomManifestGraph) {
                        !canonicalCameraPixelNodesRendered
                    } else if (matchedCameraPixelNodeIdx >= 0) {
                        false
                    } else if (!isStandaloneRecolorPass) {
                        !canonicalCameraPixelNodesRendered
                    } else {
                        // Standalone recolor pass in a custom manifest graph: only render unmatched canonical
                        // cameraPixelIndices once if no pass in passNodes explicitly targets their IDs.
                        !canonicalCameraPixelNodesRendered && cameraPixelCount > 0 &&
                            (0 until cameraPixelCount).any { k ->
                                val nId = nodes[cameraPixelIndices[k]].id
                                passNodes.none { it.id == nId }
                            }
                    }

                    if (matchedCameraPixelNodeIdx >= 0) {
                        regionMask.renderRegionsToHalfResFbo(
                            faceState = faceState,
                            viewportWidth = this.viewportWidth,
                            viewportHeight = this.viewportHeight,
                            resourceOwner = resourceOwner
                        )
                        GLES30.glViewport(0, 0, this.viewportWidth, this.viewportHeight)
                        renderNode(matchedCameraPixelNodeIdx, effect, faceState, evaluator, tierTexMaxSize)
                        renderedNodeIdsScratch.add(nodes[matchedCameraPixelNodeIdx].id)
                        passRenderedCameraPixelCount += 1
                    } else if (shouldRenderCanonicalCameraPixelNodes) {
                        if (cameraPixelCount > 0) {
                            regionMask.renderRegionsToHalfResFbo(
                                faceState = faceState,
                                viewportWidth = this.viewportWidth,
                                viewportHeight = this.viewportHeight,
                                resourceOwner = resourceOwner
                            )
                            GLES30.glViewport(0, 0, this.viewportWidth, this.viewportHeight)
                        }
                        for (a in 1 until cameraPixelCount) {
                            val keyIdx = cameraPixelIndices[a]
                            val keyOrder = nodes[keyIdx].renderOrder
                            var b = a - 1
                            while (b >= 0) {
                                val cmpIdx = cameraPixelIndices[b]
                                if (nodes[cmpIdx].renderOrder <= keyOrder) break
                                cameraPixelIndices[b + 1] = cmpIdx
                                b--
                            }
                            cameraPixelIndices[b + 1] = keyIdx
                        }
                        for (k in 0 until cameraPixelCount) {
                            val idx = cameraPixelIndices[k]
                            if (passGraph.isCustomManifestGraph && passNodes.any { it.id == nodes[idx].id && it.id != passNode.id }) {
                                continue
                            }
                            renderNode(idx, effect, faceState, evaluator, tierTexMaxSize)
                            renderedNodeIdsScratch.add(nodes[idx].id)
                            passRenderedCameraPixelCount += 1
                        }
                        canonicalCameraPixelNodesRendered = true
                    }

                    if (isStandaloneRecolorPass) {
                        val maskRefId = passNode.parsedReads.firstOrNull { it.kind == RenderTargetRef.TargetKind.MASK }?.identifier ?: "iris"
                        val cleanMaskId = maskRefId.removePrefix("\$mask:").trim().lowercase().replace('-', '_')
                        val maskExpr = effect.maskDefs[cleanMaskId]
                            ?: effect.maskDefs[maskRefId]
                            ?: RegionMaskPass.BUILTIN_MASKS[cleanMaskId]
                            ?: RegionMaskPass.BUILTIN_IRIS_MASK
                        val maskRt = regionMask.evaluateMaskToRtPool(
                            maskId = cleanMaskId,
                            expr = maskExpr,
                            faceState = faceState,
                            viewportWidth = this.viewportWidth,
                            viewportHeight = this.viewportHeight,
                            rtPool = rtPool,
                            propertySlots = effect.propertySlots,
                            scaleFactor = RtPool.RtScale.HALF.factor,
                            resourceOwner = resourceOwner
                        )
                        GLES30.glViewport(0, 0, this.viewportWidth, this.viewportHeight)
                        val synthNode = com.example.engine.scene.CompiledNode(
                            index = nodes.size + pIdx,
                            id = passNode.id,
                            parentIndex = -1,
                            tags = emptyList(),
                            type = "recolor",
                            anchor = null,
                            basePosition = floatArrayOf(0f, 0f, 0f),
                            baseRotation = floatArrayOf(0f, 0f, 0f, 1f),
                            baseScale = floatArrayOf(1f, 1f, 1f),
                            rotationFollow = "none",
                            materialId = null,
                            baseVisible = true,
                            baseOpacity = 1f,
                            renderOrder = pIdx,
                            faceMeshRegion = "\$mask:$cleanMaskId",
                            postEffect = "recolor@1",
                            postParams = resolvedParams
                        )
                        recolor.renderNode(
                            node = synthNode,
                            effect = effect,
                            faceState = faceState,
                            opacity = 1.0f,
                            viewportWidth = this.viewportWidth,
                            viewportHeight = this.viewportHeight,
                            maskTextureId = maskRt.texId
                        )
                        renderedNodeIdsScratch.add(passNode.id)
                        passRenderedCameraPixelCount += 1
                    }
                    lastRenderedCameraPixelCount += passRenderedCameraPixelCount
                }
                PassStageKind.CLEAR_DEPTH -> {
                    passOrderScratch.add(RenderPass.CLEAR_DEPTH)
                    GLES30.glDepthMask(true)
                    GLES30.glClear(GLES30.GL_DEPTH_BUFFER_BIT)
                }
                PassStageKind.OCCLUDERS -> {
                    passOrderScratch.add(RenderPass.OCCLUDERS)
                    occluder.beginPass(frontFacing = faceState.isFrontCamera)
                    var occluderCount = 0
                    for (i in 0 until nodeCount) {
                        val node = nodes[i]
                        if (node.type == "occluder" && evaluator.isNodeVisible[i]) {
                            if (!TierPolicy.isNodeVisibleInTier(i, node.baseVisible, tierOverride)) continue

                            val worldOffset = i * 16
                            Matrix.multiplyMM(nodeMvp, 0, viewProjectionMatrix, 0, evaluator.worldMatrices, worldOffset)

                            val nodeLatSegs = node.occluderSegments?.getOrNull(0) ?: 16
                            val effectiveLatSegs = minOf(nodeLatSegs, govLatSegments)

                            when (node.occluderShape) {
                                "ellipsoid" -> occluder.renderEllipsoid(nodeMvp, materials.programOccluder, effectiveLatSegs)
                                "faceMesh" -> occluder.renderFaceMesh(
                                    faceState = faceState,
                                    viewProjectionMatrix = viewProjectionMatrix,
                                    program = materials.programOccluder,
                                    pushBackFW = node.pushBackFW,
                                    viewportAspect = aspect
                                )
                                "box" -> occluder.renderBox(nodeMvp, materials.programOccluder)
                                else -> occluder.renderEllipsoid(nodeMvp, materials.programOccluder, effectiveLatSegs)
                            }
                            occluderCount++
                            renderedNodeIdsScratch.add(node.id)
                        }
                    }
                    occluder.endPass()
                    lastRenderedOccluderCount = occluderCount
                }
                PassStageKind.OPAQUE_NODES -> {
                    passOrderScratch.add(RenderPass.OPAQUE_NODES)
                    for (k in 0 until opaqueCount) {
                        val idx = opaqueIndices[k]
                        renderNode(idx, effect, faceState, evaluator, tierTexMaxSize)
                        renderedNodeIdsScratch.add(nodes[idx].id)
                    }
                    lastRenderedOpaqueCount = opaqueCount
                }
                PassStageKind.TRANSPARENT_NODES -> {
                    passOrderScratch.add(RenderPass.TRANSPARENT_NODES)
                    for (a in 1 until transparentCount) {
                        val keyIdx = transparentIndices[a]
                        val keyOrder = nodes[keyIdx].renderOrder
                        val keyZ = evaluator.worldMatrices[keyIdx * 16 + 14]
                        var b = a - 1
                        while (b >= 0) {
                            val cmpIdx = transparentIndices[b]
                            val cmpOrder = nodes[cmpIdx].renderOrder
                            val cmpZ = evaluator.worldMatrices[cmpIdx * 16 + 14]
                            val shouldSwap = (cmpOrder > keyOrder) || (cmpOrder == keyOrder && cmpZ < keyZ)
                            if (!shouldSwap) break
                            transparentIndices[b + 1] = cmpIdx
                            b--
                        }
                        transparentIndices[b + 1] = keyIdx
                    }

                    for (k in 0 until transparentCount) {
                        val idx = transparentIndices[k]
                        renderNode(idx, effect, faceState, evaluator, tierTexMaxSize)
                        renderedNodeIdsScratch.add(nodes[idx].id)
                    }
                    lastRenderedTransparentCount = transparentCount
                }
                PassStageKind.POST_NODES -> {
                    passOrderScratch.add(RenderPass.POST_NODES)
                    for (a in 1 until postCount) {
                        val keyIdx = postIndices[a]
                        val keyOrder = nodes[keyIdx].renderOrder
                        var b = a - 1
                        while (b >= 0) {
                            val cmpIdx = postIndices[b]
                            if (nodes[cmpIdx].renderOrder <= keyOrder) break
                            postIndices[b + 1] = cmpIdx
                            b--
                        }
                        postIndices[b + 1] = keyIdx
                    }

                    post.renderPass(
                        effect = effect,
                        postIndices = postIndices,
                        postCount = postCount,
                        viewportWidth = this.viewportWidth,
                        viewportHeight = this.viewportHeight,
                        resourceOwner = resourceOwner,
                        faceState = faceState,
                        regionMask = regionMask,
                        rtPool = rtPool,
                        textures = textures,
                        frameContext = frameContext
                    )
                    lastRenderedPostCount = post.lastExecutedPasses.size
                    for (p in post.lastExecutedPasses) {
                        renderedNodeIdsScratch.add(p.nodeId)
                    }
                }
                PassStageKind.OVERLAY_NODES -> {
                    passOrderScratch.add(RenderPass.OVERLAY_NODES)
                    for (a in 1 until overlayCount) {
                        val keyIdx = overlayIndices[a]
                        val keyOrder = nodes[keyIdx].renderOrder
                        var b = a - 1
                        while (b >= 0) {
                            val cmpIdx = overlayIndices[b]
                            if (nodes[cmpIdx].renderOrder <= keyOrder) break
                            overlayIndices[b + 1] = cmpIdx
                            b--
                        }
                        overlayIndices[b + 1] = keyIdx
                    }
                    for (k in 0 until overlayCount) {
                        val idx = overlayIndices[k]
                        renderNode(idx, effect, faceState, evaluator, tierTexMaxSize)
                        renderedNodeIdsScratch.add(nodes[idx].id)
                    }
                    lastRenderedOverlayCount = overlayCount
                }
                PassStageKind.FINAL_OUTPUT -> {
                    passOrderScratch.add(RenderPass.FINAL_OUTPUT)
                }
                PassStageKind.CUSTOM_PASS -> {
                    // Custom data-driven pass node (e.g. pass.lut@1, pass.grain@1, pass.sharpen@1, pass.skinSoften@1, pass.softFocus@1)
                    val executedCustom = post.renderCustomPass(
                        passNode = passNode,
                        resolvedParams = resolvedParams,
                        effect = effect,
                        viewportWidth = this.viewportWidth,
                        viewportHeight = this.viewportHeight,
                        effectiveScale = effectiveScale,
                        resourceOwner = resourceOwner,
                        faceState = faceState,
                        regionMask = regionMask,
                        rtPool = rtPool,
                        textures = textures,
                        frameContext = frameContext
                    )
                    if (executedCustom) {
                        lastRenderedPostCount = post.lastExecutedPasses.size
                        renderedNodeIdsScratch.add(passNode.id)
                    }
                }
            }

            val elapsedMs = ((System.nanoTime() - startNs).coerceAtLeast(0L)) / 1_000_000f
            executedPassNodeIdsScratch.add(passNode.id)
            executedPassRecordsScratch.add(
                PassExecutionRecord(
                    passId = passNode.id,
                    stageKind = passNode.stageKind,
                    templateId = passNode.templateId,
                    reads = passNode.reads,
                    writes = passNode.writes,
                    effectiveScaleFactor = effectiveScale,
                    durationMs = elapsedMs,
                    resolvedParams = resolvedParams
                )
            )
        }

        // Return temporary frame targets to the pool without freeing GL allocations
        rtPool.endFrame()
    }

    private fun resolveMaterialOpacity(mat: CompiledMaterial?, effect: CompiledEffect): Float {
        if (mat == null) return 1f
        var op = mat.opacity
        val tierOverride = effect.tierOverrides[effect.activeTier]
        if (tierOverride != null && tierOverride.paramOverrides.isNotEmpty()) {
            val tierVal = tierOverride.paramOverrides["mat:${mat.id}.opacity"]
                ?: tierOverride.paramOverrides["${mat.id}.opacity"]
            if (tierVal != null) op = tierVal
        }
        op = effect.propertySlots.getDynamicParam("mat:${mat.id}.opacity", op)
        return op.coerceIn(0f, 1f)
    }

    private fun renderNode(
        i: Int,
        effect: CompiledEffect,
        faceState: FaceState,
        evaluator: SceneEvaluator,
        tierTexMaxSize: Int
    ) {
        val node = effect.nodes[i]
        val worldOffset = i * 16
        Matrix.multiplyMM(nodeMvp, 0, viewProjectionMatrix, 0, evaluator.worldMatrices, worldOffset)
        val mat = ModelRenderer.resolvePrimaryModelMaterial(node, effect.materials)
        val matOpacity = resolveMaterialOpacity(mat, effect)
        val opacity = (effect.propertySlots.opacity[i] * effect.propertySlots.visible[i] * matOpacity).coerceIn(0f, 1f)

        val renderTimeSec = lastFrameContext.renderTimeSec
        val dynFrame = effect.propertySlots.getDynamicParam(
            "pass:${node.id}.frameIndex",
            effect.propertySlots.getDynamicParam("sprite:${node.id}.frameIndex", Float.NaN)
        )
        val explicitFrameIndex = if (dynFrame.isFinite() && dynFrame >= 0f) {
            kotlin.math.round(dynFrame).toInt()
        } else {
            node.spriteFrameIndex
        }
        when (node.type) {
            "sprite" -> {
                val assetId = node.spriteAssetId ?: mat?.mapAssetId
                val asset = assetId?.let { effect.assets[it] }
                val texId = asset?.let { textures.getOrLoad(it.localPath, tierTexMaxSize) } ?: 0
                val texEntry = asset?.let {
                    textures.getEntry(it.localPath, tierTexMaxSize) ?: textures.getEntryForPath(it.localPath)
                }
                SpriteRenderer.applyGlBlendMode(mat?.blend)
                if (mat?.depthTest != false) {
                    GLES30.glEnable(GLES30.GL_DEPTH_TEST)
                } else {
                    GLES30.glDisable(GLES30.GL_DEPTH_TEST)
                }
                GLES30.glDepthMask(mat?.depthWrite ?: false)

                sprites.render(
                    node = node,
                    mvpMatrix = nodeMvp,
                    textureId = texId,
                    opacity = opacity,
                    program = materials.programUnlit,
                    frontFacing = faceState.isFrontCamera,
                    renderTimeSec = renderTimeSec,
                    colorTint = mat?.color,
                    explicitFrameIndex = explicitFrameIndex,
                    texWidth = texEntry?.width ?: 0,
                    texHeight = texEntry?.height ?: 0
                )
            }
            "overlay" -> {
                val assetId = node.spriteAssetId ?: mat?.mapAssetId
                val asset = assetId?.let { effect.assets[it] }
                val texId = asset?.let { textures.getOrLoad(it.localPath, tierTexMaxSize) } ?: 0
                val texEntry = asset?.let {
                    textures.getEntry(it.localPath, tierTexMaxSize) ?: textures.getEntryForPath(it.localPath)
                }
                SpriteRenderer.applyGlBlendMode(mat?.blend)
                GLES30.glDisable(GLES30.GL_DEPTH_TEST)
                GLES30.glDepthMask(false)

                sprites.renderOverlay(
                    node = node,
                    textureId = texId,
                    opacity = opacity,
                    program = materials.programUnlit,
                    viewportWidth = viewportWidth,
                    viewportHeight = viewportHeight,
                    assetWidth = texEntry?.width ?: viewportWidth,
                    assetHeight = texEntry?.height ?: viewportHeight,
                    renderTimeSec = renderTimeSec,
                    colorTint = mat?.color,
                    explicitFrameIndex = explicitFrameIndex
                )
            }
            "faceMesh" -> {
                val aspect = viewportWidth.toFloat() / viewportHeight.toFloat().coerceAtLeast(1f)
                faceMesh.updateVertices(
                    faceState = faceState,
                    viewportAspect = aspect
                )

                val mapAsset = mat?.mapAssetId?.let { effect.assets[it] }
                val maskAsset = mat?.maskAssetId?.let { effect.assets[it] }
                val texId = mapAsset?.let { textures.getOrLoad(it.localPath, tierTexMaxSize) } ?: 0
                val maskId = maskAsset?.let { textures.getOrLoad(it.localPath, tierTexMaxSize) }
                val effectiveOpacity = opacity.coerceIn(0f, 1f)

                GLES30.glEnable(GLES30.GL_BLEND)
                GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE_MINUS_SRC_ALPHA)
                if (mat?.depthTest != false) {
                    GLES30.glEnable(GLES30.GL_DEPTH_TEST)
                } else {
                    GLES30.glDisable(GLES30.GL_DEPTH_TEST)
                }
                GLES30.glDepthMask(mat?.depthWrite ?: false)

                faceMesh.render(
                    viewProjectionMatrix = viewProjectionMatrix,
                    textureId = texId,
                    maskId = maskId,
                    opacity = effectiveOpacity,
                    program = materials.programFaceTexture,
                    tintColor = mat?.color,
                    protectEyes = mat?.protectEyes ?: true,
                    protectLips = mat?.protectLips ?: true,
                    protectMouthInterior = mat?.protectMouthInterior ?: true,
                    frontFacing = faceState.isFrontCamera,
                    faceState = faceState,
                    blendMode = mat?.blend ?: "alpha",
                    cull = mat?.cull ?: false
                )
            }
            "eyeColor", "recolor" -> {
                val rawMaskId = (node.faceMeshRegion ?: "iris").removePrefix("\$mask:").trim()
                val maskIdStr = rawMaskId.lowercase().replace('-', '_')
                val maskExpr = effect.maskDefs[maskIdStr]
                    ?: effect.maskDefs[rawMaskId]
                    ?: RegionMaskPass.BUILTIN_MASKS[maskIdStr]
                    ?: com.example.engine.pkg.FaceRegions.getRegionLoop(maskIdStr)?.let {
                        RegionMaskPass.MaskExpr.LandmarkLoop(maskIdStr)
                    }
                    ?: RegionMaskPass.BUILTIN_IRIS_MASK
                val maskRt = regionMask.evaluateMaskToRtPool(
                    maskId = maskIdStr,
                    expr = maskExpr,
                    faceState = faceState,
                    viewportWidth = viewportWidth,
                    viewportHeight = viewportHeight,
                    rtPool = rtPool,
                    propertySlots = effect.propertySlots,
                    scaleFactor = RtPool.RtScale.HALF.factor,
                    resourceOwner = resourceOwner
                )
                GLES30.glViewport(0, 0, viewportWidth, viewportHeight)
                recolor.renderNode(
                    node = node,
                    effect = effect,
                    faceState = faceState,
                    opacity = opacity,
                    viewportWidth = viewportWidth,
                    viewportHeight = viewportHeight,
                    maskTextureId = maskRt?.texId ?: 0
                )
            }
            "model" -> {
                val asset = node.modelAssetId?.let { effect.assets[it] }
                if (asset != null) {
                    val currentGen = resourceOwner.contextGeneration
                    if (loadedGlbGenerations[asset.id] != currentGen) {
                        loadedGlbModels.remove(asset.id)
                        loadedGlbGenerations.remove(asset.id)
                    }
                    val glbModel = loadedGlbModels.getOrPut(asset.id) {
                        val loaded = glbLoader.load(File(asset.localPath)) ?: GlbLoader.GlbModel(emptyList(), emptyList(), emptyList(), emptyList())
                        trackGlbModelResources(loaded)
                        loadedGlbGenerations[asset.id] = currentGen
                        loaded
                    }
                    System.arraycopy(evaluator.worldMatrices, worldOffset, scratchNodeMat, 0, 16)
                    val matMapAsset = mat?.mapAssetId?.let { effect.assets[it] }
                    val matTexId = matMapAsset?.let { textures.getOrLoad(it.localPath, tierTexMaxSize) } ?: 0
                    val normalMapAsset = mat?.normalMapAssetId?.let { effect.assets[it] }
                    val normalMapTexId = normalMapAsset?.let { textures.getOrLoad(it.localPath, tierTexMaxSize) } ?: 0
                    val mrMapAsset = mat?.metallicRoughnessMapAssetId?.let { effect.assets[it] }
                    val mrMapTexId = mrMapAsset?.let { textures.getOrLoad(it.localPath, tierTexMaxSize) } ?: 0
                    val occMapAsset = mat?.occlusionMapAssetId?.let { effect.assets[it] }
                    val occMapTexId = occMapAsset?.let { textures.getOrLoad(it.localPath, tierTexMaxSize) } ?: 0
                    val emMapAsset = mat?.emissiveMapAssetId?.let { effect.assets[it] }
                    val emMapTexId = emMapAsset?.let { textures.getOrLoad(it.localPath, tierTexMaxSize) } ?: 0

                    modelRenderer.render(
                        model = glbModel,
                        node = node,
                        worldMatrix = scratchNodeMat,
                        viewProjectionMatrix = viewProjectionMatrix,
                        material = mat,
                        lighting = effect.lighting,
                        opacity = opacity,
                        programs = materials,
                        textureId = matTexId,
                        clips = effect.behaviorRunner.clips,
                        activeTier = effect.activeTier,
                        frontFacing = faceState.isFrontCamera,
                        normalMapTexId = normalMapTexId,
                        metallicRoughnessMapTexId = mrMapTexId,
                        occlusionMapTexId = occMapTexId,
                        emissiveMapTexId = emMapTexId,
                        allMaterials = effect.materials
                    )
                }
            }
        }
    }
}
