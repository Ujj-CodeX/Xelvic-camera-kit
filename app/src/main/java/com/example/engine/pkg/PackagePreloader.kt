package com.example.engine.pkg

import android.util.Log
import com.example.engine.render.RenderGraph
import com.example.engine.render.TextureCache
import com.example.engine.scene.CompiledEffect
import org.json.JSONObject
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executors

/**
 * PackagePreloader (README Section 15 & 21 Step 6):
 *
 * Preloads upcoming carousel filters in the background following the exact sequence:
 *   Validate -> Compile -> Decode -> Build GPU resources -> Warm shaders/pipelines
 *
 * Guarantees:
 * - Spreads large GPU uploads across frames on the GL thread (`stepGlUploads`).
 * - Filter switching replaces ONLY the active `CompiledEffect` and its GPU state;
 *   camera and tracker keep running with only requested tracker output flags updated.
 * - Any preload or activation failure leaves the previous active filter untouched
 *   and records an explicit failure reason.
 */
class PackagePreloader @JvmOverloads constructor(
    private val installer: PackageInstaller? = null,
    private val maxCachedEffects: Int = 12
) {
    companion object {
        private const val TAG = "PackagePreloader"
    }

    enum class Stage {
        VALIDATED,
        COMPILED,
        DECODED,
        GPU_RESOURCES_BUILT,
        SHADERS_WARMED
    }

    sealed class PreloadResult {
        data class Success(
            val packageId: String,
            val effect: CompiledEffect,
            val completedStages: List<Stage>
        ) : PreloadResult()

        data class Failure(
            val packageId: String,
            val reason: String
        ) : PreloadResult()
    }

    data class TrackerOutputFlags(
        val requiresPose: Boolean,
        val requiresBlendshapes: Boolean,
        val requiredLandmarkCount: Int,
        val requestedBlendshapes: Set<String>
    )

    sealed class SwitchResult {
        data class Activated(
            val activeEffect: CompiledEffect,
            val trackerFlags: TrackerOutputFlags,
            val cameraRestarted: Boolean = false,
            val trackerRestarted: Boolean = false
        ) : SwitchResult()

        data class RolledBack(
            val preservedEffect: CompiledEffect?,
            val failedPackageId: String,
            val failureReason: String
        ) : SwitchResult()
    }

    data class GpuPreloadTask(
        val packageId: String,
        val assetId: String,
        val assetType: String,
        val localPath: String,
        val maxTextureSize: Int
    )

    private val ioExecutor = Executors.newSingleThreadExecutor()

    // Bounded LRU of preloaded CompiledEffects
    private val effectLru = object : LinkedHashMap<String, CompiledEffect>(maxCachedEffects + 2, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, CompiledEffect>?): Boolean {
            return size > maxCachedEffects
        }
    }

    private val failureReasons = ConcurrentHashMap<String, String>()
    private val stageHistory = ConcurrentHashMap<String, MutableList<Stage>>()
    private val pendingGpuTasks = ConcurrentLinkedQueue<GpuPreloadTask>()
    private val packageGpuGeneration = ConcurrentHashMap<String, Int>()

    @Volatile
    var isPaused: Boolean = false
        private set

    @Volatile
    var shadersWarmed: Boolean = false
        private set

    @Volatile
    var warmedContextGeneration: Int = -1
        private set

    fun onPause(textureCache: TextureCache? = null) {
        isPaused = true
        pendingGpuTasks.clear()
        textureCache?.onPause()
    }

    fun onResume(textureCache: TextureCache? = null) {
        isPaused = false
        textureCache?.onResume()
    }

    fun shutdown() {
        isPaused = true
        pendingGpuTasks.clear()
        ioExecutor.shutdown()
    }

    val isShutdown: Boolean
        get() = ioExecutor.isShutdown

    fun getPreloaded(id: String): CompiledEffect? = synchronized(effectLru) { effectLru[id] }

    fun putPreloaded(id: String, effect: CompiledEffect) {
        synchronized(effectLru) {
            effectLru[id] = effect
        }
        failureReasons.remove(id)
    }

    fun getFailureReason(id: String): String? = failureReasons[id]

    fun getCompletedStages(id: String, currentContextGeneration: Int? = null): List<Stage> {
        val stages = stageHistory[id]?.toMutableList() ?: return emptyList()
        if (currentContextGeneration != null) {
            val builtGen = packageGpuGeneration[id]
            if (builtGen != null && builtGen != currentContextGeneration) {
                stages.remove(Stage.GPU_RESOURCES_BUILT)
                stages.remove(Stage.SHADERS_WARMED)
            }
        }
        return stages
    }

    fun isPackageGpuValidForGeneration(packageId: String, currentContextGeneration: Int): Boolean {
        val builtGen = packageGpuGeneration[packageId] ?: return false
        return builtGen == currentContextGeneration
    }

    /**
     * Re-stages and rebuilds GPU resources on demand if a preloaded package's GPU objects were
     * created before an EGL context loss or ON_STOP GL release (`builtGen != currentGen`).
     */
    fun rebuildPackageGpuResourcesOnDemand(
        packageId: String,
        renderGraph: RenderGraph,
        tierMaxTextureSize: Int = 1024
    ): Boolean {
        val effect = getPreloaded(packageId) ?: return false
        val currentGen = renderGraph.resourceOwner.contextGeneration
        if (packageGpuGeneration[packageId] == currentGen) {
            return true
        }
        renderGraph.restoreEffectResourcesOnGlThread(effect, tierMaxTextureSize)
        if (warmedContextGeneration != currentGen || !shadersWarmed) {
            renderGraph.warmShaders()
            shadersWarmed = true
            warmedContextGeneration = currentGen
        }
        packageGpuGeneration[packageId] = currentGen
        val stages = stageHistory.getOrPut(packageId) { ArrayList() }
        if (!stages.contains(Stage.GPU_RESOURCES_BUILT)) {
            stages.add(Stage.GPU_RESOURCES_BUILT)
        }
        if (!stages.contains(Stage.SHADERS_WARMED)) {
            stages.add(Stage.SHADERS_WARMED)
        }
        return true
    }

    fun getPendingGpuUploadCount(): Int = pendingGpuTasks.size

    /**
     * Synchronous preload sequence for background workers or deterministic testing:
     * Validate -> Compile -> Decode -> Stage GPU resource tasks -> Mark for shader warming.
     */
    fun preloadSync(
        packageId: String,
        packageDir: File,
        textureCache: TextureCache? = null,
        tierMaxTextureSize: Int = 1024
    ): PreloadResult {
        if (isPaused) {
            val reason = "PackagePreloader is paused; skipping preload for '$packageId'"
            failureReasons[packageId] = reason
            return PreloadResult.Failure(packageId, reason)
        }
        val stages = ArrayList<Stage>()
        try {
            // 1. VALIDATE
            if (!packageDir.exists() || !packageDir.isDirectory) {
                val reason = "Package directory missing for '$packageId': ${packageDir.path}"
                failureReasons[packageId] = reason
                return PreloadResult.Failure(packageId, reason)
            }

            val manifestFile = File(packageDir, "manifest.json")
            if (!manifestFile.exists()) {
                val reason = "Missing manifest.json in package '$packageId'"
                failureReasons[packageId] = reason
                return PreloadResult.Failure(packageId, reason)
            }

            val manifestJson = try {
                JSONObject(manifestFile.readText())
            } catch (e: Exception) {
                val reason = "Invalid JSON in manifest for '$packageId': ${e.message}"
                failureReasons[packageId] = reason
                return PreloadResult.Failure(packageId, reason)
            }

            val manifestVal = PackageValidator.validateManifest(manifestJson)
            if (!manifestVal.isValid) {
                val reason = manifestVal.errorReason ?: "Manifest validation failed for '$packageId'"
                failureReasons[packageId] = reason
                return PreloadResult.Failure(packageId, reason)
            }

            val assetsArr = manifestJson.optJSONArray("assets")
            if (assetsArr != null) {
                for (i in 0 until assetsArr.length()) {
                    val a = assetsArr.getJSONObject(i)
                    val assetPath = a.getString("path")
                    val assetFile = File(packageDir, assetPath)
                    val sha = a.optString("sha256", null)
                    val bytes = if (a.has("bytes")) a.getLong("bytes") else null

                    val assetVal = PackageValidator.validateAssetFile(assetFile, sha, bytes)
                    if (!assetVal.isValid) {
                        val reason = assetVal.errorReason ?: "Asset validation failed for $assetPath"
                        failureReasons[packageId] = reason
                        return PreloadResult.Failure(packageId, reason)
                    }
                }
            }
            stages.add(Stage.VALIDATED)

            // 2. COMPILE
            val compiled = try {
                Compiler.compile(manifestJson, packageDir)
            } catch (e: Exception) {
                val reason = "Compilation failed for '$packageId': ${e.message}"
                failureReasons[packageId] = reason
                return PreloadResult.Failure(packageId, reason)
            }
            stages.add(Stage.COMPILED)

            if (isPaused) {
                val reason = "PackagePreloader paused before decode for '$packageId'"
                failureReasons[packageId] = reason
                return PreloadResult.Failure(packageId, reason)
            }

            // 3. DECODE (background CPU decode & staging)
            for (asset in compiled.assets.values) {
                if (isPaused) break
                when (asset.type) {
                    "image", "mask", "lut" -> {
                        val isLutAsset = com.example.engine.render.PostRenderer.isAssetUsedAsLut(compiled, asset.id)
                        textureCache?.preloadDecodeToStaging(asset.localPath, tierMaxTextureSize, isLut = isLutAsset)
                        pendingGpuTasks.add(
                            GpuPreloadTask(
                                packageId = packageId,
                                assetId = asset.id,
                                assetType = if (isLutAsset) "lut" else asset.type,
                                localPath = asset.localPath,
                                maxTextureSize = tierMaxTextureSize
                            )
                        )
                    }
                    "model" -> {
                        pendingGpuTasks.add(
                            GpuPreloadTask(
                                packageId = packageId,
                                assetId = asset.id,
                                assetType = "model",
                                localPath = asset.localPath,
                                maxTextureSize = tierMaxTextureSize
                            )
                        )
                    }
                }
            }
            stages.add(Stage.DECODED)

            val currentGen = textureCache?.resourceOwner?.contextGeneration ?: 0
            // If no GPU tasks were queued, mark GPU resources and shaders ready immediately
            if (pendingGpuTasks.none { it.packageId == packageId }) {
                stages.add(Stage.GPU_RESOURCES_BUILT)
                packageGpuGeneration[packageId] = currentGen
                if (shadersWarmed && warmedContextGeneration == currentGen) {
                    stages.add(Stage.SHADERS_WARMED)
                }
            }

            stageHistory[packageId] = stages
            putPreloaded(packageId, compiled)
            return PreloadResult.Success(packageId, compiled, stages.toList())
        } catch (e: Exception) {
            val reason = "Unexpected preload error for '$packageId': ${e.message}"
            failureReasons[packageId] = reason
            return PreloadResult.Failure(packageId, reason)
        }
    }

    /**
     * Asynchronous preload on background IO worker.
     */
    fun preload(
        packageId: String,
        packageDir: File? = null,
        textureCache: TextureCache? = null,
        tierMaxTextureSize: Int = 1024,
        onLoaded: (CompiledEffect?) -> Unit = {}
    ) {
        if (isPaused || ioExecutor.isShutdown) {
            onLoaded(null)
            return
        }
        val cached = getPreloaded(packageId)
        if (cached != null && packageDir == null) {
            onLoaded(cached)
            return
        }

        ioExecutor.execute {
            if (isPaused) {
                onLoaded(null)
                return@execute
            }
            val existing = getPreloaded(packageId)
            if (existing != null) {
                onLoaded(existing)
                return@execute
            }

            if (packageDir != null) {
                when (val res = preloadSync(packageId, packageDir, textureCache, tierMaxTextureSize)) {
                    is PreloadResult.Success -> onLoaded(res.effect)
                    is PreloadResult.Failure -> {
                        Log.w(TAG, "Preload failed for $packageId: ${res.reason}")
                        onLoaded(null)
                    }
                }
            } else {
                onLoaded(null)
            }
        }
    }

    /**
     * Preloads adjacent filters in the carousel in the background.
     */
    fun preloadCarouselNeighbors(
        carousel: List<Pair<String, File>>,
        selectedIndex: Int,
        radius: Int = 2,
        textureCache: TextureCache? = null,
        tierMaxTextureSize: Int = 1024
    ) {
        if (isPaused || carousel.isEmpty()) return
        val center = selectedIndex.coerceIn(0, carousel.size - 1)
        for (offset in 1..radius) {
            val nextIdx = center + offset
            if (nextIdx in carousel.indices) {
                val (id, dir) = carousel[nextIdx]
                preload(id, dir, textureCache, tierMaxTextureSize)
            }
            val prevIdx = center - offset
            if (prevIdx in carousel.indices) {
                val (id, dir) = carousel[prevIdx]
                preload(id, dir, textureCache, tierMaxTextureSize)
            }
        }
    }

    /**
     * Executed on the GL render thread once per frame:
     * 1. Warms shader programs/pipelines if not yet warmed for the current contextGeneration.
     * 2. Spreads staged texture and GLB GPU uploads across frames (`maxUploadsPerFrame`).
     */
    fun stepGlUploads(
        renderGraph: RenderGraph?,
        textureCache: TextureCache? = renderGraph?.textures,
        maxUploadsPerFrame: Int = 1
    ): Int {
        if (isPaused) return 0
        val budget = maxUploadsPerFrame.coerceAtLeast(1)
        val currentGen = renderGraph?.resourceOwner?.contextGeneration
            ?: textureCache?.resourceOwner?.contextGeneration
            ?: 0
        var workDone = 0

        if (!shadersWarmed || warmedContextGeneration != currentGen) {
            renderGraph?.warmShaders()
            shadersWarmed = true
            warmedContextGeneration = currentGen
            for (stages in stageHistory.values) {
                if (!stages.contains(Stage.SHADERS_WARMED)) {
                    stages.add(Stage.SHADERS_WARMED)
                }
            }
        }

        // 1. Drain up to `budget` staged texture uploads on the GL thread
        if (textureCache != null && textureCache.pendingUploadCount() > 0) {
            workDone += textureCache.stepFrameUploads(budget)
        }

        // 2. Process remaining budget from pendingGpuTasks
        while (workDone < budget) {
            val task = pendingGpuTasks.poll() ?: break
            when (task.assetType) {
                "image", "mask", "lut" -> {
                    textureCache?.getOrLoad(task.localPath, task.maxTextureSize, isLut = (task.assetType == "lut"))
                    workDone++
                }
                "model" -> {
                    renderGraph?.preloadModelOnGlThread(task.assetId, File(task.localPath))
                    workDone++
                }
            }

            // Record stage completion and generation when all tasks for this package have drained
            if (pendingGpuTasks.none { it.packageId == task.packageId }) {
                packageGpuGeneration[task.packageId] = currentGen
                val stages = stageHistory.getOrPut(task.packageId) { ArrayList() }
                if (!stages.contains(Stage.GPU_RESOURCES_BUILT)) {
                    stages.add(Stage.GPU_RESOURCES_BUILT)
                }
                if (shadersWarmed && !stages.contains(Stage.SHADERS_WARMED)) {
                    stages.add(Stage.SHADERS_WARMED)
                }
            }
        }

        return workDone
    }

    /**
     * Switches the active filter without restarting the camera or face tracker (README Section 3 & 15).
     * If the target filter failed validation/compilation or cannot be loaded, preserves `currentActive`
     * and returns `SwitchResult.RolledBack` with an explicit failure reason.
     */
    fun switchFilter(
        targetPackageId: String,
        targetPackageDir: File?,
        currentActive: CompiledEffect?,
        textureCache: TextureCache? = null,
        tierMaxTextureSize: Int = 1024
    ): SwitchResult {
        var effect = getPreloaded(targetPackageId)
        if (effect == null && targetPackageDir != null) {
            when (val res = preloadSync(targetPackageId, targetPackageDir, textureCache, tierMaxTextureSize)) {
                is PreloadResult.Success -> effect = res.effect
                is PreloadResult.Failure -> {
                    return SwitchResult.RolledBack(
                        preservedEffect = currentActive,
                        failedPackageId = targetPackageId,
                        failureReason = res.reason
                    )
                }
            }
        }

        if (effect == null) {
            val reason = failureReasons[targetPackageId] ?: "Filter '$targetPackageId' is not preloaded or valid"
            return SwitchResult.RolledBack(
                preservedEffect = currentActive,
                failedPackageId = targetPackageId,
                failureReason = reason
            )
        }

        effect.reset()
        val flags = TrackerOutputFlags(
            requiresPose = effect.requiresPose,
            requiresBlendshapes = effect.requiresBlendshapes,
            requiredLandmarkCount = effect.requiredLandmarkCount,
            requestedBlendshapes = effect.requestedBlendshapes
        )
        return SwitchResult.Activated(
            activeEffect = effect,
            trackerFlags = flags,
            cameraRestarted = false,
            trackerRestarted = false
        )
    }
}
