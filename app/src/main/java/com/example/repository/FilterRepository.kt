package com.example.repository

import android.content.Context
import android.util.Log
import com.example.engine.pkg.Compiler
import com.example.engine.pkg.PackageInstaller
import com.example.engine.pkg.PackagePreloader
import com.example.engine.scene.CompiledEffect
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

data class InstalledFilter(
    val id: String,
    val name: String,
    val description: String,
    val previewPath: String?,
    val packageDir: File,
    val isBundled: Boolean
)

/**
 * FilterRepository: Single source of truth for installed filters.
 * Loads bundled reference packages and user packages, compiles them,
 * and handles filter switching without restarting the camera or tracker.
 */
class FilterRepository(private val context: Context) {

    companion object {
        private const val TAG = "FilterRepository"
    }

    private val installer = PackageInstaller(context)
    val preloader = PackagePreloader(installer)

    private val _installedFilters = MutableStateFlow<List<InstalledFilter>>(emptyList())
    val installedFilters: StateFlow<List<InstalledFilter>> = _installedFilters.asStateFlow()

    private val _activeEffect = MutableStateFlow<CompiledEffect?>(null)
    val activeEffect: StateFlow<CompiledEffect?> = _activeEffect.asStateFlow()

    private val _selectedFilterId = MutableStateFlow<String?>(null)
    val selectedFilterId: StateFlow<String?> = _selectedFilterId.asStateFlow()

    private val _lastSwitchError = MutableStateFlow<String?>(null)
    val lastSwitchError: StateFlow<String?> = _lastSwitchError.asStateFlow()

    private val compiledCache = HashMap<String, CompiledEffect>()

    suspend fun initialize() = withContext(Dispatchers.IO) {
        val list = ArrayList<InstalledFilter>()
        val filtersDir = File(context.filesDir, "filters").apply { mkdirs() }

        // 1. Extract and install bundled .xfilter packages from assets
        try {
            val assetFilters = context.assets.list("filters") ?: emptyArray()
            val validPackageIds = assetFilters.map { it.removeSuffix(".xfilter") }.toSet()

            // Remove any stale directories from previous app sessions
            val currentDirs = filtersDir.listFiles { f -> f.isDirectory } ?: emptyArray()
            for (dir in currentDirs) {
                if (!validPackageIds.contains(dir.name)) {
                    dir.deleteRecursively()
                }
            }

            for (item in assetFilters) {
                if (item.endsWith(".xfilter")) {
                    val packageId = item.removeSuffix(".xfilter")
                    context.assets.open("filters/$item").use { input ->
                        val result = installer.installFromStream(input, packageId)
                        if (result is PackageInstaller.InstallResult.Success) {
                            compiledCache[packageId] = result.effect
                            preloader.putPreloaded(packageId, result.effect)
                        } else if (result is PackageInstaller.InstallResult.Failure) {
                            Log.w(TAG, "Failed to install $item: ${result.reason}")
                        }
                    }
                } else {
                    // Check if item is a directory package containing manifest.json
                    val subEntries = context.assets.list("filters/$item") ?: emptyArray()
                    if (subEntries.contains("manifest.json")) {
                        val targetDir = File(filtersDir, item)
                        copyAssetDirectory("filters/$item", targetDir)
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error discovering asset filters: ${e.message}", e)
        }

        // 2. Discover all installed filters in filesDir/filters
        val installedDirs = filtersDir.listFiles { f -> f.isDirectory && !f.name.contains("_tmp_") } ?: emptyArray()
        for (dir in installedDirs) {
            val manifestFile = File(dir, "manifest.json")
            if (manifestFile.exists()) {
                try {
                    val json = JSONObject(manifestFile.readText())
                    val id = json.optString("id", dir.name)
                    val name = json.optString("name", dir.name)
                    val desc = json.optString("description", "")
                    val prevFile = File(dir, "preview.jpg").takeIf { it.exists() }
                        ?: File(dir, "thumbnail.png").takeIf { it.exists() }

                    val filter = InstalledFilter(
                        id = id,
                        name = name,
                        description = desc,
                        previewPath = prevFile?.absolutePath,
                        packageDir = dir,
                        isBundled = true
                    )
                    list.add(filter)

                    // Compile and cache
                    if (!compiledCache.containsKey(id)) {
                        val effect = Compiler.compile(json, dir)
                        compiledCache[id] = effect
                        preloader.putPreloaded(id, effect)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to parse filter in ${dir.name}: ${e.message}", e)
                }
            }
        }

        // Sort so standalone forehead/crown/eyewear/beauty filters appear first in carousel
        list.sortBy {
            when (it.id) {
                "sunflower_forehead" -> 0
                "periwinkle_forehead" -> 1
                "heart_mood_filter" -> 2
                "furry_glasses_filter" -> 3
                "no_makeup_beauty" -> 4
                "light_lens" -> 5
                else -> 6
            }
        }
        _installedFilters.value = list

        // Select standalone filter by default
        val defaultFilter = list.find { it.id == "sunflower_forehead" } ?: list.firstOrNull()
        if (defaultFilter != null) {
            selectFilter(defaultFilter.id)
        }
    }

    /**
     * Install a .xfilter or .zip package selected by the user at runtime and activate it immediately.
     */
    suspend fun installFromUri(uri: android.net.Uri): PackageInstaller.InstallResult = withContext(Dispatchers.IO) {
        val fallbackId = "user_pkg_${System.currentTimeMillis()}"
        val inputStream = context.contentResolver.openInputStream(uri)
            ?: return@withContext PackageInstaller.InstallResult.Failure("Unable to open selected package URI")

        val result = inputStream.use { stream ->
            installer.installFromStream(stream, fallbackId)
        }

        if (result is PackageInstaller.InstallResult.Success) {
            val effect = result.effect
            val pkgDir = result.packageDir
            val prevFile = File(pkgDir, "preview.jpg").takeIf { it.exists() }
                ?: File(pkgDir, "thumbnail.png").takeIf { it.exists() }
            val newFilter = InstalledFilter(
                id = effect.packageId,
                name = effect.packageName,
                description = "",
                previewPath = prevFile?.absolutePath,
                packageDir = pkgDir,
                isBundled = false
            )
            compiledCache[effect.packageId] = effect
            preloader.putPreloaded(effect.packageId, effect)
            val updated = _installedFilters.value.filterNot { it.id == effect.packageId } + newFilter
            _installedFilters.value = updated
            selectFilter(effect.packageId)
        }
        result
    }

    /**
     * Switch active filter via PackagePreloader.
     * Guaranteed never to restart camera or face tracker, and preserves previous filter on failure!
     */
    fun selectFilter(filterId: String): PackagePreloader.SwitchResult {
        val list = _installedFilters.value
        val installed = list.find { it.id == filterId }
        val cached = compiledCache[filterId]
        if (cached != null) {
            preloader.putPreloaded(filterId, cached)
        }

        val switchRes = preloader.switchFilter(
            targetPackageId = filterId,
            targetPackageDir = installed?.packageDir,
            currentActive = _activeEffect.value
        )

        when (switchRes) {
            is PackagePreloader.SwitchResult.Activated -> {
                compiledCache[filterId] = switchRes.activeEffect
                _activeEffect.value = switchRes.activeEffect
                _selectedFilterId.value = filterId
                _lastSwitchError.value = null
                Log.i(TAG, "Switched active filter to: $filterId")

                // Preload neighbors in the carousel in background
                val selectedIdx = list.indexOfFirst { it.id == filterId }
                if (selectedIdx >= 0) {
                    val carouselPairs = list.map { it.id to it.packageDir }
                    preloader.preloadCarouselNeighbors(carouselPairs, selectedIdx)
                }
            }
            is PackagePreloader.SwitchResult.RolledBack -> {
                // Preserve previous active filter and expose explicit failure reason
                _activeEffect.value = switchRes.preservedEffect
                _lastSwitchError.value = switchRes.failureReason
                Log.w(TAG, "Filter switch to '$filterId' failed (${switchRes.failureReason}); preserved active filter")
            }
        }
        return switchRes
    }

    fun deselectFilter() {
        _activeEffect.value = null
        _selectedFilterId.value = null
        _lastSwitchError.value = null
    }

    private fun copyAssetDirectory(assetPath: String, destDir: File) {
        val entries = context.assets.list(assetPath) ?: emptyArray()
        if (entries.isEmpty()) {
            // It is a file
            destDir.parentFile?.mkdirs()
            context.assets.open(assetPath).use { input ->
                destDir.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        } else {
            destDir.mkdirs()
            for (entry in entries) {
                copyAssetDirectory("$assetPath/$entry", File(destDir, entry))
            }
        }
    }
}
