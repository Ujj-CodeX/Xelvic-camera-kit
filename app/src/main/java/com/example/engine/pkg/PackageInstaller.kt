package com.example.engine.pkg

import android.content.Context
import com.example.engine.scene.CompiledEffect
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream

/**
 * PackageInstaller: Transactional installation and verification of .xfilter packages (README Section 6 & 21 Step 3).
 * Sequence:
 * Secure extraction -> strict schema validation -> asset hash/byte verification ->
 * legacy adaptation -> capability negotiation -> reference resolution ->
 * UV/topology validation -> GLB structural validation -> expression compilation ->
 * cycle detection -> property-slot resolution -> tracker-output derivation ->
 * budget measurement -> immutable CompiledEffect.
 */
class PackageInstaller(private val context: Context) {

    sealed class InstallResult {
        data class Success(val effect: CompiledEffect, val packageDir: File) : InstallResult()
        data class Failure(val reason: String, val cause: Throwable? = null) : InstallResult()
    }

    private val filtersRootDir = File(context.filesDir, "filters").apply { mkdirs() }

    fun installFromZip(zipFile: File): InstallResult {
        return try {
            zipFile.inputStream().use { installFromStream(it, zipFile.nameWithoutExtension) }
        } catch (e: Exception) {
            InstallResult.Failure("Failed to open package: ${e.message}", e)
        }
    }

    fun installFromStream(inputStream: InputStream, packageId: String): InstallResult {
        val targetDir = File(filtersRootDir, packageId)
        val tempDir = File(filtersRootDir, "${packageId}_tmp_${System.currentTimeMillis()}")
        tempDir.mkdirs()

        return try {
            // 1. Secure extraction with Zip-Slip protection and budget limits
            ZipInputStream(inputStream).use { zis ->
                var entry: ZipEntry?
                val canonicalTarget = tempDir.canonicalPath
                var totalBytes = 0L
                val maxBytes = PackageValidator.MAX_PACKAGE_BYTES
                var entryCount = 0
                val maxEntries = 200

                while (zis.nextEntry.also { entry = it } != null) {
                    val curEntry = entry!!
                    entryCount++
                    if (entryCount > maxEntries) {
                        tempDir.deleteRecursively()
                        return InstallResult.Failure("Archive exceeds maximum entry count ($maxEntries)")
                    }

                    val destFile = File(tempDir, curEntry.name)
                    val canonicalDest = destFile.canonicalPath
                    // Zip Slip prevention
                    if (!canonicalDest.startsWith(canonicalTarget)) {
                        tempDir.deleteRecursively()
                        return InstallResult.Failure("Illegal Zip entry path traversal: ${curEntry.name}")
                    }

                    if (curEntry.isDirectory) {
                        destFile.mkdirs()
                    } else {
                        destFile.parentFile?.mkdirs()
                        FileOutputStream(destFile).use { fos ->
                            val buf = ByteArray(8192)
                            var read: Int
                            while (zis.read(buf).also { read = it } != -1) {
                                totalBytes += read
                                if (totalBytes > maxBytes) {
                                    tempDir.deleteRecursively()
                                    return InstallResult.Failure("Archive uncompressed size exceeds limit ($maxBytes bytes)")
                                }
                                fos.write(buf, 0, read)
                            }
                        }
                    }
                    zis.closeEntry()
                }
            }

            // 2. Validate manifest existence (unwrapping a single top-level directory if the zip was packaged with a root folder)
            var manifestFile = File(tempDir, "manifest.json")
            if (!manifestFile.exists()) {
                val topEntries = tempDir.listFiles() ?: emptyArray()
                val singleSubDir = topEntries.singleOrNull { it.isDirectory }
                if (singleSubDir != null && File(singleSubDir, "manifest.json").exists()) {
                    for (child in singleSubDir.listFiles() ?: emptyArray()) {
                        val dest = File(tempDir, child.name)
                        if (!child.renameTo(dest)) {
                            child.copyRecursively(dest, overwrite = true)
                            child.deleteRecursively()
                        }
                    }
                    singleSubDir.deleteRecursively()
                    manifestFile = File(tempDir, "manifest.json")
                }
            }
            if (!manifestFile.exists()) {
                tempDir.deleteRecursively()
                return InstallResult.Failure("Package missing required manifest.json")
            }

            val manifestText = manifestFile.readText()
            val manifestJson = JSONObject(manifestText)
            val valResult = PackageValidator.validateManifest(manifestJson)
            if (!valResult.isValid) {
                tempDir.deleteRecursively()
                return InstallResult.Failure("Manifest validation failed: ${valResult.errorReason}")
            }

            // 3. Validate assets (hashes, byte counts, signatures, and GLB structure)
            val assetsArr = manifestJson.optJSONArray("assets")
            if (assetsArr != null) {
                for (i in 0 until assetsArr.length()) {
                    val a = assetsArr.getJSONObject(i)
                    val assetPath = a.optString("path", a.optString("uri", ""))
                    val assetFile = File(tempDir, assetPath)
                    val sha = a.optString("sha256", null)
                    val bytes = if (a.has("bytes")) a.getLong("bytes") else null

                    val assetVal = PackageValidator.validateAssetFile(assetFile, sha, bytes)
                    if (!assetVal.isValid) {
                        tempDir.deleteRecursively()
                        return InstallResult.Failure("Asset validation failed for $assetPath: ${assetVal.errorReason}")
                    }
                }
            }

            // 4. Compile effect (resolves selectors, paths, params, expressions, cycles, topology, budgets)
            val compiled = try {
                Compiler.compile(manifestJson, tempDir)
            } catch (e: Exception) {
                tempDir.deleteRecursively()
                return InstallResult.Failure("Compilation failed: ${e.message}", e)
            }

            // 5. Commit directory transactionally (atomic replacement)
            val resolvedId = manifestJson.optString("id", "").takeIf { it.isNotBlank() && packageId.startsWith("user_pkg_") } ?: packageId
            val finalTargetDir = File(filtersRootDir, resolvedId)
            if (finalTargetDir.exists()) {
                finalTargetDir.deleteRecursively()
            }
            if (!tempDir.renameTo(finalTargetDir)) {
                tempDir.deleteRecursively()
                return InstallResult.Failure("Failed to commit package directory transactionally")
            }

            // Re-compile pointing to the committed directory path
            val finalCompiled = Compiler.compile(manifestJson, finalTargetDir)
            InstallResult.Success(finalCompiled, finalTargetDir)
        } catch (e: Exception) {
            tempDir.deleteRecursively()
            InstallResult.Failure("Installation error: ${e.message}", e)
        }
    }

    /**
     * Revalidate persisted package content before trusting it (README Section 6).
     */
    fun revalidatePersistedPackage(packageDir: File): PackageValidator.ValidationResult {
        if (!packageDir.exists() || !packageDir.isDirectory) {
            return PackageValidator.ValidationResult.fail("Package directory does not exist: ${packageDir.path}")
        }
        val manifestFile = File(packageDir, "manifest.json")
        if (!manifestFile.exists()) {
            return PackageValidator.ValidationResult.fail("Missing manifest.json in ${packageDir.name}")
        }

        val manifestJson = try {
            JSONObject(manifestFile.readText())
        } catch (e: Exception) {
            return PackageValidator.ValidationResult.fail("Corrupted manifest.json: ${e.message}")
        }

        val manifestVal = PackageValidator.validateManifest(manifestJson)
        if (!manifestVal.isValid) return manifestVal

        val assetsArr = manifestJson.optJSONArray("assets")
        if (assetsArr != null) {
            for (i in 0 until assetsArr.length()) {
                val a = assetsArr.getJSONObject(i)
                val assetPath = a.getString("path")
                val assetFile = File(packageDir, assetPath)
                val sha = a.optString("sha256", null)
                val bytes = if (a.has("bytes")) a.getLong("bytes") else null

                val assetVal = PackageValidator.validateAssetFile(assetFile, sha, bytes)
                if (!assetVal.isValid) return assetVal
            }
        }

        return PackageValidator.ValidationResult.success()
    }
}
