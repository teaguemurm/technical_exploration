package nl.wisent.technicalexploration

import android.content.ClipData
import android.content.ContentUris
import android.content.ContentValues
import android.content.Intent
import android.content.IntentFilter
import android.net.Uri
import android.os.BatteryManager
import android.os.Build
import android.provider.MediaStore
import androidx.annotation.RequiresApi
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.io.IOException

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                try {
                    when (call.method) {
                        "getDeviceIdentity" -> result.success(deviceIdentity())
                        "getBattery" -> result.success(readBattery())
                        "getThermal" -> result.success(readThermal())
                        "publishDownloads" -> result.success(publishDownloads(call))
                        "shareDownloads" -> {
                            shareDownloads(call)
                            result.success(null)
                        }
                        else -> result.notImplemented()
                    }
                } catch (error: Exception) {
                    result.error("device", error.message, null)
                }
            }
    }

    private fun deviceIdentity(): Map<String, String> {
        val market = firstNonEmpty(
            systemProp("ro.product.marketname"),
            systemProp("ro.product.vendor.marketname"),
        )
        return mapOf(
            "manufacturer" to (Build.MANUFACTURER ?: ""),
            "model" to (Build.MODEL ?: ""),
            "device" to (Build.DEVICE ?: ""),
            "product" to (Build.PRODUCT ?: ""),
            "marketName" to market,
        )
    }

    @Suppress("DEPRECATION")
    private fun readBattery(): Map<String, Any?> {
        val intent = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            ?: return mapOf("tempTenths" to null, "level" to null, "scale" to null)
        val temp = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
        val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        return mapOf(
            "tempTenths" to if (temp == Int.MIN_VALUE) null else temp,
            "level" to level,
            "scale" to scale,
        )
    }

    private fun readThermal(): Map<String, Any?> {
        val zones = mutableListOf<Map<String, String>>()
        val dir = File("/sys/class/thermal")
        if (dir.isDirectory) {
            val files = dir.listFiles()?.sortedBy { it.name } ?: emptyList()
            for (zone in files) {
                if (!zone.name.startsWith("thermal_zone")) continue
                val raw = readText(File(zone, "temp")) ?: continue
                zones.add(
                    mapOf(
                        "name" to zone.name,
                        "type" to (readText(File(zone, "type")) ?: ""),
                        "raw" to raw,
                    ),
                )
            }
        }

        var legacy: Map<String, String>? = null
        for (path in LEGACY_TEMP_PATHS) {
            val raw = readText(File(path)) ?: continue
            val type = if (path.contains("power_supply")) {
                "power_supply"
            } else {
                readText(File(File(path).parentFile, "type")) ?: ""
            }
            legacy = mapOf("path" to path, "type" to type, "raw" to raw)
            break
        }
        return mapOf("zones" to zones, "legacy" to legacy)
    }

    private fun publishDownloads(call: MethodCall): Map<String, Any?> {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            throw IllegalStateException(
                "Public downloads need Android 10 (API 29) or newer",
            )
        }
        val args = call.arguments as? Map<*, *>
            ?: throw IllegalArgumentException("missing files")
        val files = args["files"] as? List<*>
            ?: throw IllegalArgumentException("missing files")
        if (files.isEmpty()) throw IllegalArgumentException("missing files")
        val published = files.map { entry ->
            val map = entry as? Map<*, *>
                ?: throw IllegalArgumentException("bad file entry")
            val path = map["path"] as? String
                ?: throw IllegalArgumentException("missing path")
            val name = map["name"] as? String ?: File(path).name
            val mime = map["mime"] as? String ?: "application/octet-stream"
            val replace = map["replace"] as? Boolean ?: false
            publishOne(File(path), name, mime, replace)
        }
        return mapOf(
            "directory" to PUBLIC_DIRECTORY,
            "files" to published,
        )
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun publishOne(
        source: File,
        requestedName: String,
        mime: String,
        replace: Boolean,
    ): Map<String, String> {
        if (!source.isFile) {
            throw IllegalArgumentException("missing file ${source.path}")
        }
        if (replace) {
            val existing = findOwnDownload(requestedName)
            if (existing != null) {
                return finishPublish(source, existing, requestedName, overwrite = true)
            }
        }
        val name = if (replace) requestedName else uniqueDownloadName(requestedName)
        val uri = try {
            insertPending(name, mime)
        } catch (error: Exception) {
            if (replace) throw error
            val alt = uniqueDownloadName(forcedSuffix(name))
            val retry = try {
                insertPending(alt, mime)
            } catch (second: Exception) {
                second.addSuppressed(error)
                throw second
            }
            return finishPublish(source, retry, alt, overwrite = false)
        }
        return finishPublish(source, uri, name, overwrite = false)
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun finishPublish(
        source: File,
        uri: Uri,
        name: String,
        overwrite: Boolean,
    ): Map<String, String> {
        try {
            if (overwrite) markPending(uri, true)
            writeAll(source, uri)
            markPending(uri, false)
        } catch (error: Exception) {
            contentResolver.delete(uri, null, null)
            throw error
        }
        val actual = displayName(uri) ?: name
        if (overwrite) deleteOtherOwnDownloads(actual, uri)
        return mapOf("name" to actual, "uri" to uri.toString())
    }

    private fun forcedSuffix(name: String): String {
        val dot = name.lastIndexOf('.')
        val stem = if (dot > 0) name.substring(0, dot) else name
        val ext = if (dot > 0) name.substring(dot) else ""
        return "$stem (2)$ext"
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun uniqueDownloadName(name: String): String {
        if (findOwnDownload(name) == null) return name
        val dot = name.lastIndexOf('.')
        val stem = if (dot > 0) name.substring(0, dot) else name
        val ext = if (dot > 0) name.substring(dot) else ""
        for (n in 2..100) {
            val candidate = "$stem ($n)$ext"
            if (findOwnDownload(candidate) == null) return candidate
        }
        throw IOException("could not find a free Download name for $name")
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun insertPending(name: String, mime: String): Uri {
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, mime)
            put(MediaStore.MediaColumns.RELATIVE_PATH, RELATIVE_PATH)
            put(MediaStore.MediaColumns.IS_PENDING, 1)
        }
        return contentResolver.insert(downloadsCollection(), values)
            ?: throw IOException("could not create $name in $PUBLIC_DIRECTORY")
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun markPending(uri: Uri, pending: Boolean) {
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.IS_PENDING, if (pending) 1 else 0)
        }
        val updated = contentResolver.update(uri, values, null, null)
        if (updated != 1) throw IOException("could not publish ${uri}")
    }

    private fun writeAll(source: File, uri: Uri) {
        val expected = source.length()
        val stream = contentResolver.openOutputStream(uri)
            ?: throw IOException("no output stream for ${source.name}")
        stream.use { out ->
            source.inputStream().use { input ->
                val buffer = ByteArray(64 * 1024)
                var total = 0L
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    if (read == 0) continue
                    out.write(buffer, 0, read)
                    total += read
                }
                out.flush()
                if (total != expected) {
                    throw IOException("short write of ${source.name}: $total of $expected")
                }
            }
        }
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun findOwnDownload(name: String): Uri? {
        val projection = arrayOf(MediaStore.MediaColumns._ID)
        val selection = "${MediaStore.MediaColumns.DISPLAY_NAME}=? AND " +
            "${MediaStore.MediaColumns.RELATIVE_PATH}=?"
        for (relative in RELATIVE_PATHS) {
            contentResolver.query(
                downloadsCollection(),
                projection,
                selection,
                arrayOf(name, relative),
                null,
            )?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val id = cursor.getLong(0)
                    return ContentUris.withAppendedId(downloadsCollection(), id)
                }
            }
        }
        return null
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun deleteOtherOwnDownloads(name: String, keep: Uri) {
        val projection = arrayOf(MediaStore.MediaColumns._ID)
        val selection = "${MediaStore.MediaColumns.DISPLAY_NAME}=? AND " +
            "${MediaStore.MediaColumns.RELATIVE_PATH}=?"
        for (relative in RELATIVE_PATHS) {
            contentResolver.query(
                downloadsCollection(),
                projection,
                selection,
                arrayOf(name, relative),
                null,
            )?.use { cursor ->
                val idColumn = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns._ID)
                while (cursor.moveToNext()) {
                    val uri = ContentUris.withAppendedId(
                        downloadsCollection(),
                        cursor.getLong(idColumn),
                    )
                    if (uri != keep) contentResolver.delete(uri, null, null)
                }
            }
        }
    }

    private fun displayName(uri: Uri): String? {
        contentResolver.query(
            uri,
            arrayOf(MediaStore.MediaColumns.DISPLAY_NAME),
            null,
            null,
            null,
        )?.use { cursor ->
            if (cursor.moveToFirst()) return cursor.getString(0)
        }
        return null
    }

    @RequiresApi(Build.VERSION_CODES.Q)
    private fun downloadsCollection(): Uri {
        return MediaStore.Downloads.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
    }

    private fun shareDownloads(call: MethodCall) {
        val args = call.arguments as? Map<*, *>
            ?: throw IllegalArgumentException("missing uris")
        val uris = (args["uris"] as? List<*>)
            ?.mapNotNull { (it as? String)?.let(Uri::parse) }
            ?.filter { it.toString().isNotEmpty() }
            ?: emptyList()
        if (uris.isEmpty()) throw IllegalArgumentException("missing uris")
        val send = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
            type = "*/*"
            putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList(uris))
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            clipData = ClipData.newUri(contentResolver, "export", uris.first()).also { clip ->
                for (uri in uris.drop(1)) clip.addItem(ClipData.Item(uri))
            }
        }
        val chooser = Intent.createChooser(send, "Share benchmark exports")
        startActivity(chooser)
    }

    private fun systemProp(key: String): String {
        return try {
            val clazz = Class.forName("android.os.SystemProperties")
            val get = clazz.getMethod("get", String::class.java, String::class.java)
            (get.invoke(null, key, "") as? String)?.trim().orEmpty()
        } catch (_: Throwable) {
            ""
        }
    }

    private fun firstNonEmpty(vararg values: String): String {
        for (value in values) {
            if (value.isNotEmpty()) return value
        }
        return ""
    }

    private fun readText(file: File): String? {
        return try {
            if (!file.exists() || !file.canRead()) null else file.readText().trim()
        } catch (_: Exception) {
            null
        }
    }

    companion object {
        private const val CHANNEL = "nl.wisent.technicalexploration/device"
        private const val PUBLIC_DIRECTORY = "Download/technical-exploration"
        private const val RELATIVE_PATH = "$PUBLIC_DIRECTORY/"
        private val RELATIVE_PATHS = listOf(RELATIVE_PATH, PUBLIC_DIRECTORY)
        private val LEGACY_TEMP_PATHS = listOf(
            "/sys/class/power_supply/battery/temp",
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/devices/virtual/thermal/thermal_zone0/temp",
        )
    }
}
