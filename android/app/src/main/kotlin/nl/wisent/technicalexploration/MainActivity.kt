package nl.wisent.technicalexploration

import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File

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
        private val LEGACY_TEMP_PATHS = listOf(
            "/sys/class/power_supply/battery/temp",
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/devices/virtual/thermal/thermal_zone0/temp",
        )
    }
}
