import java.util.Base64

plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

private fun Project.decodedDartDefines(): Map<String, String> {
    val encoded = findProperty("dart-defines")?.toString().orEmpty()
    if (encoded.isEmpty()) return emptyMap()
    return encoded.split(",")
        .mapNotNull { token ->
            val decoded = try {
                String(Base64.getDecoder().decode(token.trim()), Charsets.UTF_8)
            } catch (_: IllegalArgumentException) {
                return@mapNotNull null
            }
            val eq = decoded.indexOf('=')
            if (eq <= 0) null else decoded.substring(0, eq) to decoded.substring(eq + 1)
        }
        .toMap()
}

val dartDefines = decodedDartDefines()
val excludeArmv92 = dartDefines["EXCLUDE_ARMV92"] == "1"
val kleidiSmeOff = dartDefines["GGML_KLEIDIAI_SME"] == "0"
val cpuArmv8 = dartDefines["CPU_LIBS"] == "armv8"
val threads4 = dartDefines["N_THREADS"] == "4"
val noMmap = dartDefines["NO_MMAP"] == "true"
val vulkan99 = dartDefines["N_GPU_LAYERS"] == "99"

android {
    namespace = "nl.wisent.technicalexploration"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        applicationId = "nl.wisent.technicalexploration"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        // Uses the version code from pubspec.yaml. When using split APKs, 1000 * ABI_VERSION
        // is added automatically by Flutter. (https://developer.android.com/studio/build/configure-apk-splits#configure-APK-versions)
        // You can force using the value of versionCode by specifying the `-P force-version-code-ignoring-abi=true`
        // flag during build.
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
            if (excludeArmv92) {
                excludes += "**/libggml-cpu-android_armv9.2_*.so"
            }
            if (cpuArmv8) {
                excludes += "**/libggml-cpu-android_armv8.6_*.so"
                excludes += "**/libggml-cpu-android_armv9.0_*.so"
                excludes += "**/libggml-cpu-android_armv9.2_*.so"
            }
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}

// Flutter always copies the packaged APK to app-release.apk. Keep a second
// copy under a name that a later variant build will not overwrite.
afterEvaluate {
    tasks.named("assembleRelease").configure {
        doLast {
            val destName = when {
                cpuArmv8 && threads4 -> "app-armv8-threads4-release.apk"
                cpuArmv8 -> "app-armv8-release.apk"
                noMmap -> "app-nommap-release.apk"
                vulkan99 -> "app-vulkan99-release.apk"
                excludeArmv92 -> "app-no-armv92-release.apk"
                kleidiSmeOff -> "app-sme0-release.apk"
                else -> return@doLast
            }
            val apkDir = layout.buildDirectory.dir("outputs/flutter-apk").get().asFile
            val src = File(apkDir, "app-release.apk")
            if (!src.exists()) return@doLast
            src.copyTo(File(apkDir, destName), overwrite = true)
        }
    }
}
