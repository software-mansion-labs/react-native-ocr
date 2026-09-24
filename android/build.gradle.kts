plugins {
    id("com.android.library")
    // Kept on the classpath but not applied here: AGP 9 ships built-in Kotlin
    // support and registers the `kotlin` extension itself, so applying the
    // Kotlin plugin on top fails with "Cannot add extension with name
    // 'kotlin'". Applied conditionally below instead.
    id("org.jetbrains.kotlin.android") apply false
    id("com.facebook.react")
}

if (project.extensions.findByName("kotlin") == null) {
    apply(plugin = "org.jetbrains.kotlin.android")
}

/**
 * Reads a variable from the root project (app level) or falls back to a local
 * project property, then to `default`.
 */
fun getExtOrDefault(name: String, default: Any): Any {
    return if (rootProject.ext.has(name)) {
        rootProject.ext.get(name)!!
    } else if (project.properties.containsKey(name)) {
        project.properties[name]!!
    } else {
        default
    }
}

/**
 * OpenCV comes from react-native-executorch: its postinstall downloads the
 * static libraries and headers into its own package directory when the app
 * declares an OpenCV feature (`ocr` does). Resolve that directory the way Node
 * would from here, so hoisted, nested and workspace layouts all work.
 */
fun resolveRneDir(): File {
    val output = providers.exec {
        commandLine(
            "node",
            "--print",
            "require.resolve('react-native-executorch/package.json', { paths: ['${projectDir.absolutePath.replace("\\", "\\\\")}'] })"
        )
    }.standardOutput.asText.get().trim()
    return File(output).parentFile
}

val rneDir = resolveRneDir()

fun requireOpencvArtifacts() {
    val libsDir = File(rneDir, "third-party/android/libs/opencv")
    val present = libsDir.listFiles()
        ?.filter { it.isDirectory && it.resolve("libopencv_core.a").exists() }
        .orEmpty()
    if (present.isNotEmpty()) return

    throw GradleException(
        """
        react-native-ocr needs the OpenCV static libraries that
        react-native-executorch downloads at install time, but none were found in

          ${libsDir.absolutePath}

        Make sure the app's package.json opts into OpenCV:

          "react-native-executorch": { "features": ["ocr"] }

        and that react-native-executorch's postinstall hook ran (pnpm 10+ and
        `--ignore-scripts` skip it):

          node node_modules/react-native-executorch/scripts/download-libs.js
        """.trimIndent()
    )
}

requireOpencvArtifacts()

/**
 * Honor the app's `reactNativeArchitectures` (Expo passes
 * `-PreactNativeArchitectures=arm64-v8a` for device builds), restricted to the
 * ABIs react-native-executorch ships OpenCV for.
 */
fun reactNativeArchitectures(): List<String> {
    val supported = listOf("arm64-v8a", "x86_64")
    val value = rootProject.findProperty("reactNativeArchitectures") as String?
    return value?.split(",")?.map { it.trim() }?.filter { it in supported }
        ?.takeIf { it.isNotEmpty() } ?: supported
}

android {
    namespace = "com.swmansion.rnocr"
    compileSdk = (getExtOrDefault("compileSdkVersion", 34) as Number).toInt()

    defaultConfig {
        // Matches react-native-executorch, whose prebuilt runtime targets API 26.
        minSdk = (getExtOrDefault("minSdkVersion", 26) as Number).toInt()
        targetSdk = (getExtOrDefault("targetSdkVersion", 34) as Number).toInt()
        consumerProguardFiles("consumer-proguard-rules.pro")

        externalNativeBuild {
            cmake {
                cppFlags("-fexceptions", "-frtti", "-std=c++20", "-Wall")
                arguments(
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON",
                    "-DRNE_DIR=${rneDir.absolutePath}"
                )

                abiFilters.addAll(reactNativeArchitectures())
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("CMakeLists.txt")
        }
    }

    sourceSets {
        getByName("main") {
            java.srcDirs("${project.buildDir}/generated/source/codegen/java")
        }
    }

    buildFeatures {
        buildConfig = true
        prefab = true
    }

    packaging {
        // Prevents "Duplicate Library" errors with the React Native JSI engine
        resources.excludes.add("**/libjsi.so")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    // No `kotlinOptions` block: its type-safe accessor is generated only when
    // the Kotlin plugin is applied from the `plugins` block, which it no longer
    // always is. Both AGP 8 and AGP 9 take the Kotlin jvmTarget from
    // `compileOptions` above.
}

repositories {
    google()
    mavenCentral()
}

dependencies {
    implementation("com.facebook.react:react-android")
    implementation("androidx.core:core-ktx:1.12.0")
}

extensions.configure<com.facebook.react.ReactExtension>("react") {
    jsRootDir = file("../src/")
    libraryName = "RnOcr"
    codegenJavaPackageName = "com.swmansion.rnocr"
}
