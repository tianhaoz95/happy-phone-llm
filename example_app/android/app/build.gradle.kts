plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.example_app"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.example.example_app"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
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
}

flutter {
    source = "../.."
}

// Get NDK path from local.properties or environment variable
val androidNdkPath = project.properties["android.ndkPath"] ?: System.getenv("ANDROID_NDK_HOME")
if (androidNdkPath == null) {
    throw GradleException("ANDROID_NDK_HOME environment variable or android.ndkPath in local.properties not set.")
}

// Path to the C++ build directory
val cppBuildDir = project.file("../../../inference")

// Define the ABIs to support
val abis = listOf("arm64-v8a", "armeabi-v7a", "x86", "x86_64")

abis.forEach { abi ->
    val buildDirForAbi = cppBuildDir.resolve("build-android-$abi")
    val jniLibsDir = file("src/main/jniLibs/$abi")

    // Task to build the C++ library for a specific ABI
    tasks.register<Exec>("buildCppForAndroid$abi") {
        workingDir(cppBuildDir)
        // Configure CMake
        commandLine("cmake",
                    "-S", cppBuildDir.absolutePath,
                    "-B", buildDirForAbi.absolutePath,
                    "-DCMAKE_TOOLCHAIN_FILE=$androidNdkPath/build/cmake/android.toolchain.cmake",
                    "-DANDROID_ABI=$abi",
                    "-DANDROID_PLATFORM=android-21",
                    "-DCMAKE_BUILD_TYPE=Release")
        // Build CMake project
        doLast {
            exec {
                workingDir(buildDirForAbi)
                commandLine("cmake", "--build", ".")
            }
        }
        // Only run if the build directory doesn't exist or CMakeLists.txt has changed
        inputs.file("../../../inference/CMakeLists.txt")
        outputs.dir(buildDirForAbi)
    }

    // Task to copy the shared library for Android
    tasks.register<Copy>("copyCppSharedLibrary$abi") {
        dependsOn("buildCppForAndroid$abi") // Ensure C++ library is built first
        from(buildDirForAbi.resolve("engine")) {
            include("libhappy_phone_llm_engine.so")
        }
        into(jniLibsDir)
    }
    tasks.getByName("preBuild").dependsOn("copyCppSharedLibrary$abi")
}
