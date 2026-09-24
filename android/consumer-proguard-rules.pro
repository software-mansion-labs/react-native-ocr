# The JNI entry point is resolved by its mangled name from C++; keep the
# class and its native methods from being renamed or stripped.
-keepclasseswithmembers class com.swmansion.rnocr.RnOcrModule {
    native <methods>;
}
