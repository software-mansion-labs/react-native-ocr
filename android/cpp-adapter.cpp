#include "RnOcr.h"
#include <jni.h>
#include <jsi/jsi.h>

extern "C" JNIEXPORT void JNICALL Java_com_swmansion_rnocr_RnOcrModule_nativeInstall(JNIEnv * /*env*/, jclass /*clazz*/, jlong jsi) {
    auto *runtime = reinterpret_cast<facebook::jsi::Runtime *>(jsi);
    rnocr::install(*runtime);
}
