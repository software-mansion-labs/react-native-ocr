#include "utils.h"

#include <string>

#ifdef __ANDROID__
#include <sys/system_properties.h>
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace rnocr::core::utils {
namespace jsi = facebook::jsi;

// On Android no single property covers every image, so three are checked: the
// build fingerprint (`generic...` for AOSP images), the hardware name
// (`goldfish`/`ranchu` are the QEMU emulator kernels, `cutf`/`vsoc` prefixes are
// Cuttlefish virtual devices), and the product model (Play-store and SDK images
// report `sdk_gphone`, `google_sdk`, `Emulator` or `Cuttlefish`). On Apple
// platforms the simulator is known at compile time.
bool isEmulator() {
#ifdef __ANDROID__
    auto readProp = [](const char *key) -> std::string {
#if __ANDROID_API__ >= 26
        const prop_info *pi = __system_property_find(key);
        if (pi == nullptr) {
            return "";
        }
        std::string result;
        __system_property_read_callback(
            pi,
            [](void *cookie, const char * /*name*/, const char *value, uint32_t /*serial*/) {
                *static_cast<std::string *>(cookie) = value;
            },
            &result);
        return result;
#else
        char value[PROP_VALUE_MAX] = {0};
        __system_property_get(key, value);
        return {value};
#endif
    };

    const auto startsWith = [](const std::string &value, const char *prefix) {
        return value.rfind(prefix, 0) == 0;
    };

    const std::string fingerprint = readProp("ro.build.fingerprint");
    if (startsWith(fingerprint, "generic") || startsWith(fingerprint, "unknown")) {
        return true;
    }

    const std::string hardware = readProp("ro.hardware");
    if (hardware == "goldfish" || hardware == "ranchu" || startsWith(hardware, "cutf") ||
        startsWith(hardware, "vsoc")) {
        return true;
    }

    const std::string model = readProp("ro.product.model");
    return model.find("sdk_gphone") != std::string::npos ||
           model.find("google_sdk") != std::string::npos ||
           model.find("Emulator") != std::string::npos ||
           model.find("Android SDK built for") != std::string::npos ||
           model.find("Cuttlefish") != std::string::npos;
#elif defined(__APPLE__) && TARGET_OS_SIMULATOR
    return true;
#else
    return false;
#endif
}

void install_isEmulator(jsi::Runtime &rt, jsi::Object &module) {
    module.setProperty(rt, "isEmulator", jsi::Value(isEmulator()));
}
} // namespace rnocr::core::utils
