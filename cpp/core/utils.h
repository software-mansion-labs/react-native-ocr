#pragma once

#include <jsi/jsi.h>

namespace rnocr::core::utils {
/**
 * Reports whether the app runs on an Android emulator or an iOS simulator.
 * Mirrors react-native-executorch's private check.
 */
bool isEmulator();

/**
 * Installs the `isEmulator` boolean on the module object.
 *
 * @param rt The active JavaScript runtime.
 * @param module The `__rnocr_jsi__` module object to install onto.
 */
void install_isEmulator(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);
} // namespace rnocr::core::utils
