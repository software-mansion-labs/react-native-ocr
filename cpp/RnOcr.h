#pragma once

#include <jsi/jsi.h>

namespace rnocr {
/**
 * Installs the package's native JSI bindings into the global JavaScript object
 * as `__rnocr_jsi__`: an object of host functions that take and return plain
 * ArrayBuffers, plus the `isEmulator` flag.
 *
 * @param jsiRuntime The active JavaScript runtime.
 */
void install(facebook::jsi::Runtime &jsiRuntime);
} // namespace rnocr
