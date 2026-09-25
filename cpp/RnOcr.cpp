#include "RnOcr.h"

#include "core/utils.h"
#include "ops/bindings.h"

namespace jsi = facebook::jsi;

namespace rnocr {
void install(jsi::Runtime &jsiRuntime) {
    jsi::Object module = jsi::Object(jsiRuntime);

    core::utils::install_isEmulator(jsiRuntime, module);
    ops::installOps(jsiRuntime, module);

    jsiRuntime.global().setProperty(jsiRuntime, "__rnocr_jsi__", std::move(module));
}
} // namespace rnocr
