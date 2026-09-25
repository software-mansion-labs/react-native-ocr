#include "guard.h"

namespace rnocr::core::error {

void throwJsiError(jsi::Runtime &rt, const RnOcrException &e) {
    auto errorCtor = rt.global().getPropertyAsFunction(rt, "Error");
    auto errObj = errorCtor.call(rt, jsi::String::createFromUtf8(rt, e.what())).asObject(rt);

    // The name react-native-executorch's `isRnExecuTorchError` matches on.
    errObj.setProperty(rt, "name", jsi::String::createFromUtf8(rt, "RnExecuTorchError"));
    errObj.setProperty(rt, "code", jsi::String::createFromUtf8(rt, errorCodeToString(e.code_)));

    throw jsi::JSError(rt, jsi::Value(std::move(errObj)));
}

} // namespace rnocr::core::error
