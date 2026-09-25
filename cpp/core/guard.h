#pragma once

#include <exception>
#include <utility>

#include <jsi/jsi.h>

#include "error.h"

/**
 * The JSI side of error handling: turns the exceptions from `error.h` into
 * coded JavaScript Errors at the host-function boundary.
 */
namespace rnocr::core::error {
namespace jsi = facebook::jsi;

/**
 * Throws `e` into JavaScript as an Error carrying `name` and `code`.
 */
[[noreturn]] void throwJsiError(jsi::Runtime &rt, const RnOcrException &e);

/**
 * Runs `fn`, translating anything it throws into a coded JavaScript Error.
 */
template <typename Fn>
auto guard(jsi::Runtime &rt, Fn &&fn) -> decltype(fn()) {
    try {
        return std::forward<Fn>(fn)();
    } catch (const RnOcrException &e) {
        throwJsiError(rt, e);
    } catch (const jsi::JSError &) {
        // Already a JavaScript value; pass it through untouched.
        throw;
    } catch (const std::exception &e) {
        throwJsiError(rt, Unknown(e.what()));
    } catch (...) {
        throwJsiError(rt, Unknown("Unknown native exception occurred"));
    }
}

/**
 * Wraps a JSI host function so that anything thrown inside it reaches
 * JavaScript as a coded Error. Apply at every `createFromHostFunction` site.
 */
inline jsi::HostFunctionType guarded(jsi::HostFunctionType fn) {
    return [fn = std::move(fn)](jsi::Runtime &rt, const jsi::Value &thisVal, const jsi::Value *args, size_t count) -> jsi::Value {
        return guard(rt, [&] { return fn(rt, thisVal, args, count); });
    };
}

} // namespace rnocr::core::error
