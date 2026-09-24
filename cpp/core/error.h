#pragma once

#include <stdexcept>
#include <string>
#include <utility>

#include <jsi/jsi.h>

namespace rnocr::core::error {
namespace jsi = facebook::jsi;

/**
 * The error codes this package raises. They are a subset of
 * react-native-executorch's `RnExecuTorchErrorCode` union and reach JavaScript
 * with `name: 'RnExecuTorchError'`, so that package's `isRnExecuTorchError`
 * recognises them and its `wrapAsync` carries them across the worklet hop.
 * Callers see one error type across both packages.
 */
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage): X-macro keeps the enum, the string mapping and the factories from drifting
#define FOR_ALL_RNOCR_ERROR_CODES(V)       \
    V(ExecutionFailed, "EXECUTION_FAILED") \
    V(InvalidArgument, "INVALID_ARGUMENT") \
    V(Unknown, "UNKNOWN")

enum class ErrorCode {
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage): helper macro for X-macro expansion
#define DEFINE_ENUM(name, str) name,
    FOR_ALL_RNOCR_ERROR_CODES(DEFINE_ENUM)
#undef DEFINE_ENUM
};

constexpr const char *errorCodeToString(ErrorCode code) {
    switch (code) {
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage): helper macro for X-macro expansion
#define DEFINE_CASE(name, str) \
    case ErrorCode::name:      \
        return str;
        FOR_ALL_RNOCR_ERROR_CODES(DEFINE_CASE)
#undef DEFINE_CASE
    }
    return "UNKNOWN";
}

/**
 * The exception every failure in the native layer is raised as. Native code
 * never throws a jsi::JSError directly; `guarded` is the only place that turns
 * an exception into a JavaScript value, so a code can never be lost on the way
 * out.
 */
class RnOcrException final : public std::runtime_error {
public:
    explicit RnOcrException(ErrorCode code, const std::string &message)
        : std::runtime_error(message), code_(code) {}

    ErrorCode code_;
};

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage): helper macro for X-macro expansion
#define DEFINE_FACTORY(name, str)                              \
    inline RnOcrException name(const std::string &message) {   \
        return RnOcrException(ErrorCode::name, message);       \
    }
FOR_ALL_RNOCR_ERROR_CODES(DEFINE_FACTORY)
#undef DEFINE_FACTORY

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
