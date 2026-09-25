#pragma once

#include <stdexcept>
#include <string>

/**
 * Error codes and the exception type shared by every native kernel.
 *
 * This header is deliberately free of JSI so the kernels under `cpp/ops/` can
 * be compiled and unit-tested on the host; `guard.h` holds the half that turns
 * an exception into a JavaScript value.
 */
namespace rnocr::core::error {

/**
 * The error codes this package raises. They are a subset of
 * react-native-executorch's `RnExecuTorchErrorCode` union and reach JavaScript
 * with `name: 'RnExecuTorchError'`, so that package's `isRnExecuTorchError`
 * recognizes them and its `wrapAsync` carries them across the worklet hop.
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
 * never throws a jsi::JSError directly; `guard` (guard.h) is the only place
 * that turns an exception into a JavaScript value, so a code can never be lost
 * on the way out.
 */
class RnOcrException final : public std::runtime_error {
public:
    explicit RnOcrException(ErrorCode code, const std::string &message)
        : std::runtime_error(message), code_(code) {}

    ErrorCode code_;
};

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage): helper macro for X-macro expansion
#define DEFINE_FACTORY(name, str)                            \
    inline RnOcrException name(const std::string &message) { \
        return RnOcrException(ErrorCode::name, message);     \
    }
FOR_ALL_RNOCR_ERROR_CODES(DEFINE_FACTORY)
#undef DEFINE_FACTORY

} // namespace rnocr::core::error
