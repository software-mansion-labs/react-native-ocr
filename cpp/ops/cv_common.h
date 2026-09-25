#pragma once

#include <cstdint>
#include <exception>
#include <string>
#include <utility>

#include <opencv2/core.hpp>

#include "core/error.h"

/**
 * Shared glue between the raw-buffer kernels and OpenCV. JSI-free, like the
 * kernels themselves.
 */
namespace rnocr::ops {

/**
 * The `CV_8UCn` type for a channel count.
 * @throws core::error::RnOcrException with code InvalidArgument when the count
 * is outside `[1, maxChannels]`.
 */
inline int cvType8U(int32_t channels, const std::string &what, int32_t maxChannels = CV_CN_MAX) {
    if (channels < 1 || channels > maxChannels) {
        throw core::error::InvalidArgument(what + ": unsupported channel count " + std::to_string(channels) +
                                           " (expected 1 to " + std::to_string(maxChannels) + ")");
    }
    return CV_MAKETYPE(CV_8U, channels);
}

/**
 * Runs `fn`, re-raising any OpenCV failure as ExecutionFailed so it reaches
 * JavaScript with a code instead of as an anonymous exception. The package's
 * own exceptions pass through untouched.
 *
 * cv::Exception is caught through its std::exception base on purpose: the
 * prebuilt mobile OpenCV libraries (react-native-executorch's Android
 * archives and the opencv-rne pod) are compiled without RTTI, so a
 * `catch (const cv::Exception &)` needs a typeinfo they do not export and the
 * app fails to link. The host build has RTTI and would not notice.
 */
template <typename Fn>
auto withCv(const char *what, Fn &&fn) -> decltype(fn()) {
    try {
        return std::forward<Fn>(fn)();
    } catch (const core::error::RnOcrException &) {
        throw;
    } catch (const std::exception &e) {
        throw core::error::ExecutionFailed(std::string(what) + ": OpenCV error: " + e.what());
    }
}

} // namespace rnocr::ops
