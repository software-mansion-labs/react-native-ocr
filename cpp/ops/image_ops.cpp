#include "image_ops.h"

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "core/buffers.h"
#include "core/error.h"

namespace rnocr::ops::image {
namespace jsi = facebook::jsi;
using namespace rnocr::core;

namespace {
int cvType(int32_t channels, const std::string &what) {
    if (channels < 1 || channels > CV_CN_MAX) {
        throw error::InvalidArgument(what + ": unsupported channel count " + std::to_string(channels));
    }
    return CV_MAKETYPE(CV_8U, channels);
}
} // namespace

void rotate(const uint8_t *src, int32_t height, int32_t width, int32_t channels, int32_t degreesClockwise, uint8_t *dst) {
    int rotateCode = 0;
    if (degreesClockwise == 90) {
        rotateCode = ::cv::ROTATE_90_CLOCKWISE;
    } else if (degreesClockwise == 180) {
        rotateCode = ::cv::ROTATE_180;
    } else if (degreesClockwise == 270) {
        rotateCode = ::cv::ROTATE_90_COUNTERCLOCKWISE;
    } else {
        throw error::InvalidArgument("rotate: degreesClockwise must be 90, 180 or 270");
    }

    const bool swap = degreesClockwise != 180;
    const int type = cvType(channels, "rotate");
    // Non-owning views: the caller's buffers are read and written in place.
    const ::cv::Mat srcMat(height, width, type, const_cast<uint8_t *>(src));
    ::cv::Mat dstMat(swap ? width : height, swap ? height : width, type, dst);
    ::cv::rotate(srcMat, dstMat, rotateCode);
}

void install_rotate(jsi::Runtime &rt, jsi::Object &module) {
    const auto *name = "rotate";
    auto fnBody = [](jsi::Runtime &rt, const jsi::Value & /*thisVal*/, const jsi::Value *args, size_t count) -> jsi::Value {
        buffers::requireArgCount(count, 5, "rotate(src, height, width, channels, degreesClockwise)");
        const int32_t height = buffers::requirePositiveInt(rt, args[1], "rotate: height");
        const int32_t width = buffers::requirePositiveInt(rt, args[2], "rotate: width");
        const int32_t channels = buffers::requirePositiveInt(rt, args[3], "rotate: channels");
        const int32_t degrees = buffers::requireNonNegativeInt(rt, args[4], "rotate: degreesClockwise");

        const size_t bytes = buffers::imageBytes(height, width, channels, "rotate");
        const buffers::ByteView src = buffers::requireBuffer(rt, args[0], "rotate: src", bytes);

        std::vector<uint8_t> out(bytes);
        rotate(src.data, height, width, channels, degrees, out.data());
        return buffers::makeArrayBuffer(rt, std::move(out));
    };
    module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), 5, error::guarded(fnBody)));
}

} // namespace rnocr::ops::image
