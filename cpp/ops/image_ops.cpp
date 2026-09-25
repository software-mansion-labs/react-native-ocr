#include "image_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "core/error.h"
#include "cv_common.h"

namespace rnocr::ops::image {
namespace error = rnocr::core::error;

namespace {
// cv::mean, Scalar fills and cv::remap handle at most four channels.
constexpr int32_t kMaxScalarChannels = 4;

void requireDims(int32_t height, int32_t width, const std::string &what) {
    if (height <= 0 || width <= 0) {
        throw error::InvalidArgument(what + ": height and width must be positive");
    }
}

int interpolationFlag(Interpolation interpolation) {
    switch (interpolation) {
    case Interpolation::Nearest:
        return ::cv::INTER_NEAREST;
    case Interpolation::Linear:
        return ::cv::INTER_LINEAR;
    case Interpolation::Cubic:
        return ::cv::INTER_CUBIC;
    }
    return ::cv::INTER_LINEAR;
}

int borderFlag(BorderMode mode) {
    return mode == BorderMode::Constant ? ::cv::BORDER_CONSTANT : ::cv::BORDER_REPLICATE;
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
    requireDims(height, width, "rotate");
    const int type = cvType8U(channels, "rotate");

    const bool swap = degreesClockwise != 180;
    // Non-owning views: the caller's buffers are read and written in place.
    const ::cv::Mat srcMat(height, width, type, const_cast<uint8_t *>(src));
    ::cv::Mat dstMat(swap ? width : height, swap ? height : width, type, dst);
    withCv("rotate", [&] { ::cv::rotate(srcMat, dstMat, rotateCode); });
}

void crop(const uint8_t *src, int32_t height, int32_t width, int32_t channels, int32_t x0, int32_t y0, int32_t x1, int32_t y1, uint8_t *dst) {
    requireDims(height, width, "crop");
    const int type = cvType8U(channels, "crop");
    if (x0 < 0 || y0 < 0 || x1 > width || y1 > height) {
        throw error::InvalidArgument("crop: box [" + std::to_string(x0) + ", " + std::to_string(y0) + ", " +
                                     std::to_string(x1) + ", " + std::to_string(y1) + "] exceeds the " +
                                     std::to_string(width) + "x" + std::to_string(height) + " image");
    }
    if (x1 <= x0 || y1 <= y0) {
        throw error::InvalidArgument("crop: box must have positive width and height");
    }

    const ::cv::Mat srcMat(height, width, type, const_cast<uint8_t *>(src));
    ::cv::Mat dstMat(y1 - y0, x1 - x0, type, dst);
    withCv("crop", [&] { srcMat(::cv::Rect(x0, y0, x1 - x0, y1 - y0)).copyTo(dstMat); });
}

void warpByGrid(const uint8_t *src, int32_t srcHeight, int32_t srcWidth, int32_t channels,
                const float *grid, int32_t gridHeight, int32_t gridWidth,
                int32_t dstHeight, int32_t dstWidth, const WarpOptions &options, uint8_t *dst) {
    requireDims(srcHeight, srcWidth, "warpByGrid: src");
    requireDims(gridHeight, gridWidth, "warpByGrid: grid");
    requireDims(dstHeight, dstWidth, "warpByGrid: dst");
    const int type = cvType8U(channels, "warpByGrid", kMaxScalarChannels);
    if (!std::isfinite(options.borderValue)) {
        throw error::InvalidArgument("warpByGrid: borderValue must be finite");
    }

    const int32_t plane = gridHeight * gridWidth;
    // Bilinearly samples channel `c` of the coarse grid at fractional (gx, gy).
    const auto sampleGrid = [&](int32_t c, float gx, float gy) -> float {
        const int32_t x0 = std::clamp(static_cast<int32_t>(std::floor(gx)), 0, gridWidth - 1);
        const int32_t y0 = std::clamp(static_cast<int32_t>(std::floor(gy)), 0, gridHeight - 1);
        const int32_t x1 = std::min(x0 + 1, gridWidth - 1);
        const int32_t y1 = std::min(y0 + 1, gridHeight - 1);
        const float dx = gx - static_cast<float>(x0);
        const float dy = gy - static_cast<float>(y0);
        const float *g = grid + c * plane;
        const float top = g[y0 * gridWidth + x0] + (g[y0 * gridWidth + x1] - g[y0 * gridWidth + x0]) * dx;
        const float bottom = g[y1 * gridWidth + x0] + (g[y1 * gridWidth + x1] - g[y1 * gridWidth + x0]) * dx;
        return top + (bottom - top) * dy;
    };

    // Grid positions are spread over the destination; the sampled values are
    // denormalized against the source, which is what align_corners=true means
    // for a backward field.
    ::cv::Mat mapX(dstHeight, dstWidth, CV_32F);
    ::cv::Mat mapY(dstHeight, dstWidth, CV_32F);
    const float gridStepY = dstHeight > 1 ? static_cast<float>(gridHeight - 1) / static_cast<float>(dstHeight - 1) : 0.0f;
    const float gridStepX = dstWidth > 1 ? static_cast<float>(gridWidth - 1) / static_cast<float>(dstWidth - 1) : 0.0f;
    const float srcSpanX = static_cast<float>(srcWidth - 1);
    const float srcSpanY = static_cast<float>(srcHeight - 1);
    for (int32_t oy = 0; oy < dstHeight; ++oy) {
        const float gy = static_cast<float>(oy) * gridStepY;
        auto *rowX = mapX.ptr<float>(oy);
        auto *rowY = mapY.ptr<float>(oy);
        for (int32_t ox = 0; ox < dstWidth; ++ox) {
            const float gx = static_cast<float>(ox) * gridStepX;
            const float nx = sampleGrid(0, gx, gy);
            const float ny = sampleGrid(1, gx, gy);
            rowX[ox] = ((nx + 1.0f) / 2.0f) * srcSpanX;
            rowY[ox] = ((ny + 1.0f) / 2.0f) * srcSpanY;
        }
    }

    const ::cv::Mat srcMat(srcHeight, srcWidth, type, const_cast<uint8_t *>(src));
    ::cv::Mat dstMat(dstHeight, dstWidth, type, dst);
    withCv("warpByGrid", [&] {
        ::cv::remap(srcMat, dstMat, mapX, mapY, interpolationFlag(options.interpolation), borderFlag(options.borderMode),
                    ::cv::Scalar::all(options.borderValue));
    });
}

void padCornerMean(const uint8_t *strip, int32_t height, int32_t width, int32_t channels, int32_t targetWidth, uint8_t *dst) {
    requireDims(height, width, "padCornerMean");
    const int type = cvType8U(channels, "padCornerMean", kMaxScalarChannels);
    if (targetWidth < width) {
        throw error::InvalidArgument("padCornerMean: targetWidth (" + std::to_string(targetWidth) +
                                     ") must be at least the strip width (" + std::to_string(width) + ")");
    }

    const ::cv::Mat content(height, width, type, const_cast<uint8_t *>(strip));
    ::cv::Mat dstMat(height, targetWidth, type, dst);
    withCv("padCornerMean", [&] {
        // Patches of a thirtieth of the short side, as the original rectifyQuad
        // did; always at least one pixel and never larger than the strip.
        const int patch = std::max(1, std::min(height, width) / 30);
        const std::array<::cv::Rect, 4> corners = {
            ::cv::Rect(0, 0, patch, patch),
            ::cv::Rect(width - patch, 0, patch, patch),
            ::cv::Rect(0, height - patch, patch, patch),
            ::cv::Rect(width - patch, height - patch, patch, patch),
        };
        ::cv::Scalar mean(0, 0, 0, 0);
        for (const auto &corner : corners) {
            mean += ::cv::mean(content(corner));
        }
        mean /= 4.0;

        if (targetWidth > width) {
            dstMat(::cv::Rect(width, 0, targetWidth - width, height)).setTo(mean);
        }
        content.copyTo(dstMat(::cv::Rect(0, 0, width, height)));
    });
}

} // namespace rnocr::ops::image
