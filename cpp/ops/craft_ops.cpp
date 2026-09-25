#include "craft_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
// OpenCV 5 moved minAreaRect out of imgproc into the geometry module; the
// 4.x headers the device builds use still have it in imgproc.
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif

#include "core/error.h"
#include "cv_common.h"

namespace rnocr::ops::craft {
namespace error = rnocr::core::error;

namespace {
// All per-component work stays inside the component's (dilation-padded) rect;
// full-frame masks per component would make the decode O(components * H * W).
std::optional<CraftBox> boxFromComponent(const ::cv::Mat &textMap, const ::cv::Mat &labels, const ::cv::Mat &stats, int32_t label,
                                         int32_t imgWidth, int32_t imgHeight, float lowTextThreshold, int32_t minArea) {
    const int32_t area = stats.at<int32_t>(label, ::cv::CC_STAT_AREA);
    if (area < minArea) {
        return std::nullopt;
    }
    const int32_t x = stats.at<int32_t>(label, ::cv::CC_STAT_LEFT);
    const int32_t y = stats.at<int32_t>(label, ::cv::CC_STAT_TOP);
    const int32_t w = stats.at<int32_t>(label, ::cv::CC_STAT_WIDTH);
    const int32_t h = stats.at<int32_t>(label, ::cv::CC_STAT_HEIGHT);
    const ::cv::Rect compRect(x, y, w, h);
    const ::cv::Mat mask = (labels(compRect) == label);
    double maxVal = 0.0;
    ::cv::minMaxLoc(textMap(compRect), nullptr, &maxVal, nullptr, nullptr, mask);
    if (maxVal < static_cast<double>(lowTextThreshold)) {
        return std::nullopt;
    }

    // Grow the component by a stroke-width estimate so the box covers the
    // glyph's full extent, not only the pixels above threshold.
    const auto dilationRadius = static_cast<int32_t>(std::sqrt(static_cast<double>(area) / std::max(w, h)) * 2);
    const int32_t sx = std::max(x - dilationRadius, 0);
    const int32_t ex = std::min(x + w + dilationRadius, imgWidth);
    const int32_t sy = std::max(y - dilationRadius, 0);
    const int32_t ey = std::min(y + h + dilationRadius, imgHeight);
    ::cv::Mat segMap = ::cv::Mat::zeros(ey - sy, ex - sx, CV_8U);
    segMap(::cv::Rect(x - sx, y - sy, w, h)).setTo(255, mask);
    const int32_t kernelSize = 1 + dilationRadius;
    const ::cv::Mat kernel = ::cv::getStructuringElement(::cv::MORPH_RECT, ::cv::Size(kernelSize, kernelSize));
    ::cv::dilate(segMap, segMap, kernel, ::cv::Point(-1, -1), 1);

    std::vector<std::vector<::cv::Point>> contours;
    // The offset restores full-frame coordinates for the ROI-local contours.
    ::cv::findContours(segMap, contours, ::cv::RETR_EXTERNAL, ::cv::CHAIN_APPROX_SIMPLE, ::cv::Point(sx, sy));
    if (contours.empty()) {
        return std::nullopt;
    }
    const ::cv::RotatedRect rotated = ::cv::minAreaRect(contours[0]);
    std::array<::cv::Point2f, 4> corners;
    rotated.points(corners.data());
    // The dilated component is clipped to the map, so the fitted rectangle can
    // only leave it by floating-point noise; clamp that away.
    const auto width = static_cast<float>(imgWidth);
    const auto height = static_cast<float>(imgHeight);
    CraftBox box;
    box.x0 = std::clamp(std::min({corners[0].x, corners[1].x, corners[2].x, corners[3].x}), 0.0f, width);
    box.y0 = std::clamp(std::min({corners[0].y, corners[1].y, corners[2].y, corners[3].y}), 0.0f, height);
    box.x1 = std::clamp(std::max({corners[0].x, corners[1].x, corners[2].x, corners[3].x}), 0.0f, width);
    box.y1 = std::clamp(std::max({corners[0].y, corners[1].y, corners[2].y, corners[3].y}), 0.0f, height);
    box.angle = rotated.angle;
    return box;
}

std::vector<CraftBox> componentBoxes(const ::cv::Mat &textMap, const ::cv::Mat &affinityMap, const CraftOptions &options) {
    const int32_t imgHeight = textMap.rows;
    const int32_t imgWidth = textMap.cols;
    ::cv::Mat textScore;
    ::cv::Mat affinityScore;
    ::cv::threshold(textMap, textScore, static_cast<double>(options.textThreshold), 1.0, ::cv::THRESH_BINARY);
    ::cv::threshold(affinityMap, affinityScore, static_cast<double>(options.linkThreshold), 1.0, ::cv::THRESH_BINARY);

    ::cv::Mat combined;
    if (options.charLevel) {
        // Subtracting the affinity separates adjacent glyphs; the erode/dilate
        // pass then heals each glyph into one solid component.
        combined = textScore - affinityScore;
        ::cv::threshold(combined, combined, 0.0, 1.0, ::cv::THRESH_TOZERO);
        ::cv::threshold(combined, combined, 1.0, 1.0, ::cv::THRESH_TRUNC);
        const ::cv::Mat kernel = ::cv::getStructuringElement(::cv::MORPH_RECT, ::cv::Size(3, 3));
        ::cv::erode(combined, combined, kernel, ::cv::Point(-1, -1), 1);
        ::cv::dilate(combined, combined, kernel, ::cv::Point(-1, -1), 4);
    } else {
        // Adding the affinity links adjacent glyphs into one line component.
        combined = textScore + affinityScore;
        ::cv::threshold(combined, combined, 0.0, 1.0, ::cv::THRESH_BINARY);
    }

    ::cv::Mat binary;
    combined.convertTo(binary, CV_8UC1);
    ::cv::Mat labels;
    ::cv::Mat stats;
    ::cv::Mat centroids;
    const int32_t labelCount = ::cv::connectedComponentsWithStats(binary, labels, stats, centroids, 4);

    std::vector<CraftBox> boxes;
    boxes.reserve(static_cast<size_t>(labelCount));
    for (int32_t label = 1; label < labelCount; ++label) {
        auto box = boxFromComponent(textMap, labels, stats, label, imgWidth, imgHeight, options.lowTextThreshold, options.minArea);
        if (!box) {
            continue;
        }
        if (options.charLevel) {
            box->angle = 0.0f; // glyphs are read upright, never rotated
        }
        boxes.push_back(*box);
    }
    return boxes;
}
} // namespace

std::vector<CraftBox> extractCraftTextBoxes(const float *heat, int32_t heatHeight, int32_t heatWidth, const CraftOptions &options) {
    if (heatHeight <= 0 || heatWidth <= 0) {
        throw error::InvalidArgument("extractCraftTextBoxes: heatHeight and heatWidth must be positive");
    }
    if (options.minArea < 0) {
        throw error::InvalidArgument("extractCraftTextBoxes: minArea must not be negative");
    }
    if (!(options.scale > 0.0f) || !std::isfinite(options.scale)) {
        throw error::InvalidArgument("extractCraftTextBoxes: scale must be a positive number");
    }

    return withCv("extractCraftTextBoxes", [&] {
        // Deinterleave the [text, affinity] channels. cv::split copies, so the
        // caller's buffer is only ever read despite the non-const view.
        const ::cv::Mat interleaved(heatHeight, heatWidth, CV_32FC2, const_cast<float *>(heat));
        std::array<::cv::Mat, 2> channels;
        ::cv::split(interleaved, channels.data());
        std::vector<CraftBox> boxes = componentBoxes(channels[0], channels[1], options);
        for (auto &box : boxes) {
            box.x0 *= options.scale;
            box.y0 *= options.scale;
            box.x1 *= options.scale;
            box.y1 *= options.scale;
        }
        return boxes;
    });
}

} // namespace rnocr::ops::craft
