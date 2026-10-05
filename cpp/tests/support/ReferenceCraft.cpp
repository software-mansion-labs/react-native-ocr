// The CRAFT branch of react-native-executorch's `ocr_ops.cpp` at commit
// 3f83e48a18bdb5c73ff5d65d364129b29e9a1578, copied verbatim apart from the
// namespace, the `Box` type moving to the header, and the OpenCV 5 include
// for `minAreaRect` (which the reference, built against OpenCV 4, did not
// need). Kept as the parity oracle for `CraftParityTest`; do not "improve" it.
#include "ReferenceCraft.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

#include <opencv2/imgproc.hpp>
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry.hpp>
#endif

namespace rnocr::tests::reference {
namespace {
// Drop CRAFT connected components smaller than this (detector-input px²).
constexpr int32_t kMinComponentArea = 10;

// ------------------------------ CRAFT branch -------------------------------
// All per-component work stays inside the component's (dilation-padded) rect —
// full-frame masks per component would make the decode O(components · H · W).
std::optional<Box> boxFromComponent(const ::cv::Mat &textMap, const ::cv::Mat &labels, const ::cv::Mat &stats, int32_t i,
                                    int32_t imgW, int32_t imgH, float lowTextThreshold) {
    const int32_t area = stats.at<int32_t>(i, ::cv::CC_STAT_AREA);
    if (area < kMinComponentArea) {
        return std::nullopt;
    }
    const int32_t x = stats.at<int32_t>(i, ::cv::CC_STAT_LEFT);
    const int32_t y = stats.at<int32_t>(i, ::cv::CC_STAT_TOP);
    const int32_t w = stats.at<int32_t>(i, ::cv::CC_STAT_WIDTH);
    const int32_t h = stats.at<int32_t>(i, ::cv::CC_STAT_HEIGHT);
    const ::cv::Rect compRect(x, y, w, h);
    ::cv::Mat mask = (labels(compRect) == i);
    double maxVal = 0.0;
    ::cv::minMaxLoc(textMap(compRect), nullptr, &maxVal, nullptr, nullptr, mask);
    if (maxVal < static_cast<double>(lowTextThreshold)) {
        return std::nullopt;
    }

    const auto dilationRadius = static_cast<int32_t>(std::sqrt(static_cast<double>(area) / std::max(w, h)) * 2);
    const int32_t sx = std::max(x - dilationRadius, 0);
    const int32_t ex = std::min(x + w + dilationRadius, imgW);
    const int32_t sy = std::max(y - dilationRadius, 0);
    const int32_t ey = std::min(y + h + dilationRadius, imgH);
    ::cv::Mat segMap = ::cv::Mat::zeros(ey - sy, ex - sx, CV_8U);
    segMap(::cv::Rect(x - sx, y - sy, w, h)).setTo(255, mask);
    const int32_t kSize = 1 + dilationRadius;
    ::cv::Mat kernel = ::cv::getStructuringElement(::cv::MORPH_RECT, ::cv::Size(kSize, kSize));
    ::cv::dilate(segMap, segMap, kernel, ::cv::Point(-1, -1), 1);

    std::vector<std::vector<::cv::Point>> contours;
    // The offset restores full-frame coordinates for the ROI-local contours.
    ::cv::findContours(segMap, contours, ::cv::RETR_EXTERNAL, ::cv::CHAIN_APPROX_SIMPLE, ::cv::Point(sx, sy));
    if (contours.empty()) {
        return std::nullopt;
    }
    ::cv::RotatedRect rr = ::cv::minAreaRect(contours[0]);
    std::array<::cv::Point2f, 4> v;
    rr.points(v.data());
    Box box;
    box.x0 = std::min({v[0].x, v[1].x, v[2].x, v[3].x});
    box.y0 = std::min({v[0].y, v[1].y, v[2].y, v[3].y});
    box.x1 = std::max({v[0].x, v[1].x, v[2].x, v[3].x});
    box.y1 = std::max({v[0].y, v[1].y, v[2].y, v[3].y});
    box.angle = rr.angle;
    return box;
}

// CRAFT text+affinity maps -> one component box each. charLevel=false ADDS
// affinity to link adjacent glyphs into line-regions (keeping the rotated-rect
// angle); charLevel=true SUBTRACTS it (+ erode/dilate) to break them into upright
// per-glyph boxes (angle 0) for the stacked-column reader.
std::vector<Box> componentBoxes(::cv::Mat &textMap, ::cv::Mat &affinityMap, float textThreshold, float linkThreshold,
                                float lowTextThreshold, bool charLevel) {
    const int32_t imgH = textMap.rows;
    const int32_t imgW = textMap.cols;
    ::cv::Mat textScore;
    ::cv::Mat affinityScore;
    ::cv::threshold(textMap, textScore, static_cast<double>(textThreshold), 1.0, ::cv::THRESH_BINARY);
    ::cv::threshold(affinityMap, affinityScore, static_cast<double>(linkThreshold), 1.0, ::cv::THRESH_BINARY);

    ::cv::Mat comb;
    if (charLevel) {
        comb = textScore - affinityScore; // subtract to separate adjacent glyphs
        ::cv::threshold(comb, comb, 0.0, 1.0, ::cv::THRESH_TOZERO);
        ::cv::threshold(comb, comb, 1.0, 1.0, ::cv::THRESH_TRUNC);
        ::cv::Mat kernel = ::cv::getStructuringElement(::cv::MORPH_RECT, ::cv::Size(3, 3));
        ::cv::erode(comb, comb, kernel, ::cv::Point(-1, -1), 1);
        ::cv::dilate(comb, comb, kernel, ::cv::Point(-1, -1), 4);
    } else {
        comb = textScore + affinityScore; // add to link adjacent glyphs into lines
        ::cv::threshold(comb, comb, 0.0, 1.0, ::cv::THRESH_BINARY);
    }

    ::cv::Mat binary;
    comb.convertTo(binary, CV_8UC1);
    ::cv::Mat labels;
    ::cv::Mat stats;
    ::cv::Mat centroids;
    const int32_t nLabels = ::cv::connectedComponentsWithStats(binary, labels, stats, centroids, 4);

    std::vector<Box> boxes;
    boxes.reserve(static_cast<std::size_t>(nLabels));
    for (int32_t i = 1; i < nLabels; ++i) {
        auto box = boxFromComponent(textMap, labels, stats, i, imgW, imgH, lowTextThreshold);
        if (box) {
            if (charLevel) {
                box->angle = 0.0f; // glyphs are read upright, never rotated
            }
            boxes.push_back(*box);
        }
    }
    return boxes;
}
} // namespace

// CRAFT half-res heatmap (text+affinity interleaved) -> component boxes in
// detector-input pixels; restoreRatio scales the half-res boxes back up. Line
// grouping and de-skew are done in TypeScript; charLevel yields upright per-glyph
// boxes. `data` points at heatW*heatH*2 floats.
std::vector<Box> extractCraft(float *data, int32_t heatW, int32_t heatH, float textThreshold, float linkThreshold,
                              float lowTextThreshold, float restoreRatio, bool charLevel) {
    // Deinterleave the [text, affinity] channels of the half-res heatmap.
    ::cv::Mat interleaved(heatH, heatW, CV_32FC2, data);
    std::array<::cv::Mat, 2> channels;
    ::cv::split(interleaved, channels);
    std::vector<Box> boxes = componentBoxes(channels[0], channels[1], textThreshold, linkThreshold, lowTextThreshold, charLevel);
    for (auto &b : boxes) {
        b.x0 *= restoreRatio;
        b.y0 *= restoreRatio;
        b.x1 *= restoreRatio;
        b.y1 *= restoreRatio;
    }
    return boxes;
}

} // namespace rnocr::tests::reference
