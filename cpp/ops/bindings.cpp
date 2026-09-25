#include "bindings.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "core/buffers.h"
#include "core/error.h"
#include "core/guard.h"
#include "core/options.h"
#include "craft_ops.h"
#include "image_ops.h"

namespace rnocr::ops {
namespace jsi = facebook::jsi;
using namespace rnocr::core;

namespace {

void define(jsi::Runtime &rt, jsi::Object &module, const char *name, unsigned paramCount, jsi::HostFunctionType body) {
    module.setProperty(rt, name, jsi::Function::createFromHostFunction(rt, jsi::PropNameID::forAscii(rt, name), paramCount, error::guarded(std::move(body))));
}

/** The three dimensions every image op starts with: `(height, width, channels)` at `args[first..first+2]`. */
struct ImageDims {
    int32_t height;
    int32_t width;
    int32_t channels;
    size_t bytes;
};

ImageDims readImageDims(jsi::Runtime &rt, const jsi::Value *args, size_t first, const std::string &op) {
    ImageDims dims{};
    dims.height = buffers::requirePositiveInt(rt, args[first], op + ": height");
    dims.width = buffers::requirePositiveInt(rt, args[first + 1], op + ": width");
    dims.channels = buffers::requirePositiveInt(rt, args[first + 2], op + ": channels");
    dims.bytes = buffers::imageBytes(dims.height, dims.width, dims.channels, op);
    return dims;
}

jsi::Value rotate(jsi::Runtime &rt, const jsi::Value & /*thisVal*/, const jsi::Value *args, size_t count) {
    buffers::requireArgCount(count, 5, "rotate(src, height, width, channels, degreesClockwise)");
    const ImageDims src = readImageDims(rt, args, 1, "rotate");
    const int32_t degrees = buffers::requireNonNegativeInt(rt, args[4], "rotate: degreesClockwise");
    const buffers::ByteView srcView = buffers::requireBuffer(rt, args[0], "rotate: src", src.bytes);

    std::vector<uint8_t> out(src.bytes);
    image::rotate(srcView.data, src.height, src.width, src.channels, degrees, out.data());
    return buffers::makeArrayBuffer(rt, std::move(out));
}

jsi::Value crop(jsi::Runtime &rt, const jsi::Value & /*thisVal*/, const jsi::Value *args, size_t count) {
    buffers::requireArgCount(count, 8, "crop(src, height, width, channels, x0, y0, x1, y1)");
    const ImageDims src = readImageDims(rt, args, 1, "crop");
    const int32_t x0 = buffers::requireNonNegativeInt(rt, args[4], "crop: x0");
    const int32_t y0 = buffers::requireNonNegativeInt(rt, args[5], "crop: y0");
    const int32_t x1 = buffers::requireNonNegativeInt(rt, args[6], "crop: x1");
    const int32_t y1 = buffers::requireNonNegativeInt(rt, args[7], "crop: y1");
    if (x1 <= x0 || y1 <= y0) {
        throw error::InvalidArgument("crop: box must have positive width and height");
    }
    const buffers::ByteView srcView = buffers::requireBuffer(rt, args[0], "crop: src", src.bytes);

    std::vector<uint8_t> out(buffers::imageBytes(y1 - y0, x1 - x0, src.channels, "crop"));
    image::crop(srcView.data, src.height, src.width, src.channels, x0, y0, x1, y1, out.data());
    return buffers::makeArrayBuffer(rt, std::move(out));
}

image::WarpOptions readWarpOptions(jsi::Runtime &rt, const jsi::Value &value) {
    image::WarpOptions result;
    if (value.isUndefined() || value.isNull()) {
        return result;
    }
    const std::string what = "warpByGrid: options";
    const jsi::Object object = options::requireObject(rt, value, what);

    if (const auto interpolation = options::optionalString(rt, object, "interpolation", what)) {
        if (*interpolation == "nearest") {
            result.interpolation = image::Interpolation::Nearest;
        } else if (*interpolation == "linear") {
            result.interpolation = image::Interpolation::Linear;
        } else if (*interpolation == "cubic") {
            result.interpolation = image::Interpolation::Cubic;
        } else {
            throw error::InvalidArgument(what + ".interpolation: expected 'nearest', 'linear' or 'cubic', got '" + *interpolation + "'");
        }
    }
    if (const auto borderMode = options::optionalString(rt, object, "borderMode", what)) {
        if (*borderMode == "replicate") {
            result.borderMode = image::BorderMode::Replicate;
        } else if (*borderMode == "constant") {
            result.borderMode = image::BorderMode::Constant;
        } else {
            throw error::InvalidArgument(what + ".borderMode: expected 'replicate' or 'constant', got '" + *borderMode + "'");
        }
    }
    if (const auto borderValue = options::optionalNumber(rt, object, "borderValue", what)) {
        result.borderValue = *borderValue;
    }
    return result;
}

jsi::Value warpByGrid(jsi::Runtime &rt, const jsi::Value & /*thisVal*/, const jsi::Value *args, size_t count) {
    if (count != 9 && count != 10) {
        throw error::InvalidArgument("Usage: warpByGrid(src, srcHeight, srcWidth, channels, grid, gridHeight, gridWidth, dstHeight, dstWidth, options?)");
    }
    const ImageDims src = readImageDims(rt, args, 1, "warpByGrid: src");
    const int32_t gridHeight = buffers::requirePositiveInt(rt, args[5], "warpByGrid: gridHeight");
    const int32_t gridWidth = buffers::requirePositiveInt(rt, args[6], "warpByGrid: gridWidth");
    const int32_t dstHeight = buffers::requirePositiveInt(rt, args[7], "warpByGrid: dstHeight");
    const int32_t dstWidth = buffers::requirePositiveInt(rt, args[8], "warpByGrid: dstWidth");
    // jsi::Value is move-only, so the optional argument cannot go through a ternary.
    const image::WarpOptions warpOptions = count == 10 ? readWarpOptions(rt, args[9]) : image::WarpOptions{};

    const size_t gridBytes = buffers::imageBytes(gridHeight, gridWidth, 2, "warpByGrid: grid") * sizeof(float);
    const size_t dstBytes = buffers::imageBytes(dstHeight, dstWidth, src.channels, "warpByGrid: dst");
    const buffers::ByteView srcView = buffers::requireBuffer(rt, args[0], "warpByGrid: src", src.bytes);
    const buffers::ByteView gridView = buffers::requireBuffer(rt, args[4], "warpByGrid: grid", gridBytes);

    // The grid arrives as raw bytes; a Float32Array's buffer is 4-byte aligned,
    // and a view with a byteOffset must be too or the typed array could not
    // have been constructed, so reinterpreting is safe.
    const auto *grid = reinterpret_cast<const float *>(gridView.data);
    std::vector<uint8_t> out(dstBytes);
    image::warpByGrid(srcView.data, src.height, src.width, src.channels, grid, gridHeight, gridWidth, dstHeight, dstWidth, warpOptions, out.data());
    return buffers::makeArrayBuffer(rt, std::move(out));
}

jsi::Value padCornerMean(jsi::Runtime &rt, const jsi::Value & /*thisVal*/, const jsi::Value *args, size_t count) {
    buffers::requireArgCount(count, 5, "padCornerMean(strip, height, width, channels, targetWidth)");
    const ImageDims strip = readImageDims(rt, args, 1, "padCornerMean");
    const int32_t targetWidth = buffers::requirePositiveInt(rt, args[4], "padCornerMean: targetWidth");
    const buffers::ByteView stripView = buffers::requireBuffer(rt, args[0], "padCornerMean: strip", strip.bytes);

    std::vector<uint8_t> out(buffers::imageBytes(strip.height, targetWidth, strip.channels, "padCornerMean"));
    image::padCornerMean(stripView.data, strip.height, strip.width, strip.channels, targetWidth, out.data());
    return buffers::makeArrayBuffer(rt, std::move(out));
}

craft::CraftOptions readCraftOptions(jsi::Runtime &rt, const jsi::Value &value) {
    const std::string what = "extractCraftTextBoxes: options";
    const jsi::Object object = options::requireObject(rt, value, what);
    craft::CraftOptions result;
    result.textThreshold = static_cast<float>(options::requireNumber(rt, object, "textThreshold", what));
    result.linkThreshold = static_cast<float>(options::requireNumber(rt, object, "linkThreshold", what));
    result.lowTextThreshold = static_cast<float>(options::requireNumber(rt, object, "lowTextThreshold", what));
    result.charLevel = options::requireBool(rt, object, "charLevel", what);
    result.scale = static_cast<float>(options::requireNumber(rt, object, "scale", what));

    const double minArea = options::requireNumber(rt, object, "minArea", what);
    if (minArea < 0 || minArea != std::floor(minArea) || minArea > static_cast<double>(INT32_MAX)) {
        throw error::InvalidArgument(what + ".minArea: expected a non-negative integer");
    }
    result.minArea = static_cast<int32_t>(minArea);
    return result;
}

jsi::Value extractCraftTextBoxes(jsi::Runtime &rt, const jsi::Value & /*thisVal*/, const jsi::Value *args, size_t count) {
    buffers::requireArgCount(count, 4, "extractCraftTextBoxes(heat, heatHeight, heatWidth, options)");
    const int32_t heatHeight = buffers::requirePositiveInt(rt, args[1], "extractCraftTextBoxes: heatHeight");
    const int32_t heatWidth = buffers::requirePositiveInt(rt, args[2], "extractCraftTextBoxes: heatWidth");
    const craft::CraftOptions craftOptions = readCraftOptions(rt, args[3]);
    const size_t heatBytes = buffers::imageBytes(heatHeight, heatWidth, 2, "extractCraftTextBoxes: heat") * sizeof(float);
    const buffers::ByteView heatView = buffers::requireBuffer(rt, args[0], "extractCraftTextBoxes: heat", heatBytes);

    const std::vector<craft::CraftBox> boxes =
        craft::extractCraftTextBoxes(reinterpret_cast<const float *>(heatView.data), heatHeight, heatWidth, craftOptions);

    // Five float32 per box: x0, y0, x1, y1, angle. CraftBox is exactly that
    // layout, so the vector's bytes are the result.
    static_assert(sizeof(craft::CraftBox) == 5 * sizeof(float), "CraftBox must be five packed floats");
    std::vector<uint8_t> out(boxes.size() * sizeof(craft::CraftBox));
    if (!boxes.empty()) {
        std::memcpy(out.data(), boxes.data(), out.size());
    }
    return buffers::makeArrayBuffer(rt, std::move(out));
}

} // namespace

void installOps(jsi::Runtime &rt, jsi::Object &module) {
    define(rt, module, "rotate", 5, rotate);
    define(rt, module, "crop", 8, crop);
    define(rt, module, "warpByGrid", 10, warpByGrid);
    define(rt, module, "padCornerMean", 5, padCornerMean);
    define(rt, module, "extractCraftTextBoxes", 4, extractCraftTextBoxes);
}

} // namespace rnocr::ops
