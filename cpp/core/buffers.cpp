#include "buffers.h"

#include <cmath>
#include <limits>
#include <memory>

#include "error.h"

namespace rnocr::core::buffers {

namespace {
/** An ArrayBuffer backing store that owns a std::vector. */
class VectorBuffer final : public jsi::MutableBuffer {
public:
    explicit VectorBuffer(std::vector<uint8_t> &&bytes) : bytes_(std::move(bytes)) {}
    size_t size() const override { return bytes_.size(); }
    uint8_t *data() override { return bytes_.data(); }

private:
    std::vector<uint8_t> bytes_;
};
} // namespace

void requireArgCount(size_t count, size_t expected, const char *usage) {
    if (count != expected) {
        throw error::InvalidArgument(std::string("Usage: ") + usage);
    }
}

double requireNumber(jsi::Runtime & /*rt*/, const jsi::Value &value, const std::string &what) {
    if (!value.isNumber()) {
        throw error::InvalidArgument(what + ": expected a number");
    }
    const double number = value.asNumber();
    if (!std::isfinite(number)) {
        throw error::InvalidArgument(what + ": expected a finite number");
    }
    return number;
}

int32_t requireNonNegativeInt(jsi::Runtime &rt, const jsi::Value &value, const std::string &what) {
    const double number = requireNumber(rt, value, what);
    if (number < 0 || number != std::floor(number) || number > std::numeric_limits<int32_t>::max()) {
        throw error::InvalidArgument(what + ": expected a non-negative integer");
    }
    return static_cast<int32_t>(number);
}

int32_t requirePositiveInt(jsi::Runtime &rt, const jsi::Value &value, const std::string &what) {
    const int32_t number = requireNonNegativeInt(rt, value, what);
    if (number == 0) {
        throw error::InvalidArgument(what + ": expected a positive integer");
    }
    return number;
}

ByteView requireBuffer(jsi::Runtime &rt, const jsi::Value &value, const std::string &what, size_t expectedBytes) {
    if (!value.isObject()) {
        throw error::InvalidArgument(what + ": expected an ArrayBuffer");
    }
    jsi::Object object = value.asObject(rt);

    size_t byteOffset = 0;
    size_t byteLength = 0;
    jsi::ArrayBuffer buffer = [&]() -> jsi::ArrayBuffer {
        if (object.isArrayBuffer(rt)) {
            auto arrayBuffer = object.getArrayBuffer(rt);
            byteLength = arrayBuffer.size(rt);
            return arrayBuffer;
        }
        // A typed-array view: unwrap it.
        jsi::Value inner = object.getProperty(rt, "buffer");
        if (!inner.isObject() || !inner.asObject(rt).isArrayBuffer(rt)) {
            throw error::InvalidArgument(what + ": expected an ArrayBuffer or a typed array");
        }
        byteOffset = static_cast<size_t>(requireNonNegativeInt(rt, object.getProperty(rt, "byteOffset"), what + ".byteOffset"));
        byteLength = static_cast<size_t>(requireNonNegativeInt(rt, object.getProperty(rt, "byteLength"), what + ".byteLength"));
        return inner.asObject(rt).getArrayBuffer(rt);
    }();

    if (byteLength != expectedBytes) {
        throw error::InvalidArgument(what + ": expected " + std::to_string(expectedBytes) + " bytes, got " +
                                     std::to_string(byteLength));
    }
    if (byteOffset + byteLength > buffer.size(rt)) {
        throw error::InvalidArgument(what + ": view exceeds its buffer");
    }
    return {buffer.data(rt) + byteOffset, byteLength};
}

jsi::Value makeArrayBuffer(jsi::Runtime &rt, std::vector<uint8_t> &&bytes) {
    return jsi::ArrayBuffer(rt, std::make_shared<VectorBuffer>(std::move(bytes)));
}

size_t imageBytes(int32_t height, int32_t width, int32_t channels, const std::string &what) {
    if (height <= 0 || width <= 0 || channels <= 0) {
        throw error::InvalidArgument(what + ": height, width and channels must be positive");
    }
    const uint64_t total = static_cast<uint64_t>(height) * static_cast<uint64_t>(width) * static_cast<uint64_t>(channels);
    if (total > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw error::InvalidArgument(what + ": image is too large");
    }
    return static_cast<size_t>(total);
}

} // namespace rnocr::core::buffers
