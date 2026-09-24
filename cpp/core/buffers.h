#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <jsi/jsi.h>

/**
 * Helpers for the buffer-in, buffer-out JSI convention every op follows.
 *
 * Arguments are read as plain ArrayBuffers (a typed-array view is accepted too
 * and unwrapped through its `buffer`/`byteOffset`/`byteLength`), validated
 * against the byte count the declared dimensions imply, and returned as a
 * non-owning view. Results are moved into a JavaScript-owned ArrayBuffer
 * without a further copy.
 */
namespace rnocr::core::buffers {
namespace jsi = facebook::jsi;

/** A non-owning view of a JavaScript buffer's bytes. */
struct ByteView {
    uint8_t *data;
    size_t size;
};

/** Throws InvalidArgument unless `count == expected`. */
void requireArgCount(size_t count, size_t expected, const char *usage);

/**
 * Reads a finite number argument.
 * @param what `"op: name"` for the error message.
 */
double requireNumber(jsi::Runtime &rt, const jsi::Value &value, const std::string &what);

/** Reads a non-negative integer argument. */
int32_t requireNonNegativeInt(jsi::Runtime &rt, const jsi::Value &value, const std::string &what);

/** Reads a strictly positive integer argument. */
int32_t requirePositiveInt(jsi::Runtime &rt, const jsi::Value &value, const std::string &what);

/**
 * Views an ArrayBuffer (or typed-array) argument, requiring exactly
 * `expectedBytes` bytes.
 */
ByteView requireBuffer(jsi::Runtime &rt, const jsi::Value &value, const std::string &what, size_t expectedBytes);

/** Moves `bytes` into a new JavaScript-owned ArrayBuffer. */
jsi::Value makeArrayBuffer(jsi::Runtime &rt, std::vector<uint8_t> &&bytes);

/** Element count of a `[height, width, channels]` image, checked for overflow. */
size_t imageBytes(int32_t height, int32_t width, int32_t channels, const std::string &what);

} // namespace rnocr::core::buffers
