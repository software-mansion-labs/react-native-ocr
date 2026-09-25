#pragma once

#include <functional>

#include <gtest/gtest.h>

#include "core/error.h"

namespace rnocr::tests {

/**
 * Asserts that `fn` throws an RnOcrException carrying `code`. A kernel that
 * throws anything else, or nothing, fails the test: the JSI guard maps only
 * RnOcrException to a coded JavaScript error, everything else becomes UNKNOWN.
 */
inline void expectThrowsCode(core::error::ErrorCode code, const std::function<void()> &fn) {
    try {
        fn();
        ADD_FAILURE() << "expected an RnOcrException with code " << core::error::errorCodeToString(code) << ", nothing was thrown";
    } catch (const core::error::RnOcrException &e) {
        EXPECT_EQ(e.code_, code) << "message: " << e.what();
    } catch (const std::exception &e) {
        ADD_FAILURE() << "expected an RnOcrException, got a std::exception: " << e.what();
    }
}

} // namespace rnocr::tests
