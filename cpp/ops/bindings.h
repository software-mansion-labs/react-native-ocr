#pragma once

#include <jsi/jsi.h>

/**
 * JSI host functions over the kernels in `image_ops.h` and `craft_ops.h`.
 *
 * Every function follows the buffer-in, buffer-out convention: images and
 * grids arrive as ArrayBuffers (or typed-array views) with their dimensions as
 * separate numbers, and the result is a freshly allocated ArrayBuffer whose
 * shape the caller derives from the arguments. `src/native/types.ts` mirrors
 * the signatures by hand.
 */
namespace rnocr::ops {

/** Installs every op as a host function on the `__rnocr_jsi__` module object. */
void installOps(facebook::jsi::Runtime &rt, facebook::jsi::Object &module);

} // namespace rnocr::ops
