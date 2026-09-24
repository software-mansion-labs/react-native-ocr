/**
 * Serial stand-in for the OpenMP entry points the prebuilt Android OpenCV
 * references.
 *
 * react-native-executorch's OpenCV build uses OpenMP as its parallel backend
 * and links the LLVM runtime statically into libRnExecutorch.so. Linking a
 * second copy into this library aborts at load time: the runtime registers
 * itself per process and refuses to start when another copy is already
 * registered ("OMP: Error #15", KMP_DUPLICATE_LIB_OK). Binding to the copy
 * exported by libRnExecutorch.so would tie this library to that binary's
 * symbol table and load order instead.
 *
 * The eight symbols below are everything libopencv_core.a and
 * libopencv_imgproc.a pull in (checked with llvm-nm), and they only serve
 * cv::parallel_for_. Reporting a single thread makes OpenCV run its bodies
 * inline, and the dispatch shim hands the whole range back once in case a
 * parallel region is entered anyway. The kernels in this package are
 * per-region crops and once-per-page warps, for which a single thread is
 * adequate.
 *
 * iOS uses the opencv2 framework, whose parallel backend is GCD, so this file
 * compiles to nothing there.
 */
#ifdef __ANDROID__

#include <cstdarg>
#include <cstdint>

extern "C" {

struct ident_t;
using kmp_int32 = int32_t;
using kmp_uint32 = uint32_t;
using kmpc_micro = void (*)(kmp_int32 *global_tid, kmp_int32 *bound_tid, ...);

int omp_get_max_threads(void) {
    return 1;
}

int omp_get_thread_num(void) {
    return 0;
}

void omp_set_dynamic(int /*dynamic_threads*/) {
}

kmp_int32 __kmpc_global_thread_num(ident_t * /*loc*/) {
    return 0;
}

void __kmpc_push_num_threads(ident_t * /*loc*/, kmp_int32 /*global_tid*/, kmp_int32 /*num_threads*/) {
}

// Runs the outlined parallel region once on the calling thread, forwarding
// the shared-variable pointers the compiler passed as varargs. Fifteen is the
// same ceiling the LLVM runtime's serial path (__kmp_invoke_microtask) uses.
void __kmpc_fork_call(ident_t * /*loc*/, kmp_int32 argc, kmpc_micro microtask, ...) {
    constexpr kmp_int32 kMaxArgs = 15;
    void *args[kMaxArgs] = {};
    va_list ap;
    va_start(ap, microtask);
    for (kmp_int32 i = 0; i < argc && i < kMaxArgs; ++i) {
        args[i] = va_arg(ap, void *);
    }
    va_end(ap);

    kmp_int32 gtid = 0;
    kmp_int32 tid = 0;
    switch (argc) {
    case 0:
        microtask(&gtid, &tid);
        break;
    case 1:
        microtask(&gtid, &tid, args[0]);
        break;
    case 2:
        microtask(&gtid, &tid, args[0], args[1]);
        break;
    case 3:
        microtask(&gtid, &tid, args[0], args[1], args[2]);
        break;
    case 4:
        microtask(&gtid, &tid, args[0], args[1], args[2], args[3]);
        break;
    case 5:
        microtask(&gtid, &tid, args[0], args[1], args[2], args[3], args[4]);
        break;
    case 6:
        microtask(&gtid, &tid, args[0], args[1], args[2], args[3], args[4], args[5]);
        break;
    case 7:
        microtask(&gtid, &tid, args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
        break;
    case 8:
        microtask(&gtid, &tid, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
        break;
    case 9:
        microtask(&gtid, &tid, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]);
        break;
    case 10:
        microtask(&gtid, &tid, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]);
        break;
    case 11:
        microtask(&gtid, &tid, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]);
        break;
    case 12:
        microtask(&gtid, &tid, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11]);
        break;
    case 13:
        microtask(&gtid, &tid, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12]);
        break;
    case 14:
        microtask(&gtid, &tid, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13]);
        break;
    default:
        microtask(&gtid, &tid, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14]);
        break;
    }
}

// schedule(dynamic) loops over an unsigned 32-bit iteration space: init
// records the range, next hands it back whole exactly once. A small stack
// keeps nested regions independent.
namespace {
struct DispatchRange {
    kmp_uint32 lower;
    kmp_uint32 upper;
    kmp_int32 stride;
    bool consumed;
};
constexpr int kMaxNesting = 8;
thread_local DispatchRange gRanges[kMaxNesting];
thread_local int gDepth = 0;
} // namespace

void __kmpc_dispatch_init_4u(ident_t * /*loc*/, kmp_int32 /*gtid*/, int /*schedule*/, kmp_uint32 lb, kmp_uint32 ub, kmp_int32 st, kmp_int32 /*chunk*/) {
    if (gDepth < kMaxNesting) {
        gRanges[gDepth] = {lb, ub, st, false};
    }
    ++gDepth;
}

int __kmpc_dispatch_next_4u(ident_t * /*loc*/, kmp_int32 /*gtid*/, kmp_int32 *p_last, kmp_uint32 *p_lb, kmp_uint32 *p_ub, kmp_int32 *p_st) {
    const int index = gDepth - 1;
    if (index < 0 || index >= kMaxNesting) {
        gDepth = index < 0 ? 0 : gDepth;
        return 0;
    }
    DispatchRange &range = gRanges[index];
    if (range.consumed) {
        --gDepth;
        return 0;
    }
    range.consumed = true;
    *p_lb = range.lower;
    *p_ub = range.upper;
    *p_st = range.stride;
    *p_last = 1;
    return 1;
}

} // extern "C"

#endif // __ANDROID__
