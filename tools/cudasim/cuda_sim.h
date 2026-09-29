// cuda_sim.h -- a tiny single-threaded CUDA runtime emulator for machines
// without an NVIDIA GPU. It is NOT a performance model; it exists only to
// check functional correctness of the kernels in this repo.
//
// How it works
//   * Every CUDA thread is a ucontext fiber. A block's threads run one after
//     another until they reach __syncthreads(), then the scheduler resumes the
//     next thread. When every live thread of the block is parked on the barrier
//     the barrier opens. So barrier semantics are exact.
//   * If some threads exit while others wait at a barrier, that is a
//     "divergent __syncthreads" (undefined behaviour on real HW) -> reported.
//   * Blocks run sequentially, so `__shared__` can simply be `static`.
//   * cudaMalloc fills memory with 0xFF bytes (NaN / -1) so kernels that read
//     uninitialised device memory produce visibly wrong results.
//   * SIM_REVERSE=1 runs threads in reverse order: combined with the default
//     order this catches most missing-__syncthreads races.
//   * Launch-config limits (1024 threads/block, 48KB dynamic smem, grid dims)
//     are checked and surface through cudaGetLastError() like the real runtime.
//
// Kernel launches `k<<<g,b,smem,stream>>>(args)` are rewritten into
// `__sim_launch(...)` by cudasim.py before compilation.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <ucontext.h>

#define __global__
#define __device__
#define __host__
#define __constant__
#define __shared__ static
#define __forceinline__ inline
#define __noinline__
#define __launch_bounds__(...)
#define __align__(n) alignas(n)

struct dim3 {
  unsigned int x, y, z;
  constexpr dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};
typedef dim3 uint3;

inline dim3 threadIdx, blockIdx, blockDim, gridDim;
static const int warpSize = 32;

// ---------------------------------------------------------------- errors
enum cudaError_t {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInvalidConfiguration = 9,
  cudaErrorLaunchFailure = 719,
};
typedef cudaError_t cudaError;
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };

inline cudaError_t __sim_last_error = cudaSuccess;
inline int __sim_fail_count = 0;
inline const char *cudaGetErrorString(cudaError_t e) {
  switch (e) {
  case cudaSuccess: return "no error";
  case cudaErrorInvalidValue: return "invalid argument";
  case cudaErrorMemoryAllocation: return "out of memory";
  case cudaErrorInvalidConfiguration: return "invalid configuration argument";
  default: return "unspecified launch failure";
  }
}
inline cudaError_t cudaGetLastError() { cudaError_t e = __sim_last_error; __sim_last_error = cudaSuccess; return e; }
inline cudaError_t cudaPeekAtLastError() { return __sim_last_error; }
inline void __sim_report(const char *what) {
  fprintf(stderr, "[cudasim] ERROR: %s\n", what);
  __sim_fail_count++;
}

// ---------------------------------------------------------------- memory
inline std::map<void *, size_t> &__sim_allocs() { static std::map<void *, size_t> m; return m; }
inline cudaError_t cudaMalloc(void **p, size_t n) {
  *p = malloc(n ? n : 1);
  if (!*p) return __sim_last_error = cudaErrorMemoryAllocation;
  memset(*p, 0xFF, n);  // poison: NaN for float, -1 for int
  __sim_allocs()[*p] = n;
  return cudaSuccess;
}
template <class T> inline cudaError_t cudaMalloc(T **p, size_t n) { return cudaMalloc((void **)p, n); }
inline cudaError_t cudaMallocHost(void **p, size_t n) { *p = malloc(n); return cudaSuccess; }
template <class T> inline cudaError_t cudaMallocHost(T **p, size_t n) { return cudaMallocHost((void **)p, n); }
inline cudaError_t cudaHostAlloc(void **p, size_t n, unsigned) { return cudaMallocHost(p, n); }
inline cudaError_t cudaFreeHost(void *p) { free(p); return cudaSuccess; }
inline cudaError_t cudaHostRegister(void *, size_t, unsigned) { return cudaSuccess; }
inline cudaError_t cudaHostUnregister(void *) { return cudaSuccess; }
#define cudaHostAllocDefault 0
#define cudaHostRegisterDefault 0
inline cudaError_t cudaFree(void *p) {
  if (!p) return cudaSuccess;
  auto &m = __sim_allocs();
  auto it = m.find(p);
  if (it == m.end()) { __sim_report("cudaFree on a pointer that was not returned by cudaMalloc"); return __sim_last_error = cudaErrorInvalidValue; }
  m.erase(it);
  free(p);
  return cudaSuccess;
}
// find the allocation that contains [p, p+n)
inline bool __sim_in_alloc(const void *p, size_t n) {
  auto &m = __sim_allocs();
  auto it = m.upper_bound((void *)p);
  if (it == m.begin()) return false;
  --it;
  const char *b = (const char *)it->first;
  return (const char *)p >= b && (const char *)p + n <= b + it->second;
}
inline cudaError_t cudaMemcpy(void *d, const void *s, size_t n, cudaMemcpyKind k) {
  if (k == cudaMemcpyHostToDevice && !__sim_in_alloc(d, n)) __sim_report("cudaMemcpy H2D: destination range is outside a device allocation");
  if (k == cudaMemcpyDeviceToHost && !__sim_in_alloc(s, n)) __sim_report("cudaMemcpy D2H: source range is outside a device allocation");
  memmove(d, s, n);
  return cudaSuccess;
}
inline cudaError_t cudaMemset(void *d, int v, size_t n) {
  if (!__sim_in_alloc(d, n)) __sim_report("cudaMemset: range outside a device allocation");
  memset(d, v, n);
  return cudaSuccess;
}
typedef struct __sim_stream *cudaStream_t;
inline cudaError_t cudaMemcpyAsync(void *d, const void *s, size_t n, cudaMemcpyKind k, cudaStream_t = 0) { return cudaMemcpy(d, s, n, k); }
inline cudaError_t cudaMemsetAsync(void *d, int v, size_t n, cudaStream_t = 0) { return cudaMemset(d, v, n); }
template <class T>
inline cudaError_t cudaMemcpyToSymbol(T &sym, const void *src, size_t n, size_t off = 0, cudaMemcpyKind = cudaMemcpyHostToDevice) {
  if (off + n > sizeof(T)) { __sim_report("cudaMemcpyToSymbol: copy larger than the __constant__ symbol"); return __sim_last_error = cudaErrorInvalidValue; }
  memcpy((char *)&sym + off, src, n);
  return cudaSuccess;
}
inline cudaError_t cudaStreamCreate(cudaStream_t *s) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaDeviceReset() { return cudaSuccess; }

typedef struct { double t; } *cudaEvent_t;
#include <chrono>
inline cudaError_t cudaEventCreate(cudaEvent_t *e) { *e = (cudaEvent_t)malloc(sizeof(double)); return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t = 0) {
  e->t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
  return cudaSuccess;
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t a, cudaEvent_t b) { *ms = (float)(b->t - a->t); return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { free(e); return cudaSuccess; }

struct cudaDeviceProp {
  char name[256];
  int major, minor;
  size_t totalGlobalMem, totalConstMem, sharedMemPerBlock;
  int maxThreadsPerBlock, maxThreadsDim[3], maxGridSize[3], warpSize, multiProcessorCount, regsPerBlock, clockRate;
};
inline cudaError_t cudaGetDeviceCount(int *n) { *n = 1; return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp *p, int) {
  memset(p, 0, sizeof(*p));
  strcpy(p->name, "cudasim (CPU emulation)");
  p->major = 7; p->minor = 0;
  p->totalGlobalMem = 16ull << 30; p->totalConstMem = 65536; p->sharedMemPerBlock = 49152;
  p->maxThreadsPerBlock = 1024; p->maxThreadsDim[0] = 1024; p->maxThreadsDim[1] = 1024; p->maxThreadsDim[2] = 64;
  p->maxGridSize[0] = 2147483647; p->maxGridSize[1] = 65535; p->maxGridSize[2] = 65535;
  p->warpSize = 32; p->multiProcessorCount = 80; p->regsPerBlock = 65536;
  return cudaSuccess;
}

// ---------------------------------------------------------------- scheduler
namespace cudasim {
struct Fiber {
  ucontext_t ctx;
  bool done = false, at_barrier = false;
  dim3 tid;
};
inline ucontext_t g_sched;
inline Fiber *g_cur = nullptr;
inline std::function<void()> *g_body = nullptr;
inline std::vector<char> g_stacks;
inline const size_t kStack = 64 * 1024;
inline unsigned char g_dyn_smem[48 * 1024 + 64];
inline bool g_divergent = false;
inline const char *g_kernel_name = "";

inline void fiber_entry() {
  (*g_body)();
  g_cur->done = true;
  swapcontext(&g_cur->ctx, &g_sched);
}
inline bool reverse_order() { static int r = getenv("SIM_REVERSE") ? atoi(getenv("SIM_REVERSE")) : 0; return r != 0; }

inline void run_block(std::function<void()> &body) {
  unsigned n = blockDim.x * blockDim.y * blockDim.z;
  static std::vector<Fiber> fibers;
  fibers.assign(n, Fiber());
  if (g_stacks.size() < n * kStack) g_stacks.resize(n * kStack);
  g_body = &body;
  for (unsigned i = 0; i < n; i++) {
    Fiber &f = fibers[i];
    f.tid = dim3(i % blockDim.x, (i / blockDim.x) % blockDim.y, i / (blockDim.x * blockDim.y));
    getcontext(&f.ctx);
    f.ctx.uc_stack.ss_sp = &g_stacks[i * kStack];
    f.ctx.uc_stack.ss_size = kStack;
    f.ctx.uc_link = nullptr;
    makecontext(&f.ctx, (void (*)())fiber_entry, 0);
  }
  bool rev = reverse_order();
  for (;;) {
    unsigned alive = 0, waiting = 0;
    for (unsigned k = 0; k < n; k++) {
      Fiber &f = fibers[rev ? n - 1 - k : k];
      if (f.done) continue;
      f.at_barrier = false;
      g_cur = &f;
      threadIdx = f.tid;
      swapcontext(&g_sched, &f.ctx);
      if (!f.done) { alive++; waiting++; }
    }
    if (alive == 0) break;
    if (waiting != n && !g_divergent) {
      // On Volta+ exited threads no longer take part in the barrier, so this
      // usually "works", but it is fragile / UB per the programming guide.
      g_divergent = true;
      fprintf(stderr, "[cudasim] WARNING: %s: __syncthreads() reached by only part of the block "
                      "(other threads already exited) -- barrier inside divergent code\n", g_kernel_name);
    }
  }
}
}  // namespace cudasim

inline void __syncthreads() {
  cudasim::Fiber *f = cudasim::g_cur;
  f->at_barrier = true;
  swapcontext(&f->ctx, &cudasim::g_sched);
}
inline void __threadfence() {}
inline void __threadfence_block() {}

inline void __sim_launch(const char *name, dim3 g, dim3 b, size_t smem, std::function<void()> body) {
  unsigned long nthreads = (unsigned long)b.x * b.y * b.z;
  if (nthreads == 0 || nthreads > 1024 || b.x > 1024 || b.y > 1024 || b.z > 64 || g.x == 0 || g.y == 0 || g.z == 0 ||
      g.y > 65535 || g.z > 65535 || smem > 48 * 1024) {
    char buf[256];
    snprintf(buf, sizeof buf, "launch of %s rejected: grid(%u,%u,%u) block(%u,%u,%u) smem=%zu (invalid configuration)", name, g.x, g.y, g.z, b.x, b.y, b.z, smem);
    fprintf(stderr, "[cudasim] %s\n", buf);
    __sim_last_error = cudaErrorInvalidConfiguration;
    return;
  }
  gridDim = g;
  blockDim = b;
  cudasim::g_kernel_name = name;
  cudasim::g_divergent = false;
  memset(cudasim::g_dyn_smem, 0xFF, sizeof cudasim::g_dyn_smem);
  for (unsigned z = 0; z < g.z; z++)
    for (unsigned y = 0; y < g.y; y++)
      for (unsigned x = 0; x < g.x; x++) {
        blockIdx = dim3(x, y, z);
        cudasim::run_block(body);
      }
}
#define __sim_dyn_smem ((void *)cudasim::g_dyn_smem)

// ---------------------------------------------------------------- device builtins
template <class T> inline T atomicAdd(T *a, T v) { T o = *a; *a = o + v; return o; }
template <class T> inline T atomicSub(T *a, T v) { T o = *a; *a = o - v; return o; }
template <class T> inline T atomicMax(T *a, T v) { T o = *a; *a = std::max(o, v); return o; }
template <class T> inline T atomicMin(T *a, T v) { T o = *a; *a = std::min(o, v); return o; }
template <class T> inline T atomicExch(T *a, T v) { T o = *a; *a = v; return o; }
template <class T> inline T atomicCAS(T *a, T c, T v) { T o = *a; if (o == c) *a = v; return o; }
template <class T> inline T __ldg(const T *p) { return *p; }
inline float __fdividef(float a, float b) { return a / b; }
inline float __saturatef(float a) { return a < 0 ? 0 : (a > 1 ? 1 : a); }
using std::max;
using std::min;
inline float max(float a, double b) { return a > b ? a : (float)b; }
inline float min(float a, double b) { return a < b ? a : (float)b; }

// ---------------------------------------------------------------- fp16
typedef _Float16 half;
typedef _Float16 __half;
struct half2 { half x, y; };
typedef half2 __half2;
inline half __float2half(float f) { return (half)f; }
inline half __float2half_rn(float f) { return (half)f; }
inline float __half2float(half h) { return (float)h; }
inline half __hadd(half a, half b) { return a + b; }
inline half __hsub(half a, half b) { return a - b; }
inline half __hmul(half a, half b) { return a * b; }
inline half __hfma(half a, half b, half c) { return (half)((float)a * (float)b + (float)c); }
inline half2 __floats2half2_rn(float a, float b) { return {(half)a, (half)b}; }
inline half2 __halves2half2(half a, half b) { return {a, b}; }
inline half2 __float2half2_rn(float a) { return {(half)a, (half)a}; }
inline half2 __hadd2(half2 a, half2 b) { return {(half)(a.x + b.x), (half)(a.y + b.y)}; }
inline half2 __hmul2(half2 a, half2 b) { return {(half)(a.x * b.x), (half)(a.y * b.y)}; }
inline half2 __hfma2(half2 a, half2 b, half2 c) { return {__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y)}; }
inline float __low2float(half2 a) { return (float)a.x; }
inline float __high2float(half2 a) { return (float)a.y; }
inline half __low2half(half2 a) { return a.x; }
inline half __high2half(half2 a) { return a.y; }
