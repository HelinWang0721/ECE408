// test_ops.cu -- standalone correctness + timing driver for the Project's
// convolution implementations (base.cu, op1..op5.cu, custom/new-forward.cu).
// It needs neither RAI nor Mini-DNN:
//
//   nvcc -O3 -arch=native -I Project/custom -DOP_ID=1 Project/test/test_ops.cu -o test_op1
//     OP_ID: 0 = base.cu, 1..5 = op1..op5.cu, 6 = custom/new-forward.cu
//     (or -DOP_FILE='"path/to/file.cu"')
//   ./test_op1                # correctness: the 4 m1/m2/m3 test cases + both LeNet layers
//   ./test_op1 bench 5000     # timing: LeNet layer1 + layer2 at batch 5000 (like ./m3 5000)
//
// The same file builds with the CPU emulator (tools/cudasim) for machines without a GPU.
// Pass -DTOL=2e-2 for FP16 implementations (op3).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <algorithm>

#if defined(OP_FILE)
#include OP_FILE
#elif OP_ID == 0
#include "../base.cu"
#elif OP_ID == 1
#include "../op1.cu"
#elif OP_ID == 2
#include "../op2.cu"
#elif OP_ID == 3
#include "../op3.cu"
#elif OP_ID == 4
#include "../op4.cu"
#elif OP_ID == 5
#include "../op5.cu"
#elif OP_ID == 6
#include "../custom/new-forward.cu"
#else
#error "define OP_ID (0..6) or OP_FILE"
#endif
#include "../custom/cpu-new-forward.cc"

#ifndef TOL
#define TOL 1e-3
#endif

struct Shape { int B, M, C, H, W, K, S; const char *name; };

static bool run_case(const Shape &s, bool bench, int check_images) {
  const int H_out = (s.H - s.K) / s.S + 1, W_out = (s.W - s.K) / s.S + 1;
  const size_t in_n = (size_t)s.B * s.C * s.H * s.W;
  const size_t out_n = (size_t)s.B * s.M * H_out * W_out;
  const size_t mask_n = (size_t)s.M * s.C * s.K * s.K;
  std::vector<float> in(in_n), mask(mask_n), out(out_n, 0.f);
  std::mt19937 rng(408);
  std::uniform_real_distribution<float> U(0.f, 1.f), Wd(-0.5f, 0.5f);
  for (auto &v : in) v = U(rng);
  for (auto &v : mask) v = Wd(rng);

  GPUInterface gpu;
  // Mini-DNN passes *uninitialised* pointers; emulate that so an implementation
  // that forgets to set *device_mask_ptr is caught when it cudaFree()s it.
  float *d_out = (float *)0x10, *d_in = (float *)0x10, *d_mask = (float *)0x10;
  auto t0 = std::chrono::steady_clock::now();
  gpu.conv_forward_gpu_prolog(out.data(), in.data(), mask.data(), &d_out, &d_in, &d_mask, s.B, s.M, s.C, s.H, s.W, s.K, s.S);
  cudaDeviceSynchronize();
  auto t1 = std::chrono::steady_clock::now();
  gpu.conv_forward_gpu(d_out, d_in, d_mask, s.B, s.M, s.C, s.H, s.W, s.K, s.S);
  cudaDeviceSynchronize();
  auto t2 = std::chrono::steady_clock::now();
  cudaError_t err = cudaGetLastError();
  gpu.conv_forward_gpu_epilog(out.data(), d_out, d_in, d_mask, s.B, s.M, s.C, s.H, s.W, s.K, s.S);
  cudaDeviceSynchronize();
  auto t3 = std::chrono::steady_clock::now();
  if (err == cudaSuccess) err = cudaGetLastError();

  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  printf("%-10s B=%-5d M=%-3d C=%-3d H=%-4d W=%-4d K=%d S=%d | prolog %8.2f ms  Op Time %8.3f ms  epilog %8.2f ms", s.name, s.B,
         s.M, s.C, s.H, s.W, s.K, s.S, ms(t0, t1), ms(t1, t2), ms(t2, t3));

  if (err != cudaSuccess) { printf("  -> CUDA error: %s\n", cudaGetErrorString(err)); return false; }

  // Reference on the first `check_images` images (the CPU conv is slow).
  int nb = std::min(s.B, check_images);
  std::vector<float> ref((size_t)nb * s.M * H_out * W_out);
  conv_forward_cpu(ref.data(), in.data(), mask.data(), nb, s.M, s.C, s.H, s.W, s.K, s.S);
  size_t bad = 0, first = (size_t)-1;
  double max_err = 0;
  for (size_t i = 0; i < ref.size(); i++) {
    double e = std::fabs((double)out[i] - ref[i]);
    if (!(e <= TOL * (1.0 + std::fabs(ref[i])))) { if (!bad) first = i; bad++; }
    if (!(e <= max_err)) max_err = e;  // also catches NaN
  }
  if (bad) {
    printf("  -> FAIL (%zu/%zu wrong, first idx %zu: got %g want %g)\n", bad, ref.size(), first, out[first], ref[first]);
    return false;
  }
  printf("  -> PASS (max abs err %.2e)\n", max_err);
  (void)bench;
  return true;
}

int main(int argc, char **argv) {
  bool bench = argc > 1 && std::string(argv[1]) == "bench";
  int ok = 0, total = 0;
  if (!bench) {
    // The four test cases that m1/m2/m3 run before inference, plus the two
    // convolution layers of the modified LeNet-5 (86x86 input) at a small batch.
    int lb = argc > 1 ? atoi(argv[1]) : 4;
    Shape cases[] = {
        {1, 3, 3, 224, 224, 3, 1, "test1"},
        {2, 3, 3, 301, 301, 3, 2, "test2"},
        {3, 3, 3, 196, 196, 3, 3, "test3"},
        {4, 3, 3, 239, 239, 3, 4, "test4"},
        {3, 5, 6, 37, 45, 5, 2, "generic"},   // non-square, K=5, C=6: generic code paths
        {lb, 4, 1, 86, 86, 7, 1, "layer1"},
        {lb, 16, 4, 40, 40, 7, 1, "layer2"},
    };
    for (auto &c : cases) { total++; ok += run_case(c, false, c.B); }
  } else {
    int B = argc > 2 ? atoi(argv[2]) : 5000;
    int reps = argc > 3 ? atoi(argv[3]) : 3;
    Shape cases[] = {{B, 4, 1, 86, 86, 7, 1, "layer1"}, {B, 16, 4, 40, 40, 7, 1, "layer2"}};
    for (int r = 0; r < reps; r++)  // first rep includes CUDA context warm-up
      for (auto &c : cases) { total++; ok += run_case(c, true, 16); }
  }
  printf("RESULT %s (%d/%d passed)\n", ok == total ? "PASS" : "FAIL", ok, total);
  return ok == total ? 0 : 1;
}
