// wb.h -- minimal stand-in for the course's libwb, so every MP can be built
// and checked outside of RAI:
//   * real GPU :  nvcc -I tools/include MPx/template.cu -o mpx
//   * no GPU   :  tools/cudasim (CPU emulator) uses the same header
// Supports exactly the subset of libwb the MPs in this repo use:
// wbArg_*, wbImport (vector / matrix / Integer|Real / PPM image), wbImage_*,
// wbLog, wbTime_*, wbSolution and CSRToJDS.
// wbSolution prints "Solution is correct." or "Solution is NOT correct" and a
// summary line "WB_RESULT PASS|FAIL" that the run scripts grep for.
#pragma once
#include <algorithm>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

// ------------------------------------------------------------------ args
struct wbArg_t {
  std::vector<std::string> inputs;
  std::string expected, type;
};
inline wbArg_t wbArg_read(int argc, char **argv) {
  wbArg_t a;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (s == "-e") a.expected = next();
    else if (s == "-t") a.type = next();
    else if (s == "-i") {
      std::stringstream ss(next());
      std::string f;
      while (std::getline(ss, f, ',')) a.inputs.push_back(f);
    }
  }
  return a;
}
inline const char *wbArg_getInputFile(const wbArg_t &a, int i) {
  if (i >= (int)a.inputs.size()) { fprintf(stderr, "wb: missing input file #%d\n", i); exit(2); }
  return a.inputs[i].c_str();
}

// ------------------------------------------------------------------ logging / timing
template <class... T> inline void wb_log_impl(const char *level, const char *file, int line, T &&...xs) {
  std::ostringstream os;
  (os << ... << xs);
  std::cout << "[" << level << "] " << os.str() << std::endl;
}
#define wbLog(level, ...) wb_log_impl(#level, __FILE__, __LINE__, __VA_ARGS__)

inline std::map<std::string, std::chrono::steady_clock::time_point> &wb_timers() {
  static std::map<std::string, std::chrono::steady_clock::time_point> m;
  return m;
}
inline void wb_time_start(const char *kind, const std::string &msg) { wb_timers()[std::string(kind) + msg] = std::chrono::steady_clock::now(); }
inline void wb_time_stop(const char *kind, const std::string &msg) {
#if defined(__CUDACC__)
  cudaDeviceSynchronize();
#endif
  auto t0 = wb_timers()[std::string(kind) + msg];
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  printf("[TIME %-8s] %9.3f ms  %s\n", kind, ms, msg.c_str());
}
#define wbTime_start(kind, msg) wb_time_start(#kind, msg)
#define wbTime_stop(kind, msg) wb_time_stop(#kind, msg)

// ------------------------------------------------------------------ import
inline std::vector<std::string> wb_tokens(const char *file) {
  std::ifstream in(file);
  if (!in) { fprintf(stderr, "wb: cannot open %s\n", file); exit(2); }
  std::vector<std::string> t;
  std::string s;
  while (in >> s) t.push_back(s);
  return t;
}
// vector file: "<n>\n v0\n v1 ..."
inline float *wbImport(const char *file, int *len) {
  auto t = wb_tokens(file);
  *len = atoi(t[0].c_str());
  float *d = (float *)malloc(sizeof(float) * (*len > 0 ? *len : 1));
  for (int i = 0; i < *len; i++) d[i] = (float)atof(t[1 + i].c_str());
  return d;
}
// matrix file: "<rows> <cols>\n values..."
inline float *wbImport(const char *file, int *rows, int *cols) {
  auto t = wb_tokens(file);
  *rows = atoi(t[0].c_str());
  *cols = atoi(t[1].c_str());
  size_t n = (size_t)*rows * *cols;
  float *d = (float *)malloc(sizeof(float) * (n ? n : 1));
  for (size_t i = 0; i < n; i++) d[i] = (float)atof(t[2 + i].c_str());
  return d;
}
// typed vector: "Integer" -> int*, "Real" -> float*
inline void *wbImport(const char *file, int *len, const char *type) {
  auto t = wb_tokens(file);
  *len = atoi(t[0].c_str());
  if (!strcmp(type, "Integer")) {
    int *d = (int *)malloc(sizeof(int) * (*len > 0 ? *len : 1));
    for (int i = 0; i < *len; i++) d[i] = atoi(t[1 + i].c_str());
    return d;
  }
  float *d = (float *)malloc(sizeof(float) * (*len > 0 ? *len : 1));
  for (int i = 0; i < *len; i++) d[i] = (float)atof(t[1 + i].c_str());
  return d;
}

// ------------------------------------------------------------------ images (binary PPM, P6)
struct wbImage_struct {
  int width, height, channels;
  float *data;  // interleaved, values in [0,1]
};
typedef wbImage_struct *wbImage_t;
inline wbImage_t wbImage_new(int w, int h, int c) {
  wbImage_t im = new wbImage_struct{w, h, c, (float *)calloc((size_t)w * h * c, sizeof(float))};
  return im;
}
inline void wbImage_delete(wbImage_t im) { free(im->data); delete im; }
inline int wbImage_getWidth(wbImage_t im) { return im->width; }
inline int wbImage_getHeight(wbImage_t im) { return im->height; }
inline int wbImage_getChannels(wbImage_t im) { return im->channels; }
inline float *wbImage_getData(wbImage_t im) { return im->data; }
inline wbImage_t wb_read_ppm(const char *file) {
  FILE *f = fopen(file, "rb");
  if (!f) { fprintf(stderr, "wb: cannot open %s\n", file); exit(2); }
  auto token = [&]() {
    std::string s;
    int c;
    for (;;) {
      c = fgetc(f);
      if (c == '#') { while (c != '\n' && c != EOF) c = fgetc(f); continue; }
      if (isspace(c)) { if (!s.empty()) break; continue; }
      if (c == EOF) break;
      s += (char)c;
    }
    return s;
  };
  std::string magic = token();
  int w = atoi(token().c_str()), h = atoi(token().c_str()), maxv = atoi(token().c_str());
  (void)maxv;
  int c = magic == "P6" ? 3 : 1;
  wbImage_t im = wbImage_new(w, h, c);
  std::vector<unsigned char> buf((size_t)w * h * c);
  size_t got = fread(buf.data(), 1, buf.size(), f);
  (void)got;
  fclose(f);
  for (size_t i = 0; i < buf.size(); i++) im->data[i] = buf[i] / 255.0f;
  return im;
}
inline wbImage_t wbImport(const char *file) { return wb_read_ppm(file); }

// ------------------------------------------------------------------ solution check
inline bool wb_close(double a, double e, double rel, double abs_) {
  if (std::isnan(a) || std::isinf(a)) return false;
  return std::fabs(a - e) <= abs_ + rel * std::fabs(e);
}
inline void wb_report(bool ok, long bad, long n, long first, double got, double exp) {
  if (ok) {
    printf("Solution is correct.\nWB_RESULT PASS\n");
  } else {
    printf("Solution is NOT correct: %ld / %ld mismatches, first at index %ld: got %g expected %g\nWB_RESULT FAIL\n", bad, n,
           first, got, exp);
  }
  fflush(stdout);
}
inline void wb_compare(const float *got, const std::vector<double> &exp, long n, double rel, double abs_) {
  if ((long)exp.size() != n) {
    printf("Solution is NOT correct: expected %zu elements, got %ld\nWB_RESULT FAIL\n", exp.size(), n);
    return;
  }
  long bad = 0, first = -1;
  for (long i = 0; i < n; i++)
    if (!wb_close(got[i], exp[i], rel, abs_)) { if (first < 0) first = i; bad++; }
  wb_report(bad == 0, bad, n, first, first < 0 ? 0 : got[first], first < 0 ? 0 : exp[first]);
}
inline void wbSolution(const wbArg_t &a, const float *out, int n) {
  auto t = wb_tokens(a.expected.c_str());
  std::vector<double> e;
  for (size_t i = 1; i < t.size(); i++) e.push_back(atof(t[i].c_str()));
  wb_compare(out, e, n, 1e-3, 1e-3);
}
inline void wbSolution(const wbArg_t &a, const float *out, int rows, int cols) {
  auto t = wb_tokens(a.expected.c_str());
  std::vector<double> e;
  for (size_t i = 2; i < t.size(); i++) e.push_back(atof(t[i].c_str()));
  wb_compare(out, e, (long)rows * cols, 1e-3, 1e-3);
}
inline void wbSolution(const wbArg_t &a, wbImage_t im) {
  wbImage_t e = wb_read_ppm(a.expected.c_str());
  if (e->width != im->width || e->height != im->height || e->channels != im->channels) {
    printf("Solution is NOT correct: image size mismatch\nWB_RESULT FAIL\n");
    return;
  }
  // PPM stores 8-bit values, so allow one intensity level of rounding.
  std::vector<double> ev(e->data, e->data + (size_t)e->width * e->height * e->channels);
  wb_compare(im->data, ev, (long)ev.size(), 0.0, 1.01 / 255.0);
  wbImage_delete(e);
}

// ------------------------------------------------------------------ CSR -> JDS (same algorithm as libwb)
inline void CSRToJDS(int dim, int *csrRowPtr, int *csrColIdx, float *csrData, int **jdsRowPerm, int **jdsRowNNZ,
                     int **jdsColStartIdx, int **jdsColIdx, float **jdsData) {
  *jdsRowPerm = (int *)malloc(sizeof(int) * dim);
  *jdsRowNNZ = (int *)malloc(sizeof(int) * dim);
  for (int r = 0; r < dim; ++r) {
    (*jdsRowPerm)[r] = r;
    (*jdsRowNNZ)[r] = csrRowPtr[r + 1] - csrRowPtr[r];
  }
  // stable sort rows by nnz, descending
  for (int i = 1; i < dim; i++)
    for (int j = i; j > 0 && (*jdsRowNNZ)[j] > (*jdsRowNNZ)[j - 1]; j--) {
      std::swap((*jdsRowNNZ)[j], (*jdsRowNNZ)[j - 1]);
      std::swap((*jdsRowPerm)[j], (*jdsRowPerm)[j - 1]);
    }
  int maxRowNNZ = dim > 0 ? (*jdsRowNNZ)[0] : 0;
  *jdsColStartIdx = (int *)malloc(sizeof(int) * (maxRowNNZ > 0 ? maxRowNNZ : 1));
  (*jdsColStartIdx)[0] = 0;
  for (int col = 0; col < maxRowNNZ - 1; ++col) {
    int count = 0;
    for (int idx = 0; idx < dim; ++idx)
      if ((*jdsRowNNZ)[idx] > col) ++count;
    (*jdsColStartIdx)[col + 1] = (*jdsColStartIdx)[col] + count;
  }
  const int NNZ = csrRowPtr[dim];
  *jdsColIdx = (int *)malloc(sizeof(int) * (NNZ > 0 ? NNZ : 1));
  *jdsData = (float *)malloc(sizeof(float) * (NNZ > 0 ? NNZ : 1));
  for (int idx = 0; idx < dim; ++idx) {
    int row = (*jdsRowPerm)[idx];
    for (int k = 0; k < (*jdsRowNNZ)[idx]; ++k) {
      int jdsPos = (*jdsColStartIdx)[k] + idx;
      int csrPos = csrRowPtr[row] + k;
      (*jdsColIdx)[jdsPos] = csrColIdx[csrPos];
      (*jdsData)[jdsPos] = csrData[csrPos];
    }
  }
}
