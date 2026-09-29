# ECE408 CUDA 知识地图 & 错题本

> 用途：CUDA 职业方向的学习资料 / 面试复习 / 查漏补缺。
> 每一节都对应仓库里一个能跑、已验证的文件，建议「读文档 → 读代码 → 自己改参数跑一遍」。
> 最后更新：2026-09（所有 `.cu` 已修复并验证，见文末「验证方式」）。

---

## 0. 一页纸总览

| 文件 | 主题 | 核心知识点 | 修复前状态 |
|---|---|---|---|
| `MP0/template.cu` | 设备查询 | `cudaGetDeviceProperties`，硬件参数的含义 | ✅ |
| `MP1/template.cu` | 向量加 | grid/block/thread 索引、边界检查、H2D/D2H 拷贝 | ✅ |
| `MP2/template.cu` | 朴素矩阵乘 | 2D 索引、全局内存访存量分析 | ❌ 主机端堆溢出 |
| `MP3/template.cu` | 分块矩阵乘 | shared memory tiling、`__syncthreads`、边界补零 | ✅ |
| `MP4/template.cu` | 3D 卷积 | constant memory、3D tiling、halo/ghost cell | ✅（已优化：tile 3→8） |
| `MP5/template.cu` | 归约 | reduction tree、控制发散、每线程处理 2 个元素 | ✅ |
| `MP6/template.cu` | 扫描（前缀和） | Brent-Kung、分层扫描（3 个 kernel） | ❌ 越界读写 |
| `MP7/template.cu` | 直方图均衡 | 原子操作、privatization、Kogge-Stone 扫描、类型转换 | ❌ 结果错误 |
| `MP8/template.cu` | 稀疏矩阵 SpMV | CSR→JDS、负载均衡、合并访存 | ✅ |
| `Project/base.cu` | 卷积 baseline（M2） | 卷积→线程映射，4D 索引宏 | ⚠️ 发散的 `__syncthreads` |
| `Project/op1.cu` | 分块共享内存卷积（2 分） | 输入 tile + halo、步长 S | ❌ 根本不是卷积 |
| `Project/op2.cu` | Unroll + 共享内存 GEMM（3 分） | im2col、GEMM 化、分批控制显存 | ❌ 没有做 unroll |
| `Project/op3.cu` | FP16（4 分） | `half`、`__hfma`、精度与累加策略 | ❌ 索引/累加错误 |
| `Project/op4.cu` | 输入通道树形归约（3 分） | 3D block、shared memory 归约 | ❌ 启动配置错，崩溃 |
| `Project/op5.cu` | 常量内存 + restrict + 循环展开（3.5 分）+ 树形归约 | `__constant__`、`__restrict__`、模板展开 | ❌ 每个 block 16384 线程 |
| `Project/custom/new-forward.cu` | 最终提交版本 | 常量内存 + restrict/展开 + 按层特化（1 分） | ❌ `cudaFree` 野指针 |

优化点合计：2 + 3 + 4 + 3 + 0.5 + 3 + 1 = **16.5 分**（M3 要求 10 分，另有 extra credit）。

---

## 1. 执行模型（MP0 / MP1）

- **层次结构**：grid → block → thread；硬件上是 SM 与 warp（32 线程）。block 会被整体调度到某个 SM 上，block 之间**不能**同步（除非用 cooperative groups 的 grid sync）。
- **全局索引**：`i = blockIdx.x * blockDim.x + threadIdx.x`；grid 大小取 `ceil(n / blockSize)`，多出来的线程必须用 `if (i < n)` 挡掉。
- **启动限制**（面试常考，MP0 会打印出来）：
  - 每个 block 最多 1024 个线程；各维上限为 x ≤ 1024、y ≤ 1024、z ≤ 64。
  - grid 的 x 维上限是 2³¹−1，y、z 维上限是 65535。
  - 静态 shared memory 默认每个 block 48 KB（更多需要用 opt-in 的动态 shared memory）。
  - constant memory 共 64 KB。
  - ❗ `op5.cu` 原来的 `BLOCK_SIZE 128` 得到 128×128 = 16384 个线程，launch 直接失败。**kernel 启动失败不会自己报错**，必须调用 `cudaGetLastError()` 去查。
- **Grid-stride loop**：`for (i = idx; i < n; i += blockDim.x * gridDim.x)`。好处是 grid 大小可以任意取，一个 kernel 适配任意 n。用例见 `op2` unroll、`op3` 类型转换、`MP7` histogram。
- **Compute capability**：Volta 是 7.0（sm_70，RAI 用的卡），Blackwell RTX PRO 6000 是 12.0（sm_120，要求 CUDA ≥ 12.8）。编译时用 `-arch=sm_XX` 指定。

## 2. 内存层次与访存优化

| 存储 | 作用域 | 延迟 | 仓库中的例子 |
|---|---|---|---|
| 寄存器 | 线程 | ~1 cycle | 累加器 `sum` / `acc` |
| shared memory | block | ~20–30 cycles | MP3、MP4、MP5、MP6、MP7、op1、op2、op4 |
| constant memory（带 cache） | grid，只读 | 同一 warp 读同一地址时可广播 | MP4 mask、op5、new-forward |
| L1/L2 → 全局内存（HBM/GDDR） | grid | 几百 cycles | 所有输入输出 |

- **合并访存（coalescing）**：同一 warp 的 32 个线程访问连续地址时，会合并成少量内存事务。
  - MP7 原代码用 `i = x * height + y`，相邻线程访问的地址相距 `height`，无法合并。改成 `y * width + x` 后就能合并。
- **算术强度（compute-to-memory ratio）**：朴素矩阵乘（MP2）每次乘加要读 2 个 float，也就是 0.25 FLOP/B。分块（MP3）后，一个 TILE 大小的块被复用 TILE 次，算术强度提高 TILE 倍。这是 roofline 分析的起点。
- **Constant memory 的正确用法**：warp 内所有线程同一时刻读取**同一个地址**才会被广播（卷积的 mask 正好满足）。如果各线程读不同地址，访问会被串行化，反而更慢。
- **Shared memory bank conflict**：共有 32 个 bank，每个 bank 4 字节宽。
  - MP4 已把 `Nds[x][y][z]` 改为 `Nds[z][y][x]`，让 `threadIdx.x` 对应连续地址。
  - op1 在步长 S > 1 时，访问地址间隔为 S，会出现 S 路 bank conflict（可作为后续优化点）。
- **`__restrict__`**：告诉编译器指针之间不存在别名，编译器因此敢把值留在寄存器里、敢重排 load，也可以走只读数据通路（`__ldg`）。
- **cudaMalloc 不会清零**：
  - ❗ MP7 原来的直方图没有 `cudaMemset`，在真卡上结果依赖显存里残留的垃圾值，时对时错，属于最难排查的一类 bug。
  - 模拟器把新分配的内存全部填成 0xFF，这类问题会稳定复现。

## 3. 同步与正确性

- `__syncthreads()` 是 **block 级 barrier**，同时保证 shared memory 的可见性。
- **规则**：同一个 block 的所有线程必须到达同一个 barrier。如果把它放进 `if (h < H_out && ...)` 里就是**发散 barrier**，编程指南明确写的是 UB。
  - `base.cu` 原来把它放在越界判断里面。在 Volta 上碰巧能跑，但完全没有必要：那段代码根本没用 shared memory。
  - `op4`、`op5`、`MP7 getColor` 也有同样的问题。
  - 正确写法（见 op1、op4）：所有线程都参与装载和 barrier，只有**计算和写回**才加边界判断。
- **tile 复用前还要再 sync 一次**：在 op1 的通道循环里，计算完成后必须再 `__syncthreads()`，才能装载下一个通道；否则快线程会覆盖慢线程还在读的数据（WAR hazard）。
- **Kogge-Stone 的读写分离**：`MP7 cdfScan` 先把 `v = s[t - stride]` 读进寄存器，然后 sync，最后再写 `s[t] += v`。少了中间那次 sync 就是数据竞争。
- **一个 block 内先写后读**：MP5、MP6 的每一层归约之间都需要 barrier。

## 4. 并行模式

### 4.1 归约（MP5，Project op4/op5）
- 每个线程先装载 2 个元素（`start = 2 * blockIdx.x * BLOCK_SIZE`），这样第一步就做了一次加法，没有空闲线程。
- stride 从 `BLOCK_SIZE` 开始逐步减半（`t < stride`）：活跃线程始终连续，warp 内不发散（这是 "convergent" 版本）。如果从 1 开始倍增，就会出现严重的控制发散。
- 进阶方向（仓库尚未实现）：最后 32 个元素改用 warp shuffle（`__shfl_down_sync`）完成，不需要 shared memory 也不需要 barrier；或者对 block 结果用 atomics 求和，一个 kernel 就能完成整个数组的归约。
- **op4 输入通道树形归约**：block 形状为 (T, T, CZ)，CZ 个通道并行计算部分和，再在 shared memory 里用 log₂CZ 步合并。适用于 C 大、H_out×W_out 小的层。

### 4.2 扫描（MP6，MP7）
- **Brent-Kung**（MP6）：up-sweep 加 down-sweep，总工作量 O(n)，步数为 2·log n。work-efficient，适合大数组。
- **Kogge-Stone**（MP7 cdfScan）：步数 log n，但工作量是 O(n log n)。适合小数组（256 个 bin）。
- **分层扫描**：任意长度的数组分三步完成：
  1. 每个 block 扫描自己的一段，并输出 block 总和到 aux。
  2. 扫描 aux。
  3. 把前缀加回每个 block。
  - ❗ MP6 原来在第 2 步把长度写死成 `2*BLOCK_SIZE`，而 aux 只有 `numBlocks` 个有效元素。输入小于 1024 时会越界读写并破坏堆。**kernel 的长度参数必须等于实际有效数据长度。**

### 4.3 直方图（MP7）
- 基础版：全局 `atomicAdd(&hist[bin], 1)`。热点 bin 上竞争严重。
- **Privatization**：每个 block 在 shared memory 里维护一份私有直方图，用 shared atomics 累加（快很多），最后再合并到全局，每个 bin 只需一次全局 atomic。
- 进阶：thread coarsening（每个线程处理多个像素，已通过 grid-stride 实现）；对连续相同值做 aggregation。

### 4.4 卷积（MP4，Project）
- **Halo / ghost cells**：输出 tile 需要的输入区域比 tile 本身大一圈，边长为 `(T−1)·S + K`。越界部分补 0（MP4 做了 padding，Project 的卷积不做 padding）。
- **3 种 tiling 策略**：
  1. block 大小 = 输入 tile；
  2. block 大小 = 输出 tile（op1、MP4 采用这种，协作装载）；
  3. 只把内部数据放 shared memory，halo 直接走 cache。
- **卷积转 GEMM**（op2）：
  - 先 unroll：`X_unroll[C·K·K][H_out·W_out]`。
  - 再算 `Y[M][H_out·W_out] = W[M][C·K·K] × X_unroll`。
  - 代价：输入被放大 K²/S² 倍（本项目是 49 倍），所以要**分批**处理，控制显存占用（`UNROLL_BUDGET`）。
  - 下一步是 **kernel fusion**（再加 2 分）：在 GEMM 装载 tile 的时候直接从原始输入里按 unroll 的下标去取，不再物化 X_unroll，省掉一次完整的写和读。
- **按层特化**（new-forward）：`template<int K, int C>` 让 LeNet 的两层（K=7, C=1/4）在编译期确定循环边界，`#pragma unroll` 可以完全展开，同时去掉除法和取模。

### 4.5 稀疏矩阵（MP8）
- **CSR**：每行一个线程（负载不均，warp 内的访存也不合并）。
- **ELL**：padding 到每行相同长度，并按列主序存放，访存可以合并，但 padding 会浪费空间。
- **JDS**（MP8）：
  - 先按每行非零元个数排序（`jdsRowPerm`），相邻线程的工作量接近；
  - 再按「第 k 个非零元」转置存储（`jdsColStart[k] + row`），访存可以合并。
  - 结果最后要按 `out[perm[row]]` 写回。
- **COO / 混合 ELL+COO**：处理极长的行。

## 5. 数值精度（op3 FP16，MP7）

- `half` 只有 11 位尾数（约 3 位十进制有效数字），最大值 65504。
- ❗ op3 原来写成 `sum += __hadd(__hmul(x, w), sum)`，相当于每一步都把 sum 翻倍（`sum = 2·sum + x·w`）。
- **长求和的误差会累积**：修复后，每个通道的 K² 个乘积用 `__hfma` 在 half 精度下累加，通道间的部分和再用 float 累加（混合精度，与 Tensor Core 的 FP16 输入 / FP32 累加思路一致）。实测最大绝对误差约 1e-2，相对误差约 1e-3。
- 更进一步的方向：用 `half2` 做 SIMD（一条指令算 2 个 FP16，吞吐翻倍）；或者使用 Tensor Core（WMMA / MMA，5 分）。
- **MP7 的教训：以数据为准**。README 的伪代码写的是 `(unsigned char)(0.21r + 0.71g + 0.07b)`（截断），但数据集的期望输出是**四舍五入**生成的。我用 numpy 逐一验证了几种假设，最终确认了这一点。遇到对不上期望输出的情况，先写 reference 模型验证假设，不要去猜。
- 浮点加法不满足结合律：并行归约和串行求和的结果会有微小差异，所以比对时要用相对误差加绝对误差，不能要求 bit-exact。

## 6. 主机端与 API 细节

- `cudaMemcpy` 的大小必须对应**目标缓冲区**的大小。
  - ❗ MP2 原来按 C 的大小往 `hostA`、`hostB` 里拷贝，当 A 或 B 比 C 小时就会发生主机堆溢出，程序在之后的 `free()` 时才崩溃，表现为 `corrupted size vs. prev_size`。
- `cudaFree` 只能释放 `cudaMalloc` 返回的指针，释放 `nullptr` 是合法的空操作。
  - ❗ `new-forward.cu` 和 `op5` 把 mask 放在 `__constant__` 里，却从没给 `*device_mask_ptr` 赋值，epilog 于是 `cudaFree(野指针)`。修复方法是在 prolog 里设 `*device_mask_ptr = nullptr`。
- `cudaMemcpyToSymbol(const_mem, src, bytes)`：拷贝前要检查 `bytes <= sizeof(symbol)`。
- 错误检查：
  - 每次 launch 之后用 `cudaGetLastError()` 检查配置错误；
  - 同步之后再检查执行错误；
  - 调试时可以设置 `CUDA_LAUNCH_BLOCKING=1`。
- kernel launch 是异步的；`cudaFree`、`cudaMemcpy`（非 Async 版本）会隐式同步。所以要计时 kernel，就得先 `cudaDeviceSynchronize()`，或者使用 `cudaEvent`。
- 用 `size_t` 或 `long long` 做大下标：`B·C·H·W` 在 B=10000 时仍在 int 范围内，但 op2 的 unroll 缓冲区（B·CKK·HW）会溢出 int。

## 7. 性能分析工具（Project M2/M3）

- **Nsight Systems**（`nsys profile --stats=true ./m3`）：看时间线，关注 kernel、memcpy、API 各占多少时间，以及能否用 stream 做重叠。
- **Nsight Compute**（`ncu --section '.*'`）：分析单个 kernel，包括
  - occupancy；
  - 访存吞吐（DRAM / L2 / L1 / shared）；
  - warp stall 原因；
  - shared memory bank conflict；
  - roofline。
- `nvcc -Xptxas -v`：查看寄存器用量和 spill。当前所有 kernel 在 sm_120 上用 35–40 个寄存器，0 spill。
- `compute-sanitizer`（旧名 `cuda-memcheck`）：检查越界、未初始化读、竞争条件（`--tool racecheck`）。

## 8. 错题本（本次修复汇总）

| # | 文件 | 症状 | 根因 | 知识点 |
|---|---|---|---|---|
| 1 | MP2 | 部分数据集崩溃 | D2H 拷贝写进了错误的主机缓冲区，而且大小不对 | 缓冲区所有权与大小 |
| 2 | MP6 | 512/1000 元素输入崩溃 | 第二级扫描长度写死成 1024 | 分层扫描；长度参数 |
| 3 | MP7 | 结果全部错误 | 直方图未清零；没有截断成 uchar；灰度需要四舍五入 | cudaMalloc 不清零；数值语义 |
| 4 | MP7 | UB | `__syncthreads` 放在 if 里面 | barrier 规则 |
| 5 | base | UB，而且更慢 | 循环里有多余的发散 barrier | 只在需要时同步 |
| 6 | op1 | 越界写，结果错误 | 把卷积写成了矩阵乘的下标 | tiled 卷积与 halo |
| 7 | op2 | 结果错误 | 缺少 unroll，下标对应的不是卷积 | im2col、GEMM |
| 8 | op3 | 结果错误 | grid 维度和下标对不上；sum 翻倍；没有乘步长 | FP16、索引一致性 |
| 9 | op4 | 崩溃 | blockDim.z = 1，动态 shared memory 是 0 字节；barrier 发散 | 3D block、动态 shared memory |
| 10 | op5 | launch 失败 | 每个 block 16384 个线程 | 启动限制 |
| 11 | op5 / new-forward | CUDA 报错 | `cudaFree` 未初始化的指针 | API 契约 |
| 12 | MP4 | 能跑但低效 | tile = 3，block 只有 27 个线程，而且下标写死 | occupancy、halo 比例 |

**共性教训**：
1. 先写 CPU reference 和自动比对，再去优化。
2. 所有 launch 都要检查错误。
3. barrier 要放在所有线程都会执行到的地方。
4. 边界判断只保护读写，不保护同步。

## 9. 面试自测题（能对着代码讲清楚就算过关）

1. 为什么 MP3 的 TILE 取 16 或 32？block 大小如何影响 occupancy？每个 SM 的寄存器和 shared memory 预算怎么算？
2. MP5 里 stride 从大到小和从小到大有什么区别？各自的 warp divergence 是什么情况？
3. Brent-Kung 和 Kogge-Stone 的工作量、步数分别是多少？各适用于什么场景？
4. 直方图 privatization 为什么快？如果 bin 数量很大（比如 64K）该怎么办？
5. constant memory 什么时候比 global memory 慢？
6. im2col 把输入放大了多少倍？kernel fusion 省掉了什么？
7. FP16 为什么要用 FP32 累加？Tensor Core 的 MMA 形状是多少（例如 16×16×16）？
8. 怎样用 stream 让 H2D 拷贝与计算重叠？为什么需要 pinned memory？
9. `__restrict__` 会改变程序语义吗？什么情况下使用它是错误的？
10. 用 Nsight Compute 看到 "long scoreboard" stall 占主导，说明了什么？

## 10. 下一步（按优先级）

- [ ] 在本机 Blackwell 上运行 `tools/run_all_windows.ps1`，记录各实现在 batch 5000 下的 Op Time，并做**参数扫描**（BLOCK_SIZE 取 8/16/32，+0.5 分）。
- [ ] **Kernel fusion**（unroll + GEMM，+2 分）：在 op2 的基础上把 unroll 融进 GEMM 的 tile 装载里。
- [ ] **寄存器分块 GEMM**（+5 分）：每个线程计算 4×4 或 8×8 个输出，shared memory 采用 double buffering。
- [ ] **Tensor Core**（+5 分）：用 WMMA 的 `fragment<matrix_a,16,16,16,half,...>` 来做 op2 的 GEMM。
- [ ] **Streams**（+4 分）：按 batch 分段，交替执行 H2D、kernel、D2H；主机端内存用 `cudaHostAlloc`。
- [ ] warp shuffle 归约、`half2` 向量化、`float4` 向量化 load。
- [ ] Blackwell 新特性：TMA、thread block cluster / distributed shared memory、FP8/FP4 Tensor Core。

---

## 附：验证方式

| 方式 | 命令 | 说明 |
|---|---|---|
| 无 GPU（CPU 模拟器） | `tools/run_mps.sh sim`、`tools/run_project.sh sim` | 线程用 fiber 模拟，barrier 语义精确；未初始化内存填 0xFF；检查启动配置；`SIM_REVERSE=1` 以反向线程顺序运行，用于暴露竞争条件 |
| Linux + GPU | `tools/run_mps.sh gpu`、`tools/run_project.sh gpu` | 使用 nvcc 编译真机运行；Project 额外跑 batch 5000 计时 |
| Windows + GPU | `powershell -ExecutionPolicy Bypass -File tools\run_all_windows.ps1` | 所有产物写到 `D:\Claude\ece408-build` |
| 单个 Project 实现 | `nvcc -O3 -arch=sm_120 -I Project/custom -DOP_ID=6 Project/test/test_ops.cu` | OP_ID：0 = base，1–5 = op1–op5，6 = new-forward |

云端验证结果（CUDA 12.9 nvcc 在 sm_70 + sm_120 下全部编译通过，零警告；CPU 模拟器在正向和反向线程顺序下都通过）：

- **MP1–MP8**：73/73 个数据集通过。
- **Project**：7 个实现 × 7 个用例全部通过。用例包括 m1/m2/m3 的 4 个测试、LeNet 的两层、一个非方形 K=5 的通用用例；op2 的分块路径也单独验证过。
- **尚未验证**：在真实 GPU 上的性能数字和 RAI 上的最终精度。
