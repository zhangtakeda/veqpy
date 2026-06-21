# VEQlib 实验记录与下一步计划

更新时间：2026-06-21  
范围：`TODO-1.md`、`TODO-2.md` 指向的 VEQlib hot path / cache / tooling 初步实验。

## 0. 最新状态：Phase 1a 已完成，PF/psin/uniform/Ip kernel 不再有 checked facade

此前已把阶段性变更提交为 git commit：

- `7566f97 Establish measurable VEQlib optimization baseline`
- `a5c4d3c Reduce geometry cache interference before chasing trig`
- `3851f62 Preserve VEQlib performance decision trail`
- `a4637fd Reprioritize VEQlib optimization by evidence`
- `4715b04 Separate VEQlib FP contracts before math experiments`
- `e161475 Make RELAXED the VEQlib performance baseline`

Phase 0 已把 FP 构建语义拆成 `STRICT` / `FMA` / `RELAXED`，并确认后续
性能 A/B 应默认只在 `RELAXED` 下比较。根据最新修正，Phase 1a 进一步
完成：VEQlib 作为 kernel 层不再返回“是否求解成功”，也不保留
checked/unchecked 双 facade。当前 `PF/psin/uniform/Ip` 路径的实现决策：

- `evaluate()`、source materialization、source update 都是唯一的 `void`
  route-kernel 路径。
- timed/source/residual kernel 内不再调用 `is_valid_magnitude()`，也不再用
  NaN/Inf/极大值 guard 或 `1e20` fallback 作为控制流。
- `math::is_finite()` 只保留 bit-level NaN/inf 诊断/测试语义；C++ kernel
  不据此判断求解成败。
- `PF/psin/uniform/Ip` 路径已移除 free/beta source 分支；外层 solver 或
  Python/C++ validation 只根据残差范数和 solver info 判断是否成功。
- 新增 `clang-release-strict` 和 `clang-release-fma` preset，默认
  `clang-release` 仍是 `RELAXED`，因此不会破坏当前性能基线口径。

验证留痕：

- compile commands 检查确认：
  - `release`: `VEQLIB_FP_MODE_RELAXED=1`，含 `-ffast-math` /
    `-ffinite-math-only` / `-fapprox-func`
  - `release-strict`: `VEQLIB_FP_MODE_STRICT=1`，含 `-fno-fast-math`
    / `-ffp-contract=off`
  - `release-fma`: `VEQLIB_FP_MODE_FMA=1`，含 `-fno-fast-math`
    / `-ffp-contract=fast`
- Debug / RELAXED release / STRICT release / FMA release CTest 均通过。
- Phase 1a 后 debug/release CTest 均通过；RELAXED release Python/C++
  `PF/psin/uniform/Ip` validation 通过，`max_abs≈5.577e-11`。
- Phase 0 RELAXED sanity benchmark（非 paired，仅确认未明显破坏基线）：
  `geometry median≈5554 ns/call`，`evaluate median≈9109 ns/call`。
- 同窗口 FP mode A/B（3 轮 median-of-medians，`repeat=25`、
  `warmup=8`、`inner=10000`）显示 RELAXED 是后续性能实验的唯一合理主基线：

  | FP mode | `geometry` ns/call | `evaluate` ns/call | `evaluate / STRICT` |
  | --- | ---: | ---: | ---: |
  | `STRICT` | 12223.9 | 16498.8 | 1.000 |
  | `FMA` | 10731.0 | 15059.6 | 0.913 |
  | `RELAXED` | 5391.8 | 8907.0 | 0.540 |

  结论：后续性能 A/B 默认只比较 RELAXED build 下的修改前/修改后；
  STRICT/FMA 只作为 correctness / error-budget 参照，不再作为热点优化的
  性能基线。

Phase 1a 的 RELAXED stage 表（三轮同窗口；每轮 `taskset -c 2`、`repeat=25`、`warmup=8`、
`inner=10000`；表中为每轮 median 的中位数）如下：

| stage | ns/call | `evaluate` share |
| --- | ---: | ---: |
| `profiles_all` | 124.5 | 1.5% |
| `geometry` | 5148.0 | 60.4% |
| `source_materialize` | 895.6 | 10.5% |
| `source_update` | 948.0 | 11.1% |
| `residual_update` | 914.9 | 10.7% |
| `residual_pack` | 135.5 | 1.6% |
| `evaluate` | 8521.5 | 100% |

相对上一轮 RELAXED baseline（`evaluate≈8907.0 ns/call`、
`geometry≈5391.8 ns/call`），route 纯化和 guard 删除让 `evaluate` 再降约
4.3%。下一步进入 Phase 1b/2：补 topology/state-ring matrix，并把 Geometry
拆成 phase synthesis / dynamic sincos / metric / stores 微阶段。

Phase 2 首轮 micro A/B：

| candidate | paired result | decision |
| --- | --- | --- |
| `JdivR = J*J/(J*R)` 复用 `inv_JR`，减少一个显式除法 | 5 组 paired median：`geometry` ratio≈0.992，`evaluate` ratio≈0.990；但 evaluate 有反向组（1.031、1.009） | 回滚；收益太小且不稳定 |
| Geometry harmonic profile reads hoist 到 `i` 层 | 5 组 paired median：`geometry` ratio≈0.984，`evaluate` ratio≈0.993；5/5 evaluate 均快 | 保留；低风险小收益，为高 `M_max` topology 预期更有价值 |

随后用独立 baseline worktree 重新测试 `a5c4d3c` 的 geometry surface layout 改动。复测仍然支持保留该改动：

| metric | baseline | current layout | current / baseline |
| --- | ---: | ---: | ---: |
| `geometry` median-of-medians ns/call | 9297.0 | 5378.6 | 0.579 |
| `evaluate` median-of-medians ns/call | 12879.2 | 8883.5 | 0.687 |
| `geometry / evaluate` share | 0.723 | 0.608 | - |

口径：`taskset -c 2` 可用时固定 CPU；5 组 paired run；每组 `repeat=30`、`warmup=8`、`inner=10000`；比较同一时间窗口内的相对变化。

在 layout 之后又测试了多条“小改动”候选，但没有新的代码改动达到端到端保留门槛：

| candidate | target stage result | `evaluate` result | decision |
| --- | ---: | ---: | --- |
| 跳过 absent Fourier orders | `geometry` ratio 0.988 | 1.001 | 回滚；stage 线索太小且端到端无收益 |
| residual load hoist | `residual_update` ratio 0.974 | 0.996 | 回滚；端到端收益不足 |
| glibc `sincos` 显式调用 | `geometry` ratio 2.386 | 1.835 | 回滚；明显退化 |
| residual physical layout 改为 `[rho][field][theta]` | `residual_update` ratio 0.927 | 0.991 | 回滚；端到端收益噪声级，且 `residual_pack` 退化 |
| source `psin_r` regularize / pass 合并 | `source_update` ratio 0.954 | 1.000 | 回滚；source stage 有线索但端到端无收益 |
| geometry hot-loop 访问器打平 | `geometry` ratio 1.003 | 1.000 | 回滚；编译器已基本消除访问器开销 |

当前结论：下一轮不要再优先做“访问器打平、单个 pass hoist、显式 `sincos`”这类微调；应转向更结构性的 geometry 数学路径实验，或在原生 Linux/PMU 可用环境中先验证 cache 和 libm 事件。

## 1. 本轮目标

本轮工作的目标不是直接大改 kernel，而是先补齐 TODO 中缺失的证据链：

1. 能稳定复现 Python/C++ 正确性基线。
2. 能把完整 `evaluate()` 拆成可重复测量的 hot-path stage。
3. 能确认当前机器上哪些性能工具可用、哪些不可用。
4. 基于实测结果决定下一步优化优先级，而不是只按直觉修改布局或写 SIMD。

## 2. 已完成工作

### 2.1 阅读与拆解 TODO

`TODO-1.md` 的核心提醒：当前 VEQlib 性能上限主要受重复计算、重复清零、中间 slab 过多、residual 重扫、source 分支/复制、geometry 标量 `sin/cos`、以及缺少逐阶段 microbenchmark 影响。它同时指出 Release FP 语义和有限性检查可能存在正确性风险。

`TODO-2.md` 的核心提醒：不要只依赖通用 profiler；应建立 `Clang AST -> LLVM/objdump -> PMU` 的证据链。它提出了一个很具体的默认 topology 风险：`Nr=32, Nt=16` 时，`Tensor<double, 9, Nr, Nt>` 的每个 field plane 为 4096 B，可能让 9 个 geometry plane 映射到同一 L1D set，从而产生 8-way L1 上的 conflict eviction。

### 2.2 新增实验入口

新增文件：

- `veqlib/stage_benchmark.cpp`

新增 CMake target：

- `veqlib_stage_benchmark`

该 target 对默认 `PF/psin/uniform/Ip` case 做 lower-level timing，不经过 CMINPACK solve loop，可单独测：

- `profiles_fixed`
- `profiles_active`
- `profiles_all`
- `geometry`
- `source_materialize`
- `source_update`
- `residual_update`
- `residual_pack`
- `evaluate`

示例：

```bash
./veqlib/build/release/veqlib_stage_benchmark \
  --stage geometry \
  --repeat 40 \
  --warmup 10 \
  --inner 10000
```

输出为 JSON，`samples_ns` 已除以 `--inner`，单位是 ns/stage-call。

### 2.3 新增 analysis build 开关

修改文件：

- `veqlib/CMakeLists.txt`
- `veqlib/README.md`

新增 CMake option：

```cmake
VEQLIB_ANALYSIS_BUILD=ON
```

用途：在保持 `-O3` 的同时禁用该 build 的 ThinLTO，并打开源码相关诊断：

- `-fno-omit-frame-pointer`
- `-fdebug-info-for-profiling`
- `-fno-unroll-loops`
- Clang vectorization remarks
- `-fsave-optimization-record=yaml`
- `-Wframe-larger-than=4096`
- `-Wlarge-by-value-copy=256`
- `-fstack-usage`

## 3. 工具状态

当前机器是 WSL2 环境，kernel 报告为：

```text
6.6.87.2-microsoft-standard-WSL2
```

已确认可用：

- `clang-tidy-18`
- `clang-query-18`
- `clang-format-18`
- `llvm-objdump-18`
- `valgrind` / Cachegrind
- Clang optimization remarks

受限或不可用：

- `/usr/bin/perf` wrapper 与当前 WSL kernel 不匹配。
- 直接调用 `/usr/lib/linux-tools/6.8.0-124-generic/perf` 可以运行，但 hardware events 显示 `<not supported>`。
- `likwid-topology` 可运行，但 `likwid-perfctr` 因 processor/MSR access 不支持而无法给出 PMU 结果。

结论：本轮不能在此环境直接验证 L1D replacement / line fill / store-RFO 等硬件 PMU 事件。4 KiB field-plane conflict 目前只能作为强假设，用 Cachegrind、objdump 和 stage timing 辅助排序；最终应在原生 Linux 或支持 PMU 的 VM 上复测。

## 4. 当前实验结果

完整原始输出保存在：

- `veqlib/experiments/summary.md`
- `veqlib/experiments/perf/`
- `veqlib/experiments/cachegrind/`
- `veqlib/experiments/tooling/`
- `veqlib/experiments/baseline/`

最终 release stage timing：

| stage | median ns/call | avg ns/call | p95/median | CV |
| --- | ---: | ---: | ---: | ---: |
| `profiles_all` | 133.7 | 134.7 | 1.082 | 0.049 |
| `geometry` | 9330.3 | 9416.4 | 1.045 | 0.036 |
| `source_materialize` | 975.4 | 982.9 | 1.034 | 0.036 |
| `source_update` | 922.4 | 922.7 | 1.030 | 0.023 |
| `residual_update` | 1073.5 | 1072.1 | 1.028 | 0.018 |
| `residual_pack` | 144.6 | 145.1 | 1.053 | 0.031 |
| `evaluate` | 12917.5 | 12919.0 | 1.034 | 0.016 |

主要结论：

- `geometry` 占完整 `evaluate` median 时间约 **72.2%**。
- `profiles_all` 和 `residual_pack` 当前不是第一墙钟瓶颈。
- `source_materialize`、`source_update`、`residual_update` 都在约 0.9--1.1 us 量级，仍值得后续做结构性简化，但第一优先级低于 geometry。
- Cachegrind 显示 `residual_update` 的 D1 miss/ref 行为值得关注，但没有 PMU 前不要把它误判为首要墙钟瓶颈。
- Release binary 的动态符号和 objdump 证据显示存在 `sincos@plt` 调用，支持 TODO 中“geometry 标量三角函数可能是当前首要瓶颈”的判断。
- Clang analysis build 显示 residual 相关循环已有 vectorization width 4 的优化 remarks；因此下一步不要先盲目手写 residual AVX2，应先针对 geometry 做 A/B。

### 4.2 后续 A/B 更新：geometry surface layout

已完成一个最小 kernel A/B：保持 logical accessor `surface_field(row, radial_node, theta_node)` 不变，把 geometry surface slab 的物理布局从 `[field][rho][theta]` 改成 `[rho][field][theta]`。这样 `theta` 仍连续，同时默认 topology 下不再让每个 field plane 起点按 4096 B 间隔排列。

同一时间窗口 paired timing，`repeat=30`、`warmup=8`、`inner=10000`、3 轮 median-of-medians：

| metric | baseline | candidate | candidate / baseline |
| --- | ---: | ---: | ---: |
| `geometry` ns/call | 9476.8 | 5457.7 | 0.576 |
| `evaluate` ns/call | 13149.6 | 8796.7 | 0.669 |
| `geometry / evaluate` share | 0.721 | 0.620 | - |

结论：layout 假设在当前工作站 wall-clock A/B 中得到强支持。由于 WSL2 下 PMU 不可用，不能把“4 KiB set conflict”机制当作硬件计数器已证实结论；但按相对变化口径，这个布局改动已经让 `geometry` 和端到端 `evaluate` 同步明显下降，应该保留并作为下一轮 baseline。

### 4.1 性能评价口径：占比和相对变化优先

当前 `veqlib_stage_benchmark` 的原始时间是进程内 `std::chrono::steady_clock` wall-clock timing，经 `--inner` 除成 ns/stage-call。这个绝对 ns 值只应作为原始观测记录，不应作为主要裁决口径。后续优化应主要看：

1. **stage 时间占比**：同一轮实验里 `stage_median / evaluate_median`。
2. **修改前后相对变化**：同一时间窗口内 `candidate_median / baseline_median`，或 speedup `baseline_median / candidate_median`。
3. **端到端收益**：最终以 `evaluate` 的相对变化作为保留优化的主标准。
4. **贡献估计**：用 `baseline_stage_share * stage_relative_improvement` 粗略估计该 stage 优化对 `evaluate` 的上限贡献。

也就是说，`geometry = 9330.3 ns/call` 这类绝对值只是留痕；更重要的是本轮 `geometry / evaluate ≈ 72.2%`，以及后续某个改动是否能让 `geometry` 和 `evaluate` 相对 baseline 同步下降。

绝对 wall-clock 值会受到以下因素影响：

- 后台进程抢占和调度迁移。
- CPU 频率、turbo、温度和功耗状态。
- WSL2 host/guest 调度和内核工具限制。
- 编译、浏览器、索引器、IO、内存压力等同时运行任务。
- 小 stage 的计时粒度和测量开销。

因此不要把单次绝对 ns 理解成永久常数。工作电脑不可能只运行 benchmark，所以后续报告默认用占比和相对变化表达结论。

后续 A/B 应采用抗噪 protocol：

1. 尽量比较同一时间窗口内的 baseline/candidate，不跨天直接比较绝对值。
2. 采用 paired run：`baseline -> candidate -> baseline -> candidate`。
3. 每个 pair 内记录：
   - `stage_share = stage_median / evaluate_median`
   - `stage_ratio = candidate_stage_median / baseline_stage_median`
   - `evaluate_ratio = candidate_evaluate_median / baseline_evaluate_median`
4. 对 `geometry`、`evaluate` 这种关键 stage 至少做 3 轮独立 pair；报告 ratio 的 median 和离散程度，而不是只报一次 avg。
5. 小于几百 ns 的 stage 增大 `--inner`，否则容易被 timer overhead 和调度噪声淹没。
6. 非隔离桌面环境下，低于约 5% 的相对改善只算线索；稳定超过约 10--15% 且 `evaluate` 同步改善，才适合作为优化决策。
7. 有条件时用 `taskset` 固定 CPU、暂停重 IO/编译任务；但不能假设用户机器只运行 benchmark。
8. 对 cache-set conflict、L1 replacement、store/RFO 等结论，仍需要原生 Linux 或 PMU 可用环境复测。

## 5. 正确性与验证留痕

本轮验证命令：

```bash
clang-format-18 --dry-run --Werror veqlib/stage_benchmark.cpp
cd veqlib && cmake --build --preset clang-release --target veqlib_stage_benchmark --clean-first
cd veqlib/build/debug && ctest --output-on-failure
.venv/bin/python veqlib/compare_pf_psin_uniform_veqpy.py \
  --cxx-exe veqlib/build/debug/veqlib_pf_psin_uniform_validation \
  --tolerance 1e-9
git diff --check
```

结果：

- Debug CTest：`3/3 passed`
- Python/C++ validation：`passed=true`
- 最大差异：`max_abs=6.7619868038271136e-12`
- `veqlib_stage_benchmark` release clean rebuild 成功
- whitespace diff check 通过

## 6. 当前工作树留痕

已提交的代码/工具改动：

```text
7566f97 Establish measurable VEQlib optimization baseline
a5c4d3c Reduce geometry cache interference before chasing trig
```

本次复测之后没有保留新的 kernel 代码改动；无效候选均已回滚，只更新 Notes 和实验 summary。当前未版本化的原始实验目录包括 paired A/B JSON、Cachegrind、tooling 输出等，位于：

```text
veqlib/experiments/*_retest_20260621/
veqlib/experiments/cachegrind/
veqlib/experiments/perf/
veqlib/experiments/tooling/
```

`veqlib/experiments/` 是实验数据目录；如果后续不希望把原始 logs/JSON 全部纳入版本库，可只保留：

- `veqlib/experiments/summary.md`
- 少量关键 JSON
- 或将完整实验数据移到外部 artifact 存储

但在做下一轮优化前，建议先保留完整目录，便于复现和比较。

## 7. 修订后的下一步计划

这份计划采纳 2026-06-21 新指导建议：`a5c4d3c` 的 layout A/B 已足够证明“allocation-free 不等于 cache-efficient”；PMU 是机制验证线，不再阻塞工程优化主线。当前首要工作从“继续随手试 geometry/trig 微调”改为先稳定 FP/correctness contract，再补齐 post-layout stage/topology 证据，最后进入结构性 geometry/residual 实验。

### Phase 0：FP 构建语义与 correctness contract（P0）

目标：先拆清楚 Release 数学语义，避免后续 vector sincos、polynomial、FMA、reciprocal 实验的误差基准不稳定。

当前风险（已处理）：Release 同时启用了 `-ffast-math`、`-ffinite-math-only`、`-fapprox-func`、`-freciprocal-math`、`-funsafe-math-optimizations`；因此 kernel 内不应依赖 NaN/Inf 或“极大值”判断来表达求解失败。

建议动作：

1. 引入明确 FP mode：
   - `STRICT`: `-fno-fast-math -ffp-contract=off`
   - `FMA`: `-fno-fast-math -ffp-contract=fast`
   - `RELAXED`: 保留历史 fast-math 性能基线，作为所有后续性能 A/B 的主模式
2. `math::is_finite` 只保留为 bit-level 诊断/测试 helper；kernel 不再用
   finite/magnitude policy 做 hot-path 控制流。
3. 每个 FP mode 都跑 Python/C++ validation，记录 max_abs、solver nfev；性能 stage timing
   只用 RELAXED 做主裁决。

停止条件：三种 FP mode 的 correctness contract 明确；RELAXED 作为性能主基线，STRICT/FMA
只用于解释误差/速度 tradeoff；route kernel 不承载求解成功/失败语义。

### Phase 1a：剥离为唯一 PF/psin/uniform/Ip route kernel 路径

目标：先把当前实验路线彻底固定为 `PF/psin/uniform/Ip`，避免 generic source
constraint / topology 逻辑、checked facade 和 hot-path validity/magnitude guard 混入判断。

建议动作：

1. route 语义静态化：benchmark target 明确只测 `PF/psin/uniform/Ip`，
   不在 timed path 中处理 beta/free-source 分支。
2. timed hot path 是唯一 evaluator：假设 route 输入由外层保证合法，不做
   `is_valid_magnitude()`、finite check 或“极大值”判断。
3. 不保留 checked evaluator / public facade；C++ kernel 只产出残差，不表达
   solve success。
4. 正确性由 timed loop 外部的 C++ smoke test、Python/C++ compare 和
   STRICT/FMA/RELAXED 差异表承担。

停止条件：确认后续 stage/topology benchmark 只测 route kernel 本身；
outer solver/validation 只从残差范数解释成功与否。

### Phase 1b：补齐 post-layout stage 表与 topology/state matrix

目标：新布局已验证 `geometry/evaluate`，但还缺 layout 后完整 stage 排序，以及 topology resonance 检查。

建议动作：

1. 输出 layout 后完整 stage 表：
   - `profiles_all`
   - `geometry`
   - `source_materialize`
   - `source_update`
   - `residual_update`
   - `residual_pack`
   - `evaluate`
2. 对旧/新布局或当前 baseline 输出同一张表：
   - stage ns
   - stage ratio
   - `stage/evaluate` share
   - Amdahl contribution estimate
   - Cachegrind D refs / D1 misses（WSL 下仅作模拟参考）
3. 增加 topology matrix，至少覆盖：
   - `Nr = 16, 32, 64`
   - `Nt = 8, 16, 24, 32, 64`
   - `Mmax = 1, 4, 8`
4. 增加 solver-state 输入模式：
   - same-x warm benchmark
   - 16--64 个有效 solver state 的 ring benchmark

停止条件：确认 `[rho][field][theta]` 在默认 topology 是默认选择；对其它 topology 只声明“已测范围内”的结论，必要时保留 `GeometryLayoutPolicy<GridType>` 设计空间。

### Phase 2：拆分 Geometry micro-stage

目标：把“继续研究 sin/cos”改成可计量的 geometry 内部阶段，而不是模糊地试 trig 优化。

建议拆成四个 micro-stage：

```text
A. Fourier phase synthesis:
   tb, tb_r, tb_t, tb_rr, tb_rt, tb_tt

B. dynamic sincos:
   sin(tb), cos(tb)

C. metric arithmetic:
   R/J/JR/grt/gtt 和导数

D. output traffic:
   九个 surface field 写入和五个 radial reduction
```

注意：固定 `sin(theta)`、`cos(theta)` 和 harmonic 表已经通过 `GridType` setup 预计算；真正动态且无法 setup 预计算的是 `sin(tb_ij)` 和 `cos(tb_ij)`，因为 `tb_ij` 依赖 active profile coefficients。因此删除“预计算 theta trig”作为主线，改为测试动态 `sincos(tb)` backend。

停止条件：能量化每次 Geometry 的实际 dynamic sincos 调用数、phase synthesis 占比、metric arithmetic 占比和 output traffic 占比。

### Phase 3：Geometry dynamic sincos backend A/B

目标：在 Phase 2 的拆分证据下，测试真正可能带来较大收益的动态数学路径。

建议顺序：

1. 捕获或生成 `tb[Nr*Nt]` 输入，单独 benchmark scalar libm。
2. 测 AVX2/SVML/SLEEF/libmvec 或受控误差的 domain-specific vector sincos backend。
3. 将 vector sincos 嵌入 Geometry，检查端到端是否仍有收益。
4. 对比 assembly/optimization remarks：vectorization width、call site、spill/reload、寄存器压力。
5. 严格跑 Python/C++ validation 与 FP mode matrix。

已删除方向：显式 glibc `::sincos`。它已严重退化，且 baseline 已有 compiler-lowered `sincos@plt` 证据。

### Phase 4：压缩 Geometry descriptor，减少 Residual 重算

目标：layout 改动改善了九个 field 的访问方式，但没有删除九个二维 field 的写入/读取。下一步应审计能否存储 residual 直接需要的派生量，而不是继续换排列。

候选派生量：

```text
qR      = -Z_t / J
qZ      =  R_t / J
R2      = R * R
dmetric = gttdivJR_r - grtdivJR_t
```

这些可以减少 Residual 中的 `1.0 / J`、`R * R`、两个 field load 后相减等重算。任何 descriptor 压缩都必须先做 correctness comparator 和 stage/evaluate A/B。

### Phase 5：Residual theta-moment fusion

目标：删除四个 residual 二维中间场，而不是再尝试单一 residual layout。

已知证据：residual physical layout 改成 `[rho][field][theta]` 时，`residual_update` 快约 7.3%，但 `residual_pack` 退化，`evaluate` 只有噪声级收益。这说明 update 和 pack 需要相反 locality：

```text
update：同一点生成多个 field
pack：固定一个 field 扫过 rho/theta
```

正确方向是删除中间 slab：

```text
G
G*psin_R
G*psin_Z
G*psin_R*sin(tb)
```

在一次 theta sweep 中直接累计 `rowwise_sum` / `rowwise_weighted_sum` 所需 moments，再做 radial projection。

### Phase 6：Source/profile route 静态化

目标：在 geometry/residual 结构性收益之后，再处理固定成本。

建议动作：

1. `PF/psin/uniform/Ip` 编译期实例已先消除不可能的 route/beta 分支；后续只处理仍有实测占比的 source/profile 固定成本。
2. fixed profiles 尽量迁到 setup/plan 阶段，只在参数变化时刷新。
3. 检查 family slab 是否在 `refresh_fixed` 和 `refresh_active` 中重复清零/重建。
4. 保持 source 单 pass 微调降级；只有端到端 `evaluate` 有同步收益才保留。

### Phase 7：PMU / native Linux 并行验证线

目标：验证 4 KiB field-plane conflict 和 libm/cache 机制，而不是决定 layout 是否保留。

当前 WSL2 不能读取硬件 PMU；`cycles`、`instructions`、`L1-dcache-load-misses` 等事件均 `<not supported>`。因此主线继续使用：

```text
paired wall-clock + correctness + assembly/objdump + Cachegrind
```

并行验证线在原生 Linux 或 PMU 可用服务器上补：

```bash
perf stat -e cycles,instructions,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,cache-misses \
  taskset -c 2 ./build/release/veqlib_stage_benchmark --stage geometry --repeat 30 --warmup 8 --inner 10000
```

PMU 的作用是确认“为什么快/慢”，不是重新决定 `a5c4d3c` 是否有效。

### 短期删除/降级项

| 工作项 | 决策 |
| --- | --- |
| Geometry accessor/pointer flatten | 删除；已测无收益 |
| 显式 glibc `sincos` | 删除；已严重退化 |
| 仅跳过 absent Fourier order | 降级；约 1% stage 线索且端到端无收益 |
| Residual 单纯 transpose | 删除；应做 fusion |
| 单个 residual load hoist | 降级；fusion 会覆盖 |
| Source 单 pass 合并 | 降级；端到端无收益 |
| 强制“所有对象放栈” | 删除；persistent workspace 更合理，当前 allocation 不在计时区间 |
| 立即开发完整自定义 AST 工具 | 后移；当前热点已明确 |

## 8. 决策原则

后续优化按以下顺序裁决：

1. 正确性先于速度：Python/C++ validation 不过，不保留性能改动。
2. `evaluate` 改善优先于单独 stage 改善。
3. 性能判断优先看 stage 占比、paired A/B ratio 和 `evaluate` 相对变化，不依赖单次 wall-clock 绝对值。
4. 先做可回滚 A/B，不做大范围重构。
5. 没有 PMU 时，cache-set conflict 只能作为假设，不作为最终结论。
6. 不新增抽象层，除非能删除重复逻辑或让 route-specific kernel 更清晰。
7. 实验输出继续放在 `veqlib/experiments/` 或其子目录，不写项目根目录。
