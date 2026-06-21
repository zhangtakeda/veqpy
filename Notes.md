# VEQlib 实验记录与下一步计划

更新时间：2026-06-21  
范围：`TODO-1.md`、`TODO-2.md` 指向的 VEQlib hot path / cache / tooling 初步实验。

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

本轮未提交 git。当前预期变更：

```text
 M veqlib/CMakeLists.txt
 M veqlib/README.md
?? Notes.md
?? veqlib/experiments/
?? veqlib/stage_benchmark.cpp
```

`veqlib/experiments/` 是本轮实验数据目录；如果后续不希望把原始 logs/JSON 全部纳入版本库，可只保留：

- `veqlib/experiments/summary.md`
- 少量关键 JSON
- 或将完整实验数据移到外部 artifact 存储

但在做下一轮优化前，建议先保留完整目录，便于复现和比较。

## 7. 下一步计划

### Phase A：先固定 baseline artifact

目标：确保后续每次 A/B 都能和本轮基线比较。

建议动作：

1. 保留 `veqlib_stage_benchmark` 作为正式开发工具。
2. 确定是否版本化 `veqlib/experiments/summary.md` 与关键 JSON。
3. 每个优化分支都至少跑：
   - `veqlib_stage_benchmark --stage geometry`
   - `veqlib_stage_benchmark --stage evaluate`
   - Python/C++ validation
   - Debug CTest

停止条件：baseline 可一键复现，且后续实验输出路径不会污染项目根目录。

### Phase B：优先做 geometry A/B

目标：验证 `geometry` 的 9.3 us median 是否主要来自三角函数和 9-plane surface 更新。

建议按以下顺序做小步 A/B：

1. **三角函数证据增强**
   - 在 analysis/objdump 中定位 `sincos@plt` 所在调用块。
   - 如果可行，构造一个只跑 theta trig 的 micro stage，估计 trig 占比。

2. **预计算或复用 theta trig**
   - 若 `theta` 相关项与当前 active profiles 的依赖允许，尝试把固定 theta 的 `sin/cos` 基础表预计算到 setup/plan。
   - 对包含 profile-dependent phase/shape 的项，先不要改变数学语义；可先做局部缓存或 recurrence A/B。

3. **geometry layout A/B**
   - Padded field plane：把每个 4096 B plane 间距改成 4160 B，验证是否改善 geometry timing。
   - Alternative layout：`[rho][field][theta]`，保持 theta 连续，同时让同 rho 的 field 聚集。
   - 这两项必须以 correctness validation + stage timing 为准；当前没有 PMU，不应仅凭推理保留。

4. **只在实测支持后再考虑 AVX2 intrinsic**
   - 当前首选让 Clang 自动向量化或使用更利于向量化的数据布局。
   - 手写 AVX2 仅在 objdump/remarks 证明 compiler 无法生成目标形态时再做。

成功标准：

- `geometry` median 明显下降。
- `evaluate` median 同步下降，而不是只优化孤立 stage。
- Python/C++ validation 仍在 `1e-9` tolerance 内。

### Phase C：Residual materialize/project 融合

目标：验证 TODO 中“写二维场再按 profile 重扫”的结构性问题。

建议动作：

1. 新增一个实验性 residual path，不替换默认实现。
2. 在一次 theta sweep 中同步累计需要的 theta moments。
3. 将 radial projection 与 moment accumulation 解耦，减少对 `Nr*Nt` surface slab 的重复扫描。
4. 对比：
   - `residual_update`
   - `residual_pack`
   - `evaluate`
   - Cachegrind D refs / D1 misses

注意：当前墙钟上 residual 不是第一瓶颈，所以应在 geometry 有结果后再投入较大重构。

### Phase D：FP 语义修正与 Release mode 分层

目标：解决 `TODO-1.md` 中 Release `-ffinite-math-only` 与有限性检查的语义冲突。

建议动作：

1. 将 FP mode 拆为：
   - `STRICT`
   - `FMA`
   - `RELAXED`
2. 不让包含 NaN/Inf validity check 的 TU 启用 `-ffinite-math-only`。
3. 将 `math::is_finite` 与 magnitude policy 分离：
   - `is_finite`: 标准有限性语义。
   - `is_valid_magnitude`: 有限且不过大。
4. 每个 FP mode 都跑 Python/C++ validation，并记录最大差异。

这属于正确性 P0，建议不要长期拖延；但它会影响 benchmark 可比性，最好作为单独分支处理。

### Phase E：Source route 静态化与 fixed profile refresh 去重

目标：减少 TODO 中指出的分支、复制和重复 refresh。

建议动作：

1. 对 `PF/psin/uniform/Ip` 编译期实例，消除不可能的 `beta`/route 分支。
2. fixed profiles 尽量迁到 plan/setup 阶段，只在参数变化时刷新。
3. 检查 `refresh_fixed -> refresh_fourier_family_fields` 与 `refresh_active -> refresh_fourier_family_fields` 是否重复清零 family slab。
4. 用 `profiles_all` 和 `evaluate` stage 验证收益。

这部分单次墙钟占比不高，但会减少每次 residual callback 的固定成本，也能降低未来更大 topology 下的成本。

### Phase F：PMU 环境复测

目标：验证 4 KiB field-plane conflict，而不是只停留在推断。

需要环境：原生 Linux 或支持硬件 PMU 的 VM/container host。

建议命令方向：

```bash
perf stat -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,cache-references,cache-misses \
  ./build/release/veqlib_stage_benchmark --stage geometry --repeat 40 --warmup 10 --inner 10000
```

如果 LIKWID 可用，可补充 L2/L3、MEM、FLOPS_DP group。只有当 padded/layout A/B 同时改善 wall time 和 L1 replacement/line fill，才应保留 layout 改动。

## 8. 决策原则

后续优化按以下顺序裁决：

1. 正确性先于速度：Python/C++ validation 不过，不保留性能改动。
2. `evaluate` 改善优先于单独 stage 改善。
3. 性能判断优先看 stage 占比、paired A/B ratio 和 `evaluate` 相对变化，不依赖单次 wall-clock 绝对值。
4. 先做可回滚 A/B，不做大范围重构。
5. 没有 PMU 时，cache-set conflict 只能作为假设，不作为最终结论。
6. 不新增抽象层，除非能删除重复逻辑或让 route-specific kernel 更清晰。
7. 实验输出继续放在 `veqlib/experiments/` 或其子目录，不写项目根目录。
