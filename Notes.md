# VEQlib 实验记录与下一步计划

更新时间：2026-06-21  
范围：`TODO-1.md`、`TODO-2.md` 指向的 VEQlib hot path / cache / tooling 初步实验。

## 0. 最新状态：kernel 纯化已完成，Residual surface layout 与 source sign pass 已保留

此前已把阶段性变更提交为 git commit：

- `7566f97 Establish measurable VEQlib optimization baseline`
- `a5c4d3c Reduce geometry cache interference before chasing trig`
- `3851f62 Preserve VEQlib performance decision trail`
- `a4637fd Reprioritize VEQlib optimization by evidence`
- `4715b04 Separate VEQlib FP contracts before math experiments`
- `e161475 Make RELAXED the VEQlib performance baseline`
- `845a3da Purify PF psin uniform Ip as a kernel contract`
- `7cf44f4 Reduce geometry harmonic profile reloads`

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
4.3%。随后已完成 `evaluate_ring`、topology-matrix 基础设施，以及 Geometry
micro-stage probe；完整 45-entry pinned topology matrix 已完成。

Geometry micro-stage probe 在引入 split/reduced-Taylor 之前的 RELAXED 排序曾显示：

| probe | median ns/call | incremental bucket |
| --- | ---: | ---: |
| `geometry_phase` | 504.6 | Fourier phase synthesis |
| `geometry_phase_sincos` | 3752.3 | +3247.7 dynamic `sin/cos(tb)` |
| `geometry_metric_no_store` | 5039.8 | +1287.5 metric/radial arithmetic |
| `geometry` | 5064.5 | +24.7 surface-output proxy |

注意：这些 probe 是 benchmark-only cumulative 近似，用于判断热点桶；
`geometry - geometry_metric_no_store` 不是 PMU store counter，也不是生产
kernel 拆分。该结果促成了后续 split-trig 和 reduced-Taylor dynamic sincos；
最新 stage 表见 Phase 3，当前最大相对占比已转向 source/residual/metric。

Phase 2 首轮 micro A/B：

| candidate | paired result | decision |
| --- | --- | --- |
| `JdivR = J*J/(J*R)` 复用 `inv_JR`，减少一个显式除法 | 早期 5 组：`geometry`≈0.992、`evaluate`≈0.990 但有反向组；reduced-Taylor 后 7 组重测：`geometry_metric_no_store`≈0.997、`geometry`≈1.018、`evaluate`≈1.006、`evaluate_ring`≈1.016 | 回滚；metric micro-probe 不足以覆盖 production codegen，端到端不支持 |
| Geometry harmonic profile reads hoist 到 `i` 层 | 5 组 paired median：`geometry` ratio≈0.984，`evaluate` ratio≈0.993；5/5 evaluate 均快 | 保留；低风险小收益，为高 `M_max` topology 预期更有价值 |
| Geometry per-rho arithmetic hoist（显式缓存 `a*rho`、`a*h`、`k+rho*k_r` 等） | 9 组 paired：`geometry` median ratio≈1.000，`evaluate`≈1.008，`evaluate_ring`≈0.990；sink diff≈0 | 回滚；编译器已基本处理该类 rho-invariant arithmetic，same-x evaluate 不支持保留 |
| Residual surface physical layout 改为 `[rho][field][theta]` | 5 组 paired median：`residual_update` ratio≈0.931，`residual_pack` ratio≈1.004，`evaluate` ratio≈0.994 | 保留；端到端收益小但未见显著回归，且 producer 语义与 geometry layout 对齐 |
| Residual theta-moment fusion，直接累计 active block moments | active-only 5 组 paired median：`residual_fused / (update+pack)`≈1.090，`evaluate` ratio≈1.007；naive 全 moments 约 1.166 / 1.021 | 回滚；当前 materialized update + vectorized pack 更快 |
| Geometry residual-ready descriptor compression（`qR/qZ/R2/dmetric`，9 fields -> 7 fields） | 5 组 paired median：`residual_update` ratio≈0.809，但 `geometry` ratio≈1.021，`evaluate` ratio≈1.004，`evaluate_ring` ratio≈1.008 | 回滚；只是把成本从 residual 移到 geometry，端到端无收益 |
| Geometry absent Fourier order static skip（仅 `harmonic_rows>2`） | 默认 `32x16x1` geometry 基本中性（短 all-stage median≈0.998，长 geometry-only median≈1.006）；`32x16x4` geometry-only median ratio≈0.925；`32x16x8` geometry-only median ratio≈0.808 | 保留；默认 topology 走原始 loop，高 `Mmax` 跳过 absent c-family order，stage sink 与 baseline 一致 |
| 去掉 `Pn_psin` 独立 array，改由 `materialized_heat_input` 作为同义源 | 7 组 paired：`source_update` median ratio≈1.008，`residual_update`≈1.003，`evaluate`≈0.993 但正负混合，`evaluate_ring`≈0.998；sink diff=0 | 回滚；减少一个重复 buffer 的语义方向可行，但当前代码形状没有稳定端到端收益 |
| 去掉 `source_psin_query/source_parameter_query` 两个同义 array，插值直接读 root `psin` | 首轮 7 组：`source_materialize`≈1.002，`evaluate`≈0.973 但 `evaluate_ring`≈1.015；追加 11 组长跑：`evaluate`≈0.997，`evaluate_ring`≈1.004；sink diff=0 | 回滚；删除重复 query buffer 未形成稳定收益，且 state-ring 口径不支持保留 |
| Geometry reduced-Taylor dynamic `sincos(tb)` | 默认 5 组 paired：`geometry_phase_split_sincos`≈0.253，`geometry`≈0.424，`evaluate`≈0.643；full matrix 45/45 改善，`evaluate` median≈0.609 | 保留；这是 split 后最大的单项收益，需继续用 Python/C++ comparator 锁误差 |
| Geometry reduced-Taylor order 降到 `sin x^11` / `cos x^10` | 默认 9 组 paired：`geometry_phase_split_sincos`≈0.928，`geometry`≈0.946，`evaluate`≈0.975，`evaluate_ring`≈0.969；45 topology `evaluate` matrix median≈0.985，37/45 改善 | 保留；`x^9/x^8` 因 `max_abs≈1.65e-7` 拒绝，`x^11/x^10` 仍通过 1e-9 comparator（`max_abs≈7.66e-10`） |

Residual layout 本轮的逐项中位数：baseline `residual_update≈912.2 ns`、candidate
`≈846.1 ns`；baseline `evaluate≈8288.7 ns`、candidate `≈8230.4 ns`。
这不是替代 Residual fusion 的结构性结论，只是把 materialized residual slab 的物理
布局改到更符合当前 producer 的 `[rho][field][theta]`。后续若做 theta-moment
fusion，这个 slab 本身仍可能被删除。

Residual fusion 随后做了两轮未保留实验：

- naive fused moments：一次 theta sweep 同时累计所有 residual moments，
  `residual_fused / (residual_update+residual_pack)≈1.166`，`evaluate≈1.021`。
- active-only fused moments：只累计当前 active blocks 需要的 moments，
  `residual_fused / (residual_update+residual_pack)≈1.090`，`evaluate≈1.007`。

结论：当前 topology 下 materialized residual surface + 后续 rowwise projection 更容易被
编译器向量化；直接在 theta sweep 内做多个 radial moment 累加会引入更长的标量
依赖链。下一次若重做 fusion，应先设计 vector-friendly / blocked moment accumulation，
而不是简单把 update 和 pack 合并。

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
| residual physical layout 改为 `[rho][field][theta]` | 旧 A/B：`residual_update` ratio 0.927；本轮：0.931 | 本轮 0.994 | 保留；语义对齐且未见显著端到端回归 |
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

进展：`veqlib_stage_benchmark` 已新增 `evaluate_ring` stage 和 `--ring-size`。
当前实现使用 deterministic synthetic solver-state ring；它用于检查 x 变化时的
callback traffic，不声称是真实 nonlinear solver trajectory。一次 release smoke
结果：same-x `evaluate≈8115.7 ns/call`，`evaluate_ring` ring-size 16
`≈8191.4 ns/call`，ratio≈1.009。

进展：`stage_benchmark.cpp` 已改为读取 CMake 生成的 `config::DefaultTopology`，
`clang-debug` / `clang-release` presets 显式固定当前默认 benchmark topology
`Nr=32, Nt=16, Mmax=1, x_size=18`。新增 `veqlib/stage_topology_matrix.py`，
可为多个 `Nr x Nt x Mmax` 生成独立 build dir 并汇总 JSON。smoke 已跑
`32x16x1`、`32x32x1` 与 `32x16x4`，输出 topology metadata 正确，且
`Mmax=4` 不再被默认 benchmark static_assert 阻塞。

代表性 production-stage matrix 已跑 9 个 topology（`repeat=12`、`warmup=4`、
`inner=5000`；只用 production `geometry/evaluate`，不把 micro-probe delta
外推到所有 topology）：

| topology | `geometry` ns | `evaluate` ns | geometry share |
| --- | ---: | ---: | ---: |
| `16x16x1` | 2510.8 | 3666.6 | 0.685 |
| `32x16x1` | 5048.1 | 8278.8 | 0.610 |
| `64x16x1` | 10157.5 | 15073.7 | 0.674 |
| `32x8x1` | 2582.0 | 6004.0 | 0.430 |
| `32x24x1` | 7656.2 | 10899.3 | 0.702 |
| `32x32x1` | 10301.8 | 14593.4 | 0.706 |
| `32x64x1` | 20502.7 | 26664.9 | 0.769 |
| `32x16x4` | 5724.1 | 9280.5 | 0.617 |
| `32x16x8` | 7743.2 | 10935.6 | 0.708 |

结论：默认 `32x16x1` 不是唯一表现；Geometry 在代表矩阵中仍是最大单项
热点，且随 `Nt` 增大 share 明显上升。`Nt=8` 时非-Geometry 固定成本占比较高，
但 `Nt>=16` 后继续优先 Geometry 是合理的。

完整 45-entry matrix 已补跑（`taskset -c 2`、`Nr={16,32,64}`、
`Nt={8,16,24,32,64}`、`Mmax={1,4,8}`；`geometry/evaluate` 各自
`repeat=6`、`warmup=3`、`inner=4000`；原始 JSON 留在 `/tmp`，不提交生成物）。
全矩阵 `geometry/evaluate` share 的 median≈0.699，范围约 0.476--0.803。
按 `Nt` 聚合最能解释热点排序：

| `Nt` | median geometry share | median `evaluate` ns |
| ---: | ---: | ---: |
| 8 | 0.528 | 5694.4 |
| 16 | 0.626 | 8682.0 |
| 24 | 0.735 | 11522.1 |
| 32 | 0.725 | 15425.7 |
| 64 | 0.774 | 27965.3 |

按 `Mmax` 聚合显示 high-Mmax 会抬高绝对时间，但并没有改变 Geometry 是主线
热点的结论：`Mmax=1/4/8` 的 median share 分别约 0.699/0.686/0.699。
只有 `32x8x{1,4,8}` 的 Geometry share 低于 0.5，说明小 `Nt` 下 source/profile/
fixed overhead 更值得单独关注；常用/default 及更大 `Nt` 仍优先 Geometry。

Source fixed-overhead 首个候选尝试删除 `Pn_psin` 独立 array：`Pn_psin` 在
PF/psin/uniform/Ip 下只是 `materialized_heat_input` 的同义值，理论上可减少
`source_update` 的一次复制和 residual/g1n 的重复读取。实现验证通过 release CTest
和 PF validation，但 paired timing 显示 `source_update`、`residual_update` 中位数反而
略退，`evaluate`/`evaluate_ring` 没有稳定正收益，因此已回滚。第二个候选删除
`source_psin_query/source_parameter_query` 两个同义 query buffers，让插值直接读
root `psin`；它同样通过 validation 且 sink 一致，但长跑中 `evaluate≈0.997`、
`evaluate_ring≈1.004`，没有达到保留门槛。后续 source cleanup 应优先看真正改变
算法结构的固定刷新策略或插值/regularization 形状，而不是单个同义 vector。

Profile staticization 的一个更小候选也已拒绝：对默认 topology 中 absent 的 `v`
profile 和 `c0` family 引入 `if constexpr` helper，避免 generic accessor 读取
zero slab。该 patch 通过 release build、PF validation 和 release CTest，sink
完全一致；但 9 组 paired timing（`taskset -c 2`、`repeat=24`、`warmup=8`、
`inner=10000`、`ring-size=16`）显示 `geometry` median ratio≈1.005，
`evaluate≈0.995` 且波动很大，`evaluate_ring≈1.003`。结论：单个 absent
core-profile helper 会增加代码形状/模板分支复杂度，却没有稳定端到端收益；已回滚。

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

进展：`veqlib_stage_benchmark` 已新增 `geometry_phase`、
`geometry_phase_sincos`、`geometry_metric_no_store` 三个 cumulative probe。
默认 topology 每次 Geometry 有 `Nr*Nt = 512` 个 dynamic `sin/cos(tb)` 点；
当前 RELAXED pinned run 显示 phase synthesis 约 505 ns、dynamic trig 增量约
3248 ns、metric/radial arithmetic 增量约 1288 ns。output traffic 只能从
probe 与生产 geometry 的差值粗略看，当前约 25 ns，说明 layout 后 surface
store 本身不再是最大桶。

停止条件：已满足用于排序的 micro-stage 量化；若要证明 cache/store 机制，仍需 native PMU 或 Cachegrind/assembly 辅助。

### Phase 3：Geometry dynamic sincos backend A/B

目标：在 Phase 2 的拆分证据下，测试真正可能带来较大收益的动态数学路径。

建议顺序：

1. 捕获或生成 `tb[Nr*Nt]` 输入，单独 benchmark scalar libm。
2. 测 AVX2/SVML/SLEEF/libmvec 或受控误差的 domain-specific vector sincos backend。
3. 将 vector sincos 嵌入 Geometry，检查端到端是否仍有收益。
4. 对比 assembly/optimization remarks：vectorization width、call site、spill/reload、寄存器压力。
5. 严格跑 Python/C++ validation 与 FP mode matrix。

已删除方向：显式 glibc `::sincos`。它已严重退化，且 baseline 已有 compiler-lowered `sincos@plt` 证据。

Phase 3 首轮 vector-math 探针：

- `libmvec` 可用，且一个独立 canonical `for` loop 在
  `-ffast-math -fveclib=libmvec` 下能生成 `_ZGVdN4v_sin` / `_ZGVdN4v_cos`。
- 但把 `veqlib_stage_benchmark` 用 `-fveclib=libmvec` 重建后，Geometry 仍只出现
  scalar `sin@plt` / `cos@plt` / `sincos@plt`，没有 `_ZGV*` vector libm 符号。
- 5 组 paired timing（`repeat=10`、`warmup=4`、`inner=5000`）显示
  `geometry_phase_sincos` median ratio≈1.018，`evaluate` ratio≈1.003；无收益。
- 对生产 `geometry.h` theta loop 加 `#pragma clang loop vectorize(enable)` 后，LTO
  仍提示 loop not vectorized，objdump 仍无 `_ZGV*`，该探针已回滚。

结论：简单打开 `-fveclib=libmvec` 或强制 pragma 不能优化当前 fused Geometry
loop。若继续做 vector trig，需要结构性拆分：先 materialize `tb[j]` 到临时数组，
再用 canonical trig loop 调 vector libm，最后 metric loop 消费；这必须单独 A/B，
因为额外数组 traffic 可能抵消 vector trig 收益。

结构性 split-trig 已完成并保留：生产 `GeometryRuntime::update()` 现在在每个
radial node 内分成三段 theta pass：

```text
phase synthesis -> canonical sin/cos(tb) loop -> metric/surface/radial accumulation
```

该结构不依赖 `-fveclib=libmvec` 即有稳定收益。默认 RELAXED build 用 9 组
paired timing（`taskset -c 2`、`repeat=24`、`warmup=8`、`inner=10000`、
`ring-size=16`）对比本阶段 baseline：

| stage | median ratio | 结果 |
| --- | ---: | --- |
| `geometry` | 0.897 | 快约 10.3% |
| `evaluate` | 0.908 | 快约 9.2% |
| `evaluate_ring` | 0.925 | 快约 7.5% |

sink diff 全部为 0，release CTest 和 PF validation 通过。`libmvec` build 下
split 本身也有效（geometry ratio≈0.902），但 final `libmvec/normal` 对照显示
`geometry≈1.021`、`evaluate≈0.988`、`evaluate_ring≈0.975`，收益小且混合；
因此不把 `-fveclib=libmvec` 纳入默认构建。后续 vector trig 若继续推进，应在
已有 split 结构上检查 assembly/PMU，而不是回到 fused loop 加 pragma。

在 split 结构稳定后，采用了一个 RELAXED-only 的 domain-specific dynamic
`sincos(tb)` backend：`tb` 先按最近的 `pi/2` 做象限规约，使余量落在
`|r|<=pi/4`，再用高阶 Taylor polynomial 同时近似 `sin(r)` 和 `cos(r)`，
最后用无分支 quadrant mapping 还原符号和交换关系。显式分支版虽已明显更快，
但 LTO 提示 switch 阻塞向量化；无分支版消除了该 warning 并进一步提速。
验证结果：

- release/debug CTest 通过，PF Python/C++ comparator 通过，
  `max_abs≈5.578e-11`（worst field: `final.x`）。
- 默认 topology 5 组 paired RELAXED A/B：
  `geometry_phase_split_sincos` median ratio≈0.253，`geometry`≈0.424，
  `evaluate`≈0.643，`evaluate_ring`≈0.635。
- full topology matrix（45 点，同上 `Nr/Nt/Mmax`；`repeat=6`、`warmup=3`、
  `inner=4000`）显示 45/45 改善：`geometry` median ratio≈0.443
  （range 0.379--0.537），`evaluate` median ratio≈0.609
  （range 0.504--0.756）。

这是目前最大的单项收益；后续如要继续提高 trig backend，应先检查生成汇编和
是否真正 vectorize，而不是退回 libmvec flag-only 路径。

2026-06-21 继续测试了 reduced-Taylor 阶数收缩。把原来的 `sin x^15` /
`cos x^14` 截断改为 `sin x^11` / `cos x^10` 后，release CTest 与 PF
Python/C++ comparator 通过，但误差预算明显更紧（`max_abs≈7.66e-10`，主要来自
`geometry_V_r`，仍低于 1e-9 门槛）。默认 topology 9 组 paired RELAXED timing
显示 `geometry_phase_split_sincos≈0.928`、`geometry≈0.946`、`evaluate≈0.975`、
`evaluate_ring≈0.969`。45 topology `evaluate` matrix 为 37/45 改善、median
ratio≈0.985、range≈0.939--1.150；其中 `64x8x1` 的非配对 +15% 回归经单独
paired 复测为 median≈0.982，`32x16x8` 复测为 median≈0.979，因此保留。
更低的 `sin x^9` / `cos x^8` 版本 PF comparator 失败（`max_abs≈1.65e-7`），
明确拒绝。

采用后又按 full topology matrix 复测（45 点：`Nr={16,32,64}`、
`Nt={8,16,24,32,64}`、`Mmax={1,4,8}`；`taskset -c 2`、`repeat=6`、
`warmup=3`、`inner=4000`；pre-split JSON 为 `/tmp/veqlib_full_matrix_*_pinned.json`，
post-split JSON 为 `/tmp/veqlib_full_matrix_*_split.json`）。结果显示 45/45
topology 的 `geometry` 和 `evaluate` 都改善：

| stage | median ratio | min ratio | max ratio | improved |
| --- | ---: | ---: | ---: | ---: |
| `geometry` | 0.906 | 0.766 | 0.944 | 45/45 |
| `evaluate` | 0.929 | 0.874 | 0.978 | 45/45 |

按 `Nt` 聚合的 `evaluate` median ratio 分别约为：`Nt=8` 0.948、
`Nt=16` 0.929、`Nt=24` 0.918、`Nt=32` 0.925、`Nt=64` 0.945。说明 split-trig
不是默认 `32x16x1` resonance，而是在已测 full matrix 内稳定改善；较大 `Nt`
的 `geometry` ratio 仍改善，但端到端收益会被其它固定/投影成本稀释。

采用 reduced-Taylor dynamic sincos 后，默认 topology 的最新 stage 表如下
（单次 pinned `--stage all`，`repeat=15`、`warmup=5`、`inner=10000`、
`ring-size=16`，原始 JSON `/tmp/veqlib_stage_all_branchless_poly.json`）：

| stage | ns/call | `evaluate` share |
| --- | ---: | ---: |
| `profiles_all` | 125.1 | 2.6% |
| `geometry_phase` | 499.9 | 10.3% |
| `geometry_phase_sincos` | 3767.5 | 77.7% |
| `geometry_phase_split_sincos` | 920.4 | 19.0% |
| `geometry_metric_no_store` | 1921.5 | 39.6% |
| `geometry` | 1911.9 | 39.4% |
| `source_materialize` | 896.4 | 18.5% |
| `source_update` | 818.6 | 16.9% |
| `residual_update` | 797.1 | 16.4% |
| `residual_pack` | 133.4 | 2.8% |
| `evaluate` | 4850.9 | 100% |
| `evaluate_ring` | 4875.5 | 100.5% |

`geometry_metric_no_store` 现在已改成与 production split 结构一致的
phase-materialize / sin-cos / metric accumulation probe，只省掉九个 surface
field 写入；因此 `geometry - geometry_metric_no_store≈23 ns` 只能作为
surface-output proxy，不能解释为硬件 store counter。新的排序显示：Geometry
已经从约 58% 降到约 39%，source materialize/update 与 residual_update 的相对
占比上升；`geometry_phase_sincos` 仍保留 scalar-libm fused reference，用来说明
reduced-Taylor split backend 的差距。

### Phase 4：压缩 Geometry descriptor，减少 Residual 重算

目标：layout 改动改善了九个 field 的访问方式，但没有删除九个二维 field 的写入/读取。下一步应审计能否存储 residual 直接需要的派生量，而不是继续换排列。

候选派生量：

```text
qR      = -Z_t / J
qZ      =  R_t / J
R2      = R * R
dmetric = gttdivJR_r - grtdivJR_t
```

已测试一版 residual-ready descriptor compression：Geometry surface 从 9 个 raw fields
改为 7 个 residual-ready fields。Python/C++ comparator 通过（`max_abs≈5.578e-11`），
但 paired RELAXED A/B 显示 `residual_update` 虽快约 19.1%，`geometry` 慢约 2.1%，
最终 `evaluate` ratio≈1.004、`evaluate_ring` ratio≈1.008，因此回滚。
结论：不要简单把 residual 重算移动到 Geometry；若再做 descriptor compression，应只
接受能改善 `evaluate` 的组合，或把派生量放在 residual 局部 vector-friendly pass 中。

### Phase 5：Residual theta-moment fusion

目标：删除四个 residual 二维中间场，而不是再尝试单一 residual layout。

已知证据：residual physical layout 已改成 `[rho][field][theta]` 并保留；paired A/B 显示 `residual_update` 快约 6.9%，`residual_pack` 小幅退化，`evaluate` 只有噪声级收益。直接 theta-moment fusion 已实测退化：active-only fused moments 仍比 materialized update+pack 慢约 9.0%，`evaluate` 慢约 0.7%。这说明 update 和 pack 仍需要相反 locality，但简单融合会丢失现有 rowwise pass 的向量化优势：

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

2026-06-21 又保留了一项小的 residual 局部效率改进：`residual_update`
在 theta loop 内把同一点的 Geometry surface field 和 `alpha1/alpha2`
加载到局部变量，避免重复 accessor/地址生成，并让 `[rho][field][theta]`
布局下的点式读取更显式。release/debug CTest 与 PF Python/C++ comparator
均通过（`max_abs≈5.578e-11`）。9 组 paired RELAXED timing
（`taskset -c 2`、`repeat=24`、`warmup=8`、`inner=10000`、`ring-size=16`）
显示 `residual_update` median ratio≈0.947，`evaluate`≈0.995，
`evaluate_ring`≈0.995，sink 全匹配。结论：保留；它是 stage-local
约 5.3% 的小收益，端到端约 0.5%，但不替代后续结构性 fusion。

### Phase 6：Source/profile route 静态化

已完成一项局部静态化：Geometry Fourier order accumulation 现在在
`harmonic_rows>2` 时按 `Shape::c_slot(order).enabled()` /
`Shape::s_slot(order).enabled()` 做 compile-time fold，absent order 不再加载或
参与 phase synthesis。为了保护当前默认 `Mmax=1` 性能，`harmonic_rows<=2`
仍保留原始小 loop。验证结果：

- default `32x16x1`：geometry 基本中性；paired geometry-only median ratio≈1.006，
  all-stage 短测 geometry median ratio≈0.998。
- `32x16x4`：geometry-only 7 组 median ratio≈0.925，all-stage evaluate median
  ratio≈0.946--0.971（按不同窗口）。
- `32x16x8`：geometry-only 7 组 median ratio≈0.808，all-stage evaluate median
  ratio≈0.885--0.894。
- `32x16x4/8` 的 stage `geometry` 和 `evaluate` sink 与 baseline 完全一致。

结论：这是高 `Mmax` topology 的真实收益，不改变默认 topology 的语义；若未来
启用 cosine family，也应继续依赖 `ProfileShape` 的 enabled slot，而不是运行期
读取 absent family 的零 slab。

2026-06-21 又保留了一项 source exact-hit 局部优化：
`local_barycentric_interpolate_pair()` 不再对 8 点 stencil 逐个比较是否命中
uniform input node，而是直接用 `round(q * (N-1))` 找最近 sample node，再做一次
`1e-14` exact-hit 检查。这个改动不改变非命中时的 barycentric stencil，也不重新
引入 route/finite/sentinel 分支。release/debug CTest 与 PF Python/C++ comparator
均通过（`max_abs≈5.578e-11`）。

默认 topology 的 7 组 paired RELAXED timing（`taskset -c 2`、`repeat=24`、
`warmup=8`、`inner=10000`、`ring-size=16`）显示：

| metric | median ratio |
| --- | ---: |
| `source_materialize` | 0.868 |
| `source_update` | 0.995 |
| `evaluate` | 0.956 |
| `evaluate_ring` | 0.971 |

完整 45 topology `evaluate` matrix（`Nr={16,32,64}`、
`Nt={8,16,24,32,64}`、`Mmax={1,4,8}`）为 40/45 改善，median ratio≈0.976，
range≈0.876--1.027。小 `Nt` 改善最大，`Nt=64` 基本中性。保留该改动，但把它
定位为 source materialization 固定成本削减，而不是新的结构性 source algorithm。

采用后默认 stage 表（`--stage all`，同 pinned 口径）为：

| Stage | ns/call | Share of `evaluate` |
| --- | ---: | ---: |
| `profiles_all` | 125.7 | 2.6% |
| `geometry` | 1926.6 | 40.5% |
| `source_materialize` | 768.5 | 16.2% |
| `source_update` | 813.6 | 17.1% |
| `residual_update` | 796.3 | 16.7% |
| `residual_pack` | 135.3 | 2.8% |
| `evaluate` | 4756.6 | 100% |
| `evaluate_ring` | 4805.8 | 101.0% |

随后测试了一个更小的 root-row copy 消除候选：`update_psin_coordinate()` 和
`psin_r` 归一化路径直接从 `source_target_root_fields` 做 matvec/dot，避免先复制
成 `RadialVector`。correctness 通过 release CTest 和 PF comparator，但 7 组默认
paired timing 只是小幅改善（`source_materialize≈0.969`、`source_update≈0.990`、
`evaluate≈0.988`、`evaluate_ring≈0.994`），完整 45 topology `evaluate` matrix
反而为 20/45 改善、median ratio≈1.003、range≈0.967--1.077。结论：回滚；
这类 root-row copy 微调没有跨 topology 的端到端收益，后续不要在没有 assembly/PMU
证据时继续追这个方向。

又测试了把 `local_uniform_stencil_start(q)` 延后到 exact-hit fast path 之后的
微调。这个改动语义等价，但 9 组默认 paired timing 只得到
`source_materialize≈1.005`、`source_update≈0.999`、`evaluate≈0.999`、
`evaluate_ring≈0.992`，没有稳定 endpoint 收益，因此回滚。source exact-hit 已保留
最近节点检查；不要继续追逐这类单指令级插值分支重排，除非新的 profile/assembly
证据显示它阻塞了向量化或分支预测。

随后测试了一个 residual pointwise 算术折叠候选：把 `psin_r_i / J_ij` 先合并成
`psin_over_J`，再生成 `psin_R` 与 `psin_Z`，理论上可减少两次乘法。该改动
通过 release CTest 与 PF Python/C++ comparator（`max_abs≈7.66e-10`），默认
9 组 paired timing 也显示 `residual_update` 有极小改善（median ratio≈0.995），
但端到端 `evaluate` 只有噪声级收益（≈0.998）。完整 45 topology matrix 的
median ratio≈0.993，不过 worst topology 的 paired 复测并不稳健：`32x8x4`
median≈1.007，`64x8x4` median≈1.002。结论：回滚；这种单点算术折叠不足以
作为保留优化，后续 residual 应继续转向结构性 moment/fusion，而不是继续追逐
乘法重排。

又测试了 residual `surface_G` row-sum cache：`update_compact()` 在生成
`G_ij` 时顺便累计每个 radial row 的 `G` 和，`pack` 中的 `block_psin/F`
复用该缓存，避免重新 `rowwise_sum(surface_G)`。该候选通过 release CTest 与
PF comparator（`max_abs≈7.66e-10`），但默认 9 组 paired timing 显示只是
阶段转移：`residual_pack` 明显变快（median ratio≈0.836），`residual_update`
变慢（≈1.014），`evaluate` 只有噪声级小幅改善（≈0.992），`evaluate_ring`
中性偏负（≈1.001）。45 topology `evaluate` matrix 为 31/45 改善、median
ratio≈0.992、range≈0.859--1.054。结论：回滚；当前只有一个 active `G`
consumer 时，这相当于把一次 vector-friendly rowwise sum 搬进 pointwise update
依赖链，并没有稳定删除端到端工作。

又测试了 `regularize_psin_r()` pass 融合：把 axis-fix 前段修正与
`1e-10` floor clamp 合并，避免先写 axis 区再全量重扫同一区间。release CTest
和 PF comparator 通过（`max_abs≈7.66e-10`），但默认 9 组 paired timing 不支持
保留：`source_materialize` median ratio≈0.992，`source_update`≈0.999，
`evaluate`≈1.008，`evaluate_ring`≈1.002。结论：回滚；当前默认 `fix_rho`
只影响很少 radial node，融合带来的少量写入节省抵不过分支/代码形状扰动。

又测试了 residual pack 静态权重表：把 `unit_weights()`、`rho_power<P>()`、
`theta_sin/cos<Order>()` 从运行期返回临时 `Vector` 改成 class-scope `inline static
constexpr` 表并返回引用。该候选通过 release CTest 与 PF comparator，但 9 组
default paired timing 明显失败：`residual_pack` median ratio≈1.390，`evaluate`≈1.009，
`evaluate_ring`≈1.018。结论：回滚；当前编译器更擅长内联/标量化小临时表，强制静态
对象反而引入全局地址读取或阻碍优化。

又测试了 reduced-Taylor `sincos(tb)` 的 Estrin 分组，把 Horner 链改成
`r2/r4/r8` 分组以尝试降低依赖链。release CTest 与 PF comparator 通过，但默认
9 组 paired timing 不支持：`geometry_phase_split_sincos` median ratio≈1.019，
`geometry`≈1.007，`evaluate`≈0.996，`evaluate_ring`≈1.006。结论：回滚；
当前 Horner 形式更容易被编译器优化，Estrin 增加的临时乘法/寄存器压力抵消了
理论依赖链优势。

本轮随后保留了一个 source sign-normalization pass reduction：在 `psin_r`
经 `Kn` 归一化的同一个循环里顺手累计加权和，用这个和决定是否整体翻符号，
从而删除一次独立的 `weighted_profile_sign(psin_r)` 扫描。该改动没有引入
route/finite/sentinel/solve-success 分支，仍然是纯 `PF/psin/uniform/Ip` kernel
路径。release/debug CTest 均通过，RELAXED release PF Python/C++ comparator
通过（`max_abs≈7.66e-10`）。默认 9 组 paired timing 显示
`source_update` median ratio≈0.977、`evaluate`≈0.995、`evaluate_ring`≈0.984。
完整 45 topology `evaluate` matrix 为 26/45 改善、median ratio≈0.995、
mean≈0.994、range≈0.851--1.035；三个 apparent worst topology 的 paired
复测未复现强回归：`32x8x8` 的 `evaluate/evaluate_ring` median≈0.998/0.994，
`16x8x4`≈0.995/0.986，`64x16x1`≈0.996/1.001。结论：保留；这是小幅但语义清晰的
source_update pass 删除，后续不要再把它包装成 checked facade 或有效值 guard。

又测试了 geometry theta-loop vectorization pragma：在 phase synthesis loop 和
metric/store loop 前显式加 `#pragma clang loop vectorize(enable)`。该候选不改公式，
release CTest 和 PF comparator 通过（`max_abs≈7.66e-10`），初轮 9 组 paired
timing 一度显示 `evaluate` median ratio≈0.988、`evaluate_ring`≈0.994，但
`geometry` 自身≈0.999，信号已经可疑。随后 7 组更长 paired 复测显示
`geometry`≈0.998、`evaluate`≈1.015、`evaluate_ring`≈1.002。结论：回滚；
当前 loop hints 没有带来稳定 geometry 改善，反而可能扰动整体 codegen。

又测试了 residual pack `unit_weights()` marker：把全 1 `RadialVector` 临时对象
换成零存储 `UnitWeights` marker，并在 `project_scaled()` 中通过 `weight_value()`
返回 1.0，试图跳过 unit 权重构造和读取。release CTest 与 PF comparator 通过，
但 9 组默认 paired timing 明显变差：`residual_pack` median ratio≈1.063、
`evaluate`≈1.006、`evaluate_ring`≈1.007。结论：回滚；这与此前静态权重表失败
一致，当前编译器更擅长处理原来的小 `RadialVector` 临时/标量化形式，不要继续追逐
unit-weight 抽象替换。

又测试了 geometry surface row padding：把物理 layout 从
`Tensor<double, radial_nodes, 9, theta_rows>` 改成每个 radial row 含 1 个未用
field padding slot，试图进一步打散 row stride/cache-set 关系。release CTest 与
PF comparator 通过，但默认 paired timing 第 1--3 组已经严重回归：
`geometry` ratio≈2.33--2.35、`residual_update`≈1.13、`evaluate`≈1.54--1.59、
`evaluate_ring`≈1.56--1.59，因此中断长跑并回滚。结论：当前 `[rho][field][theta]`
紧凑行布局已经是更好的局部性折中；不要再增加 row padding，除非原生 PMU/assembly
明确证明新的 set 冲突。

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
