# Solver

`Solver` 负责一个固定 `Operator` layout 上的非线性求解流程。它不定义
packed layout，不实现 source route，也不持有物理 runtime fields；这些属于
`Operator`。Solver 决定使用哪个 packed 初值、调用哪个 SciPy 优化器、怎样在
非线性方法内部缩放 residual、如何选择 fallback attempt，以及是否运行求解后的
collocation polish。

源码位置: `veqpy/solver/`。

## 配置与初值

`SolverConfig` 保存默认求解策略。传给 `solve()` 的单次关键字参数会生成一个
临时 config snapshot，只替换本次调用显式传入的字段；solver 保存的默认 config
本身不变。

默认 variational 方法是 SciPy `root(..., method="hybr")`。默认 fallback 是
`least_squares(..., method="lm")`。`trf` 也可以通过 `method` 或
`fallback_methods` 使用。Collocation polish 只使用 least-squares 方法，默认
collocation 方法是 `lm`。

Solver 自己持有一个 packed 向量 `Solver.x0`。构造 `Solver` 时，它由
`Operator.encode_initial_state()` 从 `Problem.profiles` 编码得到。
每次求解结束后，`Solver.x0` 会被替换为最终 packed 解；后续 warm start 和
`build_equilibrium()` 都使用这个状态。

初值来源按优先级选择:

| 来源 | 行为 |
| ---- | ---- |
| 显式 `x0` | 由 operator 校验，复制到 `Solver.x0`，并作为本次初值 |
| `initial_policy="warm"` | 复用当前 solver-owned `Solver.x0` |
| `initial_policy="zeros"` | 使用全零 packed 向量 |
| `initial_policy="homothetic"` | 使用边界形状估计 active shape 系数 |
| 默认 (`initial_policy=None`) | 重新从 `Problem.profiles` 编码 |

`homothetic` 初值是一个廉价的几何初猜，面向近似嵌套磁面。它委托给 operator
的 boundary-slope estimate: active Fourier shaping 系数会从边界 offset 推出
首项系数，`h` 在 source profile 非均匀时使用 Shafranov-shift 估计。该初值
使用 operator 侧的一套保守估计，不再暴露额外的 scale factor。

当本次求解使用显式 `x0`、zeros、homothetic 或 case-encoded 初值时，operator
会在 attempt 前使 route-local source state 失效。`warm` 则保留当前 `x0` 对应
的 source state。

## 求解流程

普通 variational solve 的流程是:

1. 合并默认配置和本次 `solve()` 覆盖参数。
2. 按上面的优先级构造 packed 初值。
3. 对 variational residual 调用主 SciPy 方法。
4. 若失败且 fallback 打开，从同一初值按配置的 fallback 方法继续尝试。
5. 选择第一个被接受的成功 attempt；若没有 attempt 被接受，则保留有有限结果且
   residual norm 最小的 attempt 用于诊断。
6. 返回 `SolverResult`，可选写入 history。

Variational solve 的成功判定是严格的: SciPy `success` 本身不够，residual norm
还必须通过 solver 接受阈值。该阈值是 `max(10 * max_residual, 1e-5)`。如果优化器
抛出异常，但初始点本身已经满足该 residual 阈值，该 attempt 仍可以作为已经求解
的状态被接受。

## Residual Normalization

Residual normalization 用于降低 packed residual 各 block 之间的幅值不平衡，
让非线性方法看到更均衡的方程组。`SolverResult` 记录的 final residual 仍是
operator 的 raw residual，不是优化器内部使用的缩放 residual。

当前模式包括:

| mode | 含义 |
| ---- | ---- |
| `none` | 直接使用 raw residual |
| `fast` / `block_rms` | 用初始 RMS 按 active residual block 缩放，最小 scale 为 1 |
| `balance` / `balanced` / `block_huber` | 用 Huber-style RMS、floor 和最大 scale ratio 构造 robust block scale |
| `safe` / `block_sensitivity` | 将 robust block amplitude 与有限差分 sensitivity probe 合并 |

默认 config 使用 `fast`。若传入 `residual_normalization=None`，也会归一到同一个
包默认值。对于 `hybr`，启用 normalization 时还会收紧初始 trust-region factor，
以减少缩放 residual 空间中的过大首步。

## Collocation Polish

`enable_collocation=True` 时，solver 先完成 variational solve，再从该结果
warm-start 第二个 least-squares solve。也就是说这是两阶段流程: 先弱形式
variational solve，再 collocation polish。Collocation 阶段关闭 fallback，并在
给定时使用 `collocation_method`、`collocation_max_residual` 和
`collocation_max_evaluations`。

`collocation_weight` 决定 polish objective:

| weight | Objective |
| ------ | --------- |
| `0` | 跳过 collocation objective，保留 variational 解 |
| `(0, 1)` | 最小化 blended vector: 到 variational 解的 coefficient-space 距离 + point-collocation residual |
| `1` | 只优化 point-collocation residual |

Blended objective 会把 polish 限制在 variational 解附近，除非 collocation
residual 的权重足以推动系数离开弱形式解。因此 collocation polish 是后处理式
改进，不改变 VEQPy 的主要求解定义。

## Result 与 History

`SolverResult` 保存初始 packed 向量、最终 packed 向量、成功标志、消息、final
residual norm、函数/Jacobian/迭代计数和耗时。每次求解后，`Solver.result`
指向最新结果，`Solver.x0` 更新为最终解。

开启 history 时，`SolverRecord` 会快照当前 case、本次 config 和 result。
`clear()` 只清空 history，不改变 `Solver.x0`；`reset()` 会把 `Solver.x0`
原地置零。
