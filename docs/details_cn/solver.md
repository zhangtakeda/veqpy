# Solver

`Solver` 负责非线性求解流程。它不定义 packed layout，不实现 residual，也不持有物理 runtime state；这些属于 `Operator`。`Solver` 只围绕 residual callable 组织初值、SciPy 方法、fallback、残差接受判据、collocation polish 和历史记录。

源码位置: `veqpy/solver/`。

## 配置与初值

`SolverConfig` 保存求解控制参数，例如主方法、最大 residual、最大函数调用次数、初值策略、fallback 开关、residual normalization 和是否启用 collocation polish。

初值来源按优先级选择:

| 来源 | 行为 |
| ---- | ---- |
| 显式 `x0` | 校验后作为当前初值 |
| `warm` | 复用上一轮最终解 |
| `zeros` | 使用全零 packed 向量 |
| `homothetic` | 使用边界形状估计初值 |
| 默认 | 从 `OperatorCase.profile_coeffs` encode 初值 |

求解结束后，`Solver.x0` 会更新为最终解，便于后续参数扫描或 continuation 使用 warm start。

## 求解流程

普通 variational solve 的流程是:

1. 合并默认配置和本次 `solve()` 覆盖参数。
2. 构造初值并调用主 SciPy 方法。
3. 若失败且 fallback 打开，按 fallback 方法继续尝试。
4. 在可用结果中选择成功且 residual 达标的结果；若都失败，保留 residual norm 最小者用于诊断。
5. 返回 `SolverResult`，可选写入 history。

成功判定不只看 SciPy 的 `success`，还会检查最终 residual norm 是否低于接受阈值。

## Residual Normalization

Residual normalization 用于降低不同 residual block 的幅值不平衡。策略通过 registry 选择，solver 主流程只依赖 `make_residual_scale()` 接口。当前可选模式包括 `none`、`fast` (`block_rms`)、`balance`/`balanced` (`block_huber`) 和 `safe` (`block_sensitivity`)。

## Collocation Polish

`enable_collocation=True` 时，solver 先完成 variational solve，再从该结果 warm-start collocation polish。`collocation_weight` 控制是否跳过、混合 coefficient-space anchor，或只优化 collocation residual。

Collocation polish 是后处理式改进，不改变 VEQPy 的主要求解定义。solver 仍只使用 operator 提供的 residual 和 block metadata，不解释 residual 的物理细节。

## Result 与 History

`SolverResult` 保存初值、最终解、成功标志、消息、residual norm、调用计数和耗时。开启 history 时，还会记录 case、config 和 result 的快照，便于比较不同求解尝试。
