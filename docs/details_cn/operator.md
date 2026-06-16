# Operator

`Operator` 是一次固定边界平衡求解的数值中心。它把 `OperatorCase`、
`Grid` 和 packed 系数向量 $x$ 转换成 `Solver` 使用的有限维
Grad--Shafranov residual。从物理数值层面看，它刷新磁面形状，根据
source route 构造源项剖面，并把强形式 residual 投影到 active 系数基上。

源码位置主要在 `veqpy/operator/`、`veqpy/layout/`、`veqpy/workspace/` 和 `veqpy/engine/`。

## OperatorCase

`OperatorCase` 描述一次固定边界求解的输入，包括 source route、source 坐标、节点语义、active profile 系数、边界、热源/电流相关输入以及可选 `Ip` 或 `beta` 约束。

`route`、`coordinate` 和 `nodes` 共同形成 source route key:

```python
(route, coordinate, nodes)
```

它决定 source kernel 和输入解释方式。`heat_input` 与 `current_input` 保持为一维数据，具体物理含义由 route 选择。

`heat_input` 总是按 pressure-like setup 数据处理；当它处在预期物理量级
范围内时，`OperatorCase` 构造阶段会乘以 `mu0`。`Ip` 也采用同样的缩放。
`current_input` 只有在 current-profile route (`PI`, `PJ1`, `PJ2`) 中会乘以
`mu0`；在其他 route 中，它已经是归一化量或场派生驱动量，例如 `FF'`、
`psin_r` 或 `q`。量级明显不符合 setup 约定的输入会在构造 operator 前被拒绝。

## Source Routes

所有 route 最终都为 residual 组装生成同一组 root fields:

- `psin`, `psin_r`, `psin_rr`: 归一化磁通坐标及其径向导数；
- `Pn_psin`: 对 `psin` 的归一化压强导数；
- `FFn_psin`: 环向场函数乘积的归一化导数；
- `alpha1`, `alpha2`: 由 route 和全局约束确定的源项缩放系数。

route 的差别在于这些场如何由一维输入重构:

| route | `heat_input` 含义 | `current_input` 含义 |
| ----- | ----------------- | -------------------- |
| `PF` | 压强梯度数据 (`P_r` 或 `P_psi`) | 环向场源项 (`FF'`) |
| `PP` | 压强梯度数据 | 归一化磁通梯度驱动 `psin_r` |
| `PI` | 压强梯度数据 | 包围环向电流 `I_tor` |
| `PJ1` | 压强梯度数据 | 环向电流密度 `j_tor` |
| `PJ2` | 压强梯度数据 | 平行电流密度 `j_parallel`，并使用当前 `F` profile |
| `PQ` | 压强梯度数据 | 安全因子 `q` |

`coordinate="rho"` 表示源项样本以径向标签为自变量；`coordinate="psin"`
表示以归一化磁通为自变量。`nodes="grid"` 表示输入已经在 operator 径向网格上；
`nodes="uniform"` 表示输入来自均匀 source 轴，需要重映射到当前查询点。
`PP/psin/uniform` route 使用 `sqrt(psin)` 作为均匀 source 参数化，使边缘附近
的输入采样更均匀。

对于非 `PJ2` 的 `psin/uniform` route (`PF`, `PP`, `PI`, `PJ1`, `PQ`)，
`psin` 必须是 active optimized profile，因为每次 residual 评估都要用当前磁通
坐标查询 source 样本。`psin/grid` 输入已经 materialized 到 operator 径向节点，
因此不拥有 active `psin` profile。active `F` 只由 `PJ2` 要求，因为
parallel-current source 会使用当前优化的环向场 profile；active `F` 和 active
`psin` 互斥。`PQ` 对环向场 profile 也更严格: 它从 `q` 和边界值 `R0 * B0`
解出 `F` 或 `F^2`，因此不接受 active `F` profile。

## Constraints 与 Scaling

没有全局约束时，source route 会从重构源项剖面的径向归一化中确定 `alpha1`
和 `alpha2`。给定 `Ip` 时，route 会选择使积分环向电流匹配目标电流的缩放。
给定 `beta` 时，route 会用体积加权压强积分和参考磁场 `B0` 确定压强缩放。
多数 route 可以同时处理 `Ip` 和 `beta`；`PF` 会拒绝这个组合，因为对该 route
可用的两个 source driver 来说这会过约束。

## Packed Layout

packed layout 定义优化变量 $x$ 中每个系数的位置。当前 profile family 包括形状剖面 `h`, `v`, `k`, `c0`, `c`, `s`，以及源/磁通相关剖面 `psin`, `F`。只有 active profile 会进入 packed 向量。

默认布局是 degree-first: 先放所有 active profile 的低阶系数，再逐阶推进。这样 residual block、profile 更新和 solver 初值都共享同一套索引语义。

形状 profile 决定连续固定边界磁面族。`h`, `v`, `k` 表示低阶径向 shaping，
`c*` 和 `s*` 表示 Fourier 谐波。可选的 `psin` 与 `F` profile 会在 case 提供
其系数时进入 active 集合；route validation 决定这种归属对所选 source model
是否有物理意义。

## Residual Pipeline

构造 `Operator` 时，active profile 集合、系数长度、source route、residual
block 和固定网格会被冻结成一次求解拓扑。若 active profile 集合、系数长度或
route 拓扑改变，应重新构造 `Operator`。替换 case 只有在保持相同 packed
拓扑时才可以复用同一个 operator。

一次 residual 调用大致分为四段:

| Stage | 作用 |
| ----- | ---- |
| profile | 从 packed $x$ 刷新 active profile |
| geometry | 从形状剖面计算几何场和磁面平均量 |
| source | 根据 route 重构 `psin`、压强、场函数和电流相关源项 |
| residual | 组装 Grad--Shafranov residual，并投影到每个 active 系数 block |

`Operator.__call__(x)` 返回 variational/Galerkin residual。每个 residual block
对应一个 active profile block，并与 packed state 使用同一套基函数顺序。这是
VEQPy 的主要求解方程。

`residual_collocation(x)` 返回用于 collocation polish 的 quadrature-scaled
pointwise residual。它在每个 `(rho, theta)` 网格点上评估局部强形式力平衡
residual，返回长度为 `Nr * Nt` 的向量。这个量是诊断或后处理目标；它不替代
Galerkin residual 作为主要求解定义。

## Runtime Memory 词表

可变 operator runtime 使用一套窄词表，让 engine binding 保持显式，同时避免
Numba kernel 接收 Python workspace object。`*_fields` 表示采样 slab，其行或轴
承载物理/数值采样值，例如 grid radial tables、geometry surface rows、residual
root rows 和 profile value/derivative rows。`*_operators` 表示径向微分、积分等
线性算子。`*_metadata` 表示 layout 决策，包括 route code、profile id、block
code、coefficient index rows、active lengths 和 grid size metadata。
`*_scratch` 是 kernel 调用内复用的临时工作区，调用结束后没有持久物理意义；
`alpha_state` 这类小型可变向量属于 state。

热路径 engine 调用直接接收 field slabs、operators、标量常数和 metadata。
`GridWorkspace.T` 或 `GeometryWorkspace.R_surface` 这类 workspace property 是
用于 debug 和 row-contract 文档化的 view accessor；主 fused runtime ABI 不依赖
这些 property。

## Snapshot 边界

求解完成后，`build_equilibrium(x)` 会用最终解刷新 runtime，然后只把快照需要的 root fields 和形状剖面写入 `Equilibrium`。runtime buffer 不会转移到模型对象中。这个边界让 operator 保持高吞吐的可变计算形态，同时让 `Equilibrium` 成为可序列化、可解释的物理快照。
