# Operator

`Operator` 是 VEQPy 的求解运行时中心。它接收 packed 向量 $x$，在固定 layout 和 workspace 上刷新 profile、geometry、source 和 residual，并返回 packed residual。与 `Equilibrium` 快照不同，`Operator` 面向热路径: 内存布局显式、数组原地更新、中间量不作为公共 property 暴露。

源码位置主要在 `veqpy/operator/`、`veqpy/layout/`、`veqpy/workspace/` 和 `veqpy/engine/`。

## OperatorCase

`OperatorCase` 描述一次固定边界求解的输入，包括 source route、source 坐标、节点语义、active profile 系数、边界、热源/电流相关输入以及可选 `Ip` 或 `beta` 约束。

`route`、`coordinate` 和 `nodes` 共同形成 source route key:

```python
(route, coordinate, nodes)
```

它决定 source kernel 和输入解释方式。`heat_input` 与 `current_input` 保持为一维数据，具体物理含义由 route 选择。

## Packed Layout

packed layout 定义优化变量 $x$ 中每个系数的位置。当前 profile family 包括形状剖面 `h`, `v`, `k`, `c0`, `c`, `s`，以及源/磁通相关剖面 `psin`, `F`。只有 active profile 会进入 packed 向量。

默认布局是 degree-first: 先放所有 active profile 的低阶系数，再逐阶推进。这样 residual block、profile 更新和 solver 初值都共享同一套索引语义。

## Build Plan 与 Pipeline

构造 `Operator` 时，`Grid` 会降低为只含数组的 workspace，`OperatorBuildPlan` 则绑定 profile layout、source route、backend ABI、residual block metadata 和各类 offset/scale。这个 plan 描述的是求解拓扑。若 active profile 集合、系数长度或 route 拓扑改变，应重新构造 `Operator`。

一次 residual 调用大致分为四段:

| Stage | 作用 |
| ----- | ---- |
| profile | 从 packed $x$ 刷新 active profile |
| geometry | 从形状剖面计算几何场和磁面平均量 |
| source | 根据 route 和约束生成磁通/源项 root fields |
| residual | 组装并打包 Grad--Shafranov residual |

`Operator.__call__(x)` 返回 variational/Galerkin residual；`residual_collocation(x)` 返回用于 collocation polish 的 pointwise residual。

## Snapshot 边界

求解完成后，`build_equilibrium(x)` 会用最终解刷新 runtime，然后只把快照需要的 root fields 和形状剖面写入 `Equilibrium`。runtime buffer 不会转移到模型对象中。这个边界让 operator 保持高吞吐的可变计算形态，同时让 `Equilibrium` 成为可序列化、可解释的物理快照。
