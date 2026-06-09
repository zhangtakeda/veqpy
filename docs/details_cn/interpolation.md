# Interpolation

`interpolation` 模块负责一维 source 数据的重映射。给定源节点上的值 $f(s_j)$ 和目标查询点 $q_i$，插值被写成矩阵或等价的局部求值过程:

$$
\hat f(q_i) \approx \sum_j H_{ij} f(s_j).
$$

这层主要服务于 source stage: 外部输入可能位于等距节点，也可能已经位于 operator 网格；source 自变量可能是 `rho`、`psin` 或由 route 派生出的坐标。实现细节见 `veqpy/math/interpolate.py`。

## 两类输入

| nodes | 含义 | 处理方式 |
| ----- | ---- | -------- |
| `grid` | 输入值已经位于当前径向网格 | 直接使用，不做重映射 |
| `uniform` | 输入值位于 $[0,1]$ 等距节点 | 按所选插值格式映射到查询点 |

当 source 坐标为 `rho` 时，查询点固定，可以预先构造重映射矩阵。坐标为 `psin` 时，查询点通常随当前磁通剖面改变，因此 source stage 会在更新后的查询点上重新求值。

## 插值格式

| 格式 | 典型用途 |
| ---- | -------- |
| 全局 Lagrange / barycentric | 任意互异节点之间的通用插值 |
| `linear`, `quadratic`, `cubic` | uniform source 的局部多项式插值 |
| `not-a-knot` | uniform source 的三次样条插值 |
| `barycentric` | uniform source 的局部 barycentric stencil，当前 operator 默认使用 |

局部格式只依赖查询点附近的有限个样本，更适合外部 source 数据的稳健重映射；全局格式保留完整多项式闭包，但每个查询点通常依赖所有源样本。样本数不足时，局部多项式和 spline 会退化到可支持的低阶形式。

## 设计边界

插值层只处理归一化的一维参数，不判断 source route 的物理含义，也不决定约束方程。`operator/source_plan.py` 负责把 route、坐标和节点语义组织成 source plan；插值模块只提供稳定的数值重映射工具。这个边界使 GEQDSK、数组输入和 coefficient-based 输入可以共享同一套 source kernel 入口。
