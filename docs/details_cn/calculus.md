# Calculus

`calculus` 模块为径向网格构造微分矩阵和积分矩阵。给定节点 $0\le x_0<\cdots<x_{n-1}\le1$，它返回两类线性算子:

$$
u_\rho \approx D u,\qquad
\int_0^\rho f(s)\,ds \approx G f .
$$

这些矩阵由 `Grid` 持有，后续被几何量、源项剖面、诊断量和 residual projection 复用。文档只说明 scheme 的语义；具体矩阵构造见 `veqpy/math/calculus.py`。

## Scheme

| Scheme              | Alias              | 基本含义                               |
| ------------------- | ------------------ | -------------------------------------- |
| Spectral Difference | `spectral`         | 基于全局 Lagrange 插值的稠密微积分矩阵 |
| CFD33               | `compact`, `cfd33` | 3 点隐式 / 3 点显式 compact stencil    |
| CFD35               | `cfd35`            | 3 点隐式 / 5 点显式 compact stencil    |
| CFD55               | `cfd55`            | 5 点隐式 / 5 点显式 compact stencil    |

Spectral Difference 适合把径向剖面视为全局多项式近似的情形。它的矩阵通常是稠密的，但在中低阶光滑问题上表达直接、误差结构清楚。

Compact Finite Difference 先构造局部 stencil 关系

$$
A u_\rho = B u,
$$

再由该关系得到 $D$ 和 $G$。局部 stencil 使格式更接近有限差分直觉，也便于在端点附近使用单侧窗口。预消元后的矩阵仍可能是稠密的，但它们保留了由局部格式诱导的离散语义。

## 积分常数

积分矩阵统一采用零积分常数约束:

$$
\int_0^0 f(s)\,ds = 0
$$

若网格不显式包含 $0$，实现会用插值约束表达这个条件。这样 `G` 的输出始终表示从磁轴侧起点到当前径向节点的累积积分，而不是只确定到任意常数的原函数。

## 设计边界

`calculus` 只负责从节点和 scheme 生成可复用线性算子。它不解释这些节点对应的物理剖面，也不决定 residual 如何投影。物理层只依赖 `Grid` 暴露的矩阵接口，因此可以在不同径向离散格式之间切换，而不改变 operator、solver 或 `Equilibrium` 的公共语义。
