# Quadrature

`quadrature` 模块构造 $[0,1]$ 上的径向求积节点和权重:

$$
\int_0^1 f(x)\,dx \approx \sum_i w_i f(x_i).
$$

这些节点进入 `Grid`，用于径向积分、磁面平均量和 residual projection。这里给出格式选择的语义；具体构造见 `veqpy/math/quadrature.py`。

## Scheme

| Scheme | 节点特征 | 端点 | 典型语义 |
| ------ | -------- | ---- | -------- |
| Legendre | Gauss-Legendre 内点 | 不含端点 | 高阶积分精度，适合光滑 integrand |
| Radau | Gauss-Radau 节点 | 含 $x=1$ | 需要保留边界端点但不保留磁轴端点 |
| Lobatto | Gauss-Lobatto 节点 | 含 $x=0,1$ | 同时显式表示磁轴侧和边界侧端点 |
| Chebyshev | Chebyshev 分布内点 | 不含端点 | 与 Chebyshev 型剖面表示配合自然 |
| Uniform | 等距梯形节点 | 含 $x=0,1$ | 便于调试、对照和外部等距数据 |

所有权重都已经按单位区间归一化，因此常数函数满足

$$
\sum_i w_i = 1.
$$

若节点先在 $[-1,1]$ 上构造，实现会统一映射到 $[0,1]$，避免上层代码混用不同区间约定。

## 选择原则

求积格式不是单纯的性能参数，它会影响 residual projection 和快照诊断中的径向平均。Legendre/Radau/Lobatto 属于 Gauss 型规则，适合以积分精度为主要目标的配置；Chebyshev 更贴近谱型剖面参数；Uniform 则更适合测试和与外部等距数据对照。

## 设计边界

`quadrature` 只生成节点和权重，不定义剖面插值、不构造微分矩阵，也不解释 source route。这些数组被 `Grid` 聚合后，才进入更高层的模型和 operator 逻辑。保持这个边界可以让积分规则作为可枚举 scheme 独立演进。
