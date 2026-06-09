# Reactive

`Reactive` 是 VEQPy 模型层表达“最小状态与公式派生”的基础设施。它把属性分成两类: root property 是构造、反序列化或用户显式写入的独立状态；derived property 是由 root 和其他 derived property 按公式给出的量。derived property 会缓存，但读取时会先检查依赖版本，必要时再重算。

源码位置: `veqpy/base/reactive.py`。

## 为什么需要

固定边界平衡中真正需要保存的独立量很少，例如网格、形状剖面、磁通剖面、源项导数和缩放系数。许多用户关心的量，如 $R,Z,J,q,\beta_t,j_\phi$ 和 GEQDSK 导出剖面，都是这些 root state 的确定结果。

如果把所有中间量都作为可变字段保存，对象会同时拥有“状态”和“公式”两份真相，并且必须维护 profile、geometry、source 和 diagnostics 的更新顺序。`Reactive` 把这个顺序问题变成依赖图问题: 写 root 只更新版本，读 derived property 时再按依赖关系验证缓存。

## 核心机制

每个子类显式声明:

```python
root_properties = {"a", "b", ...}
```

root setter 会执行输入检查、保存规范化后的值，并递增版本。非 root 的 property 被视为 derived property；类创建时会解析 `self.xxx` 访问并建立依赖图，必要时也可以用 `@depends_on(...)` 补充间接依赖。

读取 derived property 时，getter 会比较依赖 token。token 不只包含直接字段版本，也会包含嵌套 `Reactive` 对象的 revision。因此 `Equilibrium.grid` 或 `shape_profiles` 内部发生变化时，依赖它们的几何和诊断量会在下次读取时自动失效。

## 设计边界

`Reactive` 用在模型对象和求解后快照中，不用于 operator 热路径。求解器运行时需要显式 workspace 和原地数组更新；快照对象需要可解释、可序列化、按需计算的公式系统。二者分层后，VEQPy 既能保持 solver 的内存局部性，也能让公开模型对象避免缓存陈旧和重复真相。
