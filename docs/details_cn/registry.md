# Registry

`Registry` 是 VEQPy 表达“有限但可扩展的方法族”的轻量分发机制。它是一个 decorator-backed typed mapping: 创建时声明 key 类型和值类型，注册时把一个或多个 key 绑定到实现函数。字符串 key 会统一大小写，mapping 对外只读。

源码位置: `veqpy/base/registry.py`。

## 基本用法

```python
registry = Registry(str, Callable)

@registry("name", "alias")
def build(...):
    ...
```

上层 factory 负责规范化用户输入、查 registry、构造错误消息；具体实现则留在对应模块附近。这样新增求积、微分、插值、residual scale 或序列化格式时，不需要扩大中央 `if/elif` 分支。

## 方法族

VEQPy 中很多选择本身就是离散方法空间:

- quadrature scheme;
- calculus scheme;
- uniform source 插值格式;
- source route kernel;
- residual normalization;
- JSON、pickle、GEQDSK 等序列化格式。

把这些选择注册为可枚举 mapping 后，factory、测试和错误信息都可以从同一个公开集合获得支持项。实现、别名和注册声明也保持在局部，便于维护每种方法自己的数学假设和测试。

## Source Route

source route registry 的 key 是

```python
(route, coordinate, nodes)
```

其中 `route` 区分 PF/PP/PI/PJ1/PJ2/PQ 等约束路径，`coordinate` 区分 `rho` 与 `psin`，`nodes` 区分 `uniform` 与 `grid`。这个三元组不是普通字符串插件名，而是 source 建模空间的坐标化表达。

因此 operator build plan 可以在构造阶段验证 route，backend ABI 可以声明支持的组合，测试也可以枚举预期组合是否都有唯一实现。registry 的深层作用在这里体现得最清楚: 它把物理分支从隐式控制流提升为显式、有限、可验证的模型坐标。

## 设计边界

`Registry` 不负责决定某种方法是否物理合适，也不隐藏各方法的实现差异。它只提供稳定入口和受控命名空间，让 solver/operator 围绕固定接口组织，同时允许数值方法和 source route 在各自模块中独立扩展。
