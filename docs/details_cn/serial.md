# Serial

`Serial` 是 VEQPy 的统一序列化基类。它保存的是可重建模型对象的声明式 root state，而不是对象内存的完整镜像。JSON、pickle 和 GEQDSK 等格式共享 `load/read/write` 入口，具体读写器由 serializer registry 分发。

源码位置: `veqpy/base/serial.py`，GEQDSK 相关实现见 `veqpy/model/geqdsk.py`。

## Serializable Attributes

每个 `Serial` 子类声明可序列化字段:

```python
@classmethod
def serial_attributes(cls) -> dict[str, type]:
    return {"field": float, "array": np.ndarray}
```

dataclass 子类可以从类型注解推断字段。未进入 `serial_attributes()` 的派生缓存、workspace buffer 和 engine 临时数组不属于持久状态。

## 格式语义

JSON 会写出带类型名的结构，并递归处理嵌套 `Serial` 对象、dataclass、NumPy 数组和常见容器。pickle 保存字段字典。其他格式通过 registry 注册，例如 GEQDSK 文本读写。

读取后，reactive 对象会从 root state 重新建立派生缓存，而不是从文件中恢复旧缓存。

## load/read/write

`Serial.load(file)` 创建新对象并读取文件；`read(file)` 在已有对象上恢复字段；`write(file)` 根据扩展名选择 writer。构造参数复杂或 frozen dataclass 更适合使用 `load()`。

## 设计边界

`Serial` 与 `Reactive` 配合形成清晰边界: 文件保存独立状态，读取后的对象按公式重建派生量。因此公开文件格式不会随 runtime workspace 的优化而改变，也不会把一次求解或绘图过程中的缓存误认为物理事实。
