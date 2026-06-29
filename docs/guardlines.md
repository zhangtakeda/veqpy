# VEQPy 与 VEQlib 的职责边界

**核心原则**: VEQPy 负责建模与用户接口, VEQlib 负责具体的计算语义, 包括完整的求解流程与算子内核.

- **VEQPy**: 承担所有建模、可视化、API 语义的实现, 例如 `Equilibrium` 等高层抽象.
- **VEQPy/adapter**: 如果需要从 `Problem`、`Operator`、GEQDSK 等高层对象转换到 kernel 输入, 转换只能发生在 `veqlib.kernel` 之外的 adapter、benchmark 或 legacy compatibility 层.
- **VEQPy**: numba-kernel 仅作为参考实现与 VEQlib 的对照基线, 待 VEQlib 成熟后将被弃用.
- **`veqlib.kernel`**: 是与 C++/nanobind ABI 对齐的 Python bridge, 不是 VEQPy model adapter. 它只拥有 `KernelTopology`、`KernelBuild`、`KernelBoundary`、`KernelInput`、`KernelSolve`、`KernelResult`、artifact registry 和 handle lifecycle.
- **`veqlib.kernel` 禁止依赖**: `veqpy.model.Problem`、`veqpy.operator.Operator`、`source_plan`、`packed_layout` 等 VEQPy 内部语义. `Problem` 兼容若保留, 必须放在 `veqlib.kernel` 之外的独立 adapter/benchmark 层, 不得回流到 kernel package 或 VEQPy 公共 model 层.
- **VEQlib**: 以计算性能与 HPC 为唯一设计准则. 所产出的 kernel 在 kernel.solve 时, C++ 端必须做到零内存分配, 且除 index 操作与 double 运算外尽可能消除一切额外开销.
- **VEQlib**: 不面向用户, 无需考虑任何安全性兜底策略, 包括非理想边缘情况下的行为.

---

# 核心概念说明

1. **setup**: 在当前版本的 VEQPy 中, 指预热阶段产生的开销, 包括网格预计算等. 在 VEQlib 完全体实现后, VEQPy 层面将不再存在此类预计算开销, 而 VEQlib 的 setup 阶段几乎全部在编译期完成.
2. **runtime**: 指对一个已编译好的 kernel, 持续输入成千上万个 case 进行求解的阶段. runtime 的单次求解耗时是我们优先关注的性能指标.
3. **topology**: 编译 VEQlib kernel 时所依赖的全部模板元参数的集合. 每个 kernel 与其 topology 一一对应. cache 的 key 至少包含"VEQlib 源码 + topology + build options + toolchain/native ABI"的 hash.
4. **case**: 在 runtime 阶段输入的待求解问题数据, 由接口层转化为 C++ ABI 可直接消费的标量和 `float64` 数组后传入 VEQlib.

> **注意**: topology 与 case 不一定各自只有单一的类型表示, 它们均代表一类数据的聚合.

---

# VEQPy 建模语义

## topology (setup 阶段固定)

- [x] **topology.dofs**: 以字典形式描述 kernel 支持的自由度参数搭配, 例如 `{h=2, v=0, ..., c_counts: tuple/list/array, ...}`.
- [x] **topology.grid**: 径向节点数 `Nr`、环向节点数 `Nt`, 以及更细节的 quadrature(积分节点分布方案)与 calculus(微积分构造方案).
- [x] **topology.orders**: `L_max`(默认从 dofs 推断), `M_max`(默认从 dofs 推断; 显式设置时用于保留更高阶的边界影响), `K_max`(约束 c/s_family 的计算阶数).
- [x] **topology.source**: 包含 `route`(PF)、`coordinate`(rho/psin)、`sampling`(grid/uniform 及具体采样点数)、`constraint`(null/Ip/beta/Ip_beta).
- [x] **topology.layout**: 定义如何将 dofs 字典映射为展平后的向量 `x` (degree/family).
- [x] **topology.build**: 编译配置, 提供4种预设: 默认的 fastmath(全开优化), fastmath-enzyme(全开优化), debug(便于定位错误), release(遇错直接退出而非报错). 也支持在任一预设基础上对具体编译选项单独调整, 包括 enzyme 的 block 大小 (这部分隐含大量选项, CMake 中的具体配置例如 enable_unsafe_math_optimizations 等).

## case (runtime 阶段逐次输入)

- [x] **case.boundary / `KernelBoundary`**: 当前 runtime ABI 只接受已经参数化的边界:
  - `a, R0, Z0, B0, ka: double`
  - `c_offsets, s_offsets: 1D float64 ndarray`
  - `c_offsets` 与 `s_offsets` 长度不得超过 topology 中的 `M_max + 1`. `s_offsets[0]` 按约定为 0.
  - `Boundary(R0, Z0, B0, a, R: array, Z: array)` 形式是未来 planned adapter. 如果由 kernel 内部把 RZ 初始化成边界参数, 拟合阶数、拟合方法和最大 Fourier 阶数必须在 topology 阶段固定.
- [x] **case.inputs / `KernelInput`**: runtime source 输入是 C++ 已降阶后的 typed 数组, 不是 VEQPy `Profile` 或 `Problem`.
  - `scaled_heat: 1D float64 ndarray`
  - `scaled_current: 1D float64 ndarray`
  - 两者必须同 shape, C-contiguous, 且长度必须等于 `KernelTopology.sample_count`.
  - 当前过渡实现仍传入 `scaled_heat`、`scaled_current`、`scaled_Ip`: 即 Python 侧已经按当前 legacy 规则完成 `mu0` scaling. 未来成熟形态应删除 Python/C++ 两侧的 runtime scaling, 由 C++ 编译期常数和 route 语义处理单位.
  - `scaled_Ip` 与 `beta` 是 double 约束值; 未提供时使用 `NaN` 表示. `fix_rho` 是 double runtime 参数.
- [x] **case.config / `KernelSolve`**: runtime solve policy 与 case 一起进入 ABI, 不通过 JSON 热路径传递.
  - `method`: `powell`、`levenberg-marquardt`、`newton-krylov`、`newton-raphson` 或对应 int code.
  - `initial`: `cold-zeros`、`cold-geometric`、`cold`、`warm-clone` 或对应 int code.
  - `norm`: `none`、`fast`、`balanced`、`safe` 或对应 int code.
  - `max_residual`、`max_evaluations`、accepted residual 参数、residual normalization 参数均作为 double/int runtime 字段传入.

---

# 用户侧 API 形态与内核 ABI 形态

## Kernel 对象设计原则

VEQPy 的 kernel 应以对象为粒度进行管理: 同一时刻可以存在多个 kernel handle, 它们可以共享同一个编译产物(artifact/native module/dict, 内部包含元数据), 但每个 handle 私有一份 C++ 层的 solver、context、workspace、result.

Kernel 对 C++ 接口采用**惰性加载(lazy-load)**: 默认不立即挂载 native module, 也可通过参数配置为 eager 加载. 若调用者未保存 kernel 的返回值, 则 handle 立即析构, lazy 行为保证此时几乎零开销. 此外, close 释放 handle 私有 C++ workspace, 在 handle 不再被引用时(析构)释放对应 C++ 端的栈内存占用. 同一个 handle 实例, 不允许被多线程同时调用 solve. artifact 的并发安全由 per-artifact 文件锁保证. last_used_at 作为 advisory timestamp, 在 native module 挂载时于文件锁内一并写入; clean() 拿不到独占锁的 artifact 直接跳过.

## API 接口说明

- **`kernel = build(topo)`**: 先检索本地 artifact cache, 按需编译内核并挂载到 handle(默认 lazy-load native module). 若调用者未保存返回值, handle 立即析构; 在 lazy 模式下几乎无 runtime 开销. 两个进程同时对同一 artifact 触发 build 时, 通过 per-artifact 文件锁串行化: 一个进程持锁编译, 另一个阻塞等待; 锁释放后重新检查 artifact metadata 和 .so, 若可复用则直接加载结果. 编译失败(cmake 报错、OOM、进程被 kill), 锁的超时行为需要更细的设计. 不支持同进程 native reload.
- **`build(topo)`(无返回值接收)**: 仅执行编译, 将 topology 对应的 artifact 写入本地 cache. handle 析构后, 在未 lazy-load 的情况下几乎无 runtime 开销.
- **`result = kernel.solve(case, solve=...)`**: 高层 Python handle API. 它负责 build/load artifact, lazy-load native module, 调用 `set_kernel_runtime(...)` 写入 runtime case 与 solve policy, 调用 C++ `solve_direct()`, 然后把 C++ transient view copy-out 成 Python-owned `KernelResult`. `kernel.result` 与 `kernel.history[-1]` 指向这个 owned snapshot. 普通 Python 调用者应使用这一层.
- **`KernelSolver.set_kernel_runtime(...)`**: 低层 nanobind ABI setter, 不是求解接口. 它只把 `KernelInput + KernelSolve` 展开后的 scalars/ndarray/int code 一次性写入 C++ `KernelSolver` 的当前 `CaseInput/SolveContext`, 并刷新 warm-clone、cold initial state、residual scale 等 runtime 上下文. 它不返回结果, 不更新 Python `Kernel.history`, 也不做 artifact build/load. 调用它之后仍需调用 `solve_direct()` 才会真正求解.
- **`KernelSolver.set_case_json(payload)`**: debug/legacy 入口. 功能上与 `set_kernel_runtime(...)` 同属 runtime setter, 但通过 JSON 解析完成. 新 hot path 和新 API 不应依赖它.
- **`kernel.clear()`**: 清除该 kernel handle 缓存的全部 history. kernel.clear() 清的是 Python 侧的 history, 不影响 C++ 工作区, 所以 reuse 的语义不受 clear() 影响.
- **`kernel.close()`**: 释放该 handle 私有的 C++ solver、context、workspace、result. 不保证卸载已加载的 native module.
- **`kernel.residual(x)` / planned**: 传入与 dofs 长度一致的向量 `x`, 在当前 runtime case/context 下直接返回对应 residual 的 Python-owned copy. `kernel.residual_into(x, out)` 是面向 benchmark/debug 的 no-allocation 变体.
- **`kernel.jvp(x, v)` / planned**: 在当前 runtime case/context 下计算 Jacobian-vector product. AD/Enzyme 或 FD 选择是 topology/build policy, 不能在 runtime 临时改变.
- **`kernel.jacobian(x)` / planned**: 在当前 runtime case/context 下返回 dense Jacobian 的 Python-owned copy. 若后续支持 no-allocation 版本, 应命名为 `jacobian_into(x, out)`.
- **`result = solve(topo, case)`**: 临时求解一次, 完成后立即释放 kernel. 等价于 `return build(topo).solve(case)`. 它返回的 result 是 owned/copy-out Result snapshot, 因此不依赖临时 kernel 的生命周期.
- **`clean()`**: clean() 清理的是磁盘 artifact cache. 支持按最晚编译日期或最晚调用日期筛选清理范围. 拿不到独占锁的 artifact 直接跳过; 已被当前进程 import 的 native module 在 POSIX 上仍可被 unlink(当前进程不受影响, 但磁盘文件消失), Windows 上通常会删除失败.

## Result 与 lifecycle

- **C++ result view**: `solve_direct()` 返回的 `x/raw/scaled/alpha` 是 C++ mutable workspace 的 transient view. 下一次 solve、set runtime case、close 或 solver 析构后, 这些 view 不应再作为稳定结果使用.
- **Python `KernelResult`**: `Kernel.solve()` 必须立即 copy-out `x/raw/scaled/alpha`, 并复制 scalar stats (`success`、`info`、`nfev`、norms 等). `KernelResult` 是用户可持久保存的结果对象.
- **`kernel.history`**: 只保存 Python-owned `KernelResult` snapshot. 第二次 solve 不得修改第一次 result.
- **`kernel.result`**: 指向最近一次 `KernelResult`, 等价于 `kernel.history[-1]` 的 result 引用.
- **`kernel.clear()`**: 只清 Python `history/result`, 不清 C++ context/workspace, 不改变 warm-clone/reuse 语义.
- **`kernel.close()`**: 释放当前 handle 对 C++ solver/context/workspace 的引用. native module 仍由 Python import 系统和 process cache 持有, 不保证卸载.

## VEQlib ABI 约束

VEQlib 的 ABI 只接受以下类型: `double`, 1D C-contiguous `float64` ndarray, `int`/`size_t`. 所有字符串参数必须在 Python 层映射为对应的枚举整数, C++ 只消费 int code. 该映射表需在 Python 与 C++ 源码中同步维护. 未来应以一份 `.toml` 或 `.json` schema 文件作为 single source of truth, 构建时自动生成 Python 侧的枚举类和 C++ 侧的头文件. 这样版本不一致时编译直接失败.

必须区分 **ABI 校验** 与 **安全兜底**:

- **ABI 校验必须做**: nanobind/C++ 与 Python `Kernel` 边界必须检查 ndarray ndim、dtype、C-contiguous、source length 是否等于 `sample_count`、offset length 是否适配 `M_max + 1`、x/out shape 是否等于 `x_size`、enum code 是否合法、topology 是否支持当前 route/source ownership.
- **安全兜底不做**: VEQlib 不做 silent clipping、自动重采样、route fallback、profile ownership 自动修正、物理参数猜测、失败后自动切换 solver 等用户体验型容错. 这些如果需要, 应发生在 VEQPy 高层 adapter, 不进入 VEQlib hot path.
- **错误策略**: ABI contract violation 应立即抛出明确错误; 不能在 hot path 中静默修补输入.

---

> **注意**: 以下约束描述的是最终代码的成熟形态, 当前实现很可能尚未完全达到. 由于代码仍在开发/重构中, 现有 pytest 测试不应作为重构基线. 配点法相关内容暂不纳入迁移范围. 此外, 当前代码存在电流/压强的 scale 操作, 但未来 Python 与 C++ 层面均应去除, C++ 层面将 mu0 作为编译期常数编译期应该可以自动化简, 同时使内核实现的语义更加明确.
