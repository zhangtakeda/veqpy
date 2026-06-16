# Model

`model` 层保存 VEQPy 中可解释、可序列化的物理对象。它不是 operator runtime 的镜像，而是把独立输入和求解后的快照组织成稳定 API。`Problem` 保存用户侧 source、boundary 和 active-profile topology 输入，`Profile` 是求解后形状剖面快照使用的可序列化参数对象。grid 几何量和平衡诊断量由 `Reactive` property 惰性重建，文件中只保存 root state。

源码位置主要在 `veqpy/model/`。

## 对象分工

| 对象 | 责任 |
| ---- | ---- |
| `Grid` | 径向/角向离散、求积权重、微分/积分矩阵和基函数表 |
| `Profile` | 一维径向剖面的参数化表示 |
| `Boundary` | 固定边界几何参数，可由 GEQDSK boundary 拟合 |
| `Geqdsk` | GEQDSK 数据的读取、保存和转换 |
| `Equilibrium` | 求解后的连续平衡快照与诊断接口 |

`Profile` 使用 scale、power、envelope、offset 和可选 Chebyshev 系数描述一维径向剖面。operator setup 会把 active profile topology 降成 flat arrays；`Profile` 保留在 model 侧，用于 `Equilibrium.shape_profiles` 等可序列化快照。

## Equilibrium 快照

`Equilibrium` 是最重要的输出对象。它接收求解完成后的 root fields，例如:

- 几何尺度: `R0`, `Z0`, `B0`, `a`;
- 离散对象: `grid`;
- 固定边界和形状剖面: `shape_profiles`;
- 磁通与源项导数: `psin`, `psin_r`, `psin_rr`, `FFn_psin`, `Pn_psin`;
- 缩放系数: `alpha1`, `alpha2`。

这些字段足以重建常用物理量，但不会保存 operator 热路径中的临时 buffer。读取 `R`、`Z`、`F`、`P`、`q`、`Ip`、`beta_t`、`jtor`、`jpara`、`jphi`、`Psi`、`Phi` 等 property 时，对象才按公式计算所需量，并由 `Reactive` 保证依赖一致。

## 几何与诊断

模型层暴露的是稳定的快照诊断，而不是 residual 组装过程中的每个中间数组。几何上，`Equilibrium` 提供磁面映射、Jacobian、面积/体积、磁面平均几何因子等；物理上，它提供压强、环向场函数、安全因子、电流和磁通相关诊断。

少量打包几何场会作为 root-adjacent 数据保存，是因为它们对绘图、比较和 GEQDSK 导出有稳定意义。更细的局部导数组合、residual 投影矩阵和 backend workspace 则留在 operator/runtime 层，不进入公共快照 API。

## 设计边界

`model` 层的目标是“最小独立状态 + 可解释派生量”。它负责让用户拿到一个可以读取、绘图、比较和序列化的平衡对象；它不负责再次求解 Grad--Shafranov 方程，也不承担高频 residual 刷新。若未来需要新的诊断，优先在 `Equilibrium` 上增加由现有 root state 派生的 property，而不是直接暴露 solver 或 engine 的内部数组。
