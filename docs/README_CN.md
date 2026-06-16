<p align="right">
  <a href="../README.md">English</a> |
  <a href="README_CN.md">中文</a>
</p>

<p>
  <img
    align="left"
    src="../docs/assets/veqpy_banner.svg"
    alt="VEQPy logo"
  />
</p>

<br clear="left"><br>

[![arXiv](https://img.shields.io/badge/arXiv-2606.11821-b31b1b.svg)](https://arxiv.org/abs/2606.11821)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Package](https://img.shields.io/badge/package-veqpy-blue)](https://pypi.org/project/veqpy/)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-informational)](tests/)
[![Style](https://img.shields.io/badge/style-ruff-black)](https://docs.astral.sh/ruff/)

---

# VEQPy

**VEQPy** 是 **VEQ** (Veloce EQuilibrium) 的 Python 实现，是面向固定边界、轴对称托卡马克平衡的快速参数化 Grad--Shafranov 求解器。它服务于需要低延迟、连续固定边界几何信息的重复建模调用。与基于二维网格的平衡求解不同，VEQPy 的主要求解目标是有限维投影 Grad--Shafranov 残差，未知量是 MXH 型磁通面谐波以及 shifted-Chebyshev 径向剖面/源项系数；求解过程在这一有限维表示中满足投影 Grad--Shafranov 残差，并输出可重采样、可序列化、可诊断的连续平衡快照。采样局部强形式残差和可选 collocation polish 用于同一参数化表示上的诊断或后处理；它们不重新定义主要求解问题。

VEQPy 适合参数扫描、源项预处理、控制导向迭代、输运耦合和 surrogate model 等场景: 它保留比低阶形状模型更丰富的二维成形和残差诊断，又比完整求解器原生平衡或重构流水线更轻量、更易复用。

## 功能概览

- **紧凑平衡表示**: 用系数描述固定边界磁通面、形状剖面和源项相关径向剖面，求解后得到连续的 `Equilibrium` 快照。
- **统一 source route**: 支持 PF、PP、PI、PJ1、PJ2 和 PQ 等路径，将压力梯度、环向场、通量梯度、电流相关量或安全因子信息归约到共同的有限维残差装配。
- **明确的运行时边界**: `Grid + Problem -> Operator -> Solver -> Equilibrium` 将 packed 系数、运行时 workspace、非线性求解和求解后快照分层处理。`OperatorCase` 仍作为同一问题定义类型的兼容别名保留。
- **GEQDSK 工作流**: 支持 GEQDSK 读写、从 GEQDSK 边界拟合固定边界、快照导出、磁通面比较和常用诊断。
- **公式化模型对象**: `Grid`、`Profile` 和 `Equilibrium` 使用 reactive 派生属性保存最小 root state，并按公式惰性重建几何量和物理诊断量。

## 安装

VEQPy 需要 Python 3.12 或更新版本。普通用户推荐在项目本地虚拟环境中从 PyPI 安装已发布版本:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install veqpy
```

开发者可以从源码 checkout 进行 editable 安装；`.[dev]` 会把运行依赖以及 `pytest`、`ruff`、`build`、`twine`、`packaging` 等开发工具都安装到这个 venv 中。

```bash
git clone https://github.com/zhangtakeda/veqpy.git
cd veqpy
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

如果只需要从本地源码做运行时安装，可以不安装 `dev` extra:

```bash
.venv/bin/python -m pip install .
```

下面的命令都显式使用 `.venv`；是否执行 `source .venv/bin/activate` 只是个人习惯。

## 示例工作流

基础 demo:

```bash
.venv/bin/python examples/minimal_equilibrium.py
```

该脚本构造一个光滑固定边界案例，使用 PF(`psin`) source input 求解平衡，写出 `Equilibrium` JSON 快照，并生成磁通面图像。
默认输出到 `./outputs/minimal_equilibrium`；可用 `VEQPY_OUTPUT_DIR` 指定其它目录。

GEQDSK demo:

```bash
.venv/bin/python examples/geqdsk_workflow.py
```

该脚本读取 EFIT 风格 GEQDSK 文件，拟合为 VEQPy 的固定边界，使用 GEQDSK 中的一维源项剖面求解带 `Ip` 约束的 PF(`psin`) 和 PQ(`psin`) 案例，并输出双列 VEQPy 与 GEQDSK 磁通面比较图。默认读取 `./data/EFIT.geqdsk`，输出到 `./outputs/geqdsk_workflow`；可用 `VEQPY_GEQDSK` 和 `VEQPY_OUTPUT_DIR` 覆盖这些路径。**论文图像的可复现脚本会随首个公开 arXiv 版本对应的 tagged artifact 发布**。

## 开发检查

```bash
.venv/bin/python -m compileall -q veqpy tests examples
.venv/bin/ruff check veqpy tests examples
.venv/bin/python -m pytest
```

## 实现文档

设计模式与模型层:

- [[reactive.md]](details_cn/reactive.md): 最小 root state、公式化派生属性、惰性依赖验证和快照一致性。
- [[registry.md]](details_cn/registry.md): registry-backed 方法族、source route 坐标化和分发边界。
- [[serial.md]](details_cn/serial.md): root-state 序列化、格式 handler 和持久化边界。
- [[model.md]](details_cn/model.md): `Grid`、`Profile`、`Boundary`、`Geqdsk`、`Equilibrium` 的职责、快照边界和诊断接口。

热路径算子与求解器:

- [[operator.md]](details_cn/operator.md): packed layout、build plan、stage pipeline 和 runtime/snapshot 分离。
- [[solver.md]](details_cn/solver.md): 非线性求解生命周期、fallback、residual 归一化和 collocation polish。

数值构造:

- [[interpolation.md]](details_cn/interpolation.md): 一维 source 数据重映射。
- [[quadrature.md]](details_cn/quadrature.md): 径向求积 scheme 的语义。
- [[calculus.md]](details_cn/calculus.md): 径向微分/积分矩阵的 scheme 边界。

## 论文与复现资源

VEQPy 与配套论文 **"VEQ: a fast parametric Grad--Shafranov solver for fixed-boundary tokamak equilibria with flexible source inputs"** 相关。论文专用复现包会作为 tagged artifact 随首个公开 arXiv 版本发布，内容包括 figure scripts、benchmark scripts、GEQDSK 输入或生成脚本、渲染图像以及依赖元数据。

相关 VEQ 系列和表示方法论文包括:

- Ruohan Zhang, Huasheng Xie, Yueyan Li, Weiqi Meng, Feng Wang, and Zhengxiong Wang, "VEQ: a fast parametric Grad-Shafranov solver for fixed-boundary tokamak equilibria with flexible source profiles", arXiv:2606.11821, 2026. <https://arxiv.org/abs/2606.11821>
- Huasheng Xie and Yueyan Li, "What Is the Minimum Number of Parameters Required to Represent Solutions of the Grad-Shafranov Equation?", arXiv:2601.02942, 2026. <https://arxiv.org/abs/2601.02942>
- Xingyu Li, Huasheng Xie, Lai Wei, and Zhengxiong Wang, "Investigation of Toroidal Rotation Effects on Spherical Torus Equilibria using the Fast Spectral Solver VEQ-R", arXiv:2602.11422, 2026. <https://arxiv.org/abs/2602.11422>

---

<p>
<img align="left" src="../docs/assets/veqpy_icon.svg" width="150" alt="veqpy logo">

<strong>License</strong>:<br>
<em>BSD 3-Clause License</em><br>

<strong>Maintainer</strong> (rhzhang):<br>
<em>Email</em> - <em>rhzhang@mail.dlut.edu.cn</em> | <em>zhangtakeda@gmail.com</em><br>
<em>Homepage</em> - <em>https://zhangtakeda.github.io</em>

</p>
