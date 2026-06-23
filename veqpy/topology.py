from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topology:
    # profiles: ProfileTopology
    h_count: int
    v_count: int
    kappa_count: int
    psin_count: int
    F_count: int
    c_counts: tuple[int, ...]  # c0,c1,...
    s_counts: tuple[int, ...]  # s1,s2,...

    # grid: GridTopology
    Nr: int
    Nt: int
    quadrature: str = "legendre"
    calculus: str = "spectral"
    L_max: int | None  # 从 ProfileTopo 自动推断, 且不允许显式设置
    M_max: int | None = None  # 默认从 c_counts/s_counts 推断, 显式设置则代表边界 ca/sa 的最大阶数
    K_max: int | None = None  # 默认 K_max = M_max

    # source: SourceTopology
    route: str  # "PF"
    coordinate: str  # "psin"
    constraint: str | None  # "Ip"
    nodes: str = "uniform"
    sample_count: int | None  # 为grid就默认用Grid.Nr填充; uniform则必须要求给定值否则报错

    # 不进入 Cpp 内核的元数据, 因此不需要做 str -> int
    # 开启所有 fastmath 等选项
    # "release" 对应 O3但是没有浮点数交换等等
    # 还有就是 "debug" 用于python端查看kernel具体哪里报错什么的
    build: str = "fastmath"
    layout: str = "degree"  # x优先对L阶数排序; "family" 剖面种类先排序

    version: str = "v1.0.0"  # 几乎不变, 用户不能任意传入
    kernel_id: int  # 自动初始化的 int/str, 完全由上述其他数据的 hash等决定帮助 python 端快速索引
