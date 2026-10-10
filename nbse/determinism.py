"""可复现性工具：让 GPU 计算逐位可重复。

为什么需要
----------
实测（RTX 5070 Ti Laptop + torch 2.11 + cuDNN）：默认设置下**同一个输入
连续 8 次前向，得到 8 个不同的结果**，最大差约 1.6e-05（float32 量级）。
原因不是模型的 BatchNorm（已确认 running stats 不变、模型在 eval 模式），
而是 cuDNN 默认卷积算法在选定实现时引入了 run-to-run 的不确定性。

这个量级本身无害，但**在 beamforming 链路里会被放大**：掩蔽的小扰动经过
协方差估计与矩阵求逆后，权重可以相差 0.78（实测）。后果是所有评测数字
不可复现——同一个 checkpoint 跑两次得到不同的结果表，这会让
「神经方法是否优于基线」这类结论失去意义。

修复：开 `torch.use_deterministic_algorithms` + `cudnn.deterministic`。
实测代价可忽略（1.5 → 1.6 ms/batch，约 7%）。

注意 `torch.use_deterministic_algorithms(True)` 会让**没有确定性实现**的算子
直接抛错。本项目的模型（Conv2d / ConvTranspose2d / BatchNorm / GRU / Linear）
都有确定性实现，故用严格模式；`warn_only=False` 是有意的——把「悄悄用了
不确定算子」变成响亮失败，而不是让结果再次静默漂移。
"""

from __future__ import annotations


def enable_determinism(seed: int | None = None,
                       warn_only: bool = False) -> None:
    """开启确定性计算（幂等）。

    Parameters
    ----------
    seed       : 给定则同时固定 torch 的随机种子
    warn_only  : True 时对无确定性实现的算子只告警不报错。
                 默认 False —— 宁可响亮失败，也不要静默的不可复现。
    """
    import torch

    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def determinism_status() -> dict:
    """当前确定性设置，便于在日志/结果表里留痕。"""
    import torch

    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
