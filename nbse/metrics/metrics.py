"""评测指标：PESQ（宽带 16k）、STOI、SI-SDR。

SI-SDR 为纯 NumPy 实现，无外部依赖。
PESQ / STOI 依赖第三方包，采用惰性导入——未安装时只有调用它们才会报错，
不影响 SI-SDR 的使用与测试。

约定：所有函数签名为 (reference, estimate) -> float，两者均为单通道
np.ndarray，长度一致，取值归一化到 [-1, 1]。
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """尺度不变信噪比（Scale-Invariant SDR），单位 dB。

    SI-SDR = 10·log10( ||s_target||² / ||e_noise||² )
        s_target = (<est, ref> / ||ref||²) · ref
        e_noise  = est - s_target

    尺度不变性使其适合评估增强算法——算法整体放大或缩小不应影响得分。

    参考：Le Roux et al., "SDR - half-baked or well done?" (ICASSP 2019)
    """
    reference = np.asarray(reference, dtype=np.float64).ravel()
    estimate = np.asarray(estimate, dtype=np.float64).ravel()

    if reference.shape != estimate.shape:
        raise ValueError(
            f"参考与估计长度不一致：{reference.shape} vs {estimate.shape}"
        )

    ref_energy = float(np.dot(reference, reference))
    if ref_energy < EPS:
        # 参考信号近乎静音，SI-SDR 无定义
        return float("nan")

    alpha = float(np.dot(estimate, reference)) / ref_energy
    s_target = alpha * reference
    e_noise = estimate - s_target

    target_energy = float(np.dot(s_target, s_target))
    noise_energy = float(np.dot(e_noise, e_noise))

    if noise_energy < EPS:
        # 完美重建
        return float("inf")

    return 10.0 * np.log10(target_energy / noise_energy)


def si_sdr_improvement(noisy: np.ndarray, enhanced: np.ndarray,
                       reference: np.ndarray) -> float:
    """SI-SDR 提升量（dB）：增强后相对含噪输入的改善。"""
    return si_sdr(reference, enhanced) - si_sdr(reference, noisy)


def pesq_wb(reference: np.ndarray, estimate: np.ndarray, fs: int = 16000) -> float:
    """宽带 PESQ。仅在 fs=16000 下为 'wb' 模式（窄带模式需 fs=8000）。"""
    from pesq import pesq as _pesq

    if fs != 16000:
        raise ValueError(f"宽带 PESQ 要求 fs=16000，收到 {fs}")
    return float(_pesq(fs, np.asarray(reference, dtype=np.float64),
                       np.asarray(estimate, dtype=np.float64), "wb"))


def stoi(reference: np.ndarray, estimate: np.ndarray, fs: int = 16000) -> float:
    """短时客观可懂度，取值 [0, 1]。"""
    from pystoi import stoi as _stoi

    return float(_stoi(np.asarray(reference, dtype=np.float64),
                       np.asarray(estimate, dtype=np.float64), fs,
                       extended=False))


def evaluate_all(reference: np.ndarray, estimate: np.ndarray,
                 fs: int = 16000) -> dict:
    """一次性计算全部指标。

    需要 fs=16000——PESQ 宽带模式与我们的仿真采样率一致。
    """
    return {
        "pesq_wb": pesq_wb(reference, estimate, fs),
        "stoi": stoi(reference, estimate, fs),
        "si_sdr": si_sdr(reference, estimate),
    }
