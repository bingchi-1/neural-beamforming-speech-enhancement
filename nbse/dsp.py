"""时频变换工具。

波束形成在频域进行（窄带近似），因此 STFT/ISTFT 是共用基础设施。
基于 scipy.signal 实现——手写 overlap-add 容易在窗函数归一化上出错。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import istft as _istft, stft as _stft

N_FFT = 512
HOP = 128          # 75% overlap
FS_DEFAULT = 16000


def stft(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP,
         fs: int = FS_DEFAULT) -> np.ndarray:
    """单通道 STFT。

    返回 [n_freqs, n_frames] 的复数谱，n_freqs = n_fft // 2 + 1。

    必须使用 scipy 的默认边界处理（boundary='zeros', padded=True）。
    若改用 boundary=None / padded=False，会破坏 NOLA 条件导致
    STFT 不可逆——实测往返误差达 1e-1 量级，且只抛 warning 不报错，
    会静默污染所有依赖时频变换的方法。见 tests/test_dsp.py。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    _, _, Z = _stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    return Z


def stft_multi(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP,
               fs: int = FS_DEFAULT) -> np.ndarray:
    """多通道 STFT。输入 [n_mics, T] → 输出 [n_mics, n_freqs, n_frames]。"""
    x = np.atleast_2d(x)
    return np.stack([stft(ch, n_fft, hop, fs) for ch in x])


def istft(Z: np.ndarray, n_fft: int = N_FFT, hop: int = HOP,
          fs: int = FS_DEFAULT, length: int | None = None) -> np.ndarray:
    """单通道 ISTFT。length 给定时裁剪或补齐到该长度。

    与 stft 配套使用 scipy 默认参数，保证严格可逆。
    """
    _, x = _istft(Z, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
    if length is not None:
        x = _fit(x, length)
    return x


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - len(x))])


def rfft_freqs(n_fft: int = N_FFT, fs: int = FS_DEFAULT) -> np.ndarray:
    """STFT 各频点对应的频率（Hz），长度 n_fft // 2 + 1。"""
    return np.fft.rfftfreq(n_fft, d=1.0 / fs)
