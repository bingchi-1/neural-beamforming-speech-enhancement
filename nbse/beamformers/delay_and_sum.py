"""T2.1 延时求和（Delay-and-Sum）波束形成。

最经典、最稳健的波束形成器：按各阵元的传播时延补偿相位后直接相加。
它不使用任何统计信息，也不需要训练，因此是项目的**保底基线**——
即使后续所有方法都失败，这一级也一定跑得出结果。

频域实现：
    Y(f) = (1/M) · Σ_m X_m(f) · conj(d_m(f))
其中 d_m(f) = exp(-j·2π·f·τ_m) 为导向矢量（geometry.steering_vector）。
由于 |d_m| = 1，乘上共轭后各通道同相相加，而互不相关的噪声按 1/√M 衰减。
"""

from __future__ import annotations

import numpy as np

from ..dsp import HOP, N_FFT, FS_DEFAULT, istft, rfft_freqs, stft_multi
from .geometry import steering_vector


class DelayAndSum:
    """统一接口的延时求和波束形成器。

    用法：
        bf = DelayAndSum(mic_pos, center_xy=(2.0, 2.0), fs=16000)
        y = bf.enhance(multi_ch, doa_deg=37.5)

    Parameters
    ----------
    mic_pos   : [3, n_mics] 阵元坐标（米）
    center_xy : 阵列中心，导向矢量以此为参考点
    """

    name = "Delay-and-Sum"

    def __init__(self, mic_pos: np.ndarray, center_xy: tuple = (0.0, 0.0),
                 fs: int = FS_DEFAULT, n_fft: int = N_FFT, hop: int = HOP):
        self.mic_pos = np.asarray(mic_pos, dtype=np.float64)
        self.center_xy = tuple(center_xy)
        self.fs = fs
        self.n_fft = n_fft
        self.hop = hop
        self.freqs = rfft_freqs(n_fft, fs)

    @property
    def n_mics(self) -> int:
        return self.mic_pos.shape[1]

    def weights(self, doa_deg: float) -> np.ndarray:
        """各频点的加权系数 [n_freqs, n_mics]（含 1/M 归一化）。

        频率为 0（直流）时导向矢量恒为 1，不存在相位补偿问题。
        """
        d = steering_vector(self.mic_pos, doa_deg, self.freqs, self.center_xy)
        return np.conj(d) / self.n_mics

    def enhance(self, multi_ch: np.ndarray, doa_deg: float) -> np.ndarray:
        """多通道 [n_mics, T] → 单通道增强信号 [T]。"""
        multi_ch = np.atleast_2d(multi_ch)
        if multi_ch.shape[0] != self.n_mics:
            raise ValueError(
                f"通道数不符：期望 {self.n_mics}，收到 {multi_ch.shape[0]}")

        X = stft_multi(multi_ch, self.n_fft, self.hop, self.fs)   # [M, F, T]
        W = self.weights(doa_deg)                                  # [F, M]

        # Y[f, t] = Σ_m W[f, m] · X[m, f, t]
        Y = np.einsum("fm,mft->ft", W, X)

        return istft(Y, self.n_fft, self.hop, self.fs,
                     length=multi_ch.shape[1])
