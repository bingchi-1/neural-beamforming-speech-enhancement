"""T2.2 MVDR（最小方差无失真响应）波束形成。

设计文档 §5.2、实施计划 T2.2。

原理
----
在「目标方向增益为 1」这一**无失真约束**下最小化输出功率：

    w(f) = R_n(f)⁻¹ d(f) / (d(f)^H R_n(f)⁻¹ d(f))

其中 d(f) 是目标方向的导向矢量，R_n(f) 是噪声的空间协方差。
相比延时求和（各通道等权相加），MVDR 会**自适应地**在干扰方向形成零点，
因此能显著压制非目标方向的相干干扰。

两个约定（容易搞混，务必分清）
------------------------------
1. **波束形成向量 w**：输出 Y = w^H X = Σ_m conj(w_m)·X_m，满足 w^H d = 1。
   `beamforming_vector()` 返回的就是它，T2.2 的失真无畸变验收直接查它。
2. **权重矩阵 W**：本模块的 `enhance()` 用 `einsum("fm,mft->ft", W, X)`，
   即 W 直接与 X 相乘（不带共轭），故 W = conj(w)。与 `DelayAndSum.weights()`
   的约定保持一致，两者才能互换比较。

R_n 从哪来
----------
MVDR 的成败几乎全在 R_n 的估计上。本模块接受**显式的噪声观测**
（`noise_multi`，仿真管道提供）或**现成的协方差**（`noise_cov`）。
用真值噪声场估协方差属于**上界设定**，实际部署需靠 VAD 或神经网络掩蔽来估——
那正是 T4.1（mask-MVDR）要替换的一项，届时只需换掉传进来的协方差。
"""

from __future__ import annotations

import numpy as np

from ..dsp import HOP, N_FFT, FS_DEFAULT, istft, rfft_freqs, stft_multi
from .geometry import steering_vector

EPS = 1e-12


def noise_covariance(X_noise: np.ndarray, loading: float = 0.0,
                     eps: float = EPS) -> np.ndarray:
    """由噪声的多通道 STFT 估计各频点的空间协方差，形状 [n_freqs, M, M]。

        R(f) = (1/T) Σ_t X(f,t)·X(f,t)^H

    Parameters
    ----------
    X_noise : [M, F, T] 多通道复谱（`dsp.stft_multi` 的输出）
    loading : 对角加载强度，相对该频点平均功率。麦克风间距远小于波长时
              各通道高度相关、R 接近奇异，必须加载才能稳定求逆。
    eps     : 绝对下限，防止噪声恒为零时（R = 0）出现全零矩阵。
    """
    X = np.asarray(X_noise)
    if X.ndim != 3:
        raise ValueError(f"期望 [M, F, T] 的多通道复谱，收到形状 {X.shape}")

    n_frames = X.shape[2]
    if n_frames < 1:
        raise ValueError("噪声观测没有帧")

    # R[f, m, n] = mean_t X[m,f,t] · conj(X[n,f,t])
    R = np.einsum("mft,nft->fmn", X, np.conj(X)) / n_frames
    return regularize_covariance(R, loading, eps)


def regularize_covariance(R: np.ndarray, loading: float = 1e-3,
                          eps: float = EPS) -> np.ndarray:
    """对角加载：R ← R + (loading·mean_power + eps)·I。

    加载量取该频点平均功率的一定比例，因此对不同频点、不同信号电平
    都自动适配；`eps` 兜住「噪声恒为零」这一退化情形。
    """
    R = np.asarray(R)
    M = R.shape[-1]
    mean_power = np.trace(R, axis1=-2, axis2=-1).real / M     # [F]
    scale = loading * mean_power + eps
    diag = np.eye(M)[None, :, :]
    return R + scale[..., None, None] * diag


def mvdr_vector(R_n: np.ndarray, d: np.ndarray,
                loading: float = 1e-3) -> np.ndarray:
    """解 MVDR 波束形成向量 w，形状 [n_freqs, M]，满足 w^H d = 1。

    解线性方程组 R w = d 再归一化，而不是显式求逆 R⁻¹——后者在 R 条件数
    很差时数值上明显更糟，而 R_n 恰恰经常是病态的。

    R_n 传 None 或单位阵等价于「各通道噪声不相关且等功率」，
    此时 w = d / M，即退化为延时求和。
    """
    d = np.asarray(d, dtype=np.complex128)
    if d.ndim != 2:
        raise ValueError(f"期望 [F, M] 的导向矢量，收到形状 {d.shape}")

    if R_n is None:
        return d / d.shape[1]

    R = regularize_covariance(R_n, loading)

    # R[f] w[f] = d[f] —— numpy 把 [F, M, M] @ [F, M, 1] 当作批量线性求解
    w = np.linalg.solve(R, d[..., None])[..., 0]

    denom = np.einsum("fm,fm->f", np.conj(d), w)              # d^H w
    # 理论上 denom 必为实数正数；取实部以消除浮点误差带来的微小虚部
    return w / denom.real[:, None]


class MVDR:
    """统一接口的 MVDR 波束形成器。

    用法：
        bf = MVDR(mic_pos, center_xy=(2.0, 2.0), fs=16000)
        y = bf.enhance(multi_ch, doa_deg=37.5, noise_multi=noise_multi)

    Parameters
    ----------
    mic_pos   : [3, n_mics] 阵元坐标（米）
    center_xy : 阵列中心，导向矢量以此为参考点
    loading   : 对角加载强度（见 `regularize_covariance`）
    """

    name = "MVDR"

    def __init__(self, mic_pos: np.ndarray, center_xy: tuple = (0.0, 0.0),
                 fs: int = FS_DEFAULT, n_fft: int = N_FFT, hop: int = HOP,
                 loading: float = 1e-3):
        self.mic_pos = np.asarray(mic_pos, dtype=np.float64)
        self.center_xy = tuple(center_xy)
        self.fs = fs
        self.n_fft = n_fft
        self.hop = hop
        self.loading = float(loading)
        self.freqs = rfft_freqs(n_fft, fs)

    @property
    def n_mics(self) -> int:
        return self.mic_pos.shape[1]

    def steering(self, doa_deg: float) -> np.ndarray:
        """目标方向的导向矢量 d(f)，[n_freqs, M]。与延时求和共用同一套几何。"""
        return steering_vector(self.mic_pos, doa_deg, self.freqs, self.center_xy)

    def beamforming_vector(self, doa_deg: float,
                           R_n: np.ndarray | None = None) -> np.ndarray:
        """MVDR 波束形成向量 w，[n_freqs, M]，满足 w^H d(θ) = 1（T2.2 验收）。

        R_n 为 None 时退化为延时求和（各通道等权）。
        """
        return mvdr_vector(R_n, self.steering(doa_deg), self.loading)

    def weights(self, doa_deg: float,
                R_n: np.ndarray | None = None) -> np.ndarray:
        """供 `einsum("fm,mft->ft", W, X)` 使用的权重矩阵，即 conj(w)。

        与 `DelayAndSum.weights()` 的约定一致——两者都是「直接与 X 相乘」的
        形式，这样两个波束形成器的 enhance 实现可以完全对称。
        """
        return np.conj(self.beamforming_vector(doa_deg, R_n))

    def estimate_noise_covariance(self, noise_multi: np.ndarray) -> np.ndarray:
        """由纯噪声多通道观测估计 R_n，[n_freqs, M, M]。

        返回**未加载**的原始估计；对角加载在 `mvdr_vector` 里统一施加一次，
        避免加载两次（那会让实际加载量随调用路径而变）。
        """
        noise_multi = self._check_channels(noise_multi, "噪声")
        return noise_covariance(stft_multi(noise_multi, self.n_fft, self.hop,
                                           self.fs))

    def enhance(self, multi_ch: np.ndarray, doa_deg: float,
                noise_multi: np.ndarray | None = None,
                noise_cov: np.ndarray | None = None) -> np.ndarray:
        """多通道 [M, T] → 单通道增强信号 [T]。

        Parameters
        ----------
        noise_multi : 纯噪声多通道观测 [M, T]，用于估计噪声协方差
        noise_cov   : 直接给定的 [n_freqs, M, M] 协方差。给了它就用它，
                      忽略 noise_multi（mask-MVDR 走这条路）
        """
        multi_ch = self._check_channels(multi_ch, "观测")

        if noise_cov is None:
            if noise_multi is None:
                raise ValueError(
                    "MVDR 必须知道噪声协方差：请提供 noise_multi（纯噪声多通道观测）"
                    "或 noise_cov（现成的 [n_freqs, M, M]）。\n"
                    "仿真管道 simulate_sample() 的输出里就有 noise_multi 字段。\n"
                    "若想用观测协方差代替，应在调用方显式传入——那不是 MVDR 的标准用法。"
                )
            noise_cov = self.estimate_noise_covariance(noise_multi)

        W = self.weights(doa_deg, noise_cov)
        X = stft_multi(multi_ch, self.n_fft, self.hop, self.fs)     # [M, F, T]
        Y = np.einsum("fm,mft->ft", W, X)

        return istft(Y, self.n_fft, self.hop, self.fs,
                     length=multi_ch.shape[1])

    def _check_channels(self, x: np.ndarray, what: str) -> np.ndarray:
        x = np.atleast_2d(x)
        if x.shape[0] != self.n_mics:
            raise ValueError(
                f"通道数不符：{what}期望 {self.n_mics}，收到 {x.shape[0]}")
        return x
