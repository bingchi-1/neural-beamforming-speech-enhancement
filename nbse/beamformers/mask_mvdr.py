"""T4.1 mask-MVDR：用神经网络估计的掩蔽驱动 MVDR 波束形成。

设计文档 §5.2、实施计划 T4.1。

与 T2.2 的关系
--------------
`MVDR.enhance()` 已经接受 `noise_cov` 参数，其 docstring 就写明
「mask-MVDR 走这条路」。本模块要做的事只有一件：

    **把「用真值噪声场估出的 R_n」换成「用网络掩蔽加权观测估出的 R_n」**

其余（导向矢量、无失真约束、频域加权重建）完全复用，因此两条路线的
对比是干净的——差异只来自协方差估计方式。

数学口径
--------
设观测多通道复谱 X ∈ C^{M×F×T}，网络给的语音掩蔽 m ∈ [0,1]^{F×T}。

    R_s(f) = Σ_t m(f,t)² · x(f,t) x(f,t)^H  /  Σ_t m(f,t)²
    R_n(f) = Σ_t (1-m(f,t))² · x(f,t) x(f,t)^H  /  Σ_t (1-m(f,t))²

两个约定值得写清：

1. **两式都用同一套归一化（权重平方和）**，因此 R_s 与 R_n 处在同一尺度。
   这一点是必要的：MVDR 权重对 R_n 的**绝对尺度**不变（分子分母同尺度相消），
   但当 R_s 与 R_n 同时进入 `beamforming_weights` 的广义特征值形式时，
   两者的相对尺度就会影响结果。
2. **掩蔽平方而不仅是掩蔽**：这相当于按掩蔽加权的功率平均，是无偏的单快拍
   协方差估计写法。用掩蔽的一次方也可行，但会让「掩蔽接近 0 的帧」对
   R_n 的贡献被其自身权重再次放大，平方形式更稳。

为什么不用 `R_n = R_x - R_s`
--------------------------
那是另一个常见推导，但它要求 R_x、R_s 的估计在同一尺度上且残差非负定。
掩蔽加权下残差不一定半正定，求逆会出现负特征值导致的数值噪声。
直接由 (1-m) 加权估 R_n 没有这个问题。

分辨率局限（务必知晓）
--------------------
从长度 T（本项目 1 s 片段 = 126 帧）估计 M×M（M=4）空间协方差是可行的，
但帧数不算充裕。掩蔽估计一旦在某频点过度自信（m→1 或 m→0），对应协方差
就退化，因此**对角加载是必需的**而非可选——见 `beamforming_weights`。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..dsp import FS_DEFAULT, HOP, N_FFT, istft, rfft_freqs, stft_multi
from .geometry import steering_vector
from .mvdr import mvdr_vector, regularize_covariance

EPS = 1e-12


def mask_weighted_covariance(X: np.ndarray, mask: np.ndarray,
                             eps: float = EPS) -> np.ndarray:
    """掩蔽加权的空间协方差，形状 [n_freqs, M, M]。

        R(f) = Σ_t w(f,t)² · x(f,t) x(f,t)^H / Σ_t w(f,t)²

    权重按平方和归一化，使不同掩蔽下的结果落在同一尺度上。

    Parameters
    ----------
    X    : [M, F, T] 多通道复谱（`dsp.stft_multi` 的输出）
    mask : [F, T] 取值 [0,1] 的实掩蔽（语音掩蔽传 m，噪声掩蔽传 1-m）

    退化情形：某频点掩蔽全为 0（该频点没有语音/没有噪声）时，分母为 0。
    此时返回**单位阵乘以极小量**而不是零矩阵——零矩阵求逆会直接炸，
    而单位阵在对角加载后等价于「该频点不做自适应」，是安全的退让。
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"期望 [M, F, T] 的多通道复谱，收到形状 {X.shape}")

    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim != 2:
        raise ValueError(f"期望 [F, T] 的掩蔽，收到形状 {mask.shape}")

    M, F, T = X.shape
    if mask.shape != (F, T):
        raise ValueError(
            f"掩蔽形状 {mask.shape} 与复谱的 (F, T)=({F}, {T}) 不符。"
            f"掩蔽由固定长度的片段估计，若观测更长需先分段或按片段对齐。"
        )

    w2 = mask ** 2                                    # [F, T]
    denom = w2.sum(axis=1)                            # [F]
    # R[f] = Σ_t w2[f,t] · x[:,f,t] x[:,f,t]^H
    R = np.einsum("ft,mft,nft->fmn", w2, X, np.conj(X))

    ok = denom > eps
    R[ok] /= denom[ok, None, None]
    if not np.all(ok):
        # 退化频点：单位阵（对角加载后会成为「不自适应」的安全解）
        R[~ok] = np.eye(M)[None, :, :]
    return R


def beamforming_weights(R_s: np.ndarray, R_n: np.ndarray, d: np.ndarray,
                        loading: float = 1e-2,
                        mode: str = "mvdr") -> np.ndarray:
    """由估计的语音/噪声协方差解波束形成向量 w，[F, M]，满足 w^H d = 1。

    复用了 `mvdr.mvdr_vector`：MVDR 解只依赖 R_n 与 d，R_s 在纯 MVDR 中并不
    出现。之所以仍要收 R_s，是因为它提供了 `mode="gevd"` 这一更稳的解——
    对 (R_s, R_n) 做广义特征分解，取主特征向量：

        R_s w = λ R_n w   →   w ∝ 主广义特征向量

    这是掩蔽驱动波束形成的另一常见形式（尤其 R_n 病态时更稳），且它与
    MVDR 解在理论上同向。默认仍用 `mvdr`，`gevd` 留给消融。

    Parameters
    ----------
    loading : 对角加载强度。**这里默认 1e-2，比 T2.2 的 1e-3 大一个量级**：
              真值噪声协方差由整段纯噪声估计（帧数充裕、条件数好），而掩蔽加权
              的估计在掩蔽接近二值时有效帧数很少，矩阵更容易病态。
    """
    d = np.asarray(d, dtype=np.complex128)
    if d.ndim != 2:
        raise ValueError(f"期望 [F, M] 的导向矢量，收到形状 {d.shape}")

    if mode == "mvdr":
        return mvdr_vector(R_n, d, loading)

    if mode == "gevd":
        R_s = np.asarray(R_s)
        if R_s.shape != R_n.shape:
            raise ValueError(f"R_s {R_s.shape} 与 R_n {R_n.shape} 形状不符")
        Rn = regularize_covariance(R_n, loading)
        # 用白化 + 普通特征分解代替直接求广义特征值，数值上更稳：
        #   R_n = L L^H  →  解 (L⁻¹ R_s L⁻ᴴ) v = λ v，再回代 w = L⁻ᴴ v
        M = Rn.shape[-1]
        # Cholesky 需要正定；对角加载后应当正定，若失败则再加大加载
        try:
            L = np.linalg.cholesky(Rn)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(regularize_covariance(Rn, 1e-1))

        Linv = np.linalg.inv(L)
        A = Linv @ R_s @ np.conj(Linv).transpose(0, 2, 1)      # [F, M, M]
        # 取最大特征值对应的特征向量
        evals, evecs = np.linalg.eigh(A)
        v = evecs[:, :, -1]                                     # [F, M]
        w = np.conj(Linv).transpose(0, 2, 1) @ v[..., None]     # L⁻ᴴ v
        w = w[..., 0]

        denom = np.einsum("fm,fm->f", np.conj(d), w)            # d^H w
        # 必须用**完整复数** denom 归一化，这样 d^H w 恒等于 1（实部 1、虚部 0）。
        # 只除以 denom.real 会留下非零虚部——实测最大可达 88，即 w^H d ≠ 1，
        # 无失真约束被破坏（表现为输出幅度与相位随频点漂移）。
        bad = np.abs(denom) < EPS
        if np.any(bad):
            fallback = mvdr_vector(R_n, d, loading)
            w[bad] = fallback[bad]
            denom[bad] = np.einsum("fm,fm->f", np.conj(d[bad]), fallback[bad])
        return w / denom[:, None]

    raise ValueError(f"未知模式：{mode}（可选 mvdr | gevd）")


def _fit_frames(mask: np.ndarray, n_frames: int) -> np.ndarray:
    """把掩蔽的时帧维裁剪/补齐到 n_frames。

    正常路径下二者相等（掩蔽就是从同一段观测估出来的）。需要补齐的情形是
    「用 1 s 片段训练的模型去推理更长的观测」——那时调用方应先分段，
    本函数只作为兜底，避免形状不符直接抛错。
    """
    F, T = mask.shape
    if T == n_frames:
        return mask
    if T > n_frames:
        return mask[:, :n_frames]
    pad = np.ones((F, n_frames - T), dtype=mask.dtype)   # 缺失部分按「全保留」处理
    return np.concatenate([mask, pad], axis=1)


class TorchMaskPredictor:
    """把 PyTorch CRN 包装成 `mask_predictor(multi_ch) -> mask`。

    `MaskMVDR` 本身不依赖 torch（便于用解析掩蔽做单元测试），
    只有这个包装类引入 torch，并延迟到调用时才 import。
    """

    def __init__(self, model, device: str = "cuda", n_fft: int = N_FFT,
                 hop: int = HOP, fs: int = FS_DEFAULT,
                 deterministic: bool = True):
        self.model = model
        self.device = device
        self.n_fft = n_fft
        self.hop = hop
        self.fs = fs
        # 推理必须确定性：默认设置下同一输入连续 8 次前向会得到 8 个不同结果
        # （cuDNN 卷积算法选择不确定，最大差 ~1.6e-05）。这个扰动经协方差求逆
        # 放大后能让波束形成权重相差 0.78，导致评测结果不可复现。
        # 详见 nbse/determinism.py。
        if deterministic:
            from ..determinism import enable_determinism

            enable_determinism()

    @classmethod
    def from_checkpoint(cls, path: str | Path, preset: str = "default",
                        device: str = "cuda", n_freqs: int | None = None,
                        deterministic: bool = True, **kw) -> "TorchMaskPredictor":
        """从 train.py 产出的 checkpoint 构建。

        只读 `model` 权重；优化器/调度器状态与本推理路径无关。

        `n_freqs` 应从数据/FFT 推导后显式传入（`N_FFT // 2 + 1`）。省略时
        退回 CRN 的默认值 257，那**只对 n_fft=512 正确**；换 FFT 长度会
        加载出形状不符的权重并报错，或更糟——若恰好形状相同则静默用错。
        """
        import torch

        from ..models.crn import CRN, CRN_SMALL

        presets = {"default": CRN, "small": CRN_SMALL}
        if preset not in presets:
            raise ValueError(f"未知模型预设：{preset}（可选 {sorted(presets)}）")

        model = presets[preset](**({"n_freqs": n_freqs} if n_freqs else {}), **kw)
        ckpt = torch.load(Path(path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        return cls(model, device, deterministic=deterministic)

    def __call__(self, multi_ch: np.ndarray) -> np.ndarray:
        import torch

        from ..dsp import stft_multi

        X = stft_multi(np.atleast_2d(multi_ch), self.n_fft, self.hop, self.fs)
        mag = np.abs(X).astype(np.float32)[None]              # [1, M, F, T]
        with torch.no_grad():
            t = torch.from_numpy(mag).to(self.device)
            mask = self.model(t)[0].float().cpu().numpy()     # [F, T]
        return np.clip(mask, 0.0, 1.0)


class MaskMVDR:
    """掩蔽驱动的 MVDR 波束形成器。

    用法：
        pred = TorchMaskPredictor.from_checkpoint("runs/w4/best.pt")
        bf = MaskMVDR(mic_pos, center_xy=(2.0, 2.0), mask_predictor=pred)
        y = bf.enhance(multi_ch, doa_deg=37.5)

    也支持直接给掩蔽（测试与 Oracle 上界用）：
        bf = MaskMVDR(mic_pos, center_xy, mask_predictor=None)
        y = bf.enhance(multi_ch, doa_deg, mask=oracle_mask)

    Parameters
    ----------
    mask_predictor : 可调用对象 `(multi_ch) -> mask[F, T]`，或 None
    loading        : 对角加载强度（见 `beamforming_weights`）
    mode           : "mvdr"（默认）| "gevd"（后者只在 weighting="both" 下有意义）
    weighting      : **掩蔽怎么用**，这是本模块最关键的一个开关：
        "noise"（默认）—— 只用掩蔽估噪声协方差 R_n，MVDR 解只用 R_n。
                          即「掩蔽作为噪声估计器」。
        "both"         —— 同时由掩蔽估 R_s 与 R_n，用广义特征值解。
                          这是设计文档 §5.2 字面描述的做法，但**实测更差**
                          （见类 docstring 末尾的实测对比）。
        "speech"       —— 只用 R_s（配合观测协方差当 R_n），留作消融。
    use_noise_mask : 兼容旧参数名；等价于 weighting="both"/"noise"。
                     建议直接用 `weighting`。

    为什么默认是 "noise"
    -------------------
    实测（120 条测试样本，SI-SDR 均值，全部用同一个 CRN checkpoint）：
        mask-MVDR @ 常量掩蔽 0.5          -5.693
        mask-MVDR @ 预测掩蔽（weighting=both） -5.049
        mask-MVDR @ 真值 IRM（上界掩蔽）      -4.886
        mask-MVDR @ 预测掩蔽（weighting=noise）-3.645   ← 默认
        MVDR @ 真值噪声（上界设定）            -3.238

    两点判读：
    1. **掩蔽质量不是瓶颈**——换成真值 IRM 只再提升 0.163 dB，说明 CRN 学得很好。
       因此把力气花在改训练目标上收益有限。
    2. **让 R_s 参与反而有害**——同一掩蔽下换成 `weighting="noise"` 好了 1.404 dB。
       原因是掩蔽加权后 R_s 的有效帧数很少、且与 R_n 的尺度/秩不匹配，
       广义特征值解会被少数高掩蔽帧主导，产生偏斜的空间响应；
       而 R_n 的估计对掩蔽误差宽容得多（噪声掩蔽在语音间隙处更可靠）。
    """

    name = "mask-MVDR"

    def __init__(self, mic_pos: np.ndarray, center_xy: tuple = (0.0, 0.0),
                 mask_predictor=None, fs: int = FS_DEFAULT,
                 n_fft: int = N_FFT, hop: int = HOP,
                 loading: float = 1e-2, mode: str = "mvdr",
                 weighting: str = "noise",
                 use_noise_mask: bool | None = None):
        self.mic_pos = np.asarray(mic_pos, dtype=np.float64)
        self.center_xy = tuple(center_xy)
        self.mask_predictor = mask_predictor
        self.fs = fs
        self.n_fft = n_fft
        self.hop = hop
        self.loading = float(loading)
        self.mode = mode
        # 兼容旧的布尔参数：use_noise_mask=True 曾表示「用 1-m 估 R_n」，
        # 但那套实现里 R_s 始终参与。这里映射到新语义，避免旧调用静默改变行为。
        if use_noise_mask is not None:
            weighting = "both" if use_noise_mask else "speech"
        if weighting not in ("noise", "both", "speech"):
            raise ValueError(
                f"未知 weighting：{weighting}（可选 noise | both | speech）")
        self.weighting = weighting
        self.freqs = rfft_freqs(n_fft, fs)

    @property
    def n_mics(self) -> int:
        return self.mic_pos.shape[1]

    def steering(self, doa_deg: float) -> np.ndarray:
        """导向矢量 d(f)，[F, M]。与 DS / MVDR 共用同一套几何。"""
        return steering_vector(self.mic_pos, doa_deg, self.freqs, self.center_xy)

    def estimate_covariances(self, X: np.ndarray,
                             mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """由掩蔽估计 (R_s, R_n)，均为 [F, M, M]。

        按 `weighting` 决定两者的用途：

        * `"noise"`（默认）：R_n 由 1-m 加权估计；**R_s 返回观测协方差**
          （即不做语音掩蔽加权）。返回观测协方差而不是 None，是为了让调用方
          拿到的始终是一对同尺度、同形状的矩阵；而 `beamforming_weights`
          在 mvdr 模式下只用 R_n，故 R_s 具体是什么不影响结果。
        * `"both"`：R_s 与 R_n 都由掩蔽加权（设计文档 §5.2 的字面做法）。
        * `"speech"`：R_s 由掩蔽加权，R_n 取观测协方差（仅消融用）。
        """
        if self.weighting in ("noise", "both"):
            R_n = mask_weighted_covariance(X, 1.0 - mask)
        else:
            R_n = mask_weighted_covariance(X, np.ones_like(mask))

        if self.weighting in ("speech", "both"):
            R_s = mask_weighted_covariance(X, mask)
        else:
            R_s = mask_weighted_covariance(X, np.ones_like(mask))

        return R_s, R_n

    def enhance(self, multi_ch: np.ndarray, doa_deg: float,
                mask: np.ndarray | None = None) -> np.ndarray:
        """多通道 [M, T] → 单通道增强信号 [T]。

        Parameters
        ----------
        mask : 直接给定掩蔽 [F, T]；None 则调用 `mask_predictor`。
               两者都为 None 时抛错，而不是静默退化成 DS——静默退化会让
               「神经方法优于基线」这个结论变成假的。
        """
        multi_ch = np.atleast_2d(multi_ch)
        if multi_ch.shape[0] != self.n_mics:
            raise ValueError(
                f"通道数不符：期望 {self.n_mics}，收到 {multi_ch.shape[0]}")

        X = stft_multi(multi_ch, self.n_fft, self.hop, self.fs)    # [M, F, T]

        if mask is None:
            if self.mask_predictor is None:
                raise ValueError(
                    "没有掩蔽可用：请提供 mask，或在构造时给 mask_predictor。")
            mask = self.mask_predictor(multi_ch)
        mask = _fit_frames(np.asarray(mask, dtype=np.float64), X.shape[2])

        R_s, R_n = self.estimate_covariances(X, mask)
        w = beamforming_weights(R_s, R_n, self.steering(doa_deg),
                                self.loading, self.mode)

        # 与 DS / MVDR 相同的约定：einsum("fm,mft->ft", W, X)，W = conj(w)
        Y = np.einsum("fm,mft->ft", np.conj(w), X)
        return istft(Y, self.n_fft, self.hop, self.fs,
                     length=multi_ch.shape[1])

    def weights(self, multi_ch: np.ndarray, doa_deg: float,
                mask: np.ndarray | None = None) -> np.ndarray:
        """返回本样本实际使用的权重矩阵 conj(w)，[F, M]。

        单独暴露出来是为了可检验：波束图（W6 的 beampattern）与
        「权重是否真的随掩蔽变化」都靠它。
        """
        multi_ch = np.atleast_2d(multi_ch)
        X = stft_multi(multi_ch, self.n_fft, self.hop, self.fs)
        if mask is None:
            if self.mask_predictor is None:
                raise ValueError("没有掩蔽可用：请提供 mask 或 mask_predictor")
            mask = self.mask_predictor(multi_ch)
        mask = _fit_frames(np.asarray(mask, dtype=np.float64), X.shape[2])
        R_s, R_n = self.estimate_covariances(X, mask)
        w = beamforming_weights(R_s, R_n, self.steering(doa_deg),
                                self.loading, self.mode)
        return np.conj(w)
