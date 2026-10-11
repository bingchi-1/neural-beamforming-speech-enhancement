"""T5.2 DCCRN-lite 单通道增强的训练与推理目标。

设计文档 §5、实施计划 T5.2。

为什么单独一个模块而不用 `nbse/models/dataset.py`
------------------------------------------------
`dataset.py` 为**多通道掩蔽估计**服务：输出多通道幅度谱 + 理想比例掩蔽，
供 CRN 使用。DCCRN 的目标完全不同——复数谱到复数谱的重建，
输入只有一个通道。两者的数据契约没有交集，硬塞进同一个 Dataset 会
变成一堆 if 分支。因此这里独立实现，只复用落盘格式与缓存约定。

目标信号的口径（与多通道路线保持可比）
--------------------------------------
输入：`multi_ch[0]`（第一个麦克风的观测）
目标：`clean_ref`（**干声**，已按阵列中心对齐到直达路径时延）

这与评估口径一致：参考信号就是干声，因此模型要同时完成去噪与去混响。
设计文档 §4.3.1 说明了为什么必须用干声参考（混响镜像作参考会导致
排序倒挂）。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..dsp import FS_DEFAULT, HOP, N_FFT, stft

MANIFEST_NAME = "manifest.csv"
EPS = 1e-10


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """裁剪或循环补齐到长度 n（与 dataset.py 同语义）。"""
    if len(x) >= n:
        return x[:n]
    reps = int(math.ceil(n / max(len(x), 1)))
    return np.tile(x, reps)[:n]


def _segment_frames(segment_length: int, hop: int = HOP,
                    n_fft: int = N_FFT) -> int:
    """时域片段长度 → STFT 帧数。公式与 `models.dataset.segment_frames` 一致。"""
    if segment_length < n_fft:
        raise ValueError(
            f"片段长度 {segment_length} 小于 n_fft={n_fft}，"
            f"此时 scipy 会自动降低 nperseg，帧数不再服从该公式。")
    return 1 + -(-int(segment_length) // int(hop))


def build_pair(seg_mix: np.ndarray, seg_ref: np.ndarray,
               n_fft: int = N_FFT, hop: int = HOP,
               fs: int = FS_DEFAULT) -> dict:
    """由**已长度对齐**的片段产出 (混合复谱, 干净复谱)。

    约定：调用方先完成裁剪/补齐，本函数只做 STFT 与 dtype 转换。
    这样「片段长度 → 帧数」的关系只由调用方决定，避免两处各切一次
    导致帧数与 `_segment_frames` 的公式不一致。

    返回
    ----
    mix_r/mix_i     : [F, T] 观测（0 号通道）的实部与虚部
    clean_r/clean_i : [F, T] 干净参考的实部与虚部（网络的回归目标）
    """
    X = stft(np.asarray(seg_mix, dtype=np.float64), n_fft, hop, fs)
    S = stft(np.asarray(seg_ref, dtype=np.float64), n_fft, hop, fs)
    return {
        "mix_r": X.real.astype(np.float32),
        "mix_i": X.imag.astype(np.float32),
        "clean_r": S.real.astype(np.float32),
        "clean_i": S.imag.astype(np.float32),
    }


class DCCRNDataset:
    """单通道复数谱数据集。

    与 `NbseDataset` 平行：同样的 random_crop / epoch / limit 语义，
    但产出复谱对而非 (幅度谱, 掩蔽)。
    """

    def __init__(self, data_dir: str | Path, segment_length: int = FS_DEFAULT,
                 random_crop: bool = True, limit: int | None = None,
                 seed: int = 0, n_fft: int = N_FFT, hop: int = HOP,
                 fs: int = FS_DEFAULT, cache: bool = True):
        import csv

        self.data_dir = Path(data_dir)
        self.segment_length = int(segment_length)
        self.random_crop = bool(random_crop)
        self.n_fft, self.hop, self.fs = int(n_fft), int(hop), int(fs)
        self.cache = bool(cache)

        path = self.data_dir / MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"找不到 {path}。请先生成样本：\n"
                f"  python scripts/generate_dataset.py --split train --n 3000 --jobs 8")
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if limit is not None:
            rows = rows[: int(limit)]
        if not rows:
            raise ValueError(f"{self.data_dir} 的 manifest 为空")
        self.rows = rows

        self._base_seed = int(seed)
        self._epoch = 0
        self._cache: dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def n_frames(self) -> int:
        return _segment_frames(self.segment_length, self.hop, self.n_fft)

    @property
    def n_freqs(self) -> int:
        return self.n_fft // 2 + 1

    def _arrays(self, i: int) -> dict:
        if self.cache and i in self._cache:
            return self._cache[i]
        with np.load(self.data_dir / self.rows[i]["file"]) as z:
            arrs = {"multi_ch": z["multi_ch"].astype(np.float64),
                    "clean_ref": z["clean_ref"].astype(np.float64)}
        if self.cache:
            self._cache[i] = arrs
        return arrs

    def __getitem__(self, i: int) -> dict:
        arrs = self._arrays(i)
        total = arrs["multi_ch"].shape[1]
        if self.segment_length > total:
            raise ValueError(
                f"请求的片段长度 {self.segment_length} 超过样本长度 {total}"
                f"（{self.rows[i]['file']}）")

        if self.random_crop and total > self.segment_length:
            rng = np.random.default_rng([self._base_seed, self._epoch, i])
            start = int(rng.integers(0, total - self.segment_length + 1))
        else:
            start = 0

        # 先切出片段再算 STFT —— 这样帧数与 _segment_frames 严格一致
        seg_mix = arrs["multi_ch"][0, start:start + self.segment_length]
        seg_ref = arrs["clean_ref"][start:start + self.segment_length]

        out = build_pair(seg_mix, seg_ref, n_fft=self.n_fft,
                         hop=self.hop, fs=self.fs)
        row = self.rows[i]
        out["meta"] = {"index": int(row["index"]), "file": row["file"],
                       "start": start}
        return out


def collate_pairs(batch: list[dict]):
    """把若干 (复谱对) 堆成 batch，并附上 meta。"""
    import torch

    def stack(key):
        return torch.from_numpy(np.stack([b[key] for b in batch]))

    return (
        {"mix_r": stack("mix_r"), "mix_i": stack("mix_i"),
         "meta": [b["meta"] for b in batch]},
        {"clean_r": stack("clean_r"), "clean_i": stack("clean_i")},
    )


def reconstruct(spec_r: np.ndarray, spec_i: np.ndarray,
                mask_r: np.ndarray, mask_i: np.ndarray) -> np.ndarray:
    """复数掩蔽 + ISTFT，返回时域波形。

    这是评测路径用的推理工具：给定观测复谱与网络预测的复数掩蔽，
    重建增强后的单通道波形。
    """
    from ..dsp import istft

    enh = (spec_r + 1j * spec_i) * (mask_r + 1j * mask_i)
    return istft(enh, N_FFT, HOP, FS_DEFAULT)


class TorchDCCRN:
    """把 DCCRN-lite 包装成 `enhance(multi_ch) -> waveform`，供评测调用。

    与 `mask_mvdr.TorchMaskPredictor` 平行，但输出直接是波形
    （单通道增强不需要波束形成）。
    """

    def __init__(self, model, device: str = "cuda",
                 deterministic: bool = True):
        self.model = model
        self.device = device
        if deterministic:
            from ..determinism import enable_determinism

            enable_determinism()

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cuda",
                        n_freqs: int | None = None, **kw) -> "TorchDCCRN":
        import torch

        from .dccrn_lite import DCCRNLite

        model = DCCRNLite(**(dict(n_freqs=n_freqs) if n_freqs else {}), **kw)
        ckpt = torch.load(Path(path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        return cls(model, device)

    def enhance(self, multi_ch: np.ndarray) -> np.ndarray:
        """`[M, T]` 或 `[T]` → 增强波形 `[T]`（只用第一个通道）。"""
        import torch

        from ..dsp import istft

        x = np.atleast_2d(multi_ch)[0].astype(np.float64)
        X = stft(x, N_FFT, HOP, FS_DEFAULT)
        with torch.no_grad():
            t_r = torch.from_numpy(X.real.astype(np.float32))[None, None].to(self.device)
            t_i = torch.from_numpy(X.imag.astype(np.float32))[None, None].to(self.device)
            mr, mi = self.model(t_r, t_i)
        mr = mr[0].float().cpu().numpy()
        mi = mi[0].float().cpu().numpy()
        enh = X * (mr + 1j * mi)
        return istft(enh, N_FFT, HOP, FS_DEFAULT, length=len(x))
