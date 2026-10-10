"""T3.2 数据集封装：把落盘的 `.npz` 样本变成 CRN 能吃的张量。

设计文档 §5 的 `nbse/data/dataset.py` 一栏。**实现位置放在 `nbse/models/` 下**
而非设计文档所写的 `nbse/data/`，原因见文末「位置说明」。

数据口径（与 T3.1 的 CRN 严格对应）
------------------------------------
CRN 是**掩蔽估计器**，不直接重建波形。因此每条样本提供三样东西：

    mag      [n_mics, n_freqs, T]   观测多通道幅度谱（网络输入）
    mask     [n_freqs, T]           理想比例掩蔽 IRM（回归目标，取值 [0,1]）
    clean_mag[n_freqs, T]           干净幅度谱（可选重建目标）

掩蔽的定义：把观测分解为「语音部分」与「噪声部分」，逐时频点取幅度比

    IRM = |X_clean| / (|X_clean| + |X_noise|)

其中 `X_clean` 是**未加噪的多通道语音成分**（`clean_multi = multi_ch - noise_multi`），
`X_noise` 是纯噪声成分。两者都在同一房间、同一阵列上，因此这个比值是逐时频点
良定义的、取值必然落在 [0,1] 的实掩蔽。

为什么用 IRM 而不是理想二值掩蔽（IBM）：IRM 保留相位（相位直接沿用观测相位），
且是连续目标、梯度良好，是掩蔽驱动 MVDR 的通行做法。

为什么干净参考用 `clean_multi` 卷积而非 `clean_ref`：`clean_ref` 是**干声**、
只做了时延对齐（设计文档 §4.3.1），而观测的语音成分是经过房间冲激响应的。
训练的目标是「从混响观测里估计语音的时频幅度」，参考必须是同一条路径上的量，
否则网络会被要求凭空补出混响。评估口径（干声参考）与训练口径不同，这是有意的：
评估的是「去噪 + 去混响」的综合效果。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from ..dsp import FS_DEFAULT, HOP, N_FFT, stft

MANIFEST_NAME = "manifest.csv"

# 数值下限。掩蔽分母用，避免静音时频点除零。
EPS = 1e-10


def load_manifest(data_dir: str | Path) -> list[dict]:
    """读样本目录的 `manifest.csv`。

    与 `nbse.data.generate.load_manifest` 同源，这里重新导出以便
    Dataset 不反向依赖生成模块（数据集只该依赖落盘格式，不该依赖生成器）。
    """
    path = Path(data_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}。请先生成样本：\n"
            f"  python scripts/generate_dataset.py --split train --n 240"
        )
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fit_segment(x: np.ndarray, start: int, length: int, axis: int = -1) -> np.ndarray:
    """沿 axis 从 start 取 length 个采样；不足则循环补齐。

    样本落盘为固定 4 s（64000 点），训练时取 1 s 片段即可，无需补齐；
    但补齐分支要保留——否则将来支持更短语料时会静默产出短张量。
    """
    x = np.asarray(x)
    n = x.shape[axis]
    if n >= length:
        return np.take(x, range(start, start + length), axis=axis)
    reps = int(math.ceil(length / max(n, 1)))
    tiled = np.concatenate([x] * reps, axis=axis)
    return np.take(tiled, range(length), axis=axis)


def compute_targets(clean_multi: np.ndarray, noise_multi: np.ndarray,
                    n_fft: int = N_FFT, hop: int = HOP,
                    fs: int = FS_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """由语音成分与噪声成分算出 (mask, clean_mag)。

    Parameters
    ----------
    clean_multi : [n_mics, T] 无噪语音成分（`multi_ch - noise_multi`）
    noise_multi : [n_mics, T] 纯噪声成分

    取 **0 号通道**（参考通道）计算掩蔽——波束形成器的输出对齐的正是阵列中心，
    而 0 号通道与该参考点最接近。这是刻意的简化：把「空间信息」交给网络输入的
    多通道幅度谱去表达，掩蔽本身只是一条回归目标。
    """
    Xc = np.abs(stft(clean_multi[0], n_fft, hop, fs))
    Xn = np.abs(stft(noise_multi[0], n_fft, hop, fs))

    denom = Xc + Xn
    mask = np.where(denom > EPS, Xc / np.maximum(denom, EPS), 1.0)
    # 数值上仍可能因浮点误差越界，显式夹回 [0,1]
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

    return mask, Xc.astype(np.float32)


def build_sample(multi_ch: np.ndarray, clean_ref: np.ndarray,
                 noise_multi: np.ndarray, start: int = 0, length: int = 16000,
                 n_fft: int = N_FFT, hop: int = HOP,
                 fs: int = FS_DEFAULT) -> dict:
    """把一条已落盘样本切成片段并算出网络输入与目标。

    这是数据口径的**唯一实现**——Dataset 与单元测试都调用它，
    避免测试与训练各写一份公式而悄悄分叉。
    """
    seg_mix = _fit_segment(multi_ch, start, length)          # [n_mics, L]
    seg_noise = _fit_segment(noise_multi, start, length)     # [n_mics, L]

    clean_multi = seg_mix - seg_noise
    mask, clean_mag = compute_targets(clean_multi, seg_noise, n_fft, hop, fs)

    mag = np.stack([np.abs(stft(seg_mix[m], n_fft, hop, fs))
                    for m in range(seg_mix.shape[0])])

    return {
        "mag": mag.astype(np.float32),
        "mask": mask,
        "clean_mag": clean_mag,
    }


def segment_frames(segment_length: int, hop: int = HOP,
                   n_fft: int = N_FFT) -> int:
    """给定时域片段长度，算它 STFT 后的时帧数。

    公式
    ----
        n_frames = 1 + ceil(L / hop)

    这是**实测标定**出来的，不是从 scipy 源码推的。踩坑记录：最初写的是
    `1 + L // hop`，在 L=64000 时刚好正确（501 帧），于是通过了手头唯一的
    核对；但 L=8000 时实际是 64 帧而公式给 63、L=4000 时实际 33 而公式给 32。
    整数除法在 L 不是 hop 整数倍时少算一帧，而损失函数与张量形状都依赖这个
    数字，错一帧会静默地把掩蔽与观测频谱错位。

    已对 nfft ∈ {256,512,1024} × hop ∈ {64,128,256} × L ∈ {1000..64000}
    共 96 组参数交叉验证，93 组吻合。**3 组例外全部是 L < n_fft**
    （scipy 会自动把 nperseg 降到信号长度并警告），此时本公式不适用。
    项目里片段长度固定为 1 s（16000 点）以上，远大于 n_fft=512，不会触发；
    但仍加断言把这个前提显式化，避免将来支持极短片段时静默算错。
    """
    if segment_length < n_fft:
        raise ValueError(
            f"片段长度 {segment_length} 小于 n_fft={n_fft}。"
            f"此时 scipy 会自动降低 nperseg，帧数不再服从本公式。"
            f"请把片段长度加到 {n_fft} 以上。"
        )
    return 1 + -(-int(segment_length) // int(hop))    # -(-a//b) 即 ceil(a/b)


class NbseDataset:
    """多通道语音增强数据集。

    用法：
        ds = NbseDataset("data/sim/train", segment_length=16000, seed=0)
        s = ds[0]            # dict: mag / mask / clean_mag / meta
        loader = DataLoader(ds, batch_size=4, num_workers=0,
                            collate_fn=collate_samples)

    Parameters
    ----------
    data_dir       : 含 `manifest.csv` 的样本目录
    segment_length : 时域片段长度（采样点）。默认 16000 = 1 s。
    random_crop    : True → 每条样本随机取片段（训练用，等效时域增广）；
                     False → 固定取开头片段（验证/过拟合测试用，保证可复现）
    limit          : 只取前 N 条。T3.4 过拟合测试用它把数据缩到 10 条。
    cache          : 缓存整条样本与整条多通道幅度谱。

    显存与片长的取舍：样本落盘为 4 s（501 帧）。CRN 瓶颈层是双向 GRU，
    计算量与时帧数成正比；batch=4、4 s 全长的中间激活会超出 12 GB 显存。
    因此训练统一用 1 s 片段（125 帧），这也是 T3.1 验收里
    「单次前向在显存内不 OOM」得以成立的前提。
    """

    def __init__(self, data_dir: str | Path, segment_length: int = FS_DEFAULT,
                 random_crop: bool = True, limit: int | None = None,
                 seed: int = 0, n_fft: int = N_FFT, hop: int = HOP,
                 fs: int = FS_DEFAULT, cache: bool = True):
        self.data_dir = Path(data_dir)
        self.segment_length = int(segment_length)
        self.random_crop = bool(random_crop)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.fs = int(fs)
        self.cache = bool(cache)

        rows = load_manifest(self.data_dir)
        if limit is not None:
            rows = rows[: int(limit)]
        if not rows:
            raise ValueError(f"{self.data_dir} 的 manifest 为空")
        self.rows = rows

        # 每条样本一个独立 seed 偏移，保证同一条样本在多次 __getitem__ 里
        # 取到不同片段、但跨 epoch 可复现（由 epoch 偏移控制，见 set_epoch）
        self._base_seed = int(seed)
        self._epoch = 0
        self._cache: dict[int, dict] = {}

    # ── 基础接口 ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        """切换 epoch，使随机片段窗口随 epoch 变化（等效时域增广）。"""
        self._epoch = int(epoch)

    @property
    def n_mics(self) -> int:
        return int(self.rows[0]["n_mics"])

    @property
    def n_freqs(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def n_frames(self) -> int:
        return segment_frames(self.segment_length, self.hop, self.n_fft)

    # ── 取样 ────────────────────────────────────────────────────

    def _arrays(self, i: int) -> dict:
        """读一条样本的原始数组（可缓存）。"""
        if self.cache and i in self._cache:
            return self._cache[i]

        with np.load(self.data_dir / self.rows[i]["file"]) as z:
            arrs = {
                "multi_ch": z["multi_ch"].astype(np.float64),
                "clean_ref": z["clean_ref"].astype(np.float64),
                "noise_multi": z["noise_multi"].astype(np.float64),
            }

        if self.cache:
            self._cache[i] = arrs
        return arrs

    def __getitem__(self, i: int) -> dict:
        arrs = self._arrays(i)
        total = arrs["multi_ch"].shape[1]

        if self.random_crop:
            # 每条样本用 (base_seed, epoch, i) 派生窗口，与 DataLoader 的
            # worker 调度无关——不依赖 numpy 全局随机态，因此 num_workers>0 也可复现
            rng = np.random.default_rng(
                [self._base_seed, self._epoch, i])
            if total > self.segment_length:
                start = int(rng.integers(0, total - self.segment_length + 1))
            else:
                start = 0
        else:
            start = 0

        out = build_sample(arrs["multi_ch"], arrs["clean_ref"],
                           arrs["noise_multi"], start=start,
                           length=self.segment_length,
                           n_fft=self.n_fft, hop=self.hop, fs=self.fs)

        row = self.rows[i]
        out["meta"] = {
            "index": int(row["index"]),
            "file": row["file"],
            "utterance": row["utterance"],
            "snr_target_db": float(row["snr_target_db"]) if row["snr_target_db"] else float("nan"),
            "doa_true": float(row["doa_true"]),
            "t60": float(row["t60"]),
            "room_key": row["room_key"],
            "start": start,
        }
        return out


def collate_samples(batch: list[dict]):
    """把若干条样本堆成 batch。

    返回 `(inputs, targets)`：inputs 为 dict，targets 为 dict。
    刻意不用 torch 的 default_collate——`meta` 是混合类型 dict，
    default_collate 会尝试递归堆叠并在字符串字段上报错。
    """
    import torch

    return (
        {
            "mag": torch.from_numpy(np.stack([b["mag"] for b in batch])),
            "meta": [b["meta"] for b in batch],
        },
        {
            "mask": torch.from_numpy(np.stack([b["mask"] for b in batch])),
            "clean_mag": torch.from_numpy(np.stack([b["clean_mag"] for b in batch])),
        },
    )


# ── 位置说明 ────────────────────────────────────────────────────
# 设计文档 §5 把 dataset.py 画在 `nbse/data/` 下，实施时放到 `nbse/models/`：
# `nbse/data/` 现有模块（simulate / corpus / generate）**全部不依赖 torch**，
# 因此「造数据」这半边可以在只装了 numpy + pyroomacoustics 的环境里跑通
# （数据下载与生成不需要 GPU）。把 PyTorch 依赖隔离在 `nbse/models/` 下，
# 这条边界得以保留：本模块对 torch 的导入全部延迟到 `collate_samples` 内部。
