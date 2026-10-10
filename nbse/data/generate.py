"""T1.4 批量生成多通道样本并落盘。

把语料（`corpus.py`）与房间仿真（`simulate.py`）接起来，产出一批可复现的
多通道样本 + 一份 `manifest.csv`。实施计划 T1.4 的验收标准是：

    生成 ≥ 200 条多通道样本，落盘检查形状、采样率（16 kHz）、时长一致。

因此本模块除生成外还提供 `verify_batch()`——把落盘结果重新读回来逐项核对，
让「验收」是一条可执行的命令，而不是肉眼检查。

存盘格式：每个样本一个 `.npz`（float32），字段与 `simulate_sample()` 的输出对应——

    multi_ch     [n_mics, T]   多通道混合观测
    clean_ref    [T]           单通道干净参考（dry）
    noise_multi  [n_mics, T]   纯噪声多通道信号（MVDR 估噪声协方差需要）
    doa_true     float         声源方位角（度）
    snr_db       float         实测信噪比

选 npz 而非 wav：float32 往返无精度损失，且 numpy 原生，Dataset 直接读。
需要人工听感抽查时用 `wav_preview`（或 CLI 的 `--wav-preview`）另写 16-bit wav。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .corpus import Utterance, load_audio
from .simulate import SimConfig, sample_room, simulate_sample

MANIFEST_NAME = "manifest.csv"

MANIFEST_COLUMNS = [
    "index", "file", "subset", "speaker", "chapter", "utterance",
    "noise_env", "noise_file",
    "snr_target_db", "snr_db", "doa_true", "t60",
    "room_lx", "room_ly", "room_lz", "room_key",
    "distance_m", "ref_delay",
    "n_mics", "sample_rate", "n_samples",
]


@dataclass
class BatchSpec:
    """一次批量生成的参数。"""
    out_dir: Path
    n_samples: int = 200
    seed: int = 0
    # 给定则在这些 SNR 档位里等概率抽（评估用）；None → 见 _draw_snr_target
    snr_levels: tuple[float, ...] | None = None
    # None → 由 sample_room 随机；给定则固定到这些 T60 档（评估用）
    t60_levels: tuple[float, ...] | None = None
    wav_preview: int = 0        # 额外写前 N 条的 16-bit wav 以便试听


def _sample_order(n_utts: int, n_samples: int, rng: np.random.Generator) -> list[int]:
    """生成取样顺序：每轮先打乱语料，轮次用尽再开新一轮。

    语料条数少于需要的样本数时靠多轮覆盖，保证每条语料被用到的次数尽量均匀。
    """
    order: list[int] = []
    while len(order) < n_samples:
        order.extend(int(i) for i in rng.permutation(n_utts))
    return order[:n_samples]


def _source_distance(room, sim_cfg: SimConfig) -> float:
    """声源到阵列中心的距离（米）。参考信号的时延就由它决定。"""
    from .simulate import array_center

    cx, cy = array_center(room)
    center = np.array([cx, cy, sim_cfg.array.height])
    return float(np.linalg.norm(room.source_pos - center))


def _draw_snr_target(spec: BatchSpec, sim_cfg: SimConfig,
                     rng: np.random.Generator) -> float:
    """决定这一条样本的目标 SNR（dB）。

    优先级：`spec.snr_levels`（评估用的固定档）> `sim_cfg.snr_db`（显式指定）
    > `sim_cfg.train_snr_range`（训练用的随机区间）。

    关键点是**目标值由本函数决定并写进 manifest**，而不是留给
    `simulate_sample` 内部去抽——否则随机模式下 manifest 里没有目标值，
    「实测 SNR 与目标差 ±0.5 dB」这条承诺在整批规模上就无从复核。
    """
    if spec.snr_levels:
        return float(rng.choice(np.asarray(spec.snr_levels, dtype=float)))
    if sim_cfg.snr_db is not None:
        return float(sim_cfg.snr_db)
    return float(rng.uniform(*sim_cfg.train_snr_range))


def _to_int16(x: np.ndarray) -> np.ndarray:
    """float → int16，先限幅避免溢出绕回。"""
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def _write_wav_preview(path: Path, sr: int, **channels: np.ndarray) -> None:
    """写多通道 wav（soundfile 要求 [T, ch]）。"""
    import soundfile as sf

    for name, data in channels.items():
        arr = np.atleast_2d(data)
        sf.write(str(path.with_name(f"{path.stem}_{name}.wav")), _to_int16(arr.T), sr)


def generate_batch(utterances: list[Utterance], noise_files: list[Path],
                   sim_cfg: SimConfig, spec: BatchSpec,
                   cache_dir: Path | None = None,
                   progress=None) -> list[dict]:
    """生成一批样本并落盘，返回 manifest 各行（list[dict]）。

    随机数由一个种子驱动，同样的输入 + 同样的种子必然产出同样的样本。

    progress : 可选的 `f(done, total)` 回调，供 CLI 打进度（几分钟的任务
               没有任何输出会让人怀疑是不是卡住了）。
    """
    if not utterances:
        raise ValueError("语料列表为空——请检查说话人切分与 LibriSpeech 目录")
    if not noise_files:
        raise ValueError("噪声列表为空——请检查 DEMAND 目录")

    out_dir = Path(spec.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(spec.seed)
    noise_arr = np.asarray(noise_files, dtype=object)
    rows: list[dict] = []

    order = _sample_order(len(utterances), spec.n_samples, rng)
    for i, utt_idx in enumerate(order):
        utt = utterances[utt_idx]
        clean, sr = load_audio(utt.path)
        if sr != sim_cfg.sample_rate:
            raise ValueError(
                f"{utt.path} 采样率 {sr} Hz，与配置的 {sim_cfg.sample_rate} Hz 不符。"
                f"PESQ 宽带模式要求 16 kHz。"
            )

        noise_path = Path(noise_arr[int(rng.integers(len(noise_files)))])
        noise, nsr = load_audio(noise_path)
        if nsr != sim_cfg.sample_rate:
            raise ValueError(f"{noise_path} 采样率 {nsr} Hz，与配置不符")

        room = sample_room(sim_cfg, rng)
        if spec.t60_levels:
            room.t60 = float(rng.choice(np.asarray(spec.t60_levels, dtype=float)))

        snr_target = _draw_snr_target(spec, sim_cfg, rng)
        cfg_i = replace(sim_cfg, snr_db=snr_target)

        out = simulate_sample(clean, noise, cfg_i, rng, room_cfg=room,
                              cache_dir=cache_dir)

        name = f"sample_{i:05d}.npz"
        np.savez(
            out_dir / name,
            multi_ch=out["multi_ch"].astype(np.float32),
            clean_ref=out["clean_ref"].astype(np.float32),
            noise_multi=out["noise_multi"].astype(np.float32),
            doa_true=np.float64(out["doa_true"]),
            snr_db=np.float64(out["snr_db"]),
        )

        if i < spec.wav_preview:
            _write_wav_preview(out_dir / f"preview_{i:05d}", sim_cfg.sample_rate,
                               mix=out["multi_ch"], clean=out["clean_ref"],
                               noise=out["noise_multi"])

        rows.append({
            "index": i,
            "file": name,
            "subset": utt.subset,
            "speaker": utt.speaker,
            "chapter": utt.chapter,
            "utterance": utt.path.name,
            "noise_env": noise_path.parent.name,
            "noise_file": noise_path.name,
            "snr_target_db": f"{snr_target:.2f}",
            "snr_db": f"{out['snr_db']:.4f}",
            "doa_true": f"{out['doa_true']:.4f}",
            "t60": f"{room.t60:.4f}",
            "room_lx": f"{room.size[0]:.4f}",
            "room_ly": f"{room.size[1]:.4f}",
            "room_lz": f"{room.size[2]:.4f}",
            "room_key": room.key(),
            "distance_m": f"{_source_distance(room, sim_cfg):.4f}",
            "ref_delay": int(out["ref_delay"]),
            "n_mics": sim_cfg.array.n_mics,
            "sample_rate": sim_cfg.sample_rate,
            "n_samples": int(out["multi_ch"].shape[1]),
        })

        if progress is not None:
            progress(i + 1, len(order))

    _write_manifest(out_dir, rows)
    _write_run_config(out_dir, sim_cfg, spec, cache_dir)
    return rows


def _write_manifest(out_dir: Path, rows: list[dict]) -> None:
    with open(out_dir / MANIFEST_NAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_config(out_dir: Path, sim_cfg: SimConfig, spec: BatchSpec,
                      cache_dir: Path | None) -> None:
    """记录本次生成的参数，便于日后复现（§4.5 可复现性）。"""
    payload = {
        **asdict(sim_cfg),
        "batch": {k: str(v) if isinstance(v, Path) else v
                  for k, v in asdict(spec).items()},
        "rir_cache": None if cache_dir is None else str(cache_dir),
    }
    (out_dir / "run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(out_dir: str | Path) -> list[dict]:
    """读回 manifest.csv。"""
    path = Path(out_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}——该目录还没有生成过样本")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify_batch(out_dir: str | Path) -> dict:
    """T1.4 验收：把落盘的样本读回来核对形状、采样率与时长一致性。

    返回摘要 dict；任何一项不符即抛 AssertionError 并指出是哪个文件。
    """
    out_dir = Path(out_dir)
    rows = load_manifest(out_dir)
    if not rows:
        raise AssertionError(f"{out_dir} 里 manifest 为空")

    n_mics = int(rows[0]["n_mics"])
    sample_rate = int(rows[0]["sample_rate"])
    lengths: set[int] = set()
    speakers: set[str] = set()

    for row in rows:
        path = out_dir / row["file"]
        assert path.exists(), f"manifest 引用的文件缺失：{path}"

        with np.load(path) as z:
            missing = {"multi_ch", "clean_ref", "noise_multi",
                       "doa_true", "snr_db"} - set(z.files)
            assert not missing, f"{path.name} 缺字段 {sorted(missing)}"

            multi, clean, noise = z["multi_ch"], z["clean_ref"], z["noise_multi"]
            assert multi.dtype == np.float32, f"{path.name} multi_ch 应为 float32"
            assert multi.ndim == 2 and multi.shape[0] == n_mics, \
                f"{path.name} multi_ch 形状应为 [{n_mics}, T]，实为 {multi.shape}"
            assert noise.shape == multi.shape, \
                f"{path.name} noise_multi 形状 {noise.shape} 与 multi_ch 不一致"
            assert clean.shape == (multi.shape[1],), \
                f"{path.name} clean_ref 形状应为 [{multi.shape[1]}]，实为 {clean.shape}"
            assert np.isfinite(multi).all() and np.isfinite(noise).all(), \
                f"{path.name} 含 NaN/Inf"
            assert 0.0 <= float(z["doa_true"]) < 360.0, f"{path.name} doa_true 越界"

            lengths.add(int(multi.shape[1]))
            speakers.add(row["speaker"])

    assert len(lengths) == 1, (
        f"时长不一致：出现了 {len(lengths)} 种长度 {sorted(lengths)} 个采样点。"
        f"（每条样本都应是 cfg.duration × {sample_rate} Hz）"
    )
    n_per = lengths.pop()

    return {
        "out_dir": str(out_dir),
        "n_samples": len(rows),
        "n_samples_per_file": n_per,
        "duration_seconds": n_per / sample_rate,
        "sample_rate": sample_rate,
        "n_mics": n_mics,
        "n_speakers": len(speakers),
    }
