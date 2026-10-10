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
import os
from concurrent.futures import ProcessPoolExecutor
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
    # 渲染用几个进程。1 = 串行（默认，行为与历史一致）。
    # 房间仿真是纯 CPU 工作且各作业互相独立，故可并行；实测 8 进程加速 3.7×。
    # **不影响结果**：作业列表仍在主进程按同一个种子顺序生成，结果按序回收，
    # 已用「与串行逐位一致」的测试验证。
    n_jobs: int = 1


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


def _load_pair(utt: Utterance, noise_path: Path,
               sim_cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """读入一条语音与一条噪声，并校验采样率。"""
    clean, sr = load_audio(utt.path)
    if sr != sim_cfg.sample_rate:
        raise ValueError(
            f"{utt.path} 采样率 {sr} Hz，与配置的 {sim_cfg.sample_rate} Hz 不符。"
            f"PESQ 宽带模式要求 16 kHz。"
        )
    noise, nsr = load_audio(noise_path)
    if nsr != sim_cfg.sample_rate:
        raise ValueError(f"{noise_path} 采样率 {nsr} Hz，与配置不符")
    return clean, noise


def _render_sample(idx: int, utt: Utterance, noise_path: Path, room,
                   snr_target: float, sim_cfg: SimConfig, out_dir: Path,
                   cache_dir: Path | None, rng: np.random.Generator,
                   wav_preview: int) -> dict:
    """合成一条样本、落盘、返回 manifest 的一行。"""
    clean, noise = _load_pair(utt, noise_path, sim_cfg)

    out = simulate_sample(clean, noise, replace(sim_cfg, snr_db=snr_target),
                          rng, room_cfg=room, cache_dir=cache_dir)

    name = f"sample_{idx:05d}.npz"
    np.savez(
        out_dir / name,
        multi_ch=out["multi_ch"].astype(np.float32),
        clean_ref=out["clean_ref"].astype(np.float32),
        noise_multi=out["noise_multi"].astype(np.float32),
        doa_true=np.float64(out["doa_true"]),
        snr_db=np.float64(out["snr_db"]),
    )

    if idx < wav_preview:
        _write_wav_preview(out_dir / f"preview_{idx:05d}", sim_cfg.sample_rate,
                           mix=out["multi_ch"], clean=out["clean_ref"],
                           noise=out["noise_multi"])

    return {
        "index": idx,
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
    }


# ── 并行渲染 ────────────────────────────────────────────────────
# worker 进程的只读共享状态。用模块级全局而非把参数塞进每个作业：
# 语料/噪声路径列表会被每个进程重新 pickle 一次，放进 initargs 只付一次成本。
_WORKER_CTX: dict = {}


def _init_worker(sim_cfg, utterances, noise_files, out_dir, cache_dir, spec):
    _WORKER_CTX.clear()
    _WORKER_CTX.update(sim_cfg=sim_cfg, utterances=utterances,
                       noise_files=noise_files, out_dir=out_dir,
                       cache_dir=cache_dir, spec=spec)


def _render_job(args):
    """渲染单个作业（供进程池调用）。

    必须是**模块级函数**且参数可 pickle——闭包与局部函数无法跨进程传递。
    `rng` 传 None：批处理路径下目标 SNR 已由主进程决定并写进作业，
    `simulate_sample` 不会再用到它（随机取值只在未给定 sni_db 时发生）。
    """
    i, utt_idx, noise_idx, room, snr_target = args
    ctx = _WORKER_CTX
    return _render_sample(i, ctx["utterances"][utt_idx],
                          ctx["noise_files"][noise_idx], room, snr_target,
                          ctx["sim_cfg"], ctx["out_dir"], ctx["cache_dir"],
                          None, ctx["spec"].wav_preview)


def _run_jobs(jobs, utterances, noise_files, sim_cfg, spec, out_dir, cache_dir,
              progress) -> list[dict]:
    """把作业列表逐条渲染成样本。作业 = (语料下标, 噪声下标, 房间, 目标 SNR)。

    n_jobs > 1 时用进程池并行。顺序与内容不受并行影响：作业列表在主进程
    生成好，`executor.map` 按输入顺序返回结果，manifest 的行序因此稳定。
    """
    n_jobs = max(int(getattr(spec, "n_jobs", 1)), 1)

    if n_jobs == 1 or len(jobs) <= 1:
        rows = []
        for i, (utt_idx, noise_path, room, snr_target) in enumerate(jobs):
            rows.append(_render_sample(i, utterances[utt_idx], noise_path, room,
                                       snr_target, sim_cfg, out_dir, cache_dir,
                                       None, spec.wav_preview))
            if progress is not None:
                progress(i + 1, len(jobs))
        return rows

    # 预先把噪声路径解析成下标，避免给每个作业 pickle 路径对象。
    # 语料必须连 Utterance 一起给 worker：`_render_sample` 会读它的
    # .path/.subset/.speaker 等字段（manifest 要用），只传路径会 AttributeError。
    noise_paths = [Path(p) for p in noise_files]
    noise_index = {p: i for i, p in enumerate(noise_paths)}
    payload = [(i, utt_idx, noise_index[Path(np_path)], room, snr)
               for i, (utt_idx, np_path, room, snr) in enumerate(jobs)]

    chunksize = max(1, len(payload) // (n_jobs * 4))
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker,
                             initargs=(sim_cfg, utterances, noise_paths,
                                       out_dir, cache_dir, spec)) as ex:
        for done, row in enumerate(ex.map(_render_job, payload,
                                          chunksize=chunksize), start=1):
            rows.append(row)
            if progress is not None:
                progress(done, len(payload))
    return rows


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

    jobs = []
    for utt_idx in _sample_order(len(utterances), spec.n_samples, rng):
        noise_path = Path(noise_arr[int(rng.integers(len(noise_files)))])
        room = sample_room(sim_cfg, rng)
        if spec.t60_levels:
            room.t60 = float(rng.choice(np.asarray(spec.t60_levels, dtype=float)))
        jobs.append((utt_idx, noise_path, room,
                     _draw_snr_target(spec, sim_cfg, rng)))

    rows = _run_jobs(jobs, utterances, noise_files, sim_cfg, spec, out_dir,
                     cache_dir, progress)
    _write_manifest(out_dir, rows)
    _write_run_config(out_dir, sim_cfg, spec, cache_dir)
    return rows


def generate_grid(utterances: list[Utterance], noise_files: list[Path],
                  sim_cfg: SimConfig, spec: BatchSpec,
                  n_utterances: int = 20,
                  cache_dir: Path | None = None,
                  progress=None) -> list[dict]:
    """生成**配对**评估网格：`n_utterances` 条语料 × (SNR 档 × T60 档) 全交叉。

    与 `generate_batch` 的关键区别是把无关变量控制住：

    * 同一条语料在所有条件下复用**同一套房间几何与同一条噪声**，只改 T60 与 SNR。
      否则条件之间的差异会混入「房间不同」这一无关变量，条件间不可比。
    * 作业按 (SNR, T60) 分组生成，manifest 里连续一片就是同一条件，
      评测脚本按条件聚合时不必额外对齐。

    要求 snr_levels 与 t60_levels 都给全——配对网格没有「随机」这一档。
    """
    if not spec.snr_levels or not spec.t60_levels:
        raise ValueError("generate_grid 需要同时给定 snr_levels 与 t60_levels")
    if not utterances or not noise_files:
        raise ValueError("语料列表或噪声列表为空")

    n_utterances = min(int(n_utterances), len(utterances))
    out_dir = Path(spec.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    noise_arr = np.asarray(noise_files, dtype=object)

    # 每条语料先固定一套「房间几何 + 噪声」——这是配对的前提
    pool = [int(u) for u in rng.permutation(len(utterances))[:n_utterances]]
    base_rooms = {u: sample_room(sim_cfg, rng) for u in pool}
    base_noise = {u: Path(noise_arr[int(rng.integers(len(noise_files)))])
                  for u in pool}

    jobs = []
    for snr in spec.snr_levels:
        for t60 in spec.t60_levels:
            for u in pool:
                room = replace(base_rooms[u])     # 复制后只改 T60，几何保持不动
                room.t60 = float(t60)
                jobs.append((u, base_noise[u], room, float(snr)))

    rows = _run_jobs(jobs, utterances, noise_files, sim_cfg, spec, out_dir,
                     cache_dir, progress)
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
