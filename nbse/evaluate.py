"""T2.4 评测：在多条件网格上跑齐各方法，产出结果表。

设计文档 §6（评估设计）、实施计划 T2.4。

本模块只做「给定已落盘的样本 + 一组方法 → 指标表」这一件事；
样本的生成由 `scripts/generate_dataset.py` 负责，两者用 `manifest.csv` 对接。
CLI 在仓库根目录的 `eval.py`。

关于电平：PESQ / STOI 对信号幅度敏感，而波束形成器的输出幅度与参考信号
不一定同量级（DS 是相干求和，MVDR 有无失真约束）。计算指标前把估计信号的
峰值对齐到参考信号的峰值——这是语音增强评测的通行做法，也避免 PESQ 内部
转 int16 时截幅。SI-SDR 本身尺度不变，但为了三个指标口径一致也一并处理。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .beamformers.delay_and_sum import DelayAndSum
from .beamformers.mvdr import MVDR
from .data.generate import load_manifest
from .data.simulate import SimConfig, mic_positions
from .metrics import pesq_wb, si_sdr, stoi

FS_DEFAULT = 16000

# 方法名 → 实现。函数签名 (sample, mic_pos, center_xy, sim_cfg) -> 增强输出 [T]
METHOD_IMPLS = {
    "Noisy": lambda s, mic, c, cfg: s["multi_ch"][0],
    "DS": lambda s, mic, c, cfg: DelayAndSum(mic, c, fs=cfg.sample_rate)
                                  .enhance(s["multi_ch"], s["doa_true"]),
    "MVDR (Oracle DOA)": lambda s, mic, c, cfg: MVDR(mic, c, fs=cfg.sample_rate)
                                  .enhance(s["multi_ch"], s["doa_true"],
                                           noise_multi=s["noise_multi"]),
}

DEFAULT_METHODS = ["Noisy", "DS", "MVDR (Oracle DOA)"]

# 指标排序检查用的期望顺序（设计文档 §3.2 第 2 条）
EXPECTED_ORDER = ["Noisy", "DS", "MVDR (Oracle DOA)"]


def align_level(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """把估计信号的峰值对齐到参考信号，供对电平敏感的指标使用。"""
    ref_peak = float(np.max(np.abs(reference)))
    est_peak = float(np.max(np.abs(estimate)))
    if est_peak < 1e-12:
        return estimate
    return estimate * (ref_peak / est_peak)


def condition_of(row: dict) -> tuple[float, float]:
    """样本所处的 (SNR 档, T60 档)。"""
    return float(row["snr_target_db"]), float(row["t60"])


def list_conditions(rows: list[dict]) -> list[tuple[float, float]]:
    """manifest 里出现的全部条件，按 SNR 升序、再按 T60 升序。"""
    return sorted({condition_of(r) for r in rows})


def load_sample(data_dir: Path, row: dict) -> dict:
    """读一条样本，并把 manifest 的字段一并带上。"""
    with np.load(Path(data_dir) / row["file"]) as z:
        sample = {k: z[k] for k in z.files}
    sample["doa_true"] = float(row["doa_true"])
    sample["room_lx"] = float(row["room_lx"])
    sample["room_ly"] = float(row["room_ly"])
    return sample


def sample_mic_positions(sample: dict, sim_cfg: SimConfig) -> tuple[np.ndarray, tuple]:
    """由样本的房间尺寸还原阵列位置（仿真时阵列置于房间中心）。"""
    center = (sample["room_lx"] / 2.0, sample["room_ly"] / 2.0)
    return mic_positions(sim_cfg.array, center), center


def run_methods(data_dir: str | Path, rows: list[dict], sim_cfg: SimConfig,
                methods: list[str] | None = None,
                limit_per_condition: int | None = None,
                progress=None) -> list[dict]:
    """在给定样本上跑各方法，返回逐条记录。

    每条记录：method / snr_db / t60 / pesq_wb / stoi / si_sdr / utterance
    """
    methods = list(methods or DEFAULT_METHODS)
    unknown = [m for m in methods if m not in METHOD_IMPLS]
    if unknown:
        raise ValueError(f"未知方法：{unknown}（可选 {sorted(METHOD_IMPLS)}）")

    if limit_per_condition is not None:
        rows = _limit_per_condition(rows, limit_per_condition)

    records: list[dict] = []
    for i, row in enumerate(rows):
        sample = load_sample(data_dir, row)
        mic, center = sample_mic_positions(sample, sim_cfg)
        ref = sample["clean_ref"]

        for method in methods:
            est = METHOD_IMPLS[method](sample, mic, center, sim_cfg)
            est = align_level(ref, est)
            snr, t60 = condition_of(row)
            records.append({
                "method": method,
                "snr_db": snr,
                "t60": t60,
                "utterance": row["utterance"],
                "pesq_wb": pesq_wb(ref, est, sim_cfg.sample_rate),
                "stoi": stoi(ref, est, sim_cfg.sample_rate),
                "si_sdr": si_sdr(ref, est),
            })
        if progress is not None:
            progress(i + 1, len(rows))
    return records


def _limit_per_condition(rows: list[dict], limit: int) -> list[dict]:
    """每个条件只取前 limit 条（配对网格里每条语料在每条件各一条，故仍配对）。"""
    counts: dict[tuple, int] = {}
    kept = []
    for r in rows:
        key = condition_of(r)
        if counts.get(key, 0) < limit:
            counts[key] = counts.get(key, 0) + 1
            kept.append(r)
    return kept


def aggregate(records: list[dict]) -> list[dict]:
    """按 (方法, SNR, T60) 求均值，并额外给出每个方法的总计行。"""
    metrics = ("pesq_wb", "stoi", "si_sdr")
    out: list[dict] = []

    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["method"], r["snr_db"], r["t60"]), []).append(r)

    for (method, snr, t60), group in sorted(groups.items()):
        row = {"method": method, "snr_db": snr, "t60": t60, "n": len(group)}
        row.update({m: float(np.mean([g[m] for g in group])) for m in metrics})
        out.append(row)

    for method in dict.fromkeys(r["method"] for r in records):
        group = [r for r in records if r["method"] == method]
        row = {"method": method, "snr_db": "ALL", "t60": "ALL", "n": len(group)}
        row.update({m: float(np.mean([g[m] for g in group])) for m in metrics})
        out.append(row)

    return out


def write_csv(rows: list[dict], path: str | Path) -> Path:
    """落盘结果表（utf-8-sig，便于 Excel 直接打开）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["method", "snr_db", "t60", "n", "pesq_wb", "stoi", "si_sdr"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, **{m: f"{r[m]:.4f}" for m in
                                     ("pesq_wb", "stoi", "si_sdr")}})
    return path


def format_table(rows: list[dict], metric: str = "si_sdr") -> str:
    """把结果排成「方法 × 条件」的表，便于肉眼检查趋势。"""
    conditions = sorted({(r["snr_db"], r["t60"]) for r in rows
                         if r["snr_db"] != "ALL"}, key=lambda x: (x[0], x[1]))
    methods = list(dict.fromkeys(r["method"] for r in rows))

    header = f"{'方法':<20s}" + "".join(
        f"  SNR{c[0]:+5.1f}/T60 {c[1]:.1f}" for c in conditions) + "     总体"
    lines = [header, "-" * len(header)]
    for m in methods:
        cells = []
        for c in conditions:
            hit = [r for r in rows if r["method"] == m
                   and r["snr_db"] == c[0] and r["t60"] == c[1]]
            cells.append(f"{hit[0][metric]:>16.3f}" if hit else f"{'-':>16s}")
        total = [r for r in rows if r["method"] == m and r["snr_db"] == "ALL"]
        cells.append(f"{total[0][metric]:>9.3f}" if total else "")
        lines.append(f"{m:<20s}" + "".join(cells))
    return "\n".join(lines)


def check_expected_order(rows: list[dict], metric: str = "si_sdr") -> tuple[bool, str]:
    """检查总体指标是否符合期望顺序 Noisy → DS → MVDR 单调不降（§3.2 第 2 条）。"""
    totals = {r["method"]: r[metric] for r in rows if r["snr_db"] == "ALL"}
    missing = [m for m in EXPECTED_ORDER if m not in totals]
    if missing:
        return False, f"缺少方法：{missing}"

    values = [totals[m] for m in EXPECTED_ORDER]
    chain = " → ".join(f"{m} {v:.3f}" for m, v in zip(EXPECTED_ORDER, values))
    ok = all(a <= b + 1e-9 for a, b in zip(values, values[1:]))
    return ok, chain
