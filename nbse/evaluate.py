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
from .beamformers.doa import estimate_doa
from .beamformers.mask_mvdr import MaskMVDR
from .beamformers.mvdr import MVDR
from .data.generate import load_manifest
from .data.simulate import SimConfig, mic_positions
from .metrics import pesq_wb, si_sdr, stoi

FS_DEFAULT = 16000

# 对角加载强度。**两条 MVDR 路线必须用同一个值**，否则测出来的是正则化强度
# 差异而不是方法差异。
#
# 实测教训：`MaskMVDR` 与 `MVDR` 的类默认值曾是 1e-2 与 1e-3（相差 10 倍）。
# 在同一个 R_n 上，仅因加载不同就让波束形成权重相对差达 0.78，SI-SDR 相差
# 2.3 dB——足以让「谁更好」的结论完全反过来。故在此统一取值、两条路线共用。
#
# 取值 3e-4 的依据（2026-10-10 在**验证集**上扫描，避免测试集调参）：
#   loading 从 1e-2 降到 3e-4，两条路线**各提升约 1.9 dB**——原值严重过大。
#   验证集上 mask-MVDR：1e-6→−2.306、1e-5→−2.323、1e-4→−2.356、
#                       3e-4→−2.507、1e-3→−2.885、1e-2→−4.345
#   性能平台在 1e-5~1e-4，再往下不再变好。但**数值稳定性否决了取 1e-4 或更小**：
#       loading   正则化后 cond p99    ‖w‖max
#         1e-6       3,967,372          326
#         1e-4          39,790           22
#         3e-4          13,301         16.3   ← 选它
#         1e-2             401          3.5
#   1e-6 时单频点条件数达 4×10⁶、权重范数 326：输出虽仍有限，但大权重会放大
#   「假设的导向矢量」与「实际传递函数」的失配，在混响更强或 DOA 有误差时很脆弱。
#   3e-4 在性能（仅比最优差 0.2 dB）与稳健性之间取得平衡。
MVDR_LOADING = 3e-4

# mask-MVDR 的用法配置。提成模块级常量而不是写在 lambda 里，
# 这样测试能直接断言「评测用的是哪种用法」，而不必去反射 lambda 的源码。
#
# weighting="noise"：掩蔽**只用来估噪声协方差 R_n**，不让 R_s 参与。
# 注意在 mode="mvdr" 下，MVDR 解 w = R_n⁻¹d/(d^H R_n⁻¹d) 本来就不含 R_s，
# 因此 noise 与 both 在该模式下**必然给出相同结果**——这是设计使然，
# 不是 bug。两者的区别只在 mode="gevd"（广义特征值）下才体现。
MASK_WEIGHTING = "noise"
MASK_MODE = "mvdr"

# 基础方法：不需要训练。函数签名
#   (sample, mic_pos, center_xy, sim_cfg, ctx) -> 增强输出 [T]
# `ctx` 是 dict，携带可选资源：mask_predictor / dccrn。各方法按需取用，
# 不用它就忽略——这样签名统一，新增方法不必改所有调用点。
def _ctx_get(ctx, key):
    return None if ctx is None else ctx.get(key)


def _est_doa(s, mic, cfg):
    """估计方位角（GCC-PHAT）。DOA 只依赖观测与几何，与 `doa_true` 无关。"""
    return estimate_doa(s["multi_ch"], mic, fs=cfg.sample_rate)


BASE_METHOD_IMPLS = {
    "Noisy": lambda s, mic, c, cfg, ctx: s["multi_ch"][0],
    "DS": lambda s, mic, c, cfg, ctx: DelayAndSum(mic, c, fs=cfg.sample_rate)
                                .enhance(s["multi_ch"], s["doa_true"]),
    "MVDR (Oracle DOA)": lambda s, mic, c, cfg, ctx: MVDR(
                                mic, c, fs=cfg.sample_rate, loading=MVDR_LOADING
                            ).enhance(s["multi_ch"], s["doa_true"],
                                      noise_multi=s["noise_multi"]),
    # 与上一行**唯一**的差别是 DOA 来自估计而非真值。噪声协方差仍用真值噪声，
    # 因此这一对比较量化的是纯粹的「DOA 估计误差代价」。
    "MVDR (GCC-PHAT)": lambda s, mic, c, cfg, ctx: MVDR(
                                mic, c, fs=cfg.sample_rate, loading=MVDR_LOADING
                            ).enhance(s["multi_ch"], _est_doa(s, mic, cfg),
                                      noise_multi=s["noise_multi"]),
}

# 需要 mask_predictor 的方法（T4.1）
MASK_METHOD_IMPLS = {
    "mask-MVDR (Oracle DOA)": lambda s, mic, c, cfg, ctx: MaskMVDR(
        mic, c, mask_predictor=_ctx_get(ctx, "mask_predictor"),
        fs=cfg.sample_rate,
        loading=MVDR_LOADING, weighting=MASK_WEIGHTING, mode=MASK_MODE,
    ).enhance(s["multi_ch"], s["doa_true"]),
    "mask-MVDR (GCC-PHAT)": lambda s, mic, c, cfg, ctx: MaskMVDR(
        mic, c, mask_predictor=_ctx_get(ctx, "mask_predictor"),
        fs=cfg.sample_rate,
        loading=MVDR_LOADING, weighting=MASK_WEIGHTING, mode=MASK_MODE,
    ).enhance(s["multi_ch"], _est_doa(s, mic, cfg)),
}

# 需要 DCCRN-lite checkpoint 的方法（T5.2）
DCCRN_METHOD_IMPLS = {
    "DCCRN-lite": lambda s, mic, c, cfg, ctx: _ctx_get(ctx, "dccrn")
                                .enhance(s["multi_ch"]),
}

ALL_METHOD_NAMES = (list(BASE_METHOD_IMPLS) + list(MASK_METHOD_IMPLS)
                    + list(DCCRN_METHOD_IMPLS))

# 设计文档 §6.3 的最终七行，按方法依赖排序
FINAL_METHODS = ["Noisy", "DS", "MVDR (Oracle DOA)", "MVDR (GCC-PHAT)",
                 "mask-MVDR (Oracle DOA)", "mask-MVDR (GCC-PHAT)", "DCCRN-lite"]


def build_method_impls(mask_predictor=None, dccrn=None) -> dict:
    """组装方法表，按可用资源注册。

    * 没有 `mask_predictor` → **不注册** mask-MVDR 两条
    * 没有 `dccrn` → **不注册** DCCRN-lite

    刻意不提供「缺资源就退化为基线」的兜底：那会让「神经方法优于基线」
    这类结论变成假的，而表面上一切正常（结果表照常产出）。宁可让方法名
    不存在、直接报错。
    """
    impls = dict(BASE_METHOD_IMPLS)
    if mask_predictor is not None:
        impls.update(MASK_METHOD_IMPLS)
    if dccrn is not None:
        impls.update(DCCRN_METHOD_IMPLS)
    return impls


# 向后兼容：不带 predictor 时的默认方法表
METHOD_IMPLS = build_method_impls(None)

DEFAULT_METHODS = ["Noisy", "DS", "MVDR (Oracle DOA)"]

# 指标排序检查用的期望顺序（设计文档 §3.2 第 2 条）。
# mask-MVDR 不列入：T2.5 的保底交付点只要求 Noisy → DS → MVDR 单调。
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
                progress=None,
                mask_predictor=None,
                dccrn=None) -> list[dict]:
    """在给定样本上跑各方法，返回逐条记录。

    每条记录：method / snr_db / t60 / pesq_wb / stoi / si_sdr / utterance

    Parameters
    ----------
    mask_predictor : 可调用对象 `(multi_ch) -> mask[F, T]`。给了它才会注册
                     mask-MVDR；没给而 methods 里又点名要它，会明确报错。
    dccrn          : 单通道增强器 `enhance(multi_ch) -> waveform`。给了它才会
                     注册 `DCCRN-lite`。
    两者都可以缺省；缺省时对应方法**不存在**，而不是退化成基线。
    """
    ctx = {"mask_predictor": mask_predictor, "dccrn": dccrn}
    impls = build_method_impls(mask_predictor, dccrn)
    methods = list(methods or DEFAULT_METHODS)
    unknown = [m for m in methods if m not in impls]
    if unknown:
        hint = ""
        if any(m in MASK_METHOD_IMPLS for m in unknown):
            hint += ("\n其中 mask-MVDR 需要 mask_predictor（训练好的 CRN checkpoint）。"
                     "请用 eval.py --ckpt <path> 指定。")
        if any(m in DCCRN_METHOD_IMPLS for m in unknown):
            hint += ("\n其中 DCCRN-lite 需要单通道 checkpoint。"
                     "请用 eval.py --dccrn <path> 指定。")
        raise ValueError(f"未知方法：{unknown}（可选 {sorted(impls)}）{hint}")

    if limit_per_condition is not None:
        rows = _limit_per_condition(rows, limit_per_condition)

    records: list[dict] = []
    for i, row in enumerate(rows):
        sample = load_sample(data_dir, row)
        mic, center = sample_mic_positions(sample, sim_cfg)
        ref = sample["clean_ref"]

        for method in methods:
            est = impls[method](sample, mic, center, sim_cfg, ctx)
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
