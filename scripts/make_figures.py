"""生成 README 用的图。实施计划 T5.3（DOA 误差代价图）与 W6（波束图）。

用法：

    python scripts/make_figures.py doa-cost      # T5.3：DOA 误差的代价
    python scripts/make_figures.py beampattern   # W6：波束图
    python scripts/make_figures.py all

图一律输出到 `docs/figures/`，标签用**英文**（配合 W6 的英文 README，
也避免字体依赖）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nbse.beamformers.doa import angular_error, estimate_doa  # noqa: E402
from nbse.beamformers.mvdr import MVDR  # noqa: E402
from nbse.data.generate import load_manifest  # noqa: E402
from nbse.data.simulate import SimConfig  # noqa: E402
from nbse.determinism import determinism_status, enable_determinism  # noqa: E402
from nbse.evaluate import (  # noqa: E402
    MVDR_LOADING,
    align_level,
    load_sample,
    sample_mic_positions,
)
from nbse.metrics import pesq_wb, si_sdr, stoi  # noqa: E402

FIG_DIR = REPO_ROOT / "docs" / "figures"

# 统一配色：Oracle 用蓝、GCC-PHAT 用橙，全项目一致
C_ORACLE = "#1f77b4"
C_EST = "#ff7f0e"
C_GREY = "#888888"


def _prepare():
    fig_dir = FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def collect_doa_cost(data_dir: Path, sim_cfg: SimConfig,
                     limit: int | None = None) -> dict:
    """逐样本收集：DOA 误差 + 两种 DOA 来源下 MVDR 的三项指标。

    两种 MVDR **只差 DOA 的来源**（噪声协方差都用真值），
    因此这里的差值就是纯粹的「DOA 估计误差代价」。
    """
    rows = load_manifest(data_dir)
    if limit:
        rows = rows[:limit]

    rec = {k: [] for k in
           ("doa_true", "doa_est", "doa_err", "snr", "t60",
            "sdr_oracle", "sdr_est", "stoi_oracle", "stoi_est",
            "pesq_oracle", "pesq_est")}

    for row in rows:
        s = load_sample(data_dir, row)
        mic, center = sample_mic_positions(s, sim_cfg)
        ref = s["clean_ref"]

        est = estimate_doa(s["multi_ch"], mic, fs=sim_cfg.sample_rate)
        rec["doa_true"].append(s["doa_true"])
        rec["doa_est"].append(est)
        rec["doa_err"].append(angular_error(est, s["doa_true"]))
        rec["snr"].append(float(row["snr_target_db"]))
        rec["t60"].append(float(row["t60"]))

        for tag, doa in (("oracle", s["doa_true"]), ("est", est)):
            y = align_level(ref, MVDR(mic, center, fs=sim_cfg.sample_rate,
                                      loading=MVDR_LOADING)
                            .enhance(s["multi_ch"], doa,
                                     noise_multi=s["noise_multi"]))
            rec[f"sdr_{tag}"].append(si_sdr(ref, y))
            rec[f"stoi_{tag}"].append(stoi(ref, y, sim_cfg.sample_rate))
            rec[f"pesq_{tag}"].append(pesq_wb(ref, y, sim_cfg.sample_rate))

    return {k: np.asarray(v, dtype=float) for k, v in rec.items()}


def figure_doa_cost(rec: dict, out: Path) -> Path:
    """T5.3 的 DOA 误差代价图：三块面板讲一件事。

    (a) DOA 误差随 SNR 变化 —— 说明「什么时候会坏」
    (b) MVDR 指标随 SNR 变化（Oracle vs GCC-PHAT）—— 说明「坏了要付多少」
    (c) 逐样本：DOA 误差 vs SI-SDR 损失 —— 说明「误差与代价的定量关系」
    """
    snrs = sorted(set(rec["snr"].tolist()))
    err = rec["doa_err"]
    loss = rec["sdr_oracle"] - rec["sdr_est"]      # 正 = GCC-PHAT 更差

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # ── (a) DOA 误差 vs SNR ──────────────────────────────────────
    ax = axes[0]
    data = [err[rec["snr"] == s] for s in snrs]
    bp = ax.boxplot(data, positions=range(len(snrs)), widths=0.55,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor=C_GREY,
                                    markeredgecolor="none", alpha=0.55))
    for patch in bp["boxes"]:
        patch.set_facecolor(C_EST)
        patch.set_alpha(0.55)
    # 叠加逐点，避免箱线图掩盖真实的离散程度
    for i, s in enumerate(snrs):
        v = err[rec["snr"] == s]
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(
            -0.14, 0.14, len(v)), v, s=9, color="#333333", alpha=0.5, zorder=3)
    ax.axhline(5.0, color="crimson", ls="--", lw=1.2, zorder=2)
    # 标签放在左上角、线之上：放右侧会与 +5 dB 箱体的中位线重叠
    ax.text(-0.42, 8.5, "5° acceptance line", color="crimson",
            fontsize=8.5, ha="left", va="bottom")
    ax.set_xticks(range(len(snrs)))
    ax.set_xticklabels([f"{s:+.0f}" for s in snrs])
    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("DOA estimation error (deg)")
    ax.set_title("(a) DOA estimation breaks down at low SNR", fontsize=11)
    ax.set_yscale("symlog", linthresh=10)
    ax.grid(alpha=0.25, axis="y")

    # ── (b) 指标 vs SNR ─────────────────────────────────────────
    ax = axes[1]
    for tag, color, label in (("oracle", C_ORACLE, "MVDR (oracle DOA)"),
                              ("est", C_EST, "MVDR (GCC-PHAT DOA)")):
        m = [rec[f"sdr_{tag}"][rec["snr"] == s].mean() for s in snrs]
        e = [rec[f"sdr_{tag}"][rec["snr"] == s].std() for s in snrs]
        ax.errorbar(range(len(snrs)), m, yerr=e, marker="o", capsize=4,
                    color=color, label=label, lw=2)
    ax.set_xticks(range(len(snrs)))
    ax.set_xticklabels([f"{s:+.0f}" for s in snrs])
    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("SI-SDR (dB)")
    ax.set_title("(b) Cost of estimating DOA instead of knowing it",
                 fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.25)

    # ── (c) 误差 vs 代价 ────────────────────────────────────────
    ax = axes[2]
    for s in snrs:
        m = rec["snr"] == s
        ax.scatter(err[m], loss[m], s=26, alpha=0.75, label=f"SNR {s:+.0f} dB",
                   edgecolors="none")
    # 分箱均值，给出趋势
    order = np.argsort(err)
    k = max(len(err) // 6, 5)
    xs = [err[order][i * k:(i + 1) * k].mean() for i in range(len(err) // k)]
    ys = [loss[order][i * k:(i + 1) * k].mean() for i in range(len(err) // k)]
    ax.plot(xs, ys, color="black", lw=2, ls="--", marker="s", ms=4,
            label="binned mean", zorder=5)
    ax.axhline(0, color=C_GREY, lw=1)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("DOA estimation error (deg)")
    ax.set_ylabel("SI-SDR loss vs oracle (dB)")
    ax.set_title("(c) Larger DOA error costs more SI-SDR", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "DOA estimation error and its cost — 120 test samples "
        "(MVDR differs only in DOA source; oracle noise covariance in both)",
        fontsize=11.5, y=1.015)
    fig.tight_layout()
    fig.savefig(out, dpi=165, bbox_inches="tight")
    plt.close(fig)

    # 同时落一份数值摘要，便于写 README 时引用
    summary = {
        "n": int(len(err)),
        "doa_err_median_deg": float(np.median(err)),
        "doa_err_p90_deg": float(np.percentile(err, 90)),
        "doa_err_max_deg": float(err.max()),
        "by_snr": {f"{s:+.0f}": {
            "n": int((rec["snr"] == s).sum()),
            "doa_err_median": float(np.median(err[rec["snr"] == s])),
            "sdr_oracle_mean": float(rec["sdr_oracle"][rec["snr"] == s].mean()),
            "sdr_est_mean": float(rec["sdr_est"][rec["snr"] == s].mean()),
            "stoi_oracle_mean": float(rec["stoi_oracle"][rec["snr"] == s].mean()),
            "stoi_est_mean": float(rec["stoi_est"][rec["snr"] == s].mean()),
            "pesq_oracle_mean": float(rec["pesq_oracle"][rec["snr"] == s].mean()),
            "pesq_est_mean": float(rec["pesq_est"][rec["snr"] == s].mean()),
        } for s in snrs},
        "overall": {
            "sdr_oracle_mean": float(rec["sdr_oracle"].mean()),
            "sdr_est_mean": float(rec["sdr_est"].mean()),
            "stoi_oracle_mean": float(rec["stoi_oracle"].mean()),
            "stoi_est_mean": float(rec["stoi_est"].mean()),
            "pesq_oracle_mean": float(rec["pesq_oracle"].mean()),
            "pesq_est_mean": float(rec["pesq_est"].mean()),
            "corr_err_loss": float(np.corrcoef(err, loss)[0, 1]),
        },
    }
    import json

    (out.parent / "doa_cost.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def figure_beampattern(sim_cfg: SimConfig, out: Path,
                       data_dir: Path, target_err: float = 60.0) -> Path:
    """W6 波束图：真实噪声场下 DS / MVDR(真值 DOA) / MVDR(估计 DOA) 的空间响应。

    **不能用 `R_n = I` 画这张图**（初版就是这么错的）：白噪声场下 MVDR 退化为
    `w ∝ d`，与延时求和**只差一个标量**，归一化后两条曲线必然重合、且是个
    近乎完美的圆——完全没有信息量。本项目的噪声是**过 RIR 的空间相干场**，
    MVDR 与 DS 的差别正来自这里，所以必须用样本里真实的 `noise_multi`
    估协方差。

    **样本要挑**：若挑到 DOA 估得很准的样本（误差 < 1°），oracle 与 GCC-PHAT
    两条曲线会完全重合，看不出估计误差的影响。故这里主动扫描，选一条
    DOA 误差最接近 `target_err`（默认 60°）的样本——那是低 SNR 档的典型量级，
    既不是最好也不是最坏，避免挑极端样本。

    曲线为 `|w(f)^H d(θ,f)|²` 在 200–4000 Hz 上的平均，已按各自最大值归一。
    """
    from nbse.beamformers.geometry import steering_vector
    from nbse.beamformers.mvdr import MVDR
    from nbse.dsp import N_FFT

    # 扫描测试集，挑一条 DOA 误差接近 target_err 的样本
    rows = load_manifest(data_dir)
    best = None
    for row in rows:
        s = load_sample(data_dir, row)
        mic, center = sample_mic_positions(s, sim_cfg)
        e = angular_error(estimate_doa(s["multi_ch"], mic, fs=sim_cfg.sample_rate),
                          s["doa_true"])
        if best is None or abs(e - target_err) < abs(best[0] - target_err):
            best = (e, row, s, mic, center)
    err, row, sample, mic, center = best
    fs = sim_cfg.sample_rate

    bf = MVDR(mic, center, fs=fs, loading=MVDR_LOADING)
    R_n = bf.estimate_noise_covariance(sample["noise_multi"])

    doa_true = float(sample["doa_true"])
    doa_est = estimate_doa(sample["multi_ch"], mic, fs=fs)

    freqs = np.fft.rfftfreq(N_FFT, 1 / fs)
    band = (freqs >= 200.0) & (freqs <= 4000.0)
    fb = freqs[band]

    angles = np.arange(0.0, 360.0, 1.0)

    def curve(weights_fn) -> np.ndarray:
        out = []
        for a in angles:
            d = steering_vector(mic, float(a), fb, center)      # [Fb, M]
            w = weights_fn(d)
            resp = np.abs(np.einsum("fm,fm->f", np.conj(w), d)) ** 2
            out.append(float(np.mean(resp)))
        return np.asarray(out)

    def ds_w(d):
        return d / d.shape[1]

    def mvdr_w(doa):
        def fn(d):
            return bf.beamforming_vector(doa, R_n)[band]
        return fn

    curves = [
        ("Delay-and-Sum (no directivity at 4 cm)", ds_w, C_GREY, 2.6),
        (f"MVDR, oracle DOA ({doa_true:.0f}°)", mvdr_w(doa_true), C_ORACLE, 2.2),
        (f"MVDR, GCC-PHAT DOA ({doa_est:.0f}°) → error {err:.0f}°",
         mvdr_w(doa_est), C_EST, 2.2),
    ]

    fig, ax = plt.subplots(figsize=(6.8, 6.0),
                           subplot_kw={"projection": "polar"})
    for label, fn, color, lw in curves:
        r = curve(fn)
        ax.plot(np.deg2rad(angles), r / r.max(), lw=lw, color=color,
                label=label, ls="--" if color == C_ORACLE else "-")
    ax.plot(np.deg2rad([doa_true] * 2), [0, 1.10], color="crimson", lw=1.8,
            ls=":", label=f"true DOA ({doa_true:.0f}°)")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(0, 1.10)
    ax.set_title(f"Beampattern — real noise field, 200–4000 Hz\n"
                 f"sample {row['file']} (SNR {float(row['snr_target_db']):+.0f} dB, "
                 f"T60 {float(row['t60']):.1f} s)",
                 fontsize=10.5, pad=18)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.26, -0.22), fontsize=8.5,
              frameon=False)
    fig.savefig(out, dpi=165, bbox_inches="tight")
    plt.close(fig)

    print(f"  beampattern sample: {row['file']}, true DOA {doa_true:.2f}°, "
          f"estimated {doa_est:.2f}° (error {err:.2f}°)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["doa-cost", "beampattern", "all"])
    ap.add_argument("--data", default="data/sim/test")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    enable_determinism()
    print(f"determinism: {determinism_status()}")
    fig_dir = _prepare()
    sim_cfg = SimConfig.from_yaml(REPO_ROOT / args.config)

    if args.which in ("doa-cost", "all"):
        print("collecting per-sample DOA error and MVDR metrics ...")
        rec = collect_doa_cost(REPO_ROOT / args.data, sim_cfg, limit=args.limit)
        p = figure_doa_cost(rec, fig_dir / "doa_cost.png")
        print(f"  -> {p}")
        d = rec["doa_err"]
        print(f"  DOA error: median {np.median(d):.2f}°, "
              f"p90 {np.percentile(d, 90):.2f}°, max {d.max():.2f}°")
        print(f"  SI-SDR: oracle {rec['sdr_oracle'].mean():.3f} vs "
              f"GCC-PHAT {rec['sdr_est'].mean():.3f} "
              f"(cost {rec['sdr_oracle'].mean() - rec['sdr_est'].mean():.3f} dB)")

    if args.which in ("beampattern", "all"):
        p = figure_beampattern(sim_cfg, fig_dir / "beampattern.png",
                               REPO_ROOT / args.data)
        print(f"  -> {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
