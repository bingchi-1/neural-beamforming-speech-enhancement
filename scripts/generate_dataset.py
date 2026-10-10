"""T1.4：用真实语料批量生成多通道样本。

用法：

    # W1 里程碑的第一批：训练说话人 240 条，房间与 SNR 随机
    python scripts/generate_dataset.py --split train --n 240

    # 评估用：测试说话人，固定 3 档 SNR × 2 档 T60（对应设计文档 §6.2）
    python scripts/generate_dataset.py --split test --n 240 --snr -5 0 5 --t60 0.3 0.6

    # 前 3 条另写 16-bit wav，方便用播放器人工抽查混合信号/干净参考
    python scripts/generate_dataset.py --split val --n 20 --wav-preview 3

    # 只做验收核对（不生成）
    python scripts/generate_dataset.py --verify-only data/sim/train

输出目录（默认 data/sim/<split>/）：
    sample_00000.npz ...    每个样本一个 npz
    manifest.csv            每条的语料/噪声/SNR/DOA/房间参数
    run.json                本次生成的完整参数，便于复现
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nbse.data.corpus import (  # noqa: E402
    list_noise_files,
    utterances_for_split,
)
from nbse.data.generate import (  # noqa: E402
    BatchSpec,
    generate_batch,
    generate_grid,
    load_manifest,
    verify_batch,
)
from nbse.data.simulate import SimConfig  # noqa: E402


def _resolve(p: str | Path) -> Path:
    """配置里的路径都相对仓库根目录。"""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=200, help="生成样本条数（T1.4 要求 ≥200）")
    ap.add_argument("--grid", action="store_true",
                    help="配对评估网格模式：语料数次 --n-utterances × SNR 档 × T60 档，"
                         "同一语料在各条件下复用同一房间几何与噪声（需同时给 --snr 与 --t60）")
    ap.add_argument("--n-utterances", type=int, default=20,
                    help="网格模式下每个条件用多少条语料")
    ap.add_argument("--seed", type=int, default=None, help="默认取配置里的 seed")
    ap.add_argument("--out", default=None, help="默认 data/sim/<split>")
    ap.add_argument("--snr", type=float, nargs="+", default=None,
                    help="固定 SNR 档位（dB）；不给则按配置随机")
    ap.add_argument("--t60", type=float, nargs="+", default=None,
                    help="固定 T60 档位（秒）；不给则随机")
    ap.add_argument("--wav-preview", type=int, default=0,
                    help="额外为前 N 条写 16-bit wav（试听用）")
    ap.add_argument("--verify-only", metavar="DIR", default=None,
                    help="只对已有目录做验收核对，不生成")
    args = ap.parse_args()

    if args.verify_only:
        summary = verify_batch(args.verify_only)
        print("验收通过：")
        for k, v in summary.items():
            print(f"  {k:22s} {v}")
        return 0

    raw = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    sim_cfg = SimConfig.from_yaml(REPO_ROOT / args.config)
    seed = raw["seed"] if args.seed is None else args.seed

    ls_root = _resolve(raw["data"]["librispeech_root"])
    utterances = utterances_for_split(
        ls_root, args.split,
        n_train=raw["data"]["split"]["train_speakers"],
        n_val=raw["data"]["split"]["val_speakers"],
    )
    noise_files = list_noise_files(_resolve(raw["data"]["demand_root"]))

    out_dir = Path(args.out) if args.out else REPO_ROOT / "data" / "sim" / args.split
    cache_dir = _resolve(raw["data"]["rir_cache"])

    print(f"切分 {args.split}：{len(utterances)} 条语音 / "
          f"{len({u.speaker for u in utterances})} 个说话人")
    print(f"噪声：{len(noise_files)} 条（{', '.join(sorted({p.parent.name for p in noise_files}))}）")
    print(f"输出：{out_dir}")
    print(f"RIR 缓存：{cache_dir}")
    print(f"参数：n={args.n} seed={seed} "
          f"snr={'随机' if args.snr is None else args.snr} "
          f"t60={'随机' if args.t60 is None else args.t60}")
    print("-" * 60)

    spec = BatchSpec(
        out_dir=out_dir,
        n_samples=args.n,
        seed=int(seed),
        snr_levels=None if args.snr is None else tuple(args.snr),
        t60_levels=None if args.t60 is None else tuple(args.t60),
        wav_preview=args.wav_preview,
    )

    t0 = time.time()

    def _progress(done: int, total: int) -> None:
        if done % 20 == 0 or done == total:
            sys.stdout.write(f"\r  已生成 {done}/{total}  "
                             f"（{time.time() - t0:.0f} s）")
            sys.stdout.flush()

    if args.grid:
        if args.snr is None or args.t60 is None:
            ap.error("--grid 需要同时给出 --snr 与 --t60（配对网格没有「随机」这一档）")
        rows = generate_grid(utterances, noise_files, sim_cfg, spec,
                             n_utterances=args.n_utterances,
                             cache_dir=cache_dir, progress=_progress)
    else:
        rows = generate_batch(utterances, noise_files, sim_cfg, spec,
                              cache_dir=cache_dir, progress=_progress)
    print(f"\n完成，用时 {time.time() - t0:.1f} s")

    # 落盘核对（T1.4 验收）
    summary = verify_batch(out_dir)
    print("\n验收（形状 / 采样率 / 时长一致性）：")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")

    # SNR 标定精度：T1.3 承诺 ±0.5 dB，在整批规模上复核一次
    errs = [abs(float(r["snr_db"]) - float(r["snr_target_db"]))
            for r in rows if r["snr_target_db"] != ""]
    if errs:
        print(f"\nSNR 标定：{len(errs)} 条有目标档位，"
              f"最大误差 {max(errs):.3f} dB，平均 {sum(errs) / len(errs):.3f} dB")

    print(f"\nmanifest：{out_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
