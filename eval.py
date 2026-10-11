"""评测入口：在评估网格上跑齐各方法，产出结果表。

用法：

    # 1) 先生成评估网格（3 档 SNR × 2 档 T60，配对：同语料同房间同噪声）
    python scripts/generate_dataset.py --split test --grid --n-utterances 20 \
        --snr -5 0 5 --t60 0.3 0.6

    # 2) 传统波束形成基线（T2.4 的保底交付点）
    python eval.py --data data/sim/test --out results/w2_baseline.csv

    # 3) 加上 mask-MVDR（需要 T4.2 训练好的 checkpoint）
    python eval.py --data data/sim/test --ckpt runs/w4/best.pt \
        --out results/w4_nn.csv

    # 4) 完整七行（设计文档 §6.3，T5.3 的最终对比表）
    python eval.py --data data/sim/test \
        --ckpt runs/w4/best.pt --dccrn runs/dccrn/best.pt \
        --out results/final.csv

    # 冒烟：每个条件只取 2 条，几秒钟出结果
    python eval.py --data data/sim/test --limit 2

设计文档 §6（评估设计）。`results/w2_baseline.csv` 是 T2.4 的保底交付点；
`results/w4_nn.csv` 是 T4.3 的验收对象；`results/final.csv` 是 T5.3 的七行表。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from nbse.data.generate import load_manifest  # noqa: E402
from nbse.data.simulate import SimConfig  # noqa: E402
from nbse.dsp import N_FFT  # noqa: E402
from nbse.evaluate import (  # noqa: E402
    ALL_METHOD_NAMES,
    DEFAULT_METHODS,
    FINAL_METHODS,
    aggregate,
    check_expected_order,
    format_table,
    list_conditions,
    run_methods,
    write_csv,
)

METRICS = [("si_sdr", "SI-SDR (dB)"), ("stoi", "STOI"), ("pesq_wb", "PESQ-WB")]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/sim/test",
                    help="已生成的样本目录（含 manifest.csv）")
    ap.add_argument("--out", default="results/w2_baseline.csv")
    ap.add_argument("--limit", type=int, default=None,
                    help="每个条件只取前 N 条，用于冒烟测试")
    ap.add_argument("--ckpt", default=None,
                    help="训练好的 CRN checkpoint（train.py 产出）。"
                         "给了它才会评测 mask-MVDR 两条方法")
    ap.add_argument("--dccrn", default=None,
                    help="训练好的 DCCRN-lite checkpoint（train_dccrn.py 产出）。"
                         "给了它才会评测 DCCRN-lite（单通道对照）")
    ap.add_argument("--dccrn-base-channels", type=int, default=16,
                    help="DCCRN checkpoint 对应的 base_channels（须与训练时一致）")
    ap.add_argument("--model", default="default",
                    help="checkpoint 对应的模型预设：default | small")
    ap.add_argument("--methods", default=None,
                    help="逗号分隔的方法名；默认按是否给 --ckpt 自动决定")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data_dir = _resolve(args.data)
    if not (data_dir / "manifest.csv").exists():
        print(f"找不到 {data_dir / 'manifest.csv'}\n\n"
              f"请先生成评估网格：\n"
              f"  python scripts/generate_dataset.py --split test --grid \\\n"
              f"      --n-utterances 20 --snr -5 0 5 --t60 0.3 0.6\n",
              file=sys.stderr)
        return 1

    sim_cfg = SimConfig.from_yaml(REPO_ROOT / args.config)
    rows = load_manifest(data_dir)

    # 掩蔽预测器：只有给了 checkpoint 才构建，也才注册 mask-MVDR
    mask_predictor = None
    if args.ckpt:
        from nbse.beamformers.mask_mvdr import TorchMaskPredictor
        from nbse.determinism import determinism_status, enable_determinism

        ckpt_path = _resolve(args.ckpt)
        if not ckpt_path.exists():
            print(f"找不到 checkpoint：{ckpt_path}", file=sys.stderr)
            return 1
        # 必须先开确定性再构建 predictor：cuDNN 的非确定性卷积会让同一
        # checkpoint 每次评测得到不同结果表，使结论不可复核。
        enable_determinism()
        mask_predictor = TorchMaskPredictor.from_checkpoint(
            ckpt_path, preset=args.model, device=args.device,
            n_mics=sim_cfg.array.n_mics, n_freqs=N_FFT // 2 + 1)
        print(f"掩蔽预测器：{ckpt_path}（预设 {args.model}，设备 {args.device}）")
        print(f"确定性计算：{determinism_status()}")

    # 单通道 DCCRN-lite（T5.2）：给了 checkpoint 才注册该方法
    dccrn = None
    if args.dccrn:
        from nbse.determinism import enable_determinism
        from nbse.models.dccrn_data import TorchDCCRN

        d_path = _resolve(args.dccrn)
        if not d_path.exists():
            print(f"找不到 DCCRN checkpoint：{d_path}", file=sys.stderr)
            return 1
        enable_determinism()
        dccrn = TorchDCCRN.from_checkpoint(
            d_path, device=args.device, n_freqs=N_FFT // 2 + 1,
            base_channels=args.dccrn_base_channels)
        print(f"DCCRN-lite：{d_path}（设备 {args.device}）")

    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        # 默认按「有哪些资源」决定跑多少行：
        #   两个 checkpoint 都给 → 设计文档 §6.3 的完整七行
        #   只给 CRN           → 基线 + mask-MVDR 两条
        #   都不给             → 纯基线四条
        if mask_predictor is not None and dccrn is not None:
            methods = FINAL_METHODS
        elif mask_predictor is not None:
            methods = [m for m in ALL_METHOD_NAMES if m != "DCCRN-lite"]
        else:
            methods = DEFAULT_METHODS

    print(f"样本目录：{data_dir}")
    print(f"样本数：{len(rows)}")
    print(f"条件：{', '.join(f'SNR{s:+.0f}/T60 {t:.1f}' for s, t in list_conditions(rows))}")
    if args.limit:
        print(f"冒烟模式：每个条件只取前 {args.limit} 条")
    print(f"方法：{', '.join(methods)}")
    print("-" * 72)

    t0 = time.time()

    def _progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            sys.stdout.write(f"\r  评测中 {done}/{total} 条"
                             f"（{time.time() - t0:.0f} s）")
            sys.stdout.flush()

    records = run_methods(data_dir, rows, sim_cfg, methods=methods,
                          limit_per_condition=args.limit, progress=_progress,
                          mask_predictor=mask_predictor, dccrn=dccrn)
    print(f"\n完成，用时 {time.time() - t0:.1f} s")

    table_rows = aggregate(records)
    out_path = write_csv(table_rows, _resolve(args.out))

    for metric, title in METRICS:
        print(f"\n{title}")
        print(format_table(table_rows, metric))

    ok, chain = check_expected_order(table_rows, "si_sdr")
    print("\n指标顺序检查（设计文档 §3.2 第 2 条，按 SI-SDR）：")
    print(f"  {chain}")
    # 不要用 ✓/✗：Windows 控制台默认 GBK 编码，这些符号会直接让脚本崩溃
    print("  通过" if ok else "  不通过 —— 不要把这个结果带进 W3，先排查")

    print(f"\n结果表：{out_path}")
    return 0 if ok else 2


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


if __name__ == "__main__":
    sys.exit(main())
