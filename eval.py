"""T2.4 评测入口：在评估网格上跑齐各方法，产出结果表。

用法：

    # 1) 先生成评估网格（3 档 SNR × 2 档 T60，配对：同语料同房间同噪声）
    python scripts/generate_dataset.py --split test --grid --n-utterances 20 \
        --snr -5 0 5 --t60 0.3 0.6

    # 2) 评测
    python eval.py --data data/sim/test --out results/w2_baseline.csv

    # 冒烟：每个条件只取 2 条，几秒钟出结果
    python eval.py --data data/sim/test --limit 2

设计文档 §6（评估设计），实施计划 T2.4。保底交付点即本脚本产出的
`results/w2_baseline.csv`。
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
from nbse.evaluate import (  # noqa: E402
    DEFAULT_METHODS,
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

    print(f"样本目录：{data_dir}")
    print(f"样本数：{len(rows)}")
    print(f"条件：{', '.join(f'SNR{s:+.0f}/T60 {t:.1f}' for s, t in list_conditions(rows))}")
    if args.limit:
        print(f"冒烟模式：每个条件只取前 {args.limit} 条")
    print(f"方法：{', '.join(DEFAULT_METHODS)}")
    print("-" * 72)

    t0 = time.time()

    def _progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            sys.stdout.write(f"\r  评测中 {done}/{total} 条"
                             f"（{time.time() - t0:.0f} s）")
            sys.stdout.flush()

    records = run_methods(data_dir, rows, sim_cfg,
                          limit_per_condition=args.limit, progress=_progress)
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
