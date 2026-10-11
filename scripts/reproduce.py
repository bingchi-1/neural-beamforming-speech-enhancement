"""一键复现：下载 → 生成 → 训练 → 评测 → 出图。

实施计划 W6 的最后一项。

用法：

    python scripts/reproduce.py --smoke     # 端到端自检（几分钟，小规模）
    python scripts/reproduce.py all         # 完整复现（数小时，含两次训练）

设计要点
--------
* **幂等**：每个阶段先检查产物是否已存在，存在就跳过；`--force` 可强制重跑。
  这样中途失败后重跑不会白费前面的时间。
* **`--smoke` 用极小规模把整条链路走通**（含每阶段一次真实调用），
  用来验证「脚本本身没坏」，而不是产出可用的结果。
  README 里引用的数字必须来自 `all` 的完整运行，不能拿 smoke 的结果充数。
* 训练阶段用 `--resume`，中断后重跑会从 `last.pt` 接着走。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# (名称, 产物路径, 该阶段是否为目录)
STAGES = ["data", "generate", "train", "evaluate", "figures", "tests"]

# smoke 模式下的极小规模
SMOKE = {
    "train_n": 24, "val_n": 12, "test_utt": 2,
    "epochs": 1, "batch": 4, "gen_jobs": 4,
}


def _run(cmd: list[str], desc: str) -> None:
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO_ROOT)
    if r.returncode != 0:
        raise SystemExit(f"\n[失败] {desc}（退出码 {r.returncode}）")
    print(f"[完成] {desc}（{time.time() - t0:.1f} s）")


def _exists(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


def stage_data(args) -> None:
    """下载 LibriSpeech + DEMAND 语料（约 950 MB，幂等）。

    完成标记用**解包后的语料目录**（`data/LibriSpeech`、`data/DEMAND`），
    而不是下载的压缩包——压缩包在解包前就存在，拿它作标记会误判为已完成。
    """
    marker = REPO_ROOT / "data" / "LibriSpeech"
    marker2 = REPO_ROOT / "data" / "DEMAND"
    if _exists(marker, marker2) and not args.force:
        print(f"[跳过] 语料已就绪：{marker}、{marker2}")
        return
    _run([PY, "scripts/download_data.py"], "下载语料")


def stage_generate(args) -> None:
    """生成 train / val / test 三个 split 的多通道样本。

    **防误覆盖**：`--force --smoke` 会用十几条的小数据集覆盖现有的完整数据集，
    而 `results/final.csv` 与 README 的数字都依赖于那份完整测试集。
    这种覆盖是静默的、事后无法察觉，所以这里显式检查：若目标目录已有比本次
    将要生成的**更多**样本，就拒绝执行并说明原因。
    """
    import csv

    def manifest_count(d: Path) -> int:
        m = d / "manifest.csv"
        if not m.exists():
            return 0
        with open(m, newline="", encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))

    def need(split: Path, want: int) -> bool:
        have = manifest_count(split)
        if args.force and have > want:
            raise SystemExit(
                f"\n[拒绝执行] {split} 现有 {have} 条，而本次只会生成 {want} 条。\n"
                f"  覆盖会让 results/final.csv 与 README 引用的数字失效。\n"
                f"  若确实要替换，请先手动备份或删除 {split}，不要用 --force 硬来。")
        return args.force or have == 0

    if args.smoke:
        n_tr, n_va, n_te = SMOKE["train_n"], SMOKE["val_n"], SMOKE["test_utt"]
    else:
        n_tr, n_va, n_te = 3000, 120, 20

    jobs = str(SMOKE["gen_jobs"] if args.smoke else 8)

    if need(REPO_ROOT / "data/sim/train", n_tr):
        _run([PY, "scripts/generate_dataset.py", "--split", "train",
              "--n", str(n_tr), "--jobs", jobs], f"生成 train（{n_tr} 条）")
    else:
        print("[跳过] data/sim/train 已存在")

    if need(REPO_ROOT / "data/sim/val", n_va):
        _run([PY, "scripts/generate_dataset.py", "--split", "val",
              "--n", str(n_va), "--jobs", jobs], f"生成 val（{n_va} 条）")
    else:
        print("[跳过] data/sim/val 已存在")

    # test 用配对网格：n_te 个语料 × 3 SNR × 2 T60
    n_test = n_te * 6
    if need(REPO_ROOT / "data/sim/test", n_test):
        _run([PY, "scripts/generate_dataset.py", "--split", "test", "--grid",
              "--n-utterances", str(n_te), "--snr", "-5", "0", "5",
              "--t60", "0.3", "0.6", "--jobs", jobs],
             f"生成 test 配对网格（{n_te} 语料 × 6 条件 = {n_test} 条）")
    else:
        print("[跳过] data/sim/test 已存在")


def stage_train(args) -> None:
    """训练两个模型：多通道 CRN（掩蔽预测器）与单通道 DCCRN-lite。"""
    common = ["--data", "data/sim/train", "--val", "data/sim/val"]
    if args.smoke:
        common += ["--limit", str(SMOKE["train_n"]),
                   "--epochs", str(SMOKE["epochs"]),
                   "--batch-size", str(SMOKE["batch"])]

    for script, out, extra in (
        ("train.py", "runs/w4", []),
        ("train_dccrn.py", "runs/dccrn", []),
    ):
        best = REPO_ROOT / out / "best.pt"
        if best.exists() and not args.force:
            print(f"[跳过] {best} 已存在")
            continue
        cmd = [PY, script, *common, "--out", out, *extra]
        resume = REPO_ROOT / out / "last.pt"
        if resume.exists():
            cmd += ["--resume", str(resume)]
            print(f"[续训] 发现 {resume}")
        _run(cmd, f"训练 {script} -> {out}")


def stage_evaluate(args) -> None:
    """跑七行对比表。"""
    out = REPO_ROOT / "results" / "final.csv"
    if out.exists() and not args.force:
        print(f"[跳过] {out} 已存在")
        return
    cmd = [PY, "eval.py", "--data", "data/sim/test",
           "--ckpt", "runs/w4/best.pt", "--dccrn", "runs/dccrn/best.pt",
           "--out", "results/final.csv"]
    _run(cmd, "评测七行对比表")


def stage_figures(args) -> None:
    _run([PY, "scripts/make_figures.py", "all"], "生成图表")


def stage_tests(args) -> None:
    _run([PY, "-m", "pytest", "tests/", "-q"], "运行测试")


HANDLERS = {
    "data": stage_data, "generate": stage_generate, "train": stage_train,
    "evaluate": stage_evaluate, "figures": stage_figures, "tests": stage_tests,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["all", *STAGES],
                    help="要执行的阶段；all 表示按顺序全部执行")
    ap.add_argument("--smoke", action="store_true",
                    help="极小规模端到端自检（验证脚本本身，不产出可用结果）")
    ap.add_argument("--force", action="store_true",
                    help="忽略已有产物，强制重跑")
    args = ap.parse_args()

    # 缺 python 依赖时早失败，避免跑了几十分钟才在某个阶段挂掉
    import importlib.util

    missing = [m for m in ("numpy", "scipy", "torch", "pyroomacoustics",
                           "matplotlib", "pytest")
               if importlib.util.find_spec(m) is None]
    if missing:
        print(f"[警告] 缺少依赖：{missing}\n"
              f"        先执行 pip install -r requirements.txt", file=sys.stderr)

    stages = STAGES if args.stage == "all" else [args.stage]
    print("=" * 72)
    print(f"复现模式：{'SMOKE（自检）' if args.smoke else '完整'}")
    print(f"阶段：{' → '.join(stages)}")
    print(f"仓库：{REPO_ROOT}")
    print("=" * 72)

    t0 = time.time()
    for s in stages:
        print(f"\n{'─' * 72}\n[{s}]")
        HANDLERS[s](args)

    print(f"\n{'=' * 72}")
    print(f"全部完成，用时 {(time.time() - t0) / 60:.1f} 分钟")
    if args.smoke:
        print("\n注意：这是 SMOKE 运行，产物是小规模的，**不能**用于引用数字。")
        print("要得到 README 里那些结果，请跑 `python scripts/reproduce.py all`。")
    else:
        print(f"\n结果表：{REPO_ROOT / 'results' / 'final.csv'}")
        print(f"图表  ：{REPO_ROOT / 'docs' / 'figures'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
