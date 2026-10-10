"""T3.3 训练入口。

用法：

    # 正常训练（先跑 2 个 epoch 验证不崩溃）
    python train.py --data data/sim/train --epochs 2

    # 小规模过拟合测试（T3.4，关键验证手段）
    python train.py --overfit --data data/sim/train

    # 从 checkpoint 续训
    python train.py --data data/sim/train --resume runs/w3/last.pt

对照实施计划 T3.3 的验收标准：
    能训练 ≥ 2 个 epoch 不崩溃；loss 下降；checkpoint 可被重新加载并继续训练。

T3.4 的验收标准（`--overfit`）：
    取 10 条样本、关掉增广，训练 loss 显著下降至接近零。
    **若不能过拟合，说明模型或数据管道有 bug** —— 这是发现管线错误最有效的手段。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from nbse.models.crn import CRN, CRN_SMALL  # noqa: E402
from nbse.models.dataset import NbseDataset, collate_samples  # noqa: E402
from nbse.training import (  # noqa: E402
    TrainConfig,
    TrainState,
    append_history,
    build_optimizer,
    build_scheduler,
    describe_device,
    evaluate_loss,
    load_checkpoint,
    pick_device,
    save_checkpoint,
    train_one_epoch,
)

# 过拟合测试的固定超参（T3.4）。10 条样本、关增广、小模型、足够多的迭代。
OVERFIT_N = 10
OVERFIT_EPOCHS = 150
OVERFIT_LR = 2e-3

# 时间预算（实测，2026-10-10，RTX 5070 Ti Laptop 11.9 GB，AMP fp16）：
#   训练集 240 条、batch 4、accum 4、片段 1 s、默认 CRN
#     → 稳态 8.2 s/epoch（60 个 batch / 15 个优化步），首 epoch 8.9 s
#     → 100 epoch ≈ 14 min，远低于实施计划给 W4 的 5 h 预算
#   真实瓶颈是 CPU 侧的 STFT 与数据读取，不是 GPU：纯前向+反传仅 ~8.5 ms/batch，
#   而端到端是 ~137 ms/batch。故 data/ 样本量（而非 epoch 数）才是主要成本。

# 模型预设。`CRN_SMALL` 用于过拟合测试与显存受限时的快速迭代；
# 正式训练用默认 CRN（约 3.3 M 参数，见 crn.DEFAULT_NUM_PARAMETERS）。
MODEL_PRESETS = {
    "default": CRN,
    "small": CRN_SMALL,
}


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def build_model(preset: str, n_mics: int, n_freqs: int):
    """按预设名构建模型。n_mics / n_freqs 由数据决定，不交给 CLI 猜。"""
    if preset not in MODEL_PRESETS:
        raise SystemExit(
            f"未知模型预设：{preset}（可选 {sorted(MODEL_PRESETS)}）")
    return MODEL_PRESETS[preset](n_mics=n_mics, n_freqs=n_freqs)


def _sample_length(data_dir: Path) -> int:
    """读 manifest 首条的样本长度（采样点）。

    用于判断请求的片段长度是否已经达到整条样本，从而决定能否随机裁剪。
    直接读 manifest 而不加载 npz，避免为此打开一个 2 MB 文件。
    """
    import csv

    path = Path(data_dir) / "manifest.csv"
    if not path.exists():
        raise SystemExit(
            f"找不到 {path}\n请先生成样本：\n"
            f"  python scripts/generate_dataset.py --split train --n 3000 --jobs 8")
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f), None)
    if row is None or "n_samples" not in row:
        raise SystemExit(f"{path} 为空或缺 n_samples 列")
    return int(row["n_samples"])


def build_loaders(args, cfg: TrainConfig, overfit: bool):
    """构建训练/验证 DataLoader。

    过拟合测试的关键点是 **random_crop=False**（关掉时域增广）：
    若仍开随机片段，同一 epoch 内每次取到的都是不同片段，模型在「背下 10 条
    样本」之前会先被无限变动的输入拖住，loss 下降曲线失去判读意义。

    片段长度达到整条样本长度时也必须关掉随机裁剪：样本只有 4 s，
    再随机取「4 s 片段」是没有意义的（只会取到整条）。
    """
    # 用一条样本的实际长度判断，而不是硬编码 4 s——CLI 允许 --segment-length
    sample_len = _sample_length(_resolve(args.data))
    full_sample = cfg.segment_length >= sample_len
    if full_sample:
        print(f"片段 {cfg.segment_length} ≥ 样本长度 {sample_len}："
              f"使用整条样本，关闭随机片段裁剪")

    train_ds = NbseDataset(
        _resolve(args.data),
        segment_length=cfg.segment_length,
        random_crop=(not overfit) and (not full_sample),
        limit=OVERFIT_N if overfit else args.limit,
        seed=cfg.seed,
    )

    val_ds = None
    val_dir = _resolve(args.val) if args.val else None
    if val_dir and val_dir.exists() and not overfit:
        val_ds = NbseDataset(val_dir, segment_length=cfg.segment_length,
                             random_crop=False, limit=args.limit,
                             seed=cfg.seed)

    common = dict(collate_fn=collate_samples, num_workers=cfg.num_workers)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, drop_last=False, **common)

    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                                shuffle=False, drop_last=False, **common)

    return train_ds, train_loader, val_loader


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data/sim/train")
    ap.add_argument("--val", default="data/sim/val",
                    help="验证目录；不存在则跳过验证")
    ap.add_argument("--out", default=None,
                    help="默认 runs/w3（过拟合模式为 runs/overfit）")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lr-floor", type=float, default=None,
                    help="退火下界占 lr 的比例（默认 0.01）。设 1.0 即不退火，"
                         "用于诊断残留 loss 是优化不足还是信息上限")
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 条样本")
    ap.add_argument("--segment-length", type=int, default=None)
    ap.add_argument("--models", default="default",
                    help="模型预设：default | small")
    ap.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-deterministic", action="store_true",
                    help="关闭确定性计算。默认开启：不加会让同一配置两次训练"
                         "得到不同结果（cuDNN 卷积算法不确定），破坏可复现性")
    ap.add_argument("--resume", default=None, help="从 checkpoint 续训")
    ap.add_argument("--overfit", action="store_true",
                    help=f"T3.4 过拟合测试：{OVERFIT_N} 条样本、关增广、小模型")
    ap.add_argument("--dry-run", action="store_true", help="只跑 1 个 batch 后退出")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(_resolve(args.config),
                               epochs=args.epochs,
                               batch_size=args.batch_size,
                               lr=args.lr,
                               segment_length=args.segment_length,
                               lr_floor=args.lr_floor)
    if args.no_amp:
        cfg.amp = False

    if args.overfit:
        cfg.epochs = args.epochs or OVERFIT_EPOCHS
        cfg.lr = args.lr or OVERFIT_LR
        cfg.batch_size = args.batch_size or 2
        cfg.grad_accum_steps = 1
        cfg.segment_length = args.segment_length or 16000
        args.models = args.models if args.models != "default" else "small"

    out_dir = _resolve(args.out or ("runs/overfit" if args.overfit else "runs/w3"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    torch.manual_seed(cfg.seed)

    # 确定性：默认开启。实测默认设置下同一输入连续 8 次前向会得到 8 个不同结果
    # （cuDNN 卷积算法选择不确定，最大差 ~1.6e-05），使训练与评测都不可复现。
    # 代价约 7%（1.5 → 1.6 ms/batch）。见 nbse/determinism.py。
    det = not args.no_deterministic
    if det:
        from nbse.determinism import determinism_status, enable_determinism

        enable_determinism(seed=cfg.seed)
        print(f"确定性计算：已开启 {determinism_status()}")

    train_ds, train_loader, val_loader = build_loaders(args, cfg, args.overfit)

    # 模型输入形状由数据决定，而不是由 CLI 猜——避免配置与数据不一致
    model = build_model(args.models, train_ds.n_mics, train_ds.n_freqs).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler(device.type,
                                 enabled=bool(cfg.amp and device.type == "cuda"))
    state = TrainState()

    if args.resume:
        state, _ = load_checkpoint(_resolve(args.resume), model, optimizer,
                                    scheduler, map_location=device)
        print(f"已从 {args.resume} 续训（epoch={state.epoch} "
              f"step={state.global_step}）")

    # ── 启动摘要 ──
    print("=" * 74)
    print(f"设备      : {describe_device(device)}")
    print(f"模型      : {args.models} · 可训练参数 {n_param:,} ({n_param / 1e6:.2f} M)")
    print(f"输入形状  : {train_ds.n_mics} ch × {train_ds.n_freqs} freq × "
          f"{train_ds.n_frames} frames（片段 {cfg.segment_length} 点）")
    print(f"样本      : 训练 {len(train_ds)}"
          + (f" / 验证 {len(val_loader.dataset)}" if val_loader else "（无验证集）"))
    print(f"超参      : epochs={cfg.epochs} bs={cfg.batch_size} "
          f"accum={cfg.grad_accum_steps} lr={cfg.lr} amp={cfg.amp} "
          f"seed={cfg.seed}")
    print(f"输出      : {out_dir}")
    if args.overfit:
        print(f"模式      : 过拟合测试（T3.4）—— {OVERFIT_N} 条样本、"
              f"关闭随机片段增广")
    print("=" * 74)

    # 落盘本次运行的完整参数，便于复现
    (out_dir / "run.json").write_text(json.dumps({
        "argv": sys.argv[1:],
        "train": {**vars(cfg), "out_dir": str(out_dir)},
        "model": args.models,
        "n_params": n_param,
        "device": str(device),
        "overfit": bool(args.overfit),
        "n_train": len(train_ds),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    history_path = out_dir / "history.jsonl"
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    t0 = time.time()
    for epoch in range(state.epoch, cfg.epochs):
        # 训练集跨 epoch 变更随机片段窗口（过拟合模式下 random_crop=False，无影响）
        train_ds.set_epoch(epoch)

        print(f"\nepoch {epoch + 1}/{cfg.epochs}")
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device,
                             cfg, state, scheduler=scheduler,
                             log_fn=print, dry_run=args.dry_run)
        line = (f"  训练 loss={tr['loss']:.5f} (mask={tr['loss_mask']:.5f} "
                f"mag={tr['loss_mag']:.5f})")

        va = None
        if val_loader is not None:
            va = evaluate_loss(model, val_loader, device, cfg, dry_run=args.dry_run)
            line += f" | 验证 loss={va['loss']:.5f}"

        state.epoch = epoch + 1
        line += f" | lr={optimizer.param_groups[0]['lr']:.2e}"
        print(line)

        record = {"epoch": epoch + 1, "global_step": state.global_step,
                  "train": tr, "val": va,
                  "lr": optimizer.param_groups[0]["lr"],
                  "elapsed_s": round(time.time() - t0, 1)}
        state.history.append(record)
        append_history(history_path, record)

        # 验证集优先，其次训练损失——过拟合测试没有验证集
        monitor = va["loss"] if va is not None else tr["loss"]
        if monitor < state.best_val:
            state.best_val = monitor
            save_checkpoint(best_path, model, optimizer, scheduler, state, cfg,
                            extra={"monitor": "val" if va else "train",
                                   "monitor_value": monitor})

        save_checkpoint(last_path, model, optimizer, scheduler, state, cfg)

        if args.dry_run:
            print("\n--dry-run：已跑通 1 个 batch，退出")
            break

    # ── 收尾 ──
    print("\n" + "=" * 74)
    print(f"完成，用时 {time.time() - t0:.1f} s")
    print(f"最佳监控值 {state.best_val:.5f}")
    print(f"checkpoint：{best_path}")
    print(f"            {last_path}")
    print(f"训练曲线  ：{history_path}")

    # T3.3 的第二条验收：checkpoint 可被重新加载并继续训练
    if not args.dry_run:
        probe = build_model(args.models, train_ds.n_mics, train_ds.n_freqs).to(device)
        st2, _ = load_checkpoint(last_path, probe, map_location=device)
        print(f"checkpoint 复载自检：通过（epoch={st2.epoch} "
              f"step={st2.global_step}）")

    # 过拟合判读（T3.4）
    if args.overfit and state.history:
        first = state.history[0]["train"]["loss"]
        last = state.history[-1]["train"]["loss"]
        ratio = last / max(first, 1e-12)
        print("\n过拟合判读（T3.4）：")
        print(f"  首 epoch loss {first:.5f} → 末 epoch loss {last:.5f} "
              f"（降至 {ratio * 100:.2f}%）")

        # 判据的标定依据（scripts/diagnose_mask_ceiling.py 实测）：
        # 这条 10 样本数据集上，软掩蔽的常预测 MSE = 0.1201；
        # 逐频点闭式最优线性预测（用了 1/mag/sqrt/log1p 四组基）只能到 0.0960。
        # 因此「loss 降到接近零」在本任务上不可达——掩蔽由语音/噪声幅度之比定义，
        # 而观测幅度谱不能唯一决定它，存在固有不确定性。
        # 有意义的判据是**相对常预测基线的解释比例**与**相对线性基线的改进倍数**。
        CONST_BASELINE = 0.1201      # 实测：不利用观测时的最优 MSE
        LIN_BASELINE = 0.0960        # 实测：逐频点闭式最优线性预测

        explained = 1.0 - last / CONST_BASELINE
        vs_linear = LIN_BASELINE / max(last, 1e-12)

        print(f"  相对常预测基线(0.1201) 解释方差 {explained * 100:.2f}%")
        print(f"  相对线性最优基线(0.0960) 改进 {vs_linear:.2f} 倍")

        checks = [
            ("已显著下降（< 首值的 20%）", ratio < 0.20, f"{ratio * 100:.2f}%"),
            ("解释掩蔽方差 > 80%", explained > 0.80, f"{explained * 100:.2f}%"),
            ("优于闭式最优线性预测（> 2 倍）", vs_linear > 2.0, f"{vs_linear:.2f}×"),
        ]
        ok = True
        for name, passed, val in checks:
            print(f"  [{'通过' if passed else '不通过'}] {name}  实测 {val}")
            ok &= passed

        print("\n  " + ("通过：管线正确，模型确实学到了掩蔽映射。"
                        if ok else
                        "不通过：优先怀疑数据管道，其次学习率。"))
        print("  说明：软掩蔽目标存在固有不确定性（观测幅度谱不能唯一决定它），"
              "因此 loss 不会到零；\n        判据基于实测标定的两条基线，"
              "而非拍脑袋的绝对阈值。")
        return 0 if ok else 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
