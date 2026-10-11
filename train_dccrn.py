"""T5.2 训练入口：DCCRN-lite 单通道增强。

用法：

    python train_dccrn.py --data data/sim/train --val data/sim/val --epochs 100
    python train_dccrn.py --overfit --data data/sim/train     # 管线自检

对照实施计划 T5.2 的验收标准：
    在相同测试集上产出 PESQ / STOI，且**优于 Noisy 基线**。

与 `train.py`（多通道 CRN）的关系
--------------------------------
复用 `nbse.training` 的训练循环、优化器、调度器、checkpoint 与早停
—— 这些与模型无关。差异只在**数据契约**（复谱对 vs 幅度谱+掩蔽）与
**损失**（复谱重建 vs 掩蔽回归），故单独一个入口而共用底层。
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

from nbse.models.dccrn_data import (  # noqa: E402
    DCCRNDataset,
    collate_pairs,
)
from nbse.models.dccrn_lite import (  # noqa: E402
    DCCRNLite,
    apply_complex_mask,
    dccrn_lite_loss,
)
from nbse.training import (  # noqa: E402
    EarlyStopState,
    TrainConfig,
    TrainState,
    append_history,
    build_optimizer,
    build_scheduler,
    describe_device,
    load_checkpoint,
    pick_device,
    save_checkpoint,
)

OVERFIT_N = 10
OVERFIT_EPOCHS = 150
OVERFIT_LR = 2e-3


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _forward_loss(model, inputs, targets, cfg):
    """一次前向 + 损失，返回 (loss, parts, enhanced_r, enhanced_i)。"""
    with torch.autocast(device_type="cuda", enabled=cfg.amp,
                        dtype=torch.float16):
        mr, mi = model(inputs["mix_r"], inputs["mix_i"])
        er, ei = apply_complex_mask(inputs["mix_r"], inputs["mix_i"], mr, mi)
        loss, parts = dccrn_lite_loss(er, ei, targets["clean_r"],
                                      targets["clean_i"])
    return loss, parts, er, ei


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, state,
                    scheduler=None, log_fn=print, dry_run=False):
    model.train()
    totals: dict[str, float] = {}
    n_batches = n_steps = 0
    accum = max(int(cfg.grad_accum_steps), 1)
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    for i, (inputs, targets) in enumerate(loader):
        for k in ("mix_r", "mix_i", "clean_r", "clean_i"):
            if k in inputs:
                inputs[k] = inputs[k].to(device, non_blocking=True)
            if k in targets:
                targets[k] = targets[k].to(device, non_blocking=True)

        loss, parts, _, _ = _forward_loss(model, inputs, targets, cfg)
        scaled = loss / accum
        if scaler.is_enabled():
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (i + 1) % accum == 0 or (i + 1) == len(loader):
            if cfg.grad_clip and cfg.grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            state.global_step += 1
            n_steps += 1

        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1
        if cfg.log_every > 0 and (i + 1) % cfg.log_every == 0:
            log_fn(f"  [{i + 1}/{len(loader)}] loss={totals['loss'] / n_batches:.5f} "
                   f"({time.time() - t0:.1f}s)")
        if dry_run:
            break

    if scheduler is not None and n_steps > 0:
        scheduler.step()
    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_loss(model, loader, device, cfg, dry_run=False):
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    for i, (inputs, targets) in enumerate(loader):
        for k in ("mix_r", "mix_i", "clean_r", "clean_i"):
            if k in inputs:
                inputs[k] = inputs[k].to(device)
            if k in targets:
                targets[k] = targets[k].to(device)
        _, parts, _, _ = _forward_loss(model, inputs, targets, cfg)
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
        if dry_run:
            break
    return {k: v / max(n, 1) for k, v in totals.items()}


def build_loaders(args, cfg, overfit):
    train_ds = DCCRNDataset(_resolve(args.data), segment_length=cfg.segment_length,
                            random_crop=not overfit,
                            limit=OVERFIT_N if overfit else args.limit,
                            seed=cfg.seed)
    val_ds = None
    val_dir = _resolve(args.val) if args.val else None
    if val_dir and val_dir.exists() and not overfit:
        val_ds = DCCRNDataset(val_dir, segment_length=cfg.segment_length,
                              random_crop=False, limit=args.limit, seed=cfg.seed)

    common = dict(collate_fn=collate_pairs, num_workers=cfg.num_workers)
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
    ap.add_argument("--val", default="data/sim/val")
    ap.add_argument("--out", default=None,
                    help="默认 runs/dccrn（过拟合模式为 runs/dccrn_overfit）")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--segment-length", type=int, default=None)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument("--lstm-hidden", type=int, default=128)
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-deterministic", action="store_true")
    ap.add_argument("--early-stop-patience", type=int, default=None)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(_resolve(args.config), epochs=args.epochs,
                               batch_size=args.batch_size, lr=args.lr,
                               segment_length=args.segment_length,
                               early_stop_patience=args.early_stop_patience)
    if args.no_amp:
        cfg.amp = False
    if args.overfit:
        cfg.epochs = args.epochs or OVERFIT_EPOCHS
        cfg.lr = args.lr or OVERFIT_LR
        cfg.batch_size = args.batch_size or 4
        cfg.grad_accum_steps = 1
        cfg.segment_length = args.segment_length or 16000

    out_dir = _resolve(args.out or ("runs/dccrn_overfit" if args.overfit
                                    else "runs/dccrn"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    torch.manual_seed(cfg.seed)
    if not args.no_deterministic:
        from nbse.determinism import determinism_status, enable_determinism

        enable_determinism(seed=cfg.seed)
        print(f"确定性计算：已开启 {determinism_status()}")

    train_ds, train_loader, val_loader = build_loaders(args, cfg, args.overfit)

    model = DCCRNLite(n_freqs=train_ds.n_freqs, base_channels=args.base_channels,
                      lstm_hidden=args.lstm_hidden,
                      use_lstm=not args.no_lstm).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler(device.type,
                                 enabled=bool(cfg.amp and device.type == "cuda"))
    state = TrainState()

    print("=" * 74)
    print(f"设备      : {describe_device(device)}")
    print(f"模型      : DCCRN-lite · 可训练参数 {n_param:,} ({n_param / 1e6:.2f} M)")
    print(f"输入形状  : 1 ch × {train_ds.n_freqs} freq × {train_ds.n_frames} frames")
    print(f"样本      : 训练 {len(train_ds)}"
          + (f" / 验证 {len(val_loader.dataset)}" if val_loader else "（无验证集）"))
    print(f"超参      : epochs={cfg.epochs} bs={cfg.batch_size} lr={cfg.lr} "
          f"amp={cfg.amp} seed={cfg.seed}")
    _p = int(cfg.early_stop_patience)
    print(f"早停      : " + (f"验证损失连续 {_p} 个 epoch 无改善即停止"
                            if _p > 0 and val_loader is not None else
                            ("关闭" if _p <= 0 else "未启用（无验证集）")))
    print(f"输出      : {out_dir}")
    if args.overfit:
        print(f"模式      : 过拟合测试 —— {OVERFIT_N} 条样本、关闭随机片段裁剪")
    print("=" * 74)

    (out_dir / "run.json").write_text(json.dumps({
        "argv": sys.argv[1:], "train": {**vars(cfg), "out_dir": str(out_dir)},
        "model": "dccrn_lite", "base_channels": args.base_channels,
        "lstm_hidden": args.lstm_hidden, "use_lstm": not args.no_lstm,
        "n_params": n_param, "device": str(device),
        "overfit": bool(args.overfit), "n_train": len(train_ds),
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    history_path = out_dir / "history.jsonl"
    best_path, last_path = out_dir / "best.pt", out_dir / "last.pt"
    es = EarlyStopState(patience=int(cfg.early_stop_patience))

    t0 = time.time()
    for epoch in range(state.epoch, cfg.epochs):
        train_ds.set_epoch(epoch)
        print(f"\nepoch {epoch + 1}/{cfg.epochs}")
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device,
                             cfg, state, scheduler=scheduler, log_fn=print,
                             dry_run=args.dry_run)
        line = (f"  训练 loss={tr['loss']:.5f} (mag={tr['loss_mag']:.5f} "
                f"cplx={tr['loss_cplx']:.5f})")
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

        monitor = va["loss"] if va is not None else tr["loss"]
        if monitor < state.best_val - 1e-12:
            state.best_val = monitor
            save_checkpoint(best_path, model, optimizer, scheduler, state, cfg,
                            extra={"monitor": "val" if va else "train",
                                   "monitor_value": monitor, "epoch": epoch + 1})
        save_checkpoint(last_path, model, optimizer, scheduler, state, cfg)

        if args.dry_run:
            print("\n--dry-run：已跑通 1 个 batch，退出")
            break
        if es.enabled and val_loader is not None:
            if es.update(monitor, epoch + 1):
                print(f"\n早停：验证损失连续 {es.epochs_no_improve} 个 epoch 无改善"
                      f"（最优 {es.best:.5f} @ epoch {es.best_epoch}）")
                break

    print("\n" + "=" * 74)
    print(f"完成，用时 {time.time() - t0:.1f} s")
    print(f"最佳监控值 {state.best_val:.5f}")
    print(f"checkpoint：{best_path}")
    print(f"训练曲线  ：{history_path}")

    if not args.dry_run:
        probe = DCCRNLite(n_freqs=train_ds.n_freqs,
                          base_channels=args.base_channels,
                          lstm_hidden=args.lstm_hidden,
                          use_lstm=not args.no_lstm).to(device)
        st2, _ = load_checkpoint(last_path, probe, map_location=device)
        print(f"checkpoint 复载自检：通过（epoch={st2.epoch} step={st2.global_step}）")

    if args.overfit and state.history:
        first = state.history[0]["train"]["loss"]
        last = state.history[-1]["train"]["loss"]
        print(f"\n过拟合判读：首 {first:.5f} → 末 {last:.5f}"
              f"（降至 {last / max(first, 1e-12) * 100:.2f}%）")
        print("  通过" if last < 0.3 * first else "  不通过：loss 未显著下降")
        return 0 if last < 0.3 * first else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
