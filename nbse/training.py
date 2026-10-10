"""T3.3 训练基础设施：损失、训练循环、checkpoint、日志。

设计文档 §5 的 `train.py` 一栏；CLI 在仓库根目录 `train.py`，本模块放可复用实现。

训练目标
--------
网络输出掩蔽 `m_hat`，用它作用于观测幅度谱得到估计的干净幅度谱。因此有两条
互相补充的监督信号：

1. **掩蔽域**（`loss_mask`）：`MSE(m_hat, mask)`。直接约束网络输出，
   梯度尺度与掩蔽量纲一致，是主要的优化信号。
2. **幅度域**（`loss_mag`）：`MSE(mag · m_hat, clean_mag)`。约束的是**实际
   用途**——掩蔽最终要被拿去乘观测频谱，这一步才是波束形成器看到的东西。
   只监督掩蔽会允许「掩蔽对了但乘积偏了」的解（例如整体尺度偏移），
   加上这一项把两端钉在一起。

两项默认等权（各 1.0）。过拟合测试（T3.4）会验证这两项都能降到接近零——
这是「损失口径与数据口径一致」的直接证据。

为什么掩蔽用 MSE 而非 SI-SDR：SI-SDR 定义在时域波形上，需要可微的 ISTFT 与
完整相位处理（复数域掩蔽），属于 T4.1 mask-MVDR 之后才需要的能力。T3 阶段
把目标压缩到「估计掩蔽」这一件事上，让管线先跑通——这也是设计文档
「每个模块独立可测」原则的体现。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── 损失 ────────────────────────────────────────────────────────


def masked_magnitude(mag: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """观测幅度谱 × 掩蔽 → 估计的干净幅度谱。

    `mask` 是 [B, F, T]，`mag` 是 [B, M, F, T]（多通道）。广播前先取
    **0 号通道**——掩蔽的目标是按参考通道定的（见 dataset.compute_targets），
    所以这里必须用同一通道，否则监督信号与数据口径不一致。
    """
    if mag.dim() == 4:
        mag = mag[:, 0]
    return mag * mask


def crn_loss(pred_mask: torch.Tensor, target_mask: torch.Tensor,
             mag: torch.Tensor, clean_mag: torch.Tensor,
             w_mask: float = 1.0, w_mag: float = 1.0) -> tuple[torch.Tensor, dict]:
    """复合损失。返回 (总损失, 分项字典)，分项用于日志与过拟合判读。"""
    loss_mask = nn.functional.mse_loss(pred_mask, target_mask)
    loss_mag = nn.functional.mse_loss(masked_magnitude(mag, pred_mask), clean_mag)
    total = w_mask * loss_mask + w_mag * loss_mag
    parts = {
        "loss": float(total.detach()),
        "loss_mask": float(loss_mask.detach()),
        "loss_mag": float(loss_mag.detach()),
    }
    return total, parts


# ── 配置 ────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """训练超参数。字段与 `configs/default.yaml` 的 `train:` 段对应。"""
    epochs: int = 100
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    num_workers: int = 0          # Windows 上 >0 会反复重建 worker，默认 0
    segment_length: int = 16000
    seed: int = 20261009
    w_mask: float = 1.0
    w_mag: float = 1.0
    log_every: int = 10
    out_dir: str = "runs/w3"
    # 退火下界占 lr 的比例。默认 0.01（退到 1%）。
    # 诊断用：设成 1.0 即不退火，用来区分「优化不足」与「信息上限」。
    lr_floor: float = 0.01
    # 早停：验证损失连续多少个 epoch 无改善就停止。<=0 表示关闭。
    # 只监控**验证损失**——训练损失单调下降，用它做早停永远不会触发。
    #
    # 标定（用 100 epoch 的完整训练曲线回放）：最优 0.02661 出现在 epoch 91，
    # 末段仍在缓慢改善，故 patience ∈ {10..30} 都**不会触发**。即当前
    # 「3000 条 + 100 epoch」的配置本就该跑满，早停在此不省时间。
    # 它的价值在 epochs 给多了、或数据/模型变化导致早早收敛的场景。
    early_stop_patience: int = 20

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "TrainConfig":
        """从 YAML 读训练超参。

        **每个新增字段都必须在这里显式读出**，否则 YAML 里的值会被静默忽略、
        退回 dataclass 默认值。这类漏读极难发现：若配置里写的恰好等于默认值，
        行为完全看不出差别，直到有人改配置却「没反应」。

        （实测踩到：`early_stop_patience` 与 `lr_floor` 都曾漏读。
        前者因配置值与默认值同为 20 而完全隐身。）
        """
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        t = raw.get("train", {})
        cfg = cls(
            epochs=int(t.get("epochs", 100)),
            batch_size=int(t.get("batch_size", 4)),
            grad_accum_steps=int(t.get("grad_accum_steps", 4)),
            lr=float(t.get("lr", 2e-4)),
            weight_decay=float(t.get("weight_decay", 1e-4)),
            grad_clip=float(t.get("grad_clip", 1.0)),
            amp=bool(t.get("amp", True)),
            num_workers=int(t.get("num_workers", 0)),
            seed=int(raw.get("seed", 20261009)),
            lr_floor=float(t.get("lr_floor", 0.01)),
            early_stop_patience=int(t.get("early_stop_patience", 20)),
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW。权重衰减只作用于权重矩阵，不碰 BatchNorm 的仿射参数与偏置。

    对所有参数一视同仁地加 weight decay 会把 BN 的 `weight` 往 0 拉，
    在小模型 + 小数据上足以拖慢收敛并让过拟合测试误判。
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    """余弦退火（设计文档 §T3.3）。

    `eta_min` 默认取 lr 的 1%，避免末段完全停学。
    `cfg.lr_floor` 可用于**诊断**：设成 1.0 表示不退火，用来区分
    「残留 loss 是优化不足」还是「已达数据的信息上限」——若恒定学习率
    能把 loss 继续压下去，说明是衰减太快导致的欠优化。
    """
    floor = float(getattr(cfg, "lr_floor", 0.01))
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * floor)


# ── 训练循环 ────────────────────────────────────────────────────


@dataclass
class TrainState:
    """训练状态。checkpoint 的载荷就是它，保证「可被重新加载继续训练」。"""
    epoch: int = 0
    global_step: int = 0
    best_val: float = math.inf
    history: list = field(default_factory=list)


def _amp_ctx(device: torch.device, enabled: bool):
    """AMP 上下文。CPU 上 bfloat16 才可用，float16 会直接报错。"""
    if not enabled:
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, enabled=True, dtype=dtype)


def _make_grad_scaler(device: torch.device, enabled: bool):
    """GradScaler 只在 CUDA + fp16 下有意义；CPU 上必须禁用，否则会告警并空转。"""
    use = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler(device.type, enabled=use)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use)


def train_one_epoch(model: nn.Module, loader, optimizer, scaler, device,
                    cfg: TrainConfig, state: TrainState,
                    scheduler=None, log_fn=print, dry_run: bool = False) -> dict:
    """跑一个 epoch。返回该 epoch 的损失均值。"""
    model.train()
    totals: dict[str, float] = {}
    n_batches = 0
    n_steps = 0
    accum = max(int(cfg.grad_accum_steps), 1)

    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    for i, (inputs, targets) in enumerate(loader):
        mag = inputs["mag"].to(device, non_blocking=True)
        mask = targets["mask"].to(device, non_blocking=True)
        clean_mag = targets["clean_mag"].to(device, non_blocking=True)

        with _amp_ctx(device, cfg.amp):
            pred = model(mag)
            loss, parts = crn_loss(pred, mask, mag, clean_mag,
                                   cfg.w_mask, cfg.w_mag)
            # 梯度累积：把损失按累积步数摊平，等效于更大的 batch
            scaled = loss / accum

        if scaler.is_enabled():
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        is_last = (i + 1) == len(loader)
        if (i + 1) % accum == 0 or is_last:
            if cfg.grad_clip and cfg.grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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
            avg = totals["loss"] / n_batches
            log_fn(f"  [{i + 1}/{len(loader)}] loss={avg:.5f} "
                   f"mask={totals['loss_mask'] / n_batches:.5f} "
                   f"mag={totals['loss_mag'] / n_batches:.5f} "
                   f"({time.time() - t0:.1f}s)")

        if dry_run:
            break

    # 只在确实发生过 optimizer.step() 时推进调度器。
    # 梯度累积下（accum > len(loader) 或 dry_run）一个 epoch 可能一步都没走，
    # 此时调用 scheduler.step() 会触发 PyTorch 的排序警告，并静默跳过
    # 学习率序列的第一个值。
    if scheduler is not None and n_steps > 0:
        scheduler.step()

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader, device,
                  cfg: TrainConfig, dry_run: bool = False) -> dict:
    """在验证集上算损失（不反传）。"""
    model.eval()
    totals: dict[str, float] = {}
    n = 0

    for i, (inputs, targets) in enumerate(loader):
        mag = inputs["mag"].to(device)
        mask = targets["mask"].to(device)
        clean_mag = targets["clean_mag"].to(device)

        with _amp_ctx(device, cfg.amp):
            pred = model(mag)
            _, parts = crn_loss(pred, mask, mag, clean_mag, cfg.w_mask, cfg.w_mag)

        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
        if dry_run:
            break

    return {k: v / max(n, 1) for k, v in totals.items()}


# ── checkpoint ──────────────────────────────────────────────────


def save_checkpoint(path: str | Path, model: nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler,
                    state: TrainState, cfg: TrainConfig,
                    extra: dict | None = None) -> Path:
    """保存完整训练状态（不只是权重），使训练可中断续跑。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "state": asdict(state),
        "config": asdict(cfg),
        "extra": extra or {},
    }, path)
    return path


def load_checkpoint(path: str | Path, model: nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    scheduler=None, map_location="cpu") -> tuple[TrainState, dict]:
    """读回 checkpoint。返回 (训练状态, 原始载荷)。

    `weights_only=False`：载荷里有 dataclass 还原出来的普通 dict/list，
    属于受信任的本地文件。
    """
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    st = ckpt.get("state", {})
    return TrainState(epoch=st.get("epoch", 0),
                      global_step=st.get("global_step", 0),
                      best_val=st.get("best_val", math.inf),
                      history=st.get("history", [])), ckpt


def append_history(path: str | Path, record: dict) -> None:
    """把每个 epoch 的指标追加进 `history.jsonl`，便于画训练曲线。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class EarlyStopState:
    """早停状态。把判据与副作用分离，便于精确测试边界。"""
    patience: int = 20
    best: float = math.inf
    epochs_no_improve: int = 0
    stopped: bool = False
    best_epoch: int = 0

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def update(self, monitor: float, epoch: int) -> bool:
        """喂入本 epoch 的监控值，返回**是否应停止**。

        改善判据带 1e-12 容差：浮点下 `x < x` 可能因末位差异反复触发
        「改善」，让 patience 永不累加、早停形同虚设。
        """
        if monitor < self.best - 1e-12:
            self.best = monitor
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            return False

        self.epochs_no_improve += 1
        if self.enabled and self.epochs_no_improve >= self.patience:
            self.stopped = True
        return self.stopped


def pick_device(prefer: str = "cuda") -> torch.device:
    """选设备。CUDA 不可用时回落到 CPU 并明确告知。"""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "cpu":
        return torch.device("cpu")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        return (f"cuda · {p.name} · {p.total_memory / 1024 ** 3:.1f} GB · "
                f"sm_{p.major}{p.minor} · torch {torch.__version__}")
    return f"cpu · torch {torch.__version__}"
