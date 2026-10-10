"""T3.3 验收测试：训练基础设施。

这里刻意用**轻量 stub 模型**而不是真 CRN：训练循环的正确性（梯度累积、
AMP、checkpoint 往返、调度器）与网络结构无关，用 stub 可以秒级跑完，
把「循环里的 bug」和「模型里的 bug」分开定位。
真 CRN 的联调由 `tests/test_crn.py` 与 T3.4 过拟合测试负责。
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nbse.dsp import N_FFT
from nbse.models.dataset import (
    MANIFEST_NAME,
    NbseDataset,
    collate_samples,
)
from nbse.training import (
    TrainConfig,
    TrainState,
    append_history,
    build_optimizer,
    build_scheduler,
    crn_loss,
    describe_device,
    evaluate_loss,
    load_checkpoint,
    masked_magnitude,
    pick_device,
    save_checkpoint,
    train_one_epoch,
)

F = N_FFT // 2 + 1
T = 32


class StubMaskNet(nn.Module):
    """输出常量掩蔽的极简模型——足以驱动训练循环。

    刻意包含一个 BatchNorm2d，用来验证优化器确实把 BN 参数排除在
    weight decay 之外（`test_optimizer_excludes_norm_and_bias`）。
    """

    def __init__(self, n_mics: int = 4, init: float = 0.5):
        super().__init__()
        self.conv = nn.Conv2d(n_mics, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.head = nn.Conv2d(8, 1, 1)
        self.bias = nn.Parameter(torch.tensor(float(init)))

    def forward(self, mag):
        h = torch.relu(self.bn(self.conv(mag)))
        return torch.sigmoid(self.head(h).squeeze(1) + self.bias)


def _make_data(tmp_path, n=4, length=4096):
    """合成一个落盘样本目录。"""
    root = tmp_path / "sim"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        clean = np.stack([rng.normal(size=length) for _ in range(4)])
        noise = 0.3 * rng.normal(size=(4, length))
        name = f"sample_{i:05d}.npz"
        np.savez(root / name,
                 multi_ch=(clean + noise).astype(np.float32),
                 clean_ref=clean[0].astype(np.float32),
                 noise_multi=noise.astype(np.float32),
                 doa_true=np.float64(10.0 * i),
                 snr_db=np.float64(0.0))
        rows.append({"index": i, "file": name, "subset": "dev-clean",
                     "speaker": "1", "chapter": "1", "utterance": f"u{i}.flac",
                     "noise_env": "OOFFICE", "noise_file": "ch07.wav",
                     "snr_target_db": "0.00", "snr_db": "0.0000",
                     "doa_true": f"{10.0 * i:.4f}", "t60": "0.3000",
                     "room_lx": "4.0000", "room_ly": "4.0000",
                     "room_lz": "3.0000", "room_key": "k", "distance_m": "2.0",
                     "ref_delay": "100", "n_mics": 4, "sample_rate": 16000,
                     "n_samples": length})
    import csv as _csv
    with open(root / MANIFEST_NAME, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return root


def _loader(tmp_path, batch_size=2, n=4, length=4096):
    ds = NbseDataset(_make_data(tmp_path, n=n, length=length),
                     segment_length=length, random_crop=False)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=False,
                          collate_fn=collate_samples)


# ── 损失口径 ────────────────────────────────────────────────────

def test_masked_magnitude_uses_first_channel():
    """掩蔽作用于 0 号通道 —— 与 dataset.compute_targets 的口径一致。"""
    mag = torch.zeros(2, 4, F, T)
    mag[:, 0] = 3.0
    mag[:, 1:] = 7.0
    mask = torch.full((2, F, T), 0.5)
    out = masked_magnitude(mag, mask)
    assert out.shape == (2, F, T)
    assert torch.allclose(out, torch.full((2, F, T), 1.5))


def test_crn_loss_is_zero_on_perfect_prediction():
    """完美预测时两项损失都应为 0 —— 这是口径自洽的最低要求。"""
    mag = torch.rand(2, 4, F, T) + 0.1
    mask = torch.rand(2, F, T)
    clean_mag = mag[:, 0] * mask

    total, parts = crn_loss(mask, mask, mag, clean_mag)
    assert total.item() == pytest.approx(0.0, abs=1e-12)
    assert parts["loss_mask"] == pytest.approx(0.0, abs=1e-12)
    assert parts["loss_mag"] == pytest.approx(0.0, abs=1e-12)


def test_crn_loss_decreases_as_prediction_improves():
    """预测越接近目标，损失必须越小（单调性）。"""
    torch.manual_seed(0)
    mag = torch.rand(2, 4, F, T) + 0.1
    target = torch.rand(2, F, T)
    clean_mag = mag[:, 0] * target

    losses = []
    for eps in (0.5, 0.25, 0.1, 0.0):
        pred = torch.clamp(target + eps, 0, 1)
        total, _ = crn_loss(pred, target, mag, clean_mag)
        losses.append(float(total))
    assert losses == sorted(losses, reverse=True)


def test_loss_weights_are_respected():
    mag = torch.rand(1, 4, F, T) + 0.1
    target = torch.rand(1, F, T)
    clean_mag = mag[:, 0] * target
    pred = torch.zeros_like(target)

    _, p_mask_only = crn_loss(pred, target, mag, clean_mag, w_mask=1.0, w_mag=0.0)
    _, p_mag_only = crn_loss(pred, target, mag, clean_mag, w_mask=0.0, w_mag=1.0)
    _, p_both = crn_loss(pred, target, mag, clean_mag, w_mask=1.0, w_mag=1.0)

    assert p_both["loss"] == pytest.approx(p_mask_only["loss"] + p_mag_only["loss"],
                                           rel=1e-6)


# ── 优化器与调度器 ──────────────────────────────────────────────

def test_optimizer_excludes_norm_and_bias_from_weight_decay():
    """BN 的 weight/bias 与所有偏置不得进入 weight-decay 组。

    否则 weight decay 会把 BN 的缩放往 0 拉，在小模型上足以拖慢收敛并让
    T3.4 过拟合测试误判为「管线有 bug」。
    """
    model = StubMaskNet()
    cfg = TrainConfig(weight_decay=0.1)
    opt = build_optimizer(model, cfg)

    decay_ids = {id(p) for p in opt.param_groups[0]["params"]}
    nodecay_ids = {id(p) for p in opt.param_groups[1]["params"]}

    assert opt.param_groups[0]["weight_decay"] == 0.1
    assert opt.param_groups[1]["weight_decay"] == 0.0
    assert id(model.bn.weight) in nodecay_ids
    assert id(model.bn.bias) in nodecay_ids
    assert id(model.head.bias) in nodecay_ids
    assert id(model.conv.weight) in decay_ids
    # 两组必须覆盖全部可训练参数，不重不漏
    assert decay_ids | nodecay_ids == {id(p) for p in model.parameters()
                                       if p.requires_grad}
    assert not (decay_ids & nodecay_ids)


def test_scheduler_decays_lr_over_epochs():
    model = StubMaskNet()
    cfg = TrainConfig(epochs=10, lr=1e-3)
    opt = build_optimizer(model, cfg)
    sch = build_scheduler(opt, cfg)

    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(10):
        opt.step()
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])

    assert lrs[0] > lrs[-1], f"学习率未下降：{lrs}"
    assert lrs[-1] >= cfg.lr * 0.01 - 1e-12, "不应低于 eta_min"
    assert lrs == sorted(lrs, reverse=True), "余弦退火应单调不增"


# ── 训练循环 ────────────────────────────────────────────────────

@pytest.mark.parametrize("accum", [1, 2, 4])
def test_train_one_epoch_returns_finite_loss_and_steps(tmp_path, accum):
    """梯度累积不同取值下循环都要跑通并给出有限损失。"""
    ds, loader = _loader(tmp_path)
    model = StubMaskNet()
    cfg = TrainConfig(epochs=3, batch_size=2, grad_accum_steps=accum,
                      amp=False, log_every=0, lr=1e-3)
    opt = build_optimizer(model, cfg)
    state = TrainState()

    out = train_one_epoch(model, loader, opt, torch.amp.GradScaler("cpu", enabled=False),
                          torch.device("cpu"), cfg, state, log_fn=lambda *_: None)

    assert np.isfinite(out["loss"])
    assert set(out) == {"loss", "loss_mask", "loss_mag"}
    # 优化器步数 = ceil(n_batches / accum)
    expected = (len(loader) + accum - 1) // accum
    assert state.global_step == expected, \
        f"accum={accum}：期望 {expected} 步，实际 {state.global_step}"


def test_training_actually_reduces_loss(tmp_path):
    """训练若干步后损失应下降 —— 这是「管线能学」的最基本证据。"""
    ds, loader = _loader(tmp_path, n=4)
    torch.manual_seed(0)
    model = StubMaskNet()
    cfg = TrainConfig(epochs=1, batch_size=4, grad_accum_steps=1,
                      amp=False, log_every=0, lr=5e-2)
    opt = build_optimizer(model, cfg)
    state = TrainState()

    first = None
    for _ in range(20):
        out = train_one_epoch(
            model, loader, opt, torch.amp.GradScaler("cpu", enabled=False),
            torch.device("cpu"), cfg, state, log_fn=lambda *_: None)
        if first is None:
            first = out["loss"]

    assert out["loss"] < first, f"20 轮后损失未下降：{first:.5f} → {out['loss']:.5f}"


def _forward_backward(model, loader, cfg):
    """跑一次前向+反向（不 step），把损失按累积步数摊平，返回总损失。"""
    inputs, targets = next(iter(loader))
    mag = inputs["mag"]
    mask = targets["mask"]
    clean_mag = targets["clean_mag"]

    with torch.autocast(device_type="cpu", enabled=False):
        pred = model(mag)
        loss, _ = crn_loss(pred, mask, mag, clean_mag, cfg.w_mask, cfg.w_mag)
        (loss / cfg.grad_accum_steps).backward()
    return float(loss.detach())


def _grad_norm(model) -> float:
    """当前梯度总 L2 范数（不触发裁剪）。"""
    total = sum(float((p.grad.detach() ** 2).sum())
                for p in model.parameters() if p.grad is not None)
    return total ** 0.5


def test_gradient_clip_bounds_gradient_norm(tmp_path):
    """裁剪后梯度总范数必须 ≤ grad_clip（+ 数值余量）。

    两个踩坑记录：
    1. 最初想用「参数位移变小」验证裁剪，失败了——**AdamW 是尺度不变的**，
      它把梯度除以运行二阶矩，梯度整体缩放几乎不改变更新量
      （实测裁剪后位移 27.5 vs 未裁剪 29.6）。要用位移验证，优化器必须是
      SGD 这类尺度敏感的。
    2. `clip_grad_norm_()` **返回的是裁剪前的范数**，不是裁剪后的。
       拿返回值当「裁剪后的范数」会得到 0.133 这种看似合理的数字而错过 bug。
       因此这里用独立的 `_grad_norm()` 直接量裁剪后的实际范数。
    """
    ds, loader = _loader(tmp_path)
    clip = 1e-3
    cfg = TrainConfig(epochs=1, batch_size=2, grad_accum_steps=1, amp=False,
                      log_every=0, lr=1.0, grad_clip=clip)
    model = StubMaskNet()
    _forward_backward(model, loader, cfg)

    raw = _grad_norm(model)
    assert raw > 10 * clip, f"未裁剪的梯度范数 {raw:.4f} 太小，验证没有意义"

    nn.utils.clip_grad_norm_(model.parameters(), clip)

    after = _grad_norm(model)
    assert after == pytest.approx(clip, rel=1e-3), \
        f"裁剪后总范数应为 {clip}，实际 {after:.6f}"


def test_train_one_epoch_applies_clipping(tmp_path):
    """训练循环里确实调用了裁剪：把 grad_clip 设成极小值后，
    每步梯度范数被压到阈值内，因此参数位移应远小于不裁剪的情形。

    这条用 SGD 验证——只有尺度敏感的优化器才能反映裁剪效果。
    """
    def _displacement(clip):
        ds, loader = _loader(tmp_path)
        torch.manual_seed(0)
        model = StubMaskNet()
        opt = torch.optim.SGD(model.parameters(), lr=1.0)
        cfg = TrainConfig(epochs=1, batch_size=2, grad_accum_steps=1, amp=False,
                          log_every=0, grad_clip=clip)
        before = [p.detach().clone() for p in model.parameters()]

        # 直接驱动循环内的裁剪逻辑：借用 train_one_epoch，但换成 SGD
        train_one_epoch(model, loader, opt,
                        torch.amp.GradScaler("cpu", enabled=False),
                        torch.device("cpu"), cfg, TrainState(),
                        log_fn=lambda *_: None, dry_run=True)
        return sum(float((p.detach() - b).norm())
                   for b, p in zip(before, model.parameters()))

    clipped = _displacement(1e-3)
    unclipped = _displacement(0.0)
    assert clipped < unclipped, \
        f"裁剪未起作用：裁剪后 {clipped:.4f} vs 未裁剪 {unclipped:.4f}"
    assert clipped < 0.05 * unclipped, \
        f"裁剪效果不明显：{clipped:.4f} vs {unclipped:.4f}"


def test_dry_run_does_not_advance_full_epoch(tmp_path):
    ds, loader = _loader(tmp_path)
    model = StubMaskNet()
    cfg = TrainConfig(epochs=1, batch_size=2, grad_accum_steps=1, amp=False,
                      log_every=0)
    opt = build_optimizer(model, cfg)
    state = TrainState()
    train_one_epoch(model, loader, opt, torch.amp.GradScaler("cpu", enabled=False),
                    torch.device("cpu"), cfg, state, log_fn=lambda *_: None,
                    dry_run=True)
    assert state.global_step == 1, "--dry-run 应只走 1 个 batch"


def test_evaluate_loss_does_not_update_weights(tmp_path):
    ds, loader = _loader(tmp_path)
    model = StubMaskNet()
    cfg = TrainConfig(batch_size=2, amp=False)

    before = [p.detach().clone() for p in model.parameters()]
    out = evaluate_loss(model, loader, torch.device("cpu"), cfg)

    assert np.isfinite(out["loss"])
    for b, p in zip(before, model.parameters()):
        assert torch.equal(b, p.detach()), "评估阶段不应改动权重"
    assert not model.training, "评估后模型应处于 eval 模式"


# ── checkpoint ──────────────────────────────────────────────────

def test_checkpoint_roundtrip_restores_training(tmp_path):
    """T3.3 验收：checkpoint 可被重新加载并**继续训练**。

    不只看权重是否读回——还要确认优化器与调度器状态、epoch/step 都恢复，
    否则续训会从 lr 初值重启、等价于从头再来。
    """
    ds, loader = _loader(tmp_path)
    model = StubMaskNet()
    cfg = TrainConfig(epochs=5, batch_size=2, grad_accum_steps=1, amp=False,
                      log_every=0, lr=1e-2)
    opt = build_optimizer(model, cfg)
    sch = build_scheduler(opt, cfg)
    state = TrainState()

    train_one_epoch(model, loader, opt, torch.amp.GradScaler("cpu", enabled=False),
                    torch.device("cpu"), cfg, state, scheduler=sch,
                    log_fn=lambda *_: None)
    state.epoch = 1

    path = save_checkpoint(tmp_path / "ckpt" / "last.pt", model, opt, sch,
                           state, cfg)
    assert path.exists()

    model2 = StubMaskNet()
    opt2 = build_optimizer(model2, cfg)
    sch2 = build_scheduler(opt2, cfg)
    st2, payload = load_checkpoint(path, model2, opt2, sch2)

    assert st2.epoch == 1
    assert st2.global_step == state.global_step
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2), "权重未正确恢复"
    assert opt2.param_groups[0]["lr"] == pytest.approx(
        opt.param_groups[0]["lr"]), "优化器状态未恢复"
    assert payload["config"]["epochs"] == 5

    # 真的能继续训练
    st2 = TrainState(**{k: v for k, v in vars(st2).items()})
    train_one_epoch(model2, loader, opt2,
                    torch.amp.GradScaler("cpu", enabled=False),
                    torch.device("cpu"), cfg, st2, scheduler=sch2,
                    log_fn=lambda *_: None)
    assert st2.global_step > state.global_step


def test_checkpoint_can_be_created_without_scheduler(tmp_path):
    model = StubMaskNet()
    cfg = TrainConfig()
    opt = build_optimizer(model, cfg)
    path = save_checkpoint(tmp_path / "a.pt", model, opt, None, TrainState(), cfg)
    st, payload = load_checkpoint(path, StubMaskNet())
    assert payload["scheduler"] is None
    assert st.epoch == 0


# ── 日志 ────────────────────────────────────────────────────────

def test_append_history_writes_jsonl(tmp_path):
    p = tmp_path / "hist" / "history.jsonl"
    append_history(p, {"epoch": 1, "train": {"loss": 0.5}})
    append_history(p, {"epoch": 2, "train": {"loss": 0.2}})

    import json
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["epoch"] == 1
    assert json.loads(lines[1])["train"]["loss"] == 0.2


# ── 设备 ────────────────────────────────────────────────────────

def test_pick_device_falls_back_to_cpu():
    assert pick_device("cpu").type == "cpu"
    got = pick_device("cuda")
    assert got.type in ("cuda", "cpu")
    assert describe_device(got)
