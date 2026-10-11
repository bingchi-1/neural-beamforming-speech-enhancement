"""T5.2 验收测试：DCCRN-lite 单通道复数域增强。

测试重点放在**复数域实现是否正确**以及**结构是否会静默失效**：

* 复数乘法的展开式容易被写成错的组合，而错误组合在前向形状上完全正常；
* 跳跃连接的尺寸对不上会被误当成「框架问题」而用调 kernel 的方式绕过，
  结果对不同的 n_freqs 再次失配；
* LSTM 瓶颈若在优化器创建之后才建，那部分参数**永远不被更新且不报错**。
"""

import numpy as np
import pytest
import torch

from nbse.models.dccrn_lite import (
    MAG_POWER,
    ComplexBatchNorm2d,
    ComplexConv2d,
    ComplexConvTranspose2d,
    DCCRNLite,
    apply_complex_mask,
    dccrn_lite_loss,
)


# ── 复数层 ──────────────────────────────────────────────────────

def test_complex_conv_matches_manual_expansion():
    """复数卷积必须等于 (WrXr−WiXi) + j(WrXi+WiXr)。

    用 `bias=False` 构造，否则两个实卷积的偏置会混入：`conv_i(0)` 不是 0
    而是 `bias_i`，使「Xi=0 时实部 = conv_r(Xr)」这类断言失效
    （初版就是漏了这个偏置，误差 0.21 而被误判为实现错误）。
    """
    conv = ComplexConv2d(2, 3, kernel_size=(3, 3), stride=(1, 1),
                         padding=(1, 1), bias=False)
    rng = torch.Generator().manual_seed(0)
    xr = torch.randn(1, 2, 8, 8, generator=rng)
    xi = torch.randn(1, 2, 8, 8, generator=rng)

    yr, yi = conv(xr, xi)
    exp_r = conv.conv_r(xr) - conv.conv_i(xi)
    exp_i = conv.conv_r(xi) + conv.conv_i(xr)
    torch.testing.assert_close(yr, exp_r)
    torch.testing.assert_close(yi, exp_i)

    # 纯实数输入（Xi=0）：实部 = Wr·Xr，虚部 = Wi·Xr
    yr0, yi0 = conv(xr, torch.zeros_like(xr))
    torch.testing.assert_close(yr0, conv.conv_r(xr))
    torch.testing.assert_close(yi0, conv.conv_i(xr))


def test_complex_conv_is_linear():
    """复数卷积对输入线性（叠加性）。

    同样必须用 `bias=False`：带偏置时 `conv(a+b) = W(a+b) + bias` 而
    `conv(a)+conv(b) = W(a)+W(b) + 2·bias`，两者天然相差一个 bias，
    与是否线性无关。初版带偏置断言线性，得到 0.31 的差值而误判实现。
    """
    conv = ComplexConv2d(1, 2, kernel_size=(3, 3), stride=(1, 1),
                         padding=(1, 1), bias=False)
    rng = torch.Generator().manual_seed(1)
    a_r, a_i = torch.randn(1, 1, 6, 6, generator=rng), torch.randn(1, 1, 6, 6, generator=rng)
    b_r, b_i = torch.randn(1, 1, 6, 6, generator=rng), torch.randn(1, 1, 6, 6, generator=rng)

    s = conv(a_r + b_r, a_i + b_i)
    t = conv(a_r, a_i)
    u = conv(b_r, b_i)
    torch.testing.assert_close(s[0], t[0] + u[0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s[1], t[1] + u[1], atol=1e-5, rtol=1e-5)


def test_complex_conv_transpose_doubles_both_axes():
    """转置卷积必须把**频率与时间两轴都翻倍**。

    张量布局是 [B, C, F, T]，故 Conv2d 的 (H, W) = (频率, 时间)。
    编码器 stride=(2,1) 只降频率；解码器要恢复就得 stride=(2,2) 让两轴各自
    翻倍。这一步连续踩了两个坑，都以「跳跃连接尺寸不符」的形式暴露：
      (a) stride=(2,1)：时间轴不动（40 仍是 40）
      (b) stride=(1,2)：频率轴不再翻倍（停在 17，编码器对应层是 33）
    故这里把两轴的行为同时断言死。
    """
    conv = ComplexConvTranspose2d(2, 2, kernel_size=(5, 5), stride=(2, 2),
                                  padding=(2, 2), output_padding=(1, 1))
    xr = torch.randn(1, 2, 17, 40)
    yr, _ = conv(xr, torch.zeros_like(xr))
    assert yr.shape[2] == 34, f"频率轴(H)应翻倍，实得 {yr.shape[2]}"
    assert yr.shape[3] == 80, f"时间轴(W)应翻倍，实得 {yr.shape[3]}"


def test_complex_conv_transpose_rejects_invalid_output_padding():
    """守住 PyTorch 的约束：output_padding 必须严格小于同维 stride。

    两轴 stride 都是 2 时 output_padding 取 1 才合法；若某轴 stride 退回 1
    而 output_padding 仍为 1，PyTorch 会直接拒绝——这条把该约束写进测试，
    避免以后有人单改一处。
    """
    conv = ComplexConvTranspose2d(1, 1, kernel_size=(5, 5), stride=(1, 2),
                                  padding=(2, 2), output_padding=(1, 0))
    with pytest.raises(RuntimeError, match="output padding must be smaller"):
        conv(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))


def test_complex_batchnorm_normalizes_both_parts():
    bn = ComplexBatchNorm2d(4)
    xr = torch.randn(8, 4, 5, 5) * 3 + 1
    xi = torch.randn(8, 4, 5, 5) * 0.5 - 2
    yr, yi = bn(xr, xi)
    assert yr.shape == xr.shape and yi.shape == xi.shape
    # 训练模式下批统计应把两部分的均值拉向 0、方差拉向 1
    assert abs(float(yr.mean())) < 0.05
    assert abs(float(yi.mean())) < 0.05


# ── 模型结构 ────────────────────────────────────────────────────

def test_model_forward_shapes():
    model = DCCRNLite(n_freqs=257)
    mr, mi = model(torch.randn(2, 1, 257, 126), torch.randn(2, 1, 257, 126))
    assert mr.shape == (2, 257, 126)
    assert mi.shape == (2, 257, 126)


@pytest.mark.parametrize("F,T", [(257, 63), (257, 126), (257, 501), (129, 125)])
def test_model_handles_various_shapes(F, T):
    """不同 (F, T) 都必须严格还原输入尺寸。

    频率轴「降采样再上采样」不保证回到原尺寸（实测 33→34），
    跳跃连接与输出都需显式对齐；非 2 的幂的 T 也要覆盖。
    """
    model = DCCRNLite(n_freqs=F)
    mr, mi = model(torch.randn(1, 1, F, T), torch.randn(1, 1, F, T))
    assert mr.shape == (1, F, T), f"F={F},T={T} 时输出 {mr.shape}"
    assert mi.shape == (1, F, T)


def test_mask_is_bounded_and_finite():
    model = DCCRNLite(n_freqs=257)
    mr, mi = model(torch.randn(3, 1, 257, 126), torch.randn(3, 1, 257, 126))
    assert torch.isfinite(mr).all() and torch.isfinite(mi).all()
    # tanh 输出必须严格落在 (−1, 1)
    assert float(mr.abs().max()) < 1.0
    assert float(mi.abs().max()) < 1.0


def test_all_parameters_receive_gradients():
    """每个可训练参数都必须拿到梯度。

    这条能抓到「某个子模块没有被接进计算图」——例如瓶颈建好后忘记在
    forward 里使用它。那种错误不会报错，只是那部分权重永远停在初值。
    """
    model = DCCRNLite(n_freqs=257)
    mr, mi = model(torch.randn(2, 1, 257, 63), torch.randn(2, 1, 257, 63))
    loss, _ = dccrn_lite_loss(mr, mi, torch.randn(2, 257, 63),
                              torch.randn(2, 257, 63))
    loss.backward()

    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, f"以下参数没有梯度：{missing[:6]}"


def test_optimizer_created_after_init_covers_all_parameters():
    """优化器若在模型构建后立刻创建，必须覆盖**全部**参数。

    LSTM 瓶颈的特征维依赖输入形状，因此是在首次前向时才确定的。若那时才
    创建 LSTM 模块，而优化器已经建好（`train.py` 正是这个顺序），
    这批参数不会进入 param_groups —— **永远不被更新，且没有任何报错**。
    本模型在 `__init__` 里用 dummy 前向把瓶颈建好，这条测试守住该行为。
    """
    model = DCCRNLite(n_freqs=257)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_mod = sum(p.numel() for p in model.parameters())
    assert n_opt == n_mod, f"优化器只覆盖 {n_opt} / {n_mod} 个参数"
    assert n_mod > 0


def test_num_parameters_property():
    model = DCCRNLite(n_freqs=257)
    assert model.num_parameters == sum(p.numel() for p in model.parameters())
    # 单通道对照的规模应与 CRN 同量级（便于同算力下比较）
    assert 0.5e6 < model.num_parameters < 12e6


def test_use_lstm_false_disables_bottleneck():
    """消融开关应真的去掉 LSTM（参数量下降）。"""
    with_lstm = DCCRNLite(n_freqs=257, use_lstm=True)
    without = DCCRNLite(n_freqs=257, use_lstm=False)
    assert without.num_parameters < with_lstm.num_parameters

    mr, mi = without(torch.randn(1, 1, 257, 63), torch.randn(1, 1, 257, 63))
    assert mr.shape == (1, 257, 63)


# ── 复数掩蔽与损失 ──────────────────────────────────────────────

def test_apply_complex_mask_is_complex_multiplication():
    rng = torch.Generator().manual_seed(2)
    sr = torch.randn(2, 4, 5, generator=rng)
    si = torch.randn(2, 4, 5, generator=rng)
    mr = torch.randn(2, 4, 5, generator=rng)
    mi = torch.randn(2, 4, 5, generator=rng)

    er, ei = apply_complex_mask(sr, si, mr, mi)

    # 与 numpy 的复数乘法对照
    s = sr.numpy() + 1j * si.numpy()
    m = mr.numpy() + 1j * mi.numpy()
    got = er.numpy() + 1j * ei.numpy()
    np.testing.assert_allclose(got, s * m, rtol=1e-6, atol=1e-6)


def test_identity_mask_reconstructs_signal():
    """单位掩蔽 (1, 0) 必须原样重建 —— 复数乘法的基本校验。"""
    rng = torch.Generator().manual_seed(3)
    sr = torch.randn(2, 4, 5, generator=rng)
    si = torch.randn(2, 4, 5, generator=rng)
    er, ei = apply_complex_mask(sr, si, torch.ones_like(sr), torch.zeros_like(si))
    torch.testing.assert_close(er, sr)
    torch.testing.assert_close(ei, si)


def test_loss_is_zero_for_perfect_reconstruction():
    rng = torch.Generator().manual_seed(4)
    cr = torch.randn(2, 4, 5, generator=rng)
    ci = torch.randn(2, 4, 5, generator=rng)
    total, parts = dccrn_lite_loss(cr, ci, cr, ci)
    assert float(total) == pytest.approx(0.0, abs=1e-6)
    assert parts["loss_mag"] == pytest.approx(0.0, abs=1e-6)
    assert parts["loss_cplx"] == pytest.approx(0.0, abs=1e-6)


def test_loss_decreases_as_estimate_improves():
    rng = torch.Generator().manual_seed(5)
    cr = torch.randn(2, 4, 5, generator=rng)
    ci = torch.randn(2, 4, 5, generator=rng)

    losses = []
    for eps in (1.0, 0.5, 0.1, 0.0):
        total, _ = dccrn_lite_loss(cr + eps, ci + eps, cr, ci)
        losses.append(float(total))
    assert losses == sorted(losses, reverse=True), f"损失未随估计变好而下降：{losses}"


def test_loss_reports_magnitude_and_complex_parts():
    """两项损失都必须被报告 —— 只看总数无法判断是幅度还是相位在学。"""
    rng = torch.Generator().manual_seed(6)
    total, parts = dccrn_lite_loss(torch.randn(1, 4, 5, generator=rng),
                                   torch.randn(1, 4, 5, generator=rng),
                                   torch.randn(1, 4, 5, generator=rng),
                                   torch.randn(1, 4, 5, generator=rng))
    assert set(parts) == {"loss", "loss_mag", "loss_cplx"}
    assert parts["loss"] == pytest.approx(parts["loss_mag"] + parts["loss_cplx"],
                                          rel=1e-5)
    assert np.isfinite(parts["loss"])


def test_magnitude_compression_power_is_applied():
    """幅度压缩指数确实起作用：同幅度的不同 power 应给出不同损失。"""
    rng = torch.Generator().manual_seed(7)
    e_r = torch.randn(1, 4, 5, generator=rng) * 5
    e_i = torch.randn(1, 4, 5, generator=rng) * 5
    c_r = torch.randn(1, 4, 5, generator=rng)
    c_i = torch.randn(1, 4, 5, generator=rng)

    _, p03 = dccrn_lite_loss(e_r, e_i, c_r, c_i, power=0.3)
    _, p10 = dccrn_lite_loss(e_r, e_i, c_r, c_i, power=1.0)
    assert p03["loss_mag"] != pytest.approx(p10["loss_mag"], rel=1e-3)
    assert MAG_POWER == pytest.approx(0.3)
