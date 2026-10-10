"""T3.1 验收测试：多通道 CRN 掩蔽估计器。

实施计划 T3.1 的验收标准是「前向传播张量形状正确；单次前向在显存内不 OOM」。
本文件进一步覆盖几类**静默错误**——它们不抛异常、但会让掩蔽与观测频谱对不齐，
必须靠形状断言才能发现（详见 crn.py docstring 第 5~7 条）：

* 非 2 的幂的 T（如 125）：编解码整除规则不互逆，每层会多出 1 帧；
* 解码器目标尺寸错位一层：输出会静默停在第 1 层编码器的尺寸。

这些测试全部在 CPU 上跑（形状与数值逻辑与设备无关），因此不需要 GPU。
"""

import pytest
import torch

from nbse.models.crn import (
    DEFAULT_NUM_PARAMETERS,
    MASK_ACTS,
    MIN_FREQ_BINS,
    CRN,
    CRN_SMALL,
)

N_MICS = 4
N_FREQS = 257          # = dsp.N_FFT // 2 + 1（N_FFT=512）
PARAM_BUDGET = 5_000_000


@pytest.fixture(scope="module")
def model():
    """默认配置模型，eval 模式（本文件多数测试只关心形状与值域）。"""
    torch.manual_seed(0)
    m = CRN()
    m.eval()
    return m


def _mag(batch: int, t: int, n_mics: int = N_MICS,
         n_freqs: int = N_FREQS) -> torch.Tensor:
    """构造一个合法的线性幅度谱（非负）。"""
    return torch.rand(batch, n_mics, n_freqs, t)


# ── 1. 前向形状 ─────────────────────────────────────────────────

@pytest.mark.parametrize("batch,t", [(2, 250), (1, 64), (4, 100)])
def test_forward_shape(model, batch, t):
    """[B, n_mics, n_freqs, T] → 恰为 [B, n_freqs, T]。"""
    with torch.no_grad():
        mask = model(_mag(batch, t))
    assert mask.shape == (batch, N_FREQS, t), \
        f"期望 {(batch, N_FREQS, t)}，收到 {tuple(mask.shape)}"


# ── 2. 非 2 的幂的 T（关键）─────────────────────────────────────

@pytest.mark.parametrize("t", [125, 63, 127, 1, 3, 257])
def test_forward_odd_time_lengths(model, t):
    """T 非 2 的幂时同样严格返回与输入相同的 T。

    T=125 是契约里点名的情形。T=1/3 这类极小值用于确认补齐路径也能走通
    （下采样会把时间轴压到 1，解码器再补回来）。
    """
    with torch.no_grad():
        mask = model(_mag(1, t))
    assert mask.shape == (1, N_FREQS, t), \
        f"T={t} 时输出 {tuple(mask.shape)}，与输入不一致"


def test_forward_does_not_leak_time_dimension(model):
    """不同 T 的输出必须各自等于自己的 T，不能出现「都用同一个 T」的串扰。"""
    with torch.no_grad():
        a = model(_mag(1, 125))
        b = model(_mag(1, 250))
    assert a.shape[-1] == 125
    assert b.shape[-1] == 250


# ── 3. 掩蔽值域与数值健康 ───────────────────────────────────────

def test_mask_is_strictly_between_zero_and_one(model):
    """sigmoid 掩蔽必须落在开区间 (0, 1)。"""
    with torch.no_grad():
        mask = model(_mag(2, 100))
    assert torch.isfinite(mask).all(), "掩蔽出现 NaN/Inf"
    assert (mask > 0).all(), "掩蔽出现 ≤ 0 的值"
    assert (mask < 1).all(), "掩蔽出现 ≥ 1 的值"


def test_mask_is_not_constant(model):
    """掩蔽不能退化成常数——那说明网络没在起作用（例如全部权重为零）。"""
    with torch.no_grad():
        mask = model(_mag(2, 100))
    assert mask.std() > 1e-6, f"掩蔽近似常数，std={mask.std():.3e}"


def test_extreme_inputs_stay_finite(model):
    """全零谱与极大谱都不应产生 NaN/Inf（log1p 的边界行为）。"""
    with torch.no_grad():
        zero = model(torch.zeros(1, N_MICS, N_FREQS, 64))
        huge = model(torch.full((1, N_MICS, N_FREQS, 64), 1e6))
    assert torch.isfinite(zero).all()
    assert torch.isfinite(huge).all()
    assert (zero > 0).all() and (zero < 1).all()
    assert (huge > 0).all() and (huge < 1).all()


# ── 4. 参数量 ───────────────────────────────────────────────────

def test_parameter_budget(model):
    """默认配置参数量在 5 M 预算内，并与文档记录的数值一致。"""
    n = model.num_parameters
    print(f"\n默认配置可训练参数量：{n:,}")
    assert n < PARAM_BUDGET, f"参数量 {n:,} 超出预算 {PARAM_BUDGET:,}"
    assert n == DEFAULT_NUM_PARAMETERS, \
        f"参数量与 DEFAULT_NUM_PARAMETERS 不一致：{n:,} vs {DEFAULT_NUM_PARAMETERS:,}"


def test_num_parameters_counts_only_trainable(model):
    """`num_parameters` 只统计 requires_grad 的参数。"""
    expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert model.num_parameters == expected


def test_small_variant_is_smaller():
    """CRN_SMALL 必须显著小于默认配置，供过拟合测试快速跑。"""
    big, small = CRN(), CRN_SMALL()
    print(f"\nCRN_SMALL 参数量：{small.num_parameters:,}")
    assert small.num_parameters < big.num_parameters
    assert small.n_mics == big.n_mics and small.n_freqs == big.n_freqs, \
        "小模型应保持 n_mics/n_freqs 与正式配置一致，否则会掩盖管道问题"


# ── 5. batch 规模 ───────────────────────────────────────────────

@pytest.mark.parametrize("batch", [1, 3])
def test_small_batches_run(model, batch):
    """B=1 与 B=3 都能跑通（BatchNorm 在 B=1 时仍应有合理行为）。"""
    with torch.no_grad():
        mask = model(_mag(batch, 80))
    assert mask.shape == (batch, N_FREQS, 80)
    assert torch.isfinite(mask).all()


# ── 6. eval 模式的确定性 ────────────────────────────────────────

def test_eval_mode_is_deterministic(model):
    """eval() 下两次前向结果必须一致（BatchNorm 用 running stats，无 dropout）。"""
    mag = _mag(2, 100)
    with torch.no_grad():
        a = model(mag)
        b = model(mag)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_train_mode_updates_running_stats():
    """train() 下 BatchNorm 的 running stats 应被更新——否则训练不会收敛。"""
    torch.manual_seed(0)
    m = CRN_SMALL()
    before = m.gru_norm.running_mean.clone()
    m.train()
    m(_mag(2, 64))
    assert not torch.equal(before, m.gru_norm.running_mean), \
        "train() 模式下 running_mean 未更新"


# ── 7. 反向传播与梯度 ───────────────────────────────────────────

def test_backward_reaches_all_parameters():
    """反传应覆盖全部可训练参数——有参数拿不到梯度就说明该模块成了死支路。"""
    torch.manual_seed(0)
    m = CRN_SMALL()
    m.train()
    m(_mag(2, 64)).mean().backward()

    missing = [name for name, p in m.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not missing, f"以下参数未获得有效梯度：{missing}"


# ── 8. 输入校验 ─────────────────────────────────────────────────

def test_rejects_wrong_ndim(model):
    with pytest.raises(ValueError, match="4 维"):
        model(torch.rand(N_MICS, N_FREQS, 100))


def test_rejects_wrong_mic_count(model):
    with pytest.raises(ValueError, match="通道数不符"):
        model(torch.rand(1, 6, N_FREQS, 100))


def test_rejects_wrong_freq_bins(model):
    with pytest.raises(ValueError, match="频率轴不符"):
        model(torch.rand(1, N_MICS, 129, 100))


def test_rejects_unknown_mask_act():
    with pytest.raises(ValueError, match="未知掩蔽激活"):
        CRN(mask_act="tanh")


# ── 9. 构造期行为 ───────────────────────────────────────────────

def test_depth_is_limited_by_freqs():
    """频率轴下采样层数应受 n_freqs 限制，且最深层保留 ≥ MIN_FREQ_BINS。"""
    m = CRN(n_freqs=257, encoder_channels=(16, 32, 64, 128, 256, 512))
    assert m.depth == len(m.encoder)
    assert m.enc_freq_sizes[-1] >= MIN_FREQ_BINS, \
        f"最深层频率轴只剩 {m.enc_freq_sizes[-1]} 个频点"

    # 频率轴很小时不应强行下采样
    small = CRN(n_freqs=9, encoder_channels=(16, 32, 64))
    assert small.depth == 1
    with torch.no_grad():
        out = small(torch.rand(1, N_MICS, 9, 20))
    assert out.shape == (1, 9, 20)


def test_freq_sizes_are_monotonic_and_match_simulation():
    """逐层频率轴长度必须单调不增，且首层等于卷积公式的结果。"""
    m = CRN()
    sizes = m.enc_freq_sizes
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] == (m.n_freqs - 1) // 2 + 1 == 129, \
        f"首层频率轴应为 129，收到 {sizes[0]}"


def test_non_power_of_two_freqs_also_shapes_correctly():
    """非 2 的幂的 n_freqs（如 129）也必须精确还原形状。"""
    m = CRN(n_freqs=129)
    m.eval()
    with torch.no_grad():
        out = m(torch.rand(1, N_MICS, 129, 77))
    assert out.shape == (1, 129, 77)


def test_base_channels_affects_decoder_width():
    """base_channels 必须真正生效（契约要求不留无用参数）。"""
    narrow = CRN(base_channels=16)
    default = CRN(base_channels=32)
    assert narrow._dec_channels != default._dec_channels, \
        "base_channels 未影响解码器宽度，参数是死参数"
    assert narrow.num_parameters < default.num_parameters
    assert narrow._dec_channels == [max(1, round(c * 16 / 32))
                                   for c in default._dec_channels]


# ── 10. mask_act 备选 ───────────────────────────────────────────

def test_relu_mask_act_has_no_upper_bound():
    """relu 掩蔽无上界（契约允许），但形状与有限性必须保住。"""
    m = CRN(mask_act="relu")
    m.eval()
    with torch.no_grad():
        mask = m(_mag(1, 64))
    assert mask.shape == (1, N_FREQS, 64)
    assert torch.isfinite(mask).all()
    assert (mask >= 0).all()


def test_mask_acts_list_matches_supported():
    """文档里的 MASK_ACTS 与实际实现一致。"""
    assert set(MASK_ACTS) == {"sigmoid", "relu"}
