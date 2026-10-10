"""T4.1 验收测试：mask-MVDR。

测试策略
--------
核心行为用一个**解析可构造**的场景验证：目标语音与干扰源占据不相交的频段，
于是「理想二值掩蔽」就是频率选择器，mask-MVDR 应当据此把干扰压掉、把目标留下。

这比只测张量形状有说服力得多：它能抓住「掩蔽没用上」「协方差口径反了」
「导向矢量符号错了」这一类静默错误——这些错误形状全都正常。

掩蔽与协方差的口径细节（单位阵退化、同尺度归一化）单独测，因为它们是数值
稳定性的关键，出错时表现为偶发 NaN 而不是明显报错。
"""

import numpy as np
import pytest

from nbse.beamformers.geometry import steering_delays, steering_vector
from nbse.beamformers.mask_mvdr import (
    MaskMVDR,
    TorchMaskPredictor,
    beamforming_weights,
    mask_weighted_covariance,
)
from nbse.data.simulate import ArrayConfig, mic_positions
from nbse.dsp import HOP, N_FFT, istft, rfft_freqs, stft, stft_multi, stft_multi
from nbse.metrics import si_sdr

FS = 16000
CENTER = (2.0, 2.0)
F = N_FFT // 2 + 1


@pytest.fixture
def array():
    cfg = ArrayConfig(type="circular", n_mics=4, radius=0.04, height=1.0)
    return mic_positions(cfg, CENTER)


def plane_wave(clean, mic_pos, doa_deg, center=CENTER):
    """构造远场平面波多通道观测（与 test_beamformers.py 同一手法）。"""
    tau = steering_delays(mic_pos, doa_deg, center)
    freqs = rfft_freqs(N_FFT, FS)
    S = stft(clean, N_FFT, HOP, FS)
    chans = []
    for t in tau:
        X = S * np.exp(-2j * np.pi * freqs[:, None] * t)
        chans.append(istft(X, N_FFT, HOP, FS, length=len(clean)))
    return np.stack(chans)


def band_mask(freqs, lo, hi):
    """频带选择掩蔽：频段内为 1，其余为 0。"""
    m = ((freqs >= lo) & (freqs <= hi)).astype(np.float64)
    return m


# ── 掩蔽加权协方差 ──────────────────────────────────────────────

def test_mask_weighted_covariance_shapes():
    X = np.random.default_rng(0).normal(size=(4, F, 30)) + \
        1j * np.random.default_rng(1).normal(size=(4, F, 30))
    mask = np.random.default_rng(2).random(size=(F, 30))

    R = mask_weighted_covariance(X, mask)
    assert R.shape == (F, 4, 4)
    # 协方差矩阵必须 Hermitian
    np.testing.assert_allclose(R, np.conj(R.transpose(0, 2, 1)), atol=1e-10)
    # 且主对角（各通道功率）非负
    assert np.all(np.diagonal(R, axis1=1, axis2=2).real >= -1e-12)


def test_all_ones_mask_gives_plain_covariance():
    """掩蔽全 1 时，加权协方差应等于普通样本协方差。"""
    X = np.random.default_rng(3).normal(size=(4, F, 25)) + \
        1j * np.random.default_rng(4).normal(size=(4, F, 25))
    mask = np.ones((F, 25))

    R = mask_weighted_covariance(X, mask)
    R_ref = np.einsum("mft,nft->fmn", X, np.conj(X)) / X.shape[2]
    np.testing.assert_allclose(R, R_ref, rtol=1e-10, atol=1e-12)


def test_all_zero_mask_returns_identity_not_zeros():
    """掩蔽全 0 时返回单位阵，而不是零矩阵。

    零矩阵求逆会直接产生 inf/nan 并污染整条链路；单位阵在对角加载后等价于
    「该频点不做自适应」，是安全的退让。
    """
    X = np.random.default_rng(5).normal(size=(4, F, 20)) + \
        1j * np.random.default_rng(6).normal(size=(4, F, 20))
    R = mask_weighted_covariance(X, np.zeros((F, 20)))

    for f in range(F):
        np.testing.assert_allclose(R[f], np.eye(4), atol=1e-12)
    assert np.isfinite(R).all()


def test_covariance_normalization_uses_weight_sum_not_frame_count():
    """归一化分母必须是「权重平方和」，不是帧数 —— 尺度一致性的回归测试。

    为什么这条重要：R_s 与 R_n 必须处在同一尺度，广义特征值形式
    （`mode="gevd"`）才成立。而**纯 MVDR 路径对 R_n 的绝对尺度不变**，
    所以就算归一化写错，默认路径看起来也完全正常，只有 gevd 会静默变差。

    构造上要注意：用**均匀掩蔽**测不出这个问题——那时加权平均退化为普通
    平均功率，除以帧数还是除以权重和结果相同。必须用**非均匀掩蔽**，
    让两种归一化产生不同结果。

    解析可算的构造：只保留一帧（掩蔽在该帧为 1、其余为 0），
    则权重平方和 = 1，R 应恰等于该帧的外积；
    若误用帧数归一化，会得到外积 / n_frames，相差 n_frames 倍。
    """
    rng = np.random.default_rng(7)
    T = 24
    X = rng.normal(size=(4, F, T)) + 1j * rng.normal(size=(4, F, T))

    # 只在第 5 帧有掩蔽
    mask = np.zeros((F, T))
    mask[:, 5] = 1.0

    R = mask_weighted_covariance(X, mask)

    # 期望：恰好是该帧的外积（权重平方和 = 1）
    x5 = X[:, :, 5]                                   # [M, F]
    expected = np.einsum("mf,nf->fmn", x5, np.conj(x5))
    np.testing.assert_allclose(R, expected, rtol=1e-10, atol=1e-12)

    # 反向确认：若分母误用帧数，结果会小 T 倍
    wrong = expected / T
    assert not np.allclose(R, wrong, rtol=1e-6), \
        "结果与「按帧数归一化」的错值相同，说明分母可能写错了"

    # 迹也必须是该帧的通道功率和，而不是它的 1/T
    tr = np.trace(R, axis1=1, axis2=2).real
    np.testing.assert_allclose(tr, np.sum(np.abs(x5) ** 2, axis=0), rtol=1e-10)


def test_speech_and_noise_covariances_are_same_scale():
    """非均匀掩蔽下，R_s 与 R_n 的尺度关系必须与权重平方和之比一致。

    这是尺度一致性的**性质版**测试（上一条测的是单帧这种极端构造）。

    推导：R 的迹 = Σ_t w²·‖x‖² / Σ_t w²，即「按 w² 加权的平均帧功率」，
    是**频率相关**的（取决于该频点各帧的功率分布），不能简化成权重比的常数。
    因此测试要独立算出这个加权平均来比对，而不是假设它只依赖掩蔽。

    构造：让时间包络按 (T-1-t) 衰减，于是 ‖x[f,t]‖² ∝ (T-1-t)²，
    加权平均可解析计算。
    """
    rng = np.random.default_rng(17)
    T = 30
    env = (T - 1 - np.arange(T)).astype(np.float64)     # 线性衰减
    base = rng.normal(size=(4, F)) + 1j * rng.normal(size=(4, F))
    X = base[:, :, None] * env[None, None, :]           # [M, F, T]

    mask = rng.random(size=(F, T))                       # 非均匀

    power = np.sum(np.abs(X) ** 2, axis=0)               # [F, T]（各帧总功率）
    w_s = mask ** 2
    w_n = (1.0 - mask) ** 2

    def weighted_mean(w):
        return (w * power).sum(axis=1) / w.sum(axis=1)

    R_s = mask_weighted_covariance(X, mask)
    R_n = mask_weighted_covariance(X, 1.0 - mask)

    tr_s = np.trace(R_s, axis1=1, axis2=2).real
    tr_n = np.trace(R_n, axis1=1, axis2=2).real

    np.testing.assert_allclose(tr_s, weighted_mean(w_s), rtol=1e-9)
    np.testing.assert_allclose(tr_n, weighted_mean(w_n), rtol=1e-9)

    # 比值 = 两个加权平均之比（不是常数，故只断言按上式成立）
    np.testing.assert_allclose(
        tr_s / tr_n, weighted_mean(w_s) / weighted_mean(w_n), rtol=1e-9)


def test_covariance_rejects_mask_shape_mismatch():
    X = np.zeros((4, F, 10), dtype=complex)
    with pytest.raises(ValueError, match="掩蔽形状"):
        mask_weighted_covariance(X, np.ones((F, 11)))
    with pytest.raises(ValueError, match=r"\[M, F, T\]"):
        mask_weighted_covariance(np.zeros((4, F)), np.ones((F, 10)))


# ── 波束形成权重 ────────────────────────────────────────────────

def test_weights_satisfy_distortionless_constraint(array):
    """两种模式都必须满足 w^H d = 1（与 T2.2 同一验收标准）。"""
    rng = np.random.default_rng(8)
    freqs = rfft_freqs(N_FFT, FS)
    d = steering_vector(array, 45.0, freqs, CENTER)

    A = rng.normal(size=(F, 4, 4)) + 1j * rng.normal(size=(F, 4, 4))
    R_s = A @ np.conj(A.transpose(0, 2, 1))          # 半正定
    B = rng.normal(size=(F, 4, 4)) + 1j * rng.normal(size=(F, 4, 4))
    R_n = B @ np.conj(B.transpose(0, 2, 1)) + 0.1 * np.eye(4)

    for mode in ("mvdr", "gevd"):
        w = beamforming_weights(R_s, R_n, d, loading=1e-2, mode=mode)
        assert w.shape == (F, 4)
        got = np.einsum("fm,fm->f", np.conj(d), w)
        np.testing.assert_allclose(got.real, 1.0, atol=1e-6,
                                   err_msg=f"{mode} 未满足失真无畸变约束")
        np.testing.assert_allclose(got.imag, 0.0, atol=1e-6)
        assert np.isfinite(w).all()


def test_weights_reject_unknown_mode(array):
    freqs = rfft_freqs(N_FFT, FS)
    d = steering_vector(array, 0.0, freqs, CENTER)
    R = np.tile(np.eye(4), (F, 1, 1))
    with pytest.raises(ValueError, match="未知模式"):
        beamforming_weights(R, R, d, mode="nope")


# ── T4.1 核心行为：掩蔽确实在用 ─────────────────────────────────

def test_mask_mvdr_suppresses_out_of_band_interferer(array):
    """核心行为：目标 1 kHz、干扰 4 kHz，理想频带掩蔽应把干扰压掉。

    这条测的是整条链路——掩蔽 → 协方差 → 权重 → 重建。任何一环口径写反
    （例如噪声掩蔽误用 m、权重误取 conj 之外的形式）都会让它失败。
    """
    rng = np.random.default_rng(9)
    n = FS
    t = np.arange(n) / FS
    target = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    interf = 0.5 * np.sin(2 * np.pi * 4000.0 * t)

    doa_target, doa_interf = 0.0, 90.0
    multi = plane_wave(target, array, doa_target) + \
        plane_wave(interf, array, doa_interf)

    # 观测（0 号通道）作为基线
    noisy = multi[0]
    base = si_sdr(target, noisy)

    # 理想频带掩蔽：保留目标频段
    freqs = rfft_freqs(N_FFT, FS)
    mask = np.tile(band_mask(freqs, 700.0, 1400.0)[:, None], (1, stft(target).shape[1]))

    bf = MaskMVDR(array, CENTER, mask_predictor=None, loading=1e-2)
    y = bf.enhance(multi, doa_target, mask=mask)

    gain = si_sdr(target, y) - base
    assert gain > 10.0, f"掩蔽未生效：SI-SDR 仅提升 {gain:.1f} dB"
    assert si_sdr(target, y) > 10.0, f"增强后 SI-SDR 仅 {si_sdr(target, y):.1f} dB"


def test_mask_matters_by_construction(array):
    """换一个掩蔽必须换一个结果 —— 否则说明掩蔽没进入权重。"""
    rng = np.random.default_rng(10)
    n = FS
    t = np.arange(n) / FS
    target = 0.4 * np.sin(2 * np.pi * 1000.0 * t)
    interf = 0.4 * np.sin(2 * np.pi * 4000.0 * t)
    multi = plane_wave(target, array, 0.0) + plane_wave(interf, array, 90.0)

    freqs = rfft_freqs(N_FFT, FS)
    T = stft(target).shape[1]
    m_target = np.tile(band_mask(freqs, 700.0, 1400.0)[:, None], (1, T))
    m_interf = np.tile(band_mask(freqs, 3500.0, 4500.0)[:, None], (1, T))

    bf = MaskMVDR(array, CENTER, mask_predictor=None)
    w_good = bf.weights(multi, 0.0, mask=m_target)
    w_bad = bf.weights(multi, 0.0, mask=m_interf)

    assert not np.allclose(w_good, w_bad), "掩蔽没有影响权重"
    # 用错掩蔽（保留了干扰频段）的效果必须明显更差
    y_good = bf.enhance(multi, 0.0, mask=m_target)
    y_bad = bf.enhance(multi, 0.0, mask=m_interf)
    assert si_sdr(target, y_good) > si_sdr(target, y_bad) + 5.0, \
        "正确掩蔽与错误掩蔽的结果差异过小"


def test_mask_mvdr_beats_observation_covariance_mvdr(array):
    """掩蔽驱动估的 R_n 应优于「直接用观测协方差当噪声协方差」。"""
    rng = np.random.default_rng(11)
    n = FS
    t = np.arange(n) / FS
    target = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    interf = 0.5 * np.sin(2 * np.pi * 4000.0 * t)
    multi = plane_wave(target, array, 0.0) + plane_wave(interf, array, 90.0)

    freqs = rfft_freqs(N_FFT, FS)
    T = stft(target).shape[1]
    mask = np.tile(band_mask(freqs, 700.0, 1400.0)[:, None], (1, T))

    bf_masked = MaskMVDR(array, CENTER, mask_predictor=None, weighting="noise")
    bf_plain = MaskMVDR(array, CENTER, mask_predictor=None, weighting="speech")

    masked = si_sdr(target, bf_masked.enhance(multi, 0.0, mask=mask))
    plain = si_sdr(target, bf_plain.enhance(multi, 0.0, mask=mask))

    assert masked > plain, \
        f"掩蔽估 R_n（{masked:.1f} dB）未优于观测协方差当 R_n（{plain:.1f} dB）"


def test_weighting_noise_vs_both_uses_rs_only_in_gevd(array):
    """`R_s` 只在 gevd 模式下影响结果；mvdr 模式下必须完全不影响。

    这是默认语义的回归测试。注意区分两件事：
      * `mvdr` 模式的解是 `w = R_n⁻¹ d / (d^H R_n⁻¹ d)`，**根本不出现 R_s**，
        所以 weighting 换成 both 结果必须逐位相同——初版测试误以为这里会有差别；
      * `gevd` 模式解 (R_s, R_n) 的广义特征值问题，**才会**用到 R_s。
    """
    rng = np.random.default_rng(18)
    multi = rng.normal(size=(4, 4000))
    T = stft(multi[0]).shape[1]
    mask = np.clip(rng.random(size=(F, T)), 0.05, 0.95)

    # mvdr 模式：R_s 不参与 → 两种 weighting 完全一致
    a = MaskMVDR(array, CENTER, mask_predictor=None, weighting="noise",
                 mode="mvdr")
    b = MaskMVDR(array, CENTER, mask_predictor=None, weighting="both",
                 mode="mvdr")
    np.testing.assert_allclose(a.enhance(multi, 0.0, mask=mask),
                               b.enhance(multi, 0.0, mask=mask), atol=1e-12)

    # gevd 模式：R_s 参与 → 必须不同
    c = MaskMVDR(array, CENTER, mask_predictor=None, weighting="noise",
                 mode="gevd")
    d = MaskMVDR(array, CENTER, mask_predictor=None, weighting="both",
                 mode="gevd")
    assert not np.allclose(c.enhance(multi, 0.0, mask=mask),
                           d.enhance(multi, 0.0, mask=mask)), \
        "gevd 模式下 weighting 未改变结果，说明 R_s 没被用上"

    # weighting="noise" 时返回的 R_s 是观测协方差（与掩蔽无关）
    X = stft_multi(multi)
    R_s, _ = a.estimate_covariances(X, mask)
    np.testing.assert_allclose(R_s, mask_weighted_covariance(
        X, np.ones((F, T))), atol=1e-12)


def test_unknown_weighting_raises(array):
    with pytest.raises(ValueError, match="未知 weighting"):
        MaskMVDR(array, CENTER, mask_predictor=None, weighting="magic")


def test_legacy_use_noise_mask_flag_maps_to_weighting(array):
    """旧的布尔参数必须映射到新语义，不能静默改变行为。"""
    assert MaskMVDR(array, CENTER, use_noise_mask=True).weighting == "both"
    assert MaskMVDR(array, CENTER, use_noise_mask=False).weighting == "speech"
    assert MaskMVDR(array, CENTER).weighting == "noise"


# ── 接口与错误处理 ──────────────────────────────────────────────

def test_missing_mask_and_predictor_raises(array):
    """两者都缺时必须报错，不能静默退化成 DS。

    静默退化会让「神经方法优于基线」这一结论变成假的——那正是最危险的失败模式。
    """
    bf = MaskMVDR(array, CENTER, mask_predictor=None)
    with pytest.raises(ValueError, match="没有掩蔽可用"):
        bf.enhance(np.zeros((4, 1000)), 0.0)
    with pytest.raises(ValueError, match="没有掩蔽可用"):
        bf.weights(np.zeros((4, 1000)), 0.0)


def test_predictor_is_used_when_mask_not_given(array):
    """给了 predictor 时应被调用，且其输出被当作掩蔽使用。"""
    calls = {"n": 0}

    def fake_predictor(multi_ch):
        calls["n"] += 1
        T = stft(multi_ch[0]).shape[1]
        return np.full((F, T), 0.7)

    bf = MaskMVDR(array, CENTER, mask_predictor=fake_predictor)
    multi = np.random.default_rng(12).normal(size=(4, 2000))

    y = bf.enhance(multi, 30.0)
    assert calls["n"] == 1
    assert y.shape == (2000,)
    assert np.isfinite(y).all()


def test_wrong_channel_count_raises(array):
    bf = MaskMVDR(array, CENTER, mask_predictor=None)
    with pytest.raises(ValueError, match="通道数不符"):
        bf.enhance(np.zeros((3, 1000)), 0.0, mask=np.ones((F, 9)))


def test_mask_frame_count_tolerated(array):
    """掩蔽帧数与观测帧数不一致时应能容错（裁剪/补齐），而不是直接抛错。

    这是为「用 1 s 片段训练的模型推理更长观测」留的兜底路径。
    """
    bf = MaskMVDR(array, CENTER, mask_predictor=None)
    multi = np.random.default_rng(13).normal(size=(4, 4000))
    T_obs = stft(multi[0]).shape[1]

    short = np.full((F, T_obs - 3), 0.5)
    long = np.full((F, T_obs + 5), 0.5)

    for m in (short, long):
        y = bf.enhance(multi, 0.0, mask=m)
        assert y.shape == (4000,)
        assert np.isfinite(y).all()


def test_output_length_matches_input(array):
    rng = np.random.default_rng(14)
    for n in (16000, 8000, 4000):
        multi = rng.normal(size=(4, n))
        mask = np.full((F, stft(multi[0]).shape[1]), 0.5)
        bf = MaskMVDR(array, CENTER, mask_predictor=None)
        assert bf.enhance(multi, 0.0, mask=mask).shape == (n,)


def test_no_nan_with_degenerate_masks(array):
    """全 0 / 全 1 掩蔽都不能产生 NaN —— 退化掩蔽在真实推理里很常见。"""
    rng = np.random.default_rng(15)
    multi = rng.normal(size=(4, 4000))
    T = stft(multi[0]).shape[1]
    bf = MaskMVDR(array, CENTER, mask_predictor=None)

    for val in (0.0, 1.0):
        y = bf.enhance(multi, 0.0, mask=np.full((F, T), val))
        assert np.isfinite(y).all(), f"掩蔽全 {val} 时出现非有限值"
