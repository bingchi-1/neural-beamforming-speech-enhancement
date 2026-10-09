"""T2.3 验收测试：评测指标。

SI-SDR 部分不依赖第三方包，可在任何环境运行。
PESQ / STOI 若无对应包则跳过。
"""

import numpy as np
import pytest

from nbse.metrics import si_sdr, si_sdr_improvement


def _orthogonal_pair(n=100, noise_amp=0.1):
    """构造解析可解的信号对：ref 与 noise 正交。

    取 ref 为全 1，noise 为交替 ±a —— 两者的内积恒为 0。
    于是 s_target 恰为 ref、e_noise 恰为 noise，
    SI-SDR = 10·log10(||ref||² / ||noise||²) = -20·log10(a)。
    """
    ref = np.ones(n)
    noise = np.empty(n)
    noise[0::2] = noise_amp
    noise[1::2] = -noise_amp
    return ref, noise


def test_si_sdr_matches_analytic_value():
    """估计 = 参考 + 正交噪声时，SI-SDR 应等于解析解 -20·log10(a)。"""
    ref, noise = _orthogonal_pair(noise_amp=0.1)
    est = ref + noise

    expected = -20 * np.log10(0.1)  # = 20 dB
    assert si_sdr(ref, est) == pytest.approx(expected, abs=1e-6)


def test_si_sdr_scale_invariance():
    """整体缩放不应改变 SI-SDR —— 这正是「尺度不变」的含义。"""
    rng = np.random.default_rng(0)
    ref = rng.normal(size=512)
    est = ref + 0.3 * rng.normal(size=512)

    base = si_sdr(ref, est)
    assert si_sdr(3.7 * ref, est) == pytest.approx(base, abs=1e-9)
    assert si_sdr(ref, 3.7 * est) == pytest.approx(base, abs=1e-9)


def test_si_sdr_perfect_reconstruction_is_infinite():
    """完美重建（含整体缩放）应返回 +inf。"""
    rng = np.random.default_rng(1)
    ref = rng.normal(size=256)
    assert si_sdr(ref, ref) == float("inf")
    assert si_sdr(ref, -2.5 * ref) == float("inf")


def test_si_sdr_silent_reference_is_nan():
    """静音参考下 SI-SDR 无定义，应返回 nan 而不是崩溃或返回假值。"""
    assert np.isnan(si_sdr(np.zeros(128), np.random.default_rng(2).normal(size=128)))


def test_si_sdr_length_mismatch_raises():
    with pytest.raises(ValueError, match="长度不一致"):
        si_sdr(np.ones(100), np.ones(101))


def test_si_sdr_improvement():
    """提升量 = 增强后 SI-SDR − 含噪输入 SI-SDR。"""
    ref, noise = _orthogonal_pair(noise_amp=0.1)
    noisy = ref + noise
    enhanced = ref + 0.1 * noise  # 噪声幅度降到 1/10

    gain = si_sdr_improvement(noisy, enhanced, ref)
    # 幅度 1/10 → 能量 1/100 → 提升 20 dB（SI-SDR 度量的是能量比）
    assert gain == pytest.approx(20.0, abs=1e-6)


# ── 依赖第三方包的部分：缺失时跳过 ──────────────────────────────

def test_pesq_self_comparison_is_maximal():
    pytest.importorskip("pesq")
    from nbse.metrics import pesq_wb

    rng = np.random.default_rng(3)
    ref = 0.3 * rng.normal(size=16000)
    score = pesq_wb(ref, ref.copy(), fs=16000)
    assert score > 4.0, "同一信号自比，PESQ 应接近上界 4.5"


def test_stoi_self_comparison_is_one():
    pytest.importorskip("pystoi")
    from nbse.metrics import stoi

    rng = np.random.default_rng(4)
    ref = 0.3 * rng.normal(size=16000)
    assert stoi(ref, ref.copy(), fs=16000) == pytest.approx(1.0, abs=1e-6)
