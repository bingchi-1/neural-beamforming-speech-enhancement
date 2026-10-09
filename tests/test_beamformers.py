"""T2.1 验收测试：延时求和波束形成。

核心思路：在频域**精确**构造远场平面波观测（各通道为干净信号的纯延迟版本），
这样波束形成器若补偿正确，就应当几乎无损地还原出原始信号。
用频域构造而非时域插值，是为了让延迟精确到机器精度，避免测试自身的误差
掩盖实现的问题。
"""

import numpy as np
import pytest

from nbse.beamformers.delay_and_sum import DelayAndSum
from nbse.beamformers.geometry import (
    direction_vector,
    steering_delays,
    steering_vector,
)
from nbse.data.simulate import ArrayConfig, mic_positions
from nbse.dsp import HOP, N_FFT, istft, rfft_freqs, stft
from nbse.metrics import si_sdr

FS = 16000
CENTER = (2.0, 2.0)


@pytest.fixture
def array():
    cfg = ArrayConfig(type="circular", n_mics=4, radius=0.04, height=1.0)
    return mic_positions(cfg, CENTER)


def plane_wave_observation(clean, mic_pos, doa_deg, center=CENTER,
                           fs=FS, n_fft=N_FFT, hop=HOP):
    """构造远场平面波多通道观测：X_m(f) = S(f)·exp(-j2πf·τ_m)。"""
    tau = steering_delays(mic_pos, doa_deg, center)
    freqs = rfft_freqs(n_fft, fs)
    S = stft(clean, n_fft, hop, fs)

    chans = []
    for t in tau:
        X = S * np.exp(-2j * np.pi * freqs[:, None] * t)
        chans.append(istft(X, n_fft, hop, fs, length=len(clean)))
    return np.stack(chans)


# ── 几何 ────────────────────────────────────────────────────────

def test_direction_vector_is_unit():
    for az in (0.0, 45.0, 137.0, 300.0):
        v = direction_vector(az)
        assert np.linalg.norm(v) == pytest.approx(1.0)
        assert v[2] == 0.0


def test_steering_vector_has_unit_magnitude(array):
    freqs = rfft_freqs(N_FFT, FS)
    d = steering_vector(array, 42.0, freqs, CENTER)
    np.testing.assert_allclose(np.abs(d), 1.0, atol=1e-12)


def test_steering_delay_sign_convention(array):
    """声源在 0° 方向时，位于 0° 的阵元离声源最近，应**最先**收到信号。"""
    tau = steering_delays(array, 0.0, CENTER)
    r = 0.04
    # 第一个阵元在 φ=0，即 [cx+r, cy]；沿 x 轴正方向距声源最近
    assert tau[0] == pytest.approx(-r / 343.0, rel=1e-9)
    # 它应当是所有通道中时延最小的
    assert tau[0] == tau.min()


def test_steering_delays_are_bounded_by_radius(array):
    """球面波下任意方位的时延绝对值不超过阵列半径 / 声速。"""
    limit = 0.04 / 343.0
    for az in np.linspace(0, 360, 37):
        tau = steering_delays(array, az, CENTER)
        assert np.abs(tau).max() <= limit + 1e-12


# ── T2.1 核心验收 ───────────────────────────────────────────────

def test_ds_recovers_clean_in_noiseless_single_source(array):
    """无噪声、单声源、无混响条件下，DS 输出应与干净信号高度一致。

    这是实施计划 T2.1 的验收标准：SI-SDR > 20 dB。
    """
    rng = np.random.default_rng(0)
    clean = rng.normal(size=FS)  # 1 秒宽带信号

    for doa in (0.0, 37.5, 90.0, 213.0, 359.0):
        multi = plane_wave_observation(clean, array, doa)
        bf = DelayAndSum(array, CENTER, fs=FS)
        y = bf.enhance(multi, doa)

        score = si_sdr(clean, y)
        assert score > 20.0, f"DOA={doa}° 时 SI-SDR 仅 {score:.1f} dB"


def test_ds_output_length_matches_input(array):
    rng = np.random.default_rng(1)
    clean = rng.normal(size=FS // 2)
    multi = plane_wave_observation(clean, array, 60.0)

    y = DelayAndSum(array, CENTER, fs=FS).enhance(multi, 60.0)
    assert y.shape == (len(clean),)


def test_ds_rejects_wrong_channel_count(array):
    bf = DelayAndSum(array, CENTER, fs=FS)
    with pytest.raises(ValueError, match="通道数不符"):
        bf.enhance(np.zeros((3, 1000)), 0.0)


def test_ds_suppresses_source_in_wrong_direction(array):
    """用错误的 DOA 做波束形成，输出应明显劣于用正确 DOA。

    这验证了「导向矢量确实在起作用」，而不是恒等变换。
    """
    rng = np.random.default_rng(2)
    clean = rng.normal(size=FS)

    true_doa = 30.0
    multi = plane_wave_observation(clean, array, true_doa)
    bf = DelayAndSum(array, CENTER, fs=FS)

    good = si_sdr(clean, bf.enhance(multi, true_doa))
    # 180° 反向：指向与声源相反的方向
    bad = si_sdr(clean, bf.enhance(multi, true_doa + 180.0))

    assert good - bad > 10.0, \
        f"正确/错误 DOA 的 SI-SDR 差距仅 {good - bad:.1f} dB，导向矢量可能未生效"


def test_ds_attenuates_uncorrelated_noise(array):
    """对互不相关的噪声，DS 应带来约 10·log10(M) dB 的增益。

    4 阵元理论上约 6 dB。这是波束形成最基本的性质，
    若此测试失败说明归一化方式有误。
    """
    rng = np.random.default_rng(3)
    n = FS
    clean = rng.normal(size=n)
    multi = plane_wave_observation(clean, array, 45.0)

    # 加入各通道独立的噪声
    noise_amp = 0.5
    noisy = multi + noise_amp * rng.normal(size=multi.shape)

    bf = DelayAndSum(array, CENTER, fs=FS)
    y = bf.enhance(noisy, 45.0)

    # 与「只取一个通道」对比
    ref_channel = noisy[0]
    gain = si_sdr(clean, y) - si_sdr(clean, ref_channel)

    assert gain > 3.0, f"阵列增益仅 {gain:.1f} dB，低于 4 阵元的预期（约 6 dB）"
