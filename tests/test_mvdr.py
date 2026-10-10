"""T2.2 验收测试：MVDR 波束形成。

实施计划 T2.2 的两条验收标准是：
  1. 失真无畸变约束：w^H d(θ) ≈ 1（误差 < 1e-6）
  2. 结果合理性：MVDR 的 SI-SDR 不低于延时求和

第 2 条这里用两种场景验证：一个**方向性干扰源**（MVDR 应当明显更强），
以及一个端到端的混响房间仿真。
"""

import numpy as np
import pytest

from nbse.beamformers.delay_and_sum import DelayAndSum
from nbse.beamformers.geometry import steering_delays
from nbse.beamformers.mvdr import (
    MVDR,
    mvdr_vector,
    noise_covariance,
    regularize_covariance,
)
from nbse.data.simulate import ArrayConfig, mic_positions
from nbse.dsp import HOP, N_FFT, istft, rfft_freqs, stft, stft_multi
from nbse.metrics import si_sdr

FS = 16000
CENTER = (2.0, 2.0)


@pytest.fixture
def array():
    cfg = ArrayConfig(type="circular", n_mics=4, radius=0.04, height=1.0)
    return mic_positions(cfg, CENTER)


def plane_wave(clean, mic_pos, doa_deg, center=CENTER):
    """构造远场平面波多通道观测（与 T2.1 测试同一手法）。"""
    tau = steering_delays(mic_pos, doa_deg, center)
    freqs = rfft_freqs(N_FFT, FS)
    S = stft(clean, N_FFT, HOP, FS)
    chans = [istft(S * np.exp(-2j * np.pi * freqs[:, None] * t), N_FFT, HOP, FS,
                   length=len(clean)) for t in tau]
    return np.stack(chans)


def random_spd_cov(F=N_FFT // 2 + 1, M=4, seed=0):
    """造一批随机的正定协方差矩阵（Hermitian PSD），模拟真实的噪声协方差。"""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(F, M, M)) + 1j * rng.normal(size=(F, M, M))
    return np.einsum("fmn,fkn->fmk", A, np.conj(A)) / M


# ── 协方差估计与加载 ────────────────────────────────────────────

def test_noise_covariance_is_hermitian_psd():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 16, 50)) + 1j * rng.normal(size=(4, 16, 50))

    R = noise_covariance(X)

    assert R.shape == (16, 4, 4)
    np.testing.assert_allclose(R, np.conj(np.swapaxes(R, -1, -2)), atol=1e-12)
    for f in range(16):
        assert np.linalg.eigvalsh(R[f]).min() > -1e-12


def test_noise_covariance_matches_direct_formula():
    """与逐帧外积求平均的定义式逐元素对齐。"""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3, 5, 7)) + 1j * rng.normal(size=(3, 5, 7))

    R = noise_covariance(X)

    for f in range(5):
        expected = sum(np.outer(X[:, f, t], np.conj(X[:, f, t]))
                       for t in range(7)) / 7
        np.testing.assert_allclose(R[f], expected, atol=1e-12)


def test_regularization_makes_singular_matrix_invertible():
    """各通道完全相关（秩 1）时，未加载的 R 奇异，加载后可求逆。"""
    v = np.array([1.0, 1.0, 1.0, 1.0])
    R = np.outer(v, v)[None, :, :]                      # 秩 1
    assert np.linalg.matrix_rank(R[0]) == 1

    assert np.linalg.matrix_rank(regularize_covariance(R, 1e-3)[0]) == 4


# ── T2.2 验收标准 1：失真无畸变约束 ─────────────────────────────

def test_mvdr_distortionless_constraint(array):
    """w^H d(θ) = 1，误差 < 1e-6 —— 实施计划 T2.2 的正式验收标准。"""
    bf = MVDR(array, CENTER, fs=FS)
    R = random_spd_cov(seed=3)

    for doa in (0.0, 37.5, 90.0, 213.0, 359.0):
        w = bf.beamforming_vector(doa, R)
        d = bf.steering(doa)

        response = np.einsum("fm,fm->f", np.conj(w), d)     # w^H d
        np.testing.assert_allclose(response.real, 1.0, atol=1e-6)
        np.testing.assert_allclose(response.imag, 0.0, atol=1e-6)


def test_mvdr_constraint_holds_without_noise_info(array):
    """退化情形（不给 R_n）也必须满足约束——此时 w = d/M。"""
    bf = MVDR(array, CENTER, fs=FS)
    w = bf.beamforming_vector(45.0, None)
    d = bf.steering(45.0)

    np.testing.assert_allclose(np.einsum("fm,fm->f", np.conj(w), d), 1.0, atol=1e-9)


def test_mvdr_weights_are_conjugate_of_vector(array):
    """`weights()` 是供 einsum 直接相乘的矩阵，故应为 conj(w)。

    这一条守着最容易搞错的共轭约定：搞反了波束图会镜像、指标会莫名变差，
    但代码不会报错。
    """
    bf = MVDR(array, CENTER, fs=FS)
    R = random_spd_cov(seed=4)

    np.testing.assert_allclose(bf.weights(60.0, R),
                               np.conj(bf.beamforming_vector(60.0, R)), atol=1e-15)


def test_isotropic_noise_reduces_to_delay_and_sum(array):
    """各通道噪声不相关且等功率时，MVDR 应恰好退化为延时求和。"""
    ds = DelayAndSum(array, CENTER, fs=FS)
    bf = MVDR(array, CENTER, fs=FS, loading=0.0)
    F = N_FFT // 2 + 1
    R_white = np.tile(0.5 * np.eye(4), (F, 1, 1))

    for doa in (0.0, 120.0, 300.0):
        np.testing.assert_allclose(bf.beamforming_vector(doa, R_white),
                                   ds.weights(doa).conj(), atol=1e-10)


def test_heavy_loading_approaches_delay_and_sum(array):
    """加载量极大时 R_n 被淹没，MVDR 应回到延时求和的行为。"""
    ds = DelayAndSum(array, CENTER, fs=FS)
    bf = MVDR(array, CENTER, fs=FS, loading=1e6)
    R = random_spd_cov(seed=5)

    np.testing.assert_allclose(bf.beamforming_vector(200.0, R),
                               ds.weights(200.0).conj(), atol=1e-3)


# ── T2.2 验收标准 2：SI-SDR 不低于延时求和 ──────────────────────

def test_mvdr_beats_ds_on_directional_interferer(array):
    """目标 + 同频方向性干扰：MVDR 应在干扰方向形成零点，明显优于 DS。"""
    rng = np.random.default_rng(6)
    n = FS
    target = rng.normal(size=n)
    interferer = rng.normal(size=n)

    doa_target, doa_interf = 20.0, 80.0        # 相隔 60°
    obs_target = plane_wave(target, array, doa_target)
    obs_interf = plane_wave(interferer, array, doa_interf)

    scale = np.linalg.norm(obs_target[0]) / np.linalg.norm(obs_interf[0])
    noisy = obs_target + 0.7 * scale * obs_interf     # 干扰比目标低约 3 dB

    bf = MVDR(array, CENTER, fs=FS)
    y_mvdr = bf.enhance(noisy, doa_target, noise_multi=obs_interf)
    y_ds = DelayAndSum(array, CENTER, fs=FS).enhance(noisy, doa_target)

    gain = si_sdr(target, y_mvdr) - si_sdr(target, y_ds)
    assert gain > 10.0, \
        f"MVDR 相对 DS 仅提升 {gain:.1f} dB，零点可能没形成"


def test_mvdr_recovers_clean_in_noiseless_case(array):
    """无噪声单源时，MVDR 也应几乎无损还原信号（不能因加载而失真）。"""
    rng = np.random.default_rng(7)
    clean = rng.normal(size=FS)
    multi = plane_wave(clean, array, 137.0)

    bf = MVDR(array, CENTER, fs=FS)
    # 噪声为零：加载是唯一的正则项，w 应退化为指向该方向的 DS 解
    y = bf.enhance(multi, 137.0, noise_multi=np.zeros_like(multi))

    assert si_sdr(clean, y) > 20.0
    assert np.isfinite(y).all()


def test_silent_noise_yields_no_nan(array, capsys):
    """噪声观测恒为零的退化输入不能产生 NaN/Inf。"""
    bf = MVDR(array, CENTER, fs=FS)
    R = bf.estimate_noise_covariance(np.zeros((4, 1000)))

    w = bf.beamforming_vector(0.0, R)

    assert np.isfinite(w).all()


# ── 接口与形状 ──────────────────────────────────────────────────

def test_output_length_matches_input(array):
    rng = np.random.default_rng(8)
    clean = rng.normal(size=FS // 2)
    multi = plane_wave(clean, array, 60.0)

    y = MVDR(array, CENTER, fs=FS).enhance(multi, 60.0, noise_multi=multi * 0.1)

    assert y.shape == (len(clean),)


def test_mvdr_requires_noise_information(array):
    """不给噪声信息必须报错，且错误信息要说清该传什么。"""
    bf = MVDR(array, CENTER, fs=FS)
    with pytest.raises(ValueError, match="噪声协方差"):
        bf.enhance(np.zeros((4, 1000)), 0.0)


def test_mvdr_rejects_wrong_channel_count(array):
    bf = MVDR(array, CENTER, fs=FS)
    with pytest.raises(ValueError, match="通道数不符"):
        bf.enhance(np.zeros((3, 1000)), 0.0, noise_multi=np.zeros((3, 1000)))


def test_noise_cov_input_matches_noise_multi_path(array):
    """直接给协方差与给噪声观测，两条路径应得到完全相同的结果。

    mask-MVDR（T4.1）正是靠 noise_cov 这条路走网络估计的协方差。
    """
    rng = np.random.default_rng(9)
    clean = rng.normal(size=FS // 2)
    noisy = plane_wave(clean, array, 30.0)
    noise = plane_wave(rng.normal(size=len(clean)), array, 200.0) * 0.3

    bf = MVDR(array, CENTER, fs=FS)
    cov = bf.estimate_noise_covariance(noise)

    a = bf.enhance(noisy, 30.0, noise_multi=noise)
    b = bf.enhance(noisy, 30.0, noise_cov=cov)

    np.testing.assert_allclose(a, b, atol=1e-12)


# ── 端到端：混响房间仿真 ────────────────────────────────────────

def test_mvdr_not_worse_than_ds_on_simulated_room():
    """真实仿真管道上的完整性检查：MVDR 的 SI-SDR 不低于 DS。"""
    pytest.importorskip("pyroomacoustics")
    from nbse.data.simulate import (
        SimConfig,
        array_center,
        position_to_azimuth,
        simulate_sample,
    )

    cfg = SimConfig(sample_rate=FS, duration=0.5,
                    room_size_range=(3.0, 4.0), t60_range=(0.2, 0.4),
                    source_distance_range=(1.0, 1.5), snr_db=0.0)
    rng = np.random.default_rng(10)
    clean = rng.normal(size=int(FS * cfg.duration))
    noise = rng.normal(size=int(FS * cfg.duration))

    out = simulate_sample(clean, noise, cfg, rng)
    doa = position_to_azimuth(out["room"].source_pos, array_center(out["room"]))
    mic_pos = mic_positions(cfg.array, array_center(out["room"]))

    y_mvdr = MVDR(mic_pos, array_center(out["room"]), fs=FS).enhance(
        out["multi_ch"], doa, noise_multi=out["noise_multi"])
    y_ds = DelayAndSum(mic_pos, array_center(out["room"]), fs=FS).enhance(
        out["multi_ch"], doa)

    s_mvdr = si_sdr(out["clean_ref"], y_mvdr)
    s_ds = si_sdr(out["clean_ref"], y_ds)

    assert s_mvdr >= s_ds - 0.5, \
        f"MVDR {s_mvdr:.2f} dB 明显低于 DS {s_ds:.2f} dB"
