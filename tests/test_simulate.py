"""T1.3 验收测试：房间仿真。

分两类：
  1. 几何与信噪比标定 —— 纯 NumPy，无需 pyroomacoustics，任何环境都能跑。
  2. 端到端仿真 —— 需要 pyroomacoustics，缺库时跳过。

设计文档对 T1.3 的验收标准是「仿真管道产出的实际 SNR 应接近设定值（±0.5 dB）」。
"""

import numpy as np
import pytest

from nbse.data.simulate import (
    ArrayConfig,
    SimConfig,
    azimuth_to_position,
    convolve_rir,
    measure_snr,
    mic_positions,
    position_to_azimuth,
    sample_room,
    scale_to_snr,
    _fit_length,
)


# ── 1. 纯 NumPy 部分 ────────────────────────────────────────────

def test_circular_array_geometry():
    """4 元圆阵：4 个阵元，且各阵元到中心距离均为半径。"""
    cfg = ArrayConfig(type="circular", n_mics=4, radius=0.04, center_xy=(0, 0))
    pos = mic_positions(cfg)

    assert pos.shape == (3, 4)
    r = np.linalg.norm(pos[:2] - np.array([[0], [0]]), axis=0)
    np.testing.assert_allclose(r, 0.04, atol=1e-12)
    # 圆阵所有阵元等高
    np.testing.assert_allclose(pos[2], pos[2][0])


def test_linear_array_geometry():
    """线阵阵元等间距排布。"""
    cfg = ArrayConfig(type="linear", n_mics=8, spacing=0.04, center_xy=(0, 0))
    pos = mic_positions(cfg)

    assert pos.shape == (3, 8)
    np.testing.assert_allclose(np.diff(pos[0]),
                               np.full(7, 0.04), atol=1e-12)
    # 以中心对称
    assert pos[0].sum() == pytest.approx(0.0, abs=1e-12)


def test_azimuth_position_roundtrip():
    """方位角 → 坐标 → 方位角 应闭合（Oracle DOA 依赖这一点）。"""
    center = (4.0, 4.0)
    for az in (0.0, 37.5, 90.0, 180.0, 271.3, 359.9):
        pos = azimuth_to_position(az, 2.0, center, 1.0)
        back = position_to_azimuth(pos, center)
        assert back == pytest.approx(az, abs=1e-9)


def test_distance_is_preserved():
    pos = azimuth_to_position(123.0, 2.5, (0.0, 0.0), 1.0)
    assert np.hypot(pos[0], pos[1]) == pytest.approx(2.5, abs=1e-12)


def test_snr_scaling_hits_target_within_half_db():
    """T1.3 核心验收：标定后的实际 SNR 与目标值误差在 ±0.5 dB 内。"""
    rng = np.random.default_rng(0)
    clean = rng.normal(size=16000)
    noise = rng.normal(size=16000)

    for target in (-5.0, 0.0, 5.0, -10.0, 15.0):
        scale = scale_to_snr(clean, noise, target)
        measured = measure_snr(clean, noise * scale)
        assert measured == pytest.approx(target, abs=0.5), \
            f"目标 {target} dB，实测 {measured:.2f} dB"


def test_snr_scaling_survives_rir_convolution():
    """经 RIR 卷积后（含混响拖尾）SNR 标定依然准确 —— 这是仿真管道的真实路径。"""
    rng = np.random.default_rng(1)
    clean = rng.normal(size=8000)
    noise = rng.normal(size=8000)
    # 人为构造两条不同长度的 RIR，模拟语音与噪声走不同路径
    rir_a = np.concatenate([[1.0], 0.3 * rng.normal(size=200)])
    rir_b = np.concatenate([[0.8], 0.5 * rng.normal(size=500)])

    c = convolve_rir(clean, rir_a)
    n = convolve_rir(noise, rir_b)

    for target in (-5.0, 0.0, 5.0):
        scale = scale_to_snr(c, n, target)
        assert measure_snr(c, n * scale) == pytest.approx(target, abs=0.5)


def test_convolve_with_delta_is_identity():
    """与单位冲激卷积应原样返回（验证 FFT 卷积实现正确）。"""
    rng = np.random.default_rng(2)
    x = rng.normal(size=1000)
    delta = np.array([1.0])

    np.testing.assert_allclose(convolve_rir(x, delta), x, atol=1e-10)


def test_convolve_matches_numpy_convolve():
    """FFT 卷积应与 np.convolve 的对应区间一致。"""
    rng = np.random.default_rng(3)
    x = rng.normal(size=500)
    h = rng.normal(size=64)

    expected = np.convolve(x, h)[: len(x)]
    np.testing.assert_allclose(convolve_rir(x, h), expected, atol=1e-9)


def test_fit_length_pads_and_truncates():
    x = np.arange(10.0)
    np.testing.assert_array_equal(_fit_length(x, 5), np.arange(5.0))
    assert len(_fit_length(x, 25)) == 25
    np.testing.assert_array_equal(_fit_length(x, 20)[:10], x)  # 循环补齐


def test_sampled_room_keeps_sources_inside():
    """采样的房间配置中，声源与噪声源必须落在房间内部。"""
    cfg = SimConfig()
    rng = np.random.default_rng(4)

    for _ in range(50):
        room = sample_room(cfg, rng)
        margin = 0.3 - 1e-9
        for pos in (room.source_pos, room.noise_pos):
            assert np.all(pos >= margin), f"声源越界：{pos}"
            assert np.all(pos <= room.size - margin + 1e-9), f"声源越界：{pos}"
        assert cfg.t60_range[0] <= room.t60 <= cfg.t60_range[1]


def test_room_key_is_deterministic_and_discriminative():
    """RIR 缓存依赖 key 的稳定性：同配置同 key，异配置异 key。"""
    cfg = SimConfig()
    rng = np.random.default_rng(5)
    a = sample_room(cfg, rng)
    b = sample_room(cfg, rng)
    assert a.key() == a.key()
    assert a.key() != b.key()


# ── 2. 端到端仿真（需要 pyroomacoustics）────────────────────────

@pytest.fixture
def sim_env(tmp_path):
    """构造一个短时长、低混响的小房间，让端到端测试跑得快。"""
    pra = pytest.importorskip("pyroomacoustics")

    cfg = SimConfig(sample_rate=16000, duration=0.5,
                    room_size_range=(3.0, 3.5),
                    t60_range=(0.2, 0.3),
                    source_distance_range=(1.0, 1.5),
                    snr_db=0.0)
    rng = np.random.default_rng(42)
    clean = rng.normal(size=int(cfg.sample_rate * cfg.duration))
    noise = rng.normal(size=int(cfg.sample_rate * cfg.duration))
    return cfg, rng, clean, noise, tmp_path


def test_end_to_end_snr_is_accurate(sim_env):
    """端到端仿真产出的 SNR 应接近设定值 —— T1.3 的正式验收标准。"""
    from nbse.data.simulate import simulate_sample

    cfg, rng, clean, noise, cache = sim_env
    for target in (-5.0, 0.0, 5.0):
        cfg.snr_db = target
        out = simulate_sample(clean, noise, cfg, rng, cache_dir=cache)

        assert out["snr_db"] == pytest.approx(target, abs=0.5), \
            f"目标 {target} dB，实测 {out['snr_db']:.2f} dB"


def test_end_to_end_shapes_and_determinism(sim_env):
    from nbse.data.simulate import simulate_sample

    cfg, rng, clean, noise, cache = sim_env
    cfg.snr_db = 0.0

    a = simulate_sample(clean, noise, cfg, rng, cache_dir=cache)
    n = int(cfg.sample_rate * cfg.duration)

    assert a["multi_ch"].shape == (cfg.array.n_mics, n)
    assert a["noise_multi"].shape == (cfg.array.n_mics, n)
    assert a["clean_ref"].shape == (n,)
    assert 0.0 <= a["doa_true"] < 360.0
    assert not np.isnan(a["multi_ch"]).any()

    # 同一房间配置重跑应完全一致（RIR 缓存不能破坏可复现性）
    b = simulate_sample(clean, noise, cfg, rng, room_cfg=a["room"], cache_dir=cache)
    np.testing.assert_allclose(a["multi_ch"], b["multi_ch"], atol=1e-12)


def test_multi_channel_is_not_identical_across_mics(sim_env):
    """各通道必须互不相同 —— 否则阵列仿真没有意义。"""
    from nbse.data.simulate import simulate_sample

    cfg, rng, clean, noise, cache = sim_env
    cfg.snr_db = 0.0
    out = simulate_sample(clean, noise, cfg, rng, cache_dir=cache)
    multi = out["multi_ch"]

    for i in range(multi.shape[0]):
        for j in range(i + 1, multi.shape[0]):
            assert not np.allclose(multi[i], multi[j]), \
                f"通道 {i} 与 {j} 完全相同，阵列仿真未生效"
