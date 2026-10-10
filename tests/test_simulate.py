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


def test_array_signature_tracks_geometry():
    """RIR 缓存文件名必须随阵列几何/采样率变化。

    只按阵元数命名时，把半径从 4 cm 改成 6 cm 会命中同一个缓存文件，
    静默复用错误的房间响应——这类错误不会报错，只会让结果悄悄失真。
    """
    from nbse.data.simulate import array_signature

    base = SimConfig()
    assert array_signature(base) == array_signature(SimConfig())

    for change in (
        dict(array=ArrayConfig(type="circular", n_mics=4, radius=0.06)),
        dict(array=ArrayConfig(type="circular", n_mics=4, radius=0.04, height=1.5)),
        dict(array=ArrayConfig(type="linear", n_mics=4, radius=0.04)),
        dict(sample_rate=8000),
        dict(array=ArrayConfig(type="circular", n_mics=8, radius=0.04)),
    ):
        assert array_signature(SimConfig(**change)) != array_signature(base), \
            f"{change} 没有改变阵列签名"


def test_array_signature_tracks_max_order_cap():
    """RIR 缓存文件名必须随 max_order_cap 变化。

    改上限会改变 RIR 的抽头数与拖尾。若不计入签名，改了口径后会读到旧口径的
    缓存，新旧 RIR 混在同一批数据里且**不会有任何报错**——只是结果悄悄失真。
    这条测试保证旧缓存会自动失效。
    """
    from nbse.data.simulate import array_signature

    for cap in (12, 20, 30, 50, None):
        assert array_signature(SimConfig(max_order_cap=cap)) != \
            array_signature(SimConfig(max_order_cap=31)), \
            f"max_order_cap={cap} 与 31 的签名相同，缓存不会失效"

    assert array_signature(SimConfig(max_order_cap=None)) != \
        array_signature(SimConfig(max_order_cap=30)), "None（完整仿真）必须与 30 区分"
    # 同值稳定
    assert array_signature(SimConfig(max_order_cap=30)) == \
        array_signature(SimConfig(max_order_cap=30))


def test_max_order_cap_actually_shrinks_rir_and_keeps_energy():
    """上限确实降低仿真规模，且几乎不损失 RIR 能量。

    两条容易被写错的性质，都在这里有断言：
      1. 自然阶数 **低于** 上限时，上限必须**不改变任何东西**（等价于完整仿真）；
      2. 自然阶数 **高于** 上限时才截断，且能量保留 > 99%。
    初版测试漏了 (1)，在一个自然阶数恰好 ≤ 30 的房间上断言「结果必须不同」，
    于是失败——那不是 bug，是把上限误当成「总会生效」。
    """
    pytest.importorskip("pyroomacoustics")
    import pyroomacoustics as pra
    from nbse.data.simulate import _room_and_rirs, sample_room

    rng = np.random.default_rng(20)
    cfg_full = SimConfig(max_order_cap=None)
    room = sample_room(cfg_full, rng)

    natural = pra.inverse_sabine(room.t60, room.size)[1]
    rir_full = _room_and_rirs(room, cfg_full, n_sources=2)

    # (1) 上限高于自然阶数 → 必须与完整仿真逐位相同
    rir_high = _room_and_rirs(room, SimConfig(max_order_cap=natural + 5),
                              n_sources=2)
    np.testing.assert_array_equal(
        rir_full, rir_high,
        err_msg="上限高于自然阶数时不应改变结果（否则说明截断逻辑写错了）")

    # 找一个自然阶数明显超过 30 的房间来验证截断效果
    cfg_cap = SimConfig(max_order_cap=30)
    big = None
    for _ in range(40):
        cand = sample_room(cfg_full, rng)
        if pra.inverse_sabine(cand.t60, cand.size)[1] > 30:
            big = cand
            break
    if big is None:
        pytest.skip("这一批随机房间里没有自然阶数 > 30 的，无法验证截断")

    full = _room_and_rirs(big, cfg_full, n_sources=2)
    cap = _room_and_rirs(big, cfg_cap, n_sources=2)

    assert cap.shape[-1] < full.shape[-1], \
        f"截断后 RIR 应变短：{cap.shape[-1]} vs {full.shape[-1]}"

    L = max(full.shape[-1], cap.shape[-1])
    a = np.zeros((*full.shape[:2], L)); a[..., :full.shape[-1]] = full
    b = np.zeros((*cap.shape[:2], L)); b[..., :cap.shape[-1]] = cap
    energy = np.linalg.norm(b) / np.linalg.norm(a)
    assert energy > 0.99, f"能量保留仅 {energy:.4f}，截断丢掉了有效信息"


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


def test_clean_ref_is_time_aligned_with_observation(sim_env):
    """参考信号必须与观测在时间上对齐 —— 否则所有逐样本指标全部失效。

    历史 bug：clean_ref 曾是未延迟的干声，而声源距阵列 1–3 m（传播时延
    3–9 ms）。SI-SDR 对整体时延零容忍，实测一条只是延迟 117 点的完美信号
    SI-SDR 只有 −59 dB，于是整批数据的指标被压在 −30 ~ −60 dB 的无意义区间。
    """
    from nbse.data.simulate import simulate_sample
    from nbse.metrics import si_sdr

    cfg, rng, clean, noise, cache = sim_env
    cfg.snr_db = 10.0                      # 高 SNR，让互相关由语音主导

    # 用带限信号（0–3.4 kHz，即语音频带）而非 fixture 里的白噪声：
    # 白噪声在任意非零整数滞后上都近乎正交，2 个采样点的残差就能把
    # SI-SDR 打到 −25 dB，无法反映真实语音的行为。
    n = int(cfg.sample_rate * cfg.duration)
    spec = np.fft.rfft(rng.normal(size=n))
    freqs = np.fft.rfftfreq(n, 1.0 / cfg.sample_rate)
    spec[freqs > 3400] = 0.0
    clean = np.fft.irfft(spec, n)

    out = simulate_sample(clean, noise, cfg, rng, cache_dir=cache)

    ref, obs = out["clean_ref"], out["multi_ch"][0]
    assert out["ref_delay"] > 0, "参考信号的时延应为正（声源总比参考点远）"

    # 在 ±8 采样点内找互相关峰
    lags = np.arange(-8, 9)
    n = len(ref)
    corr = []
    for k in lags:
        a, b = (ref[: n - k], obs[k:]) if k >= 0 else (ref[-k:], obs[: n + k])
        a, b = a - a.mean(), b - b.mean()
        corr.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    best = int(np.argmax(corr))
    best_lag, best_rho = int(lags[best]), corr[best]

    # 两条判据缺一不可：
    #   容差 3 点——参考信号对齐的是**阵列中心**（波束形成器的参考点），
    #   而这里比的是 0 号麦克风，两者相距半径 4 cm ≈ 1.9 个采样点。
    #   相关性——修复前参考信号整体晚了 50–150 点，在上述 ±8 窗口内相关性
    #   近乎为零，仅靠 argmax 会因噪声而在窗口内随机取值，必须配合相关性判据。
    assert abs(best_lag) <= 3, f"参考与观测相差 {best_lag} 个采样点，未对齐"
    assert best_rho >= 0.3, (
        f"最佳滞后处相关性仅 {best_rho:.3f}，参考信号可能整体错位（实测正常值约 0.5）"
    )

    # 不用 SI-SDR 设阈值：小房间里混响主导，对齐后它也只有几 dB 且随房间波动大，
    # 阈值会变得难以稳定。混响对该指标的影响由 T2.4 按 T60 分档来体现。
    assert np.isfinite(si_sdr(ref, obs))
