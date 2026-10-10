"""T3.2 验收测试：数据集封装。

验收标准（实施计划 T3.2）：
    `DataLoader` 能产出 batch；随机取 3 个 batch 检查形状、值域、无 NaN。

本文件不依赖真实数据目录——除最后一个「真实数据」测试外，
其余用临时目录合成样本，保证数据集逻辑本身可独立测试。
"""

import csv

import numpy as np
import pytest

from nbse.dsp import HOP, N_FFT
from nbse.models.dataset import (
    MANIFEST_NAME,
    NbseDataset,
    build_sample,
    collate_samples,
    compute_targets,
    load_manifest,
    segment_frames,
)

FS = 16000


def _write_fake_sample(root, idx, n_mics=4, n=64000, seed=0, snr_target=0.0):
    """合成一条落盘样本：语音成分 + 噪声成分按目标 SNR 混合。

    构造方式刻意做成「可解析」——语音是 1 kHz 纯音、噪声是白噪，
    这样掩蔽的理论值可以独立算出来做交叉验证。
    """
    rng = np.random.default_rng(seed + idx)
    t = np.arange(n) / FS
    tone = np.sin(2 * np.pi * 1000.0 * t)

    clean = np.stack([np.roll(tone, m) for m in range(n_mics)])
    noise = 0.3 * rng.normal(size=(n_mics, n))

    # 按 0 号通道标定到目标 SNR
    sc = np.linalg.norm(clean[0])
    sn = np.linalg.norm(noise[0])
    noise = noise * (sc / (sn * 10 ** (snr_target / 20.0)))

    multi = clean + noise
    name = f"sample_{idx:05d}.npz"
    np.savez(root / name,
             multi_ch=multi.astype(np.float32),
             clean_ref=tone.astype(np.float32),
             noise_multi=noise.astype(np.float32),
             doa_true=np.float64(37.5),
             snr_db=np.float64(snr_target))

    return {
        "index": idx, "file": name, "subset": "dev-clean", "speaker": "1272",
        "chapter": "135031", "utterance": f"fake-{idx}.flac",
        "noise_env": "OOFFICE", "noise_file": "ch07.wav",
        "snr_target_db": f"{snr_target:.2f}", "snr_db": f"{snr_target:.4f}",
        "doa_true": "37.5000", "t60": "0.3000",
        "room_lx": "4.0000", "room_ly": "4.0000", "room_lz": "3.0000",
        "room_key": "deadbeef", "distance_m": "2.0000", "ref_delay": "100",
        "n_mics": n_mics, "sample_rate": FS, "n_samples": n,
    }


@pytest.fixture
def fake_data(tmp_path):
    """5 条合成样本 + manifest。"""
    root = tmp_path / "sim"
    root.mkdir()
    rows = [_write_fake_sample(root, i, seed=100, snr_target=-5.0 + 2.5 * i)
            for i in range(5)]
    with open(root / MANIFEST_NAME, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return root


# ── manifest ────────────────────────────────────────────────────

def test_load_manifest_reads_rows(fake_data):
    rows = load_manifest(fake_data)
    assert len(rows) == 5
    assert rows[0]["file"] == "sample_00000.npz"
    assert int(rows[0]["n_mics"]) == 4


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_manifest(tmp_path)


def test_segment_frames_matches_scipy():
    """片段帧数必须与 scipy 实际输出逐个长度吻合。

    这条测试是护栏而非形式：初版公式 `1 + L // hop` 只在 L 是 hop 整数倍时
    正确（如 64000 → 501），换成 8000/4000 就少算一帧。硬编码单个期望值
    抓不到这种错，所以这里对整组长度逐一比对 scipy 的真实输出。
    """
    from nbse.dsp import stft

    for length in (2000, 4000, 6000, 8000, 12000, 16000, 24000, 64000):
        x = np.random.default_rng(0).normal(size=length)
        assert stft(x).shape[1] == segment_frames(length), \
            f"L={length} 时帧数与 scipy 不符"


def test_segment_frames_rejects_too_short_segment():
    """片段短于 n_fft 时 scipy 会静默降低 nperseg，公式失效——必须显式报错。"""
    with pytest.raises(ValueError, match="小于 n_fft"):
        segment_frames(N_FFT - 1)


def test_segment_frames_uses_ceiling_not_integer_division():
    """非整数倍长度时必须是向上取整（回归测试）。"""
    assert segment_frames(16000) == 126      # 16000/128 = 125 整除 → 126
    assert segment_frames(8000) == 64        # 8000/128 = 62.5 → 63? 实测 64
    assert segment_frames(4000) == 33        # 4000/128 = 31.25 → 32? 实测 33
    # 关键性质：帧数单调不减，且不小于 1 + L//hop
    prev = 0
    for length in range(N_FFT, 20000, 977):
        got = segment_frames(length)
        assert got >= 1 + length // HOP
        assert got >= prev
        prev = got


# ── 掩蔽口径 ────────────────────────────────────────────────────

def test_mask_is_in_unit_interval_and_no_nan(fake_data):
    ds = NbseDataset(fake_data, segment_length=8000, random_crop=False)
    for i in range(len(ds)):
        s = ds[i]
        assert s["mask"].shape == (N_FFT // 2 + 1, segment_frames(8000))
        assert np.all(s["mask"] >= 0.0) and np.all(s["mask"] <= 1.0)
        assert not np.isnan(s["mask"]).any()
        assert not np.isnan(s["mag"]).any()


def test_mask_is_bounded_even_with_zero_noise():
    """噪声恒为零时掩蔽应恒为 1，而不是除零变成 inf/nan。

    这是最容易被忽略的退化情形：静音段在真实数据里大量存在。
    """
    n = 4000
    clean = np.random.default_rng(0).normal(size=(4, n))
    noise = np.zeros((4, n))

    mask, clean_mag = compute_targets(clean, noise)
    assert not np.isnan(mask).any()
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
    # 无噪声 → 掩蔽处处为 1
    np.testing.assert_allclose(mask, 1.0, atol=1e-6)
    assert not np.isnan(clean_mag).any()


def test_mask_drops_to_zero_when_speech_absent():
    """纯噪声段掩蔽应为 0（该处没有语音可保留）。"""
    n = 4000
    clean = np.zeros((4, n))
    noise = np.random.default_rng(1).normal(size=(4, n))

    mask, _ = compute_targets(clean, noise)
    np.testing.assert_allclose(mask, 0.0, atol=1e-6)


def test_mask_rises_with_snr(fake_data):
    """SNR 越高，平均掩蔽应越大——验证掩蔽确实反映了信噪比。"""
    ds = NbseDataset(fake_data, segment_length=8000, random_crop=False)
    means = [float(ds[i]["mask"].mean()) for i in range(len(ds))]

    # manifest 里 SNR 从 -5 dB 递增到 +5 dB
    assert means == sorted(means), f"掩蔽均值未随 SNR 单调上升：{means}"
    assert means[0] < means[-1]


def test_mag_channel_count_equals_n_mics(fake_data):
    ds = NbseDataset(fake_data, segment_length=8000, random_crop=False)
    assert ds.n_mics == 4
    assert ds[0]["mag"].shape[0] == 4


# ── 随机片段与可复现性 ──────────────────────────────────────────

def test_fixed_crop_is_deterministic(fake_data):
    ds = NbseDataset(fake_data, segment_length=8000, random_crop=False)
    a, b = ds[0], ds[0]
    assert a["meta"]["start"] == 0
    np.testing.assert_array_equal(a["mag"], b["mag"])


def test_random_crop_changes_with_epoch_but_is_reproducible(fake_data):
    """同 epoch 可复现、跨 epoch 变化 —— 这是「仿真即增广」的时域部分。"""
    ds = NbseDataset(fake_data, segment_length=8000, random_crop=True, seed=7)

    ds.set_epoch(0)
    a1, a2 = ds[0]["meta"]["start"], ds[0]["meta"]["start"]
    assert a1 == a2, "同一 epoch 内重复取样应一致"

    starts = {a1}
    for e in range(1, 12):
        ds.set_epoch(e)
        starts.add(ds[0]["meta"]["start"])
    assert len(starts) > 1, f"跨 epoch 片段窗口始终不变：{starts}"


def test_limit_truncates(fake_data):
    ds = NbseDataset(fake_data, segment_length=8000, limit=3)
    assert len(ds) == 3


def test_full_length_segment_makes_random_crop_a_noop(fake_data):
    """片段长度等于整条样本时，随机裁剪必须退化为「取整条」。

    这是长片段实验的前提：`random_crop=True` 下若片段不短于样本，
    total - segment_length + 1 == 0 会让 rng.integers(0, 0) 报错（或取到
    长度 0）。`train.py` 因此在 full_sample 时关闭随机裁剪，本测试锁住
    数据集这一侧的行为——即此时无论 epoch 如何变，取到的都是同一条全长样本。
    """
    total = 64000                     # fake_data 里每条样本就是 64000 点
    ds = NbseDataset(fake_data, segment_length=total, random_crop=True, seed=3)

    ds.set_epoch(0)
    a = ds[0]
    assert a["meta"]["start"] == 0
    assert a["mag"].shape[-1] == segment_frames(total)

    for e in (1, 5, 50):
        ds.set_epoch(e)
        b = ds[0]
        assert b["meta"]["start"] == 0, "片段已覆盖整条，起点必须恒为 0"
        np.testing.assert_array_equal(a["mag"], b["mag"])


def test_segment_longer_than_sample_is_rejected(fake_data):
    """片段长于样本时必须报错，而不是静默循环补齐。

    `_fit_segment` 对不足长度会循环平铺，那会造出「句尾接句首」的假音频。
    与其静默产出错误数据，不如直接拦住。
    """
    ds = NbseDataset(fake_data, segment_length=64000 + 128, random_crop=False)
    with pytest.raises(ValueError, match="超过样本长度"):
        ds[0]


def test_empty_limit_raises(fake_data):
    with pytest.raises(ValueError, match="为空"):
        NbseDataset(fake_data, limit=0)


# ── T3.2 核心验收：DataLoader 产 batch ──────────────────────────

def test_dataloader_produces_valid_batches(fake_data):
    """T3.2 验收：随机取 3 个 batch，检查形状、值域、无 NaN。

    5 条样本、batch=2 → 第 3 个 batch 只有 1 条（不 drop_last）。
    因此批量维断言写成「≤ batch_size 且 > 0」，而不是恒等于 2。
    """
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    ds = NbseDataset(fake_data, segment_length=8000, random_crop=True, seed=1)
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0,
                        collate_fn=collate_samples)

    n_seen = 0
    for inputs, targets in loader:
        mag, mask = inputs["mag"], targets["mask"]
        b = mag.shape[0]
        assert 0 < b <= 2, f"批量维异常：{b}"

        assert mag.shape == (b, 4, N_FFT // 2 + 1, segment_frames(8000))
        assert mask.shape == (b, N_FFT // 2 + 1, segment_frames(8000))
        assert torch.isfinite(mag).all() and torch.isfinite(mask).all()
        assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0
        assert len(inputs["meta"]) == b

        n_seen += 1
        if n_seen == 3:
            break

    assert n_seen == 3, "未能取出 3 个 batch"


def test_collate_preserves_batch_size_one(fake_data):
    pytest.importorskip("torch")
    ds = NbseDataset(fake_data, segment_length=4000, limit=1)
    inputs, targets = collate_samples([ds[0]])
    assert inputs["mag"].shape[0] == 1
    assert targets["mask"].shape[0] == 1


# ── 真实数据（无样本目录时跳过）─────────────────────────────────

REAL_DIR = "data/sim/train"


def test_real_sample_directory_if_present():
    """对真实落盘样本跑一遍——形状、值域、无 NaN。"""
    import pathlib

    if not (pathlib.Path(REAL_DIR) / MANIFEST_NAME).exists():
        pytest.skip(f"没有真实样本目录 {REAL_DIR}")

    ds = NbseDataset(REAL_DIR, segment_length=16000, random_crop=False, limit=3)
    assert ds.n_mics == 4
    for i in range(len(ds)):
        s = ds[i]
        assert s["mag"].shape == (4, 257, segment_frames(16000))
        assert s["mask"].shape == (257, segment_frames(16000))
        assert np.isfinite(s["mag"]).all()
        assert np.isfinite(s["mask"]).all()
        assert 0.0 <= float(s["mask"].min()) and float(s["mask"].max()) <= 1.0
        assert s["meta"]["t60"] > 0.0
