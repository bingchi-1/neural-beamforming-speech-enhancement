"""T1.4 验收测试之二：批量生成与落盘校验。

用假语料 + 小房间跑**真实的仿真路径**（含 pyroomacoustics 算 RIR），
只是把时长压到 0.25 s 让它快。真实语料上的 240 条由 CLI 生成。
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nbse.data.corpus import list_noise_files, list_utterances
from nbse.data.generate import (
    BatchSpec,
    generate_batch,
    load_manifest,
    verify_batch,
)
from nbse.data.simulate import SimConfig

pytest.importorskip("pyroomacoustics")

SR = 16000
DURATION = 0.25
N_MICS = 4


def _write_wav(path: Path, seconds: float, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x = (0.1 * rng.normal(size=(int(SR * seconds), 1))).astype(np.float32)
    sf.write(str(path), x, SR)


@pytest.fixture
def tiny_corpus(tmp_path):
    """2 条语音 + 2 条噪声，外加一个短时长的小房间配置。"""
    rng = np.random.default_rng(0)
    spk = "174"
    for utt in ("0001", "0002"):
        _write_wav(tmp_path / "LibriSpeech" / "dev-clean" / spk / "1"
                   / f"{spk}-1-{utt}.flac", seconds=1.0, seed=int(rng.integers(999)))
    for env, i in (("DKITCHEN", 1), ("OOFFICE", 2)):
        _write_wav(tmp_path / "DEMAND" / env / f"ch{i:02d}.wav",
                   seconds=1.0, seed=int(rng.integers(999)))

    utterances = list_utterances(tmp_path / "LibriSpeech", "dev-clean")
    noise_files = list_noise_files(tmp_path / "DEMAND")
    sim_cfg = SimConfig(
        sample_rate=SR, duration=DURATION,
        room_size_range=(3.0, 3.5), t60_range=(0.2, 0.3),
        source_distance_range=(1.0, 1.5), snr_db=0.0,
    )
    return tmp_path, utterances, noise_files, sim_cfg


def _spec(tmp_path, **kw) -> BatchSpec:
    defaults = dict(out_dir=tmp_path / "out", n_samples=4, seed=7)
    defaults.update(kw)
    return BatchSpec(**defaults)


def test_generate_writes_npz_manifest_and_run_config(tiny_corpus):
    root, utterances, noise_files, sim_cfg = tiny_corpus
    out = root / "out"

    rows = generate_batch(utterances, noise_files, sim_cfg, _spec(root),
                          cache_dir=root / "rirs")

    assert len(rows) == 4
    assert len(list(out.glob("sample_*.npz"))) == 4
    assert (out / "manifest.csv").exists()
    assert (out / "run.json").exists()
    assert len(load_manifest(out)) == 4

    # 落盘内容：形状、dtype、有限性
    with np.load(out / "sample_00000.npz") as z:
        n = int(SR * DURATION)
        assert z["multi_ch"].shape == (N_MICS, n)
        assert z["noise_multi"].shape == (N_MICS, n)
        assert z["clean_ref"].shape == (n,)
        assert z["multi_ch"].dtype == np.float32
        assert np.isfinite(z["multi_ch"]).all()
        assert 0.0 <= float(z["doa_true"]) < 360.0


def test_verify_batch_passes_on_generated_batch(tiny_corpus):
    root, utterances, noise_files, sim_cfg = tiny_corpus
    generate_batch(utterances, noise_files, sim_cfg, _spec(root))

    summary = verify_batch(root / "out")

    assert summary["n_samples"] == 4
    assert summary["n_samples_per_file"] == int(SR * DURATION)
    assert summary["duration_seconds"] == pytest.approx(DURATION)
    assert summary["sample_rate"] == SR
    assert summary["n_mics"] == N_MICS


def test_verify_batch_catches_inconsistent_length(tiny_corpus):
    """验收校验必须真的会失败 —— 否则它只是个摆设。"""
    root, utterances, noise_files, sim_cfg = tiny_corpus
    out = root / "out"
    generate_batch(utterances, noise_files, sim_cfg, _spec(root))

    # 把某条样本截短，模拟「时长不一致」
    bad = out / "sample_00001.npz"
    with np.load(bad) as z:
        data = {k: z[k] for k in z.files}
    data["multi_ch"] = data["multi_ch"][:, :-100]
    np.savez(bad, **data)

    with pytest.raises(AssertionError):
        verify_batch(out)


def test_snr_targets_are_hit_on_every_sample(tiny_corpus):
    """固定 SNR 档位时，每条样本的实测 SNR 都应在 ±0.5 dB 内（T1.3 的承诺）。"""
    root, utterances, noise_files, sim_cfg = tiny_corpus
    rows = generate_batch(utterances, noise_files, sim_cfg,
                          _spec(root, n_samples=6, snr_levels=(0.0, -5.0)))

    for r in rows:
        assert float(r["snr_db"]) == pytest.approx(float(r["snr_target_db"]), abs=0.5)


def test_manifest_always_records_snr_target(tiny_corpus):
    """随机 SNR 模式下也要写下目标值 —— 否则整批的标定精度无从复核。"""
    root, utterances, noise_files, sim_cfg = tiny_corpus
    rows = generate_batch(utterances, noise_files, sim_cfg, _spec(root, n_samples=4))

    for r in rows:
        target = float(r["snr_target_db"])
        assert sim_cfg.train_snr_range[0] <= target <= sim_cfg.train_snr_range[1]
        assert float(r["snr_db"]) == pytest.approx(target, abs=0.5)


def test_explicit_sim_snr_wins_over_random_range(tiny_corpus):
    """显式指定的 sim_cfg.snr_db 应优先于随机区间。"""
    root, utterances, noise_files, sim_cfg = tiny_corpus
    fixed = SimConfig(**{**sim_cfg.__dict__, "snr_db": 3.0})

    rows = generate_batch(utterances, noise_files, fixed, _spec(root, n_samples=3))

    assert {float(r["snr_target_db"]) for r in rows} == {3.0}


def test_t60_levels_are_respected(tiny_corpus):
    """给定 T60 档位时（评估用），房间的 t60 必须落在给定值上。"""
    root, utterances, noise_files, sim_cfg = tiny_corpus
    rows = generate_batch(utterances, noise_files, sim_cfg,
                          _spec(root, n_samples=4, t60_levels=(0.3,)))

    assert {float(r["t60"]) for r in rows} == {0.3}


def test_generation_is_deterministic(tiny_corpus):
    """同种子两次生成必须逐位一致 —— 可复现性（§4.5）的基础。"""
    root, utterances, noise_files, sim_cfg = tiny_corpus
    cache = root / "rirs"

    a = generate_batch(utterances, noise_files, sim_cfg,
                       _spec(root, out_dir=root / "a"), cache_dir=cache)
    b = generate_batch(utterances, noise_files, sim_cfg,
                       _spec(root, out_dir=root / "b"), cache_dir=cache)

    assert [r["file"] for r in a] == [r["file"] for r in b]
    assert [r["utterance"] for r in a] == [r["utterance"] for r in b]
    assert [r["noise_file"] for r in a] == [r["noise_file"] for r in b]
    assert [r["doa_true"] for r in a] == [r["doa_true"] for r in b]

    for row in a:
        with np.load(root / "a" / row["file"]) as za, \
             np.load(root / "b" / row["file"]) as zb:
            np.testing.assert_array_equal(za["multi_ch"], zb["multi_ch"])


def test_empty_corpus_raises(tiny_corpus):
    root, _, noise_files, sim_cfg = tiny_corpus
    with pytest.raises(ValueError, match="语料列表为空"):
        generate_batch([], noise_files, sim_cfg, _spec(root))
