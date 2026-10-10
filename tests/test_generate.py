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


# ── 配对评估网格 ────────────────────────────────────────────────

def test_grid_covers_every_condition_for_every_utterance(tiny_corpus):
    """网格 = 语料数 × SNR 档 × T60 档，且每条语料在每档条件下都出现一次。"""
    from nbse.data.generate import generate_grid

    root, utterances, noise_files, sim_cfg = tiny_corpus
    spec = _spec(root, snr_levels=(-5.0, 0.0, 5.0), t60_levels=(0.3, 0.6))
    rows = generate_grid(utterances, noise_files, sim_cfg, spec, n_utterances=2)

    assert len(rows) == 2 * 3 * 2
    seen = {(r["utterance"], r["snr_target_db"], r["t60"]) for r in rows}
    assert len(seen) == len(rows), "条件组合有重复"


def test_grid_holds_room_and_noise_fixed_across_conditions(tiny_corpus):
    """配对的关键：同一语料在各条件下必须复用同一套房间几何与同一条噪声。

    否则条件之间的差异会混入「房间不同」这一无关变量，条件间不可比。
    T60 与 SNR 是刻意操纵的变量，理应不同。
    """
    from nbse.data.generate import generate_grid

    root, utterances, noise_files, sim_cfg = tiny_corpus
    spec = _spec(root, snr_levels=(-5.0, 5.0), t60_levels=(0.3, 0.6))
    rows = generate_grid(utterances, noise_files, sim_cfg, spec, n_utterances=2)

    by_utt = {}
    for r in rows:
        by_utt.setdefault(r["utterance"], []).append(r)

    for utt, group in by_utt.items():
        assert len(group) == 4, f"{utt} 的条件数不对"
        for field in ("noise_file", "doa_true", "room_lx", "room_ly", "room_lz",
                      "distance_m"):
            assert len({r[field] for r in group}) == 1, \
                f"{utt} 的 {field} 在不同条件下不一致，配对被打断"
        assert {r["t60"] for r in group} == {"0.3000", "0.6000"}
        assert {r["snr_target_db"] for r in group} == {"-5.00", "5.00"}


def test_grid_requires_both_levels(tiny_corpus):
    from nbse.data.generate import generate_grid

    root, utterances, noise_files, sim_cfg = tiny_corpus
    with pytest.raises(ValueError, match="snr_levels"):
        spec = _spec(root, snr_levels=(-5.0,))       # 少了 t60_levels
        generate_grid(utterances, noise_files, sim_cfg, spec, n_utterances=2)


# ── 并行生成（n_jobs > 1）───────────────────────────────────────
# 并行的唯一风险是「结果变了」：作业顺序、随机抽样、缓存读写都可能受影响。
# 因此核心断言是**与串行逐位一致**，而不只是「跑得通」。

def test_parallel_matches_serial_bit_for_bit(tiny_corpus, tmp_path):
    """n_jobs=2 的产出必须与 n_jobs=1 逐位一致。

    这是并行化的验收标准：加速不能以改变数据为代价。若失败，优先检查
    作业列表是否仍在主进程按同一种子生成、以及结果是否按输入顺序回收。
    """
    root, utterances, noise_files, sim_cfg = tiny_corpus

    out_s = tmp_path / "serial"
    rows_s = generate_batch(utterances, noise_files, sim_cfg,
                            BatchSpec(out_dir=out_s, n_samples=6, seed=11,
                                      n_jobs=1),
                            cache_dir=tmp_path / "rirs_s")

    out_p = tmp_path / "parallel"
    rows_p = generate_batch(utterances, noise_files, sim_cfg,
                            BatchSpec(out_dir=out_p, n_samples=6, seed=11,
                                      n_jobs=2),
                            cache_dir=tmp_path / "rirs_p")

    assert len(rows_s) == len(rows_p) == 6

    # manifest 逐字段一致（index 也一致：行序必须稳定）
    for a, b in zip(rows_s, rows_p):
        assert a == b, f"manifest 行不一致：\n  {a}\n  {b}"

    # 落盘数组逐位一致
    for a, b in zip(rows_s, rows_p):
        with np.load(out_s / a["file"]) as za, np.load(out_p / b["file"]) as zb:
            assert set(za.files) == set(zb.files)
            for k in za.files:
                np.testing.assert_array_equal(
                    za[k], zb[k], err_msg=f"{a['file']} 的 {k} 与串行不一致")


def test_parallel_with_one_job_is_serial_path(tiny_corpus, tmp_path):
    """n_jobs=1 与缺省行为一致（走串行分支，不启动进程池）。"""
    root, utterances, noise_files, sim_cfg = tiny_corpus
    rows = generate_batch(utterances, noise_files, sim_cfg,
                          BatchSpec(out_dir=tmp_path / "a", n_samples=3, seed=5),
                          cache_dir=tmp_path / "rirs")
    assert len(rows) == 3
    assert (tmp_path / "a" / "manifest.csv").exists()


def test_run_config_records_max_order_cap(tiny_corpus, tmp_path):
    """run.json 必须记录 max_order_cap —— 改口径后要能追溯每批数据用的什么。

    若不记录，日后看到 RIR 抽头数变化将无从判断是配置改了还是出了 bug。
    """
    import json

    root, utterances, noise_files, sim_cfg = tiny_corpus
    out = tmp_path / "out"
    generate_batch(utterances, noise_files, sim_cfg,
                   BatchSpec(out_dir=out, n_samples=2, seed=3),
                   cache_dir=tmp_path / "rirs")

    cfg = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert "max_order_cap" in cfg, "run.json 未记录 max_order_cap"
    assert cfg["max_order_cap"] == sim_cfg.max_order_cap
