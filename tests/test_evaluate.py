"""T2.4 验收测试：评测入口。

纯函数（聚合、排序检查、CSV 落盘）直接测；端到端部分用真实仿真管道
生成 2 条短样本再评测，确认「读样本 → 跑方法 → 算指标 → 出表」整条链路通。
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nbse.data.corpus import list_noise_files, list_utterances
from nbse.data.generate import BatchSpec, generate_batch, load_manifest
from nbse.data.simulate import SimConfig
from nbse.evaluate import (
    DEFAULT_METHODS,
    aggregate,
    align_level,
    check_expected_order,
    condition_of,
    format_table,
    list_conditions,
    run_methods,
    write_csv,
)

SR = 16000


# ── 纯函数 ──────────────────────────────────────────────────────

def test_align_level_matches_peaks():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=100)
    est = rng.normal(size=100) * 0.01

    out = align_level(ref, est)

    assert np.max(np.abs(out)) == pytest.approx(np.max(np.abs(ref)))


def test_align_level_leaves_silent_estimate_alone():
    out = align_level(np.ones(10), np.zeros(10))
    assert np.all(out == 0)


def test_condition_and_list_conditions():
    rows = [{"snr_target_db": "0.00", "t60": "0.6000"},
            {"snr_target_db": "-5.00", "t60": "0.3000"},
            {"snr_target_db": "-5.00", "t60": "0.3000"}]

    assert condition_of(rows[0]) == (0.0, 0.6)
    # 去重 + 按 SNR 升序、T60 升序
    assert list_conditions(rows) == [(-5.0, 0.3), (0.0, 0.6)]


def _record(method, snr, t60, si_sdr, pesq=1.0, stoi=0.5):
    return {"method": method, "snr_db": snr, "t60": t60, "utterance": "x",
            "pesq_wb": pesq, "stoi": stoi, "si_sdr": si_sdr}


def test_aggregate_means_and_adds_overall_row():
    records = [
        _record("DS", -5.0, 0.3, -10.0), _record("DS", -5.0, 0.3, -6.0),
        _record("MVDR", -5.0, 0.3, -2.0),
    ]

    rows = aggregate(records)

    cell = [r for r in rows if r["method"] == "DS" and r["snr_db"] == -5.0][0]
    assert cell["n"] == 2 and cell["si_sdr"] == pytest.approx(-8.0)

    overall = {r["method"]: r for r in rows if r["snr_db"] == "ALL"}
    assert overall["DS"]["n"] == 2
    assert overall["DS"]["si_sdr"] == pytest.approx(-8.0)
    assert overall["MVDR"]["si_sdr"] == pytest.approx(-2.0)


def test_check_expected_order_passes_and_fails():
    good = [_record(m, "ALL", "ALL", v) for m, v in
            zip(DEFAULT_METHODS, (-10.0, -8.0, -3.0))]
    ok, chain = check_expected_order(good)
    assert ok and "Noisy" in chain

    # 排序倒挂必须被抓住 —— 否则这个检查只是摆设
    bad = [_record(m, "ALL", "ALL", v) for m, v in
           zip(DEFAULT_METHODS, (-3.0, -8.0, -10.0))]
    assert not check_expected_order(bad)[0]


def test_check_expected_order_reports_missing_method():
    ok, msg = check_expected_order([_record("DS", "ALL", "ALL", -8.0)])
    assert not ok and "缺少方法" in msg


def test_write_csv_roundtrip(tmp_path):
    rows = aggregate([_record("DS", -5.0, 0.3, -10.0)])

    path = write_csv(rows, tmp_path / "sub" / "out.csv")

    assert path.exists()
    text = path.read_text(encoding="utf-8-sig")
    assert "method,snr_db,t60,n,pesq_wb,stoi,si_sdr" in text
    assert "ALL" in text


def test_format_table_contains_methods_and_conditions():
    rows = aggregate([_record("DS", -5.0, 0.3, -10.0),
                      _record("Noisy", -5.0, 0.3, -12.0)])

    table = format_table(rows, "si_sdr")

    assert "DS" in table and "Noisy" in table
    assert "SNR -5.0/T60 0.3" in table


# ── 端到端：真实仿真管道 → 评测 ─────────────────────────────────

@pytest.fixture
def eval_dir(tmp_path):
    pytest.importorskip("pyroomacoustics")

    rng = np.random.default_rng(0)
    for utt in ("0001", "0002"):
        p = tmp_path / "LibriSpeech" / "dev-clean" / "174" / "1" / f"174-1-{utt}.flac"
        p.parent.mkdir(parents=True, exist_ok=True)
        x = (0.1 * rng.normal(size=(SR, 1))).astype(np.float32)
        sf.write(str(p), x, SR)
    for env, i in (("DKITCHEN", 1), ("OOFFICE", 2)):
        p = tmp_path / "DEMAND" / env / f"ch{i:02d}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), (0.1 * rng.normal(size=(SR, 1))).astype(np.float32), SR)

    cfg = SimConfig(sample_rate=SR, duration=0.5, room_size_range=(3.0, 3.5),
                    t60_range=(0.2, 0.3), source_distance_range=(1.0, 1.5))
    out = tmp_path / "out"
    generate_batch(list_utterances(tmp_path / "LibriSpeech", "dev-clean"),
                   list_noise_files(tmp_path / "DEMAND"), cfg,
                   BatchSpec(out_dir=out, n_samples=3, seed=1,
                             snr_levels=(0.0,), t60_levels=(0.3,)),
                   cache_dir=tmp_path / "rirs")
    return out, cfg


def test_run_methods_produces_finite_records(eval_dir):
    data_dir, cfg = eval_dir
    rows = load_manifest(data_dir)

    records = run_methods(data_dir, rows, cfg)

    assert len(records) == len(rows) * len(DEFAULT_METHODS)
    for r in records:
        assert r["method"] in DEFAULT_METHODS
        assert np.isfinite(r["si_sdr"]) and np.isfinite(r["stoi"])
        assert np.isfinite(r["pesq_wb"])


def test_run_methods_limit_keeps_conditions_balanced(eval_dir):
    data_dir, cfg = eval_dir
    rows = load_manifest(data_dir)

    records = run_methods(data_dir, rows, cfg, limit_per_condition=2)

    per_condition = {}
    for r in records:
        per_condition.setdefault((r["snr_db"], r["t60"]), 0)
        per_condition[(r["snr_db"], r["t60"])] += 1
    # 每个条件 2 条 × 3 个方法
    assert all(v == 6 for v in per_condition.values())


def test_run_methods_rejects_unknown_method(eval_dir):
    data_dir, cfg = eval_dir
    rows = load_manifest(data_dir)

    with pytest.raises(ValueError, match="未知方法"):
        run_methods(data_dir, rows, cfg, methods=["Noisy", "MagicBeamformer"])
