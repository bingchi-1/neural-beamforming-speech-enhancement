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
from nbse.dsp import N_FFT, stft
from nbse.evaluate import (
    ALL_METHOD_NAMES,
    DEFAULT_METHODS,
    FINAL_METHODS,
    aggregate,
    align_level,
    build_method_impls,
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


# ── mask-MVDR 的注册与接线（T4.3）────────────────────────────────

def test_mask_methods_not_registered_without_predictor():
    """没有 predictor 时 mask-MVDR 不应出现在方法表里。

    这是刻意的：若提供「缺 predictor 就退化为 DS」的兜底，
    「神经方法优于基线」这一结论就会变成假的，而表面上一切正常。
    """
    impls = build_method_impls(None)
    assert "mask-MVDR (Oracle DOA)" not in impls
    assert "mask-MVDR (GCC-PHAT)" not in impls
    # 只应剩三条不需训练、也不需要神经网络资源的方法
    assert set(impls) == {"Noisy", "DS", "MVDR (Oracle DOA)", "MVDR (GCC-PHAT)"}


def test_dccrn_not_registered_without_checkpoint():
    """没有 DCCRN checkpoint 时不应注册 DCCRN-lite —— 同样的理由。"""
    impls = build_method_impls(lambda multi_ch: None)
    assert "DCCRN-lite" not in impls


def test_all_methods_registered_with_all_resources():
    """两类资源都给齐时，应注册设计文档 §6.3 的**全部七行**。"""
    impls = build_method_impls(lambda multi_ch: None, dccrn=object())
    assert set(FINAL_METHODS) <= set(impls), \
        f"缺少方法：{set(FINAL_METHODS) - set(impls)}"
    assert len(FINAL_METHODS) == 7


def test_every_final_method_is_registered_when_resources_present():
    """七行表的每一项都必须真的有实现，且都能在资源齐全时注册。"""
    impls = build_method_impls(lambda mc: None, dccrn=object())
    for name in FINAL_METHODS:
        assert name in impls, f"{name} 没有实现"
        assert callable(impls[name])


def test_requesting_mask_method_without_predictor_names_the_cause(eval_dir):
    """点名 mask-MVDR 但不给 predictor 时，报错必须指出缺什么。

    只说「未知方法」会让人以为是拼写错误，白查半天。
    """
    data_dir, cfg = eval_dir
    rows = load_manifest(data_dir)

    with pytest.raises(ValueError, match="mask_predictor"):
        run_methods(data_dir, rows, cfg,
                    methods=["Noisy", "mask-MVDR (Oracle DOA)"])


def test_run_methods_with_predictor_produces_mask_records(eval_dir):
    """给了 predictor 时，mask-MVDR 应真正跑出有限指标。"""
    data_dir, cfg = eval_dir
    rows = load_manifest(data_dir)

    # 常量掩蔽预测器：足以驱动完整链路，且结果确定
    from nbse.dsp import N_FFT, stft

    def const_predictor(multi_ch):
        T = stft(np.atleast_2d(multi_ch)[0]).shape[1]
        return np.full((N_FFT // 2 + 1, T), 0.6)

    records = run_methods(data_dir, rows, cfg,
                          methods=["MVDR (Oracle DOA)",
                                   "mask-MVDR (Oracle DOA)"],
                          mask_predictor=const_predictor)

    mask_recs = [r for r in records if r["method"].startswith("mask-MVDR")]
    assert mask_recs, "没有产出 mask-MVDR 记录"
    for r in mask_recs:
        assert np.isfinite(r["si_sdr"])
        assert np.isfinite(r["stoi"])
        assert np.isfinite(r["pesq_wb"])


def test_base_methods_ignore_predictor(eval_dir):
    """给 predictor 不应改变基线方法的结果（签名统一但不影响语义）。"""
    data_dir, cfg = eval_dir
    rows = load_manifest(data_dir)

    a = run_methods(data_dir, rows, cfg, methods=["DS"])
    b = run_methods(data_dir, rows, cfg, methods=["DS"],
                    mask_predictor=lambda mc: np.ones((129, 10)))

    assert a[0]["si_sdr"] == pytest.approx(b[0]["si_sdr"], abs=1e-12)


def test_mask_and_plain_mvdr_share_the_same_loading():
    """两条 MVDR 路线必须用同一个对角加载强度，且都取自 `MVDR_LOADING`。

    这是让「mask-MVDR vs MVDR」这个对比有意义的前提。实测教训：
    两个类的默认值曾是 1e-2 与 1e-3（差 10 倍），仅此一项就让波束形成权重
    相对差 0.78、SI-SDR 相差约 2.3 dB——测出来的是正则化强度差异，
    而不是方法差异，结论可以直接反过来。
    """
    from types import SimpleNamespace

    from nbse.beamformers.mask_mvdr import MaskMVDR
    from nbse.beamformers.mvdr import MVDR
    from nbse.evaluate import (
        BASE_METHOD_IMPLS,
        MASK_METHOD_IMPLS,
        MVDR_LOADING,
    )

    sample = {"multi_ch": np.zeros((4, 8000)), "noise_multi": np.zeros((4, 8000)),
              "doa_true": 0.0}
    mic = np.zeros((3, 4))
    cfg = SimpleNamespace(sample_rate=16000)
    pred = lambda mc: np.full((N_FFT // 2 + 1, stft(np.zeros(8000)).shape[1]), 0.5)

    seen = {}
    for name, cls, table in (("mask-MVDR (Oracle DOA)", MaskMVDR, MASK_METHOD_IMPLS),
                             ("MVDR (Oracle DOA)", MVDR, BASE_METHOD_IMPLS)):
        real = cls.__init__

        def spy(self, *a, _real=real, _name=name, **kw):
            seen[_name] = kw.get("loading")
            return _real(self, *a, **kw)

        cls.__init__ = spy
        try:
            try:
                # 第 5 个参数是 ctx（dict），不再是裸的 predictor
                table[name](sample, mic, (0.0, 0.0), cfg,
                            {"mask_predictor": pred})
            except Exception:
                pass          # 只关心传了什么参数，不关心后续计算
        finally:
            cls.__init__ = real

    assert seen.get("mask-MVDR (Oracle DOA)") == MVDR_LOADING, \
        f"mask-MVDR 未使用统一加载强度：{seen}"
    assert seen.get("MVDR (Oracle DOA)") == MVDR_LOADING, \
        f"MVDR 未使用统一加载强度：{seen}"


def test_eval_uses_the_measured_best_mask_weighting():
    """评测必须用实测最优的掩蔽用法（weighting="noise"）。

    这条是防「改了默认值但评测没跟着变」的回归测试——本轮就踩过一次：
    把 `MaskMVDR` 的默认 weighting 改掉后重跑评测，数字**逐位不变**，
    因为 `MASK_METHOD_IMPLS` 的 lambda 在导入时已固化、未显式传参。
    把配置提成模块级常量并断言它，才能拦住这类静默失效。
    """
    from nbse.evaluate import MASK_MODE, MASK_WEIGHTING

    assert MASK_WEIGHTING == "noise", \
        "评测未使用实测最优的 weighting；改动前请重跑真实评测确认有改善"
    assert MASK_MODE == "mvdr"


def test_mvdr_loading_stays_in_the_validated_range():
    """`MVDR_LOADING` 必须落在实测验证过的区间内。

    这不是形式检查——该值两端都有实测的失效模式：

    * **过大**（≥1e-2）：两条 MVDR 路线各损失约 1.9 dB（验证集上 mask-MVDR
      从 −2.507 退化到 −4.345），等于白扔性能。
    * **过小**（≤1e-5）：性能不再提升，但正则化后的条件数爆炸
      （1e-6 时单频点 cond p99 达 3.97×10⁶、权重范数 326），
      大权重会放大导向矢量与实际传递函数的失配，混响更强或 DOA 有误差时很脆弱。

    实测（40 条测试样本）：
        loading    cond(正则化后) p99    ‖w‖max
          1e-6       3,967,372          326
          1e-4          39,790           22
          3e-4          13,301         16.3
          1e-2             401          3.5
    当前取 3e-4：性能仅比平台最优差 0.2 dB，条件数与权重范数都在安全区。
    """
    from nbse.evaluate import MVDR_LOADING

    assert 1e-4 <= MVDR_LOADING <= 1e-3, (
        f"MVDR_LOADING={MVDR_LOADING:g} 超出实测验证区间 [1e-4, 1e-3]。"
        f"调大需重跑评测确认性能未退化；调小需重测条件数与权重范数。")


def test_mvdr_loading_keeps_inverse_well_conditioned():
    """实际检验：该 loading 下正则化后的协方差矩阵应保持可控。

    比上一条更直接——不只看常量，而是真的构造一个病态的掩蔽加权协方差
    （掩蔽接近二值会让有效帧数极少、矩阵近乎奇异），看加载后能否稳定求逆。
    """
    from nbse.beamformers.mask_mvdr import mask_weighted_covariance
    from nbse.beamformers.mvdr import regularize_covariance
    from nbse.evaluate import MVDR_LOADING

    rng = np.random.default_rng(0)
    F, T, M = 64, 40, 4
    X = rng.normal(size=(M, F, T)) + 1j * rng.normal(size=(M, F, T))

    # 制造病态：掩蔽只在少数帧为 1、其余为 0 → 噪声权重几乎全零
    mask = np.zeros((F, T))
    mask[:, :2] = 1.0
    R_n = mask_weighted_covariance(X, mask)

    R_reg = regularize_covariance(R_n, MVDR_LOADING)
    conds = np.linalg.cond(R_reg)

    assert np.isfinite(conds).all(), "出现非有限条件数"
    assert np.linalg.eigvalsh(R_reg).min() > 0, "正则化后矩阵仍非正定"
    # 阈值 1e7：1e-6 时实测 p99 已达 4e6，当前值应低一到两个量级
    assert np.percentile(conds, 99) < 1e7, \
        f"正则化后条件数过大（p99={np.percentile(conds, 99):.3g}），loading 可能过小"


def test_mask_method_impl_receives_weighting(monkeypatch, eval_dir):
    """确认 lambda 真的把 weighting 传给了 MaskMVDR（而不只是常量写对）。"""
    from nbse.beamformers import mask_mvdr as mm
    from nbse.evaluate import MASK_METHOD_IMPLS

    seen = {}
    real_init = mm.MaskMVDR.__init__

    def spy(self, *a, **kw):
        seen.update(kw)
        return real_init(self, *a, **kw)

    monkeypatch.setattr(mm.MaskMVDR, "__init__", spy)

    data_dir, cfg = eval_dir
    rows = load_manifest(data_dir)[:1]
    sample = {"multi_ch": np.zeros((4, 8000)),
              "noise_multi": np.zeros((4, 8000)),
              "doa_true": 0.0}
    mic = np.zeros((3, 4))
    n_freqs = N_FFT // 2 + 1
    T = stft(np.zeros(8000)).shape[1]
    MASK_METHOD_IMPLS["mask-MVDR (Oracle DOA)"](
        sample, mic, (0.0, 0.0), cfg,
        {"mask_predictor": lambda mc: np.full((n_freqs, T), 0.5)})

    assert seen.get("weighting") == "noise"
    assert seen.get("mode") == "mvdr"
