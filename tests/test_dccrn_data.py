"""T5.2 数据契约与推理切块的测试。

重点守两类**不会报错、只会静默劣化**的问题：

1. **推理序列长度与训练不一致**：LSTM 对长度敏感，整条前向会让 SI-SDR
   掉约 10 dB，而训练 loss 完全正常。故 `chunk_frames` 必须从 checkpoint
   记录的训练片段长度推出。
2. **帧数公式与实际不符**：`_segment_frames` 若写成 `L // hop`，会与实际
   STFT 帧数差 1，导致掩蔽与观测对不齐。
"""

import numpy as np
import pytest

from nbse.dsp import HOP, N_FFT, stft
from nbse.models.dccrn_data import (
    TorchDCCRN,
    _segment_frames,
    build_pair,
)


# ── 帧数公式 ────────────────────────────────────────────────────

@pytest.mark.parametrize("length,expected", [
    (16000, 126),          # 项目训练片段：1 秒
    (32000, 251),
    (48000, 376),
    (64000, 501),          # 4 秒样本，也是评测时的长度
])
def test_segment_frames_matches_actual_stft(length, expected):
    """帧数公式必须与 `stft` 的真实输出一致（差 1 就会错位）。"""
    x = np.zeros(length)
    actual = stft(x, N_FFT, HOP).shape[1]
    assert _segment_frames(length) == actual
    assert actual == expected, f"{length} 点应为 {expected} 帧，实得 {actual}"


def test_segment_frames_rejects_too_short():
    with pytest.raises(ValueError, match="小于 n_fft"):
        _segment_frames(N_FFT - 1)


# ── build_pair ──────────────────────────────────────────────────

def test_build_pair_shapes_and_dtype():
    rng = np.random.default_rng(0)
    mix = rng.normal(size=16000)
    ref = rng.normal(size=16000)
    out = build_pair(mix, ref)
    T = stft(np.zeros(16000), N_FFT, HOP).shape[1]
    for k in ("mix_r", "mix_i", "clean_r", "clean_i"):
        assert out[k].shape == (N_FFT // 2 + 1, T)
        assert out[k].dtype == np.float32


def test_build_pair_is_linear_and_consistent():
    """复谱对必须真的来自各自信号：把 ref 换成 mix 时两者应相等。"""
    rng = np.random.default_rng(1)
    mix = rng.normal(size=16000)
    out = build_pair(mix, mix)
    np.testing.assert_allclose(out["mix_r"], out["clean_r"])
    np.testing.assert_allclose(out["mix_i"], out["clean_i"])


# ── 推理切块（T5.2 修复的核心）──────────────────────────────────

class _StubModel:
    """按输入时间帧数改变输出的假模型，用来暴露「是否真的切了块」。

    真实模型里 LSTM 对长度敏感；这里用一个显式的长度依赖模拟同一现象：
    输入长于 200 帧时给出不同的掩蔽值。

    返回形状必须与真实 `DCCRNLite` 一致 —— `(B, F, T)`，不是 `(B, 1, F, T)`；
    `TorchDCCRN._mask` 会取 `[0]` 把它降成 `(F, T)`。
    """

    def __call__(self, spec_r, spec_i):
        import torch

        B, _, F, T = spec_r.shape
        val = 0.1 if T <= 200 else 0.9
        mr = torch.full((B, F, T), val)
        mi = torch.zeros(B, F, T)
        return mr, mi


def _wrapper(model, chunk_frames):
    return TorchDCCRN(model, device="cpu", deterministic=False,
                      chunk_frames=chunk_frames)


def test_chunked_inference_differs_from_whole_forward():
    """切块必须真的改变结果 —— 否则「修复」是空的。

    用长度敏感的假模型：整条 4 秒前向给出 mask=0.9，按 126 帧切块则每块
    都走 mask=0.1 分支，两者输出必须不同。
    """
    rng = np.random.default_rng(2)
    x = rng.normal(size=64000)          # 4 秒 → 501 帧
    model = _StubModel()

    whole = _wrapper(model, chunk_frames=None).enhance(x)
    chunked = _wrapper(model, chunk_frames=126).enhance(x)

    assert whole.shape == chunked.shape == (64000,)
    assert not np.allclose(whole, chunked), \
        "切块与整条前向结果相同，说明切块没有生效"


def test_chunked_inference_uses_training_length():
    """切块后每块的输出应与「按该长度直接前向」一致。

    这样把「切块正确」与「模型行为」解耦：只要每块都被当成 126 帧处理，
    拼起来的结果就应与逐块单独处理相同。
    """
    rng = np.random.default_rng(3)
    x = rng.normal(size=64000)
    model = _StubModel()
    chunked = _wrapper(model, chunk_frames=126).enhance(x)

    # 手动按 126 帧逐块处理再拼，应与 chunked 完全一致
    X = stft(x, N_FFT, HOP)
    from nbse.dsp import istft

    mr = np.empty_like(X.real, dtype=np.float32)
    mi = np.empty_like(X.imag, dtype=np.float32)
    for s in range(0, X.shape[1], 126):
        e = min(s + 126, X.shape[1])
        r, i = model(X[:, s:e].real.astype(np.float32)[None, None],
                     X[:, s:e].imag.astype(np.float32)[None, None])
        mr[:, s:e], mi[:, s:e] = r[0].numpy(), i[0].numpy()
    manual = istft(X * (mr + 1j * mi), N_FFT, HOP, length=len(x))

    np.testing.assert_allclose(chunked, manual, rtol=1e-5, atol=1e-6)


def test_short_input_is_not_chunked():
    """短于一个切块时走整条前向（此时与训练条件一致，不该强行切）。"""
    rng = np.random.default_rng(4)
    x = rng.normal(size=16000)          # 恰好 126 帧
    model = _StubModel()
    a = _wrapper(model, chunk_frames=126).enhance(x)
    b = _wrapper(model, chunk_frames=None).enhance(x)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


def test_chunk_frames_read_from_checkpoint(tmp_path):
    """`chunk_frames` 必须能从 checkpoint 的 extra 里读出来。

    `save_checkpoint` 把额外字段嵌在 "extra" 下，故读取时要下探一层。
    这条守住「训练片段长度自动传递到推理」这一机制——
    它正是本次 10 dB 修复的关键，坏了不会报错、只会静默劣化。
    """
    import torch

    from nbse.models.dccrn_lite import DCCRNLite

    model = DCCRNLite(n_freqs=257, base_channels=4, lstm_hidden=8,
                      use_lstm=False)
    p = tmp_path / "ck.pt"
    torch.save({"model": model.state_dict(),
                "extra": {"segment_length": 16000}}, p)

    w = TorchDCCRN.from_checkpoint(p, device="cpu", base_channels=4,
                                   lstm_hidden=8, use_lstm=False)
    assert w.chunk_frames == _segment_frames(16000) == 126


def test_missing_segment_length_warns_and_defaults(tmp_path):
    """旧格式 checkpoint 缺该字段时必须**警告**，而不是静默按错误长度推理。"""
    import torch

    from nbse.models.dccrn_lite import DCCRNLite

    model = DCCRNLite(n_freqs=257, base_channels=4, lstm_hidden=8,
                      use_lstm=False)
    p = tmp_path / "old.pt"
    torch.save({"model": model.state_dict()}, p)

    with pytest.warns(RuntimeWarning, match="未记录训练片段长度"):
        w = TorchDCCRN.from_checkpoint(p, device="cpu", base_channels=4,
                                       lstm_hidden=8, use_lstm=False)
    assert w.chunk_frames == _segment_frames(16000)


def test_explicit_chunk_frames_overrides_checkpoint(tmp_path):
    """显式指定的值优先于 checkpoint 记录。"""
    import torch

    from nbse.models.dccrn_lite import DCCRNLite

    model = DCCRNLite(n_freqs=257, base_channels=4, lstm_hidden=8,
                      use_lstm=False)
    p = tmp_path / "ck.pt"
    torch.save({"model": model.state_dict(),
                "extra": {"segment_length": 16000}}, p)

    w = TorchDCCRN.from_checkpoint(p, device="cpu", base_channels=4,
                                   lstm_hidden=8, use_lstm=False,
                                   chunk_frames=64)
    assert w.chunk_frames == 64
