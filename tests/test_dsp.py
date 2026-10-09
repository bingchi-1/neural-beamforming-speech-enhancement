"""时频变换的可逆性测试。

这组测试的存在有具体原因：dsp.py 早期版本用 boundary=None / padded=False
破坏了 NOLA 条件，STFT 变得不可逆（往返误差 1e-1），而 scipy 只抛 warning
不报错。结果 DS 波束形成的验收测试误判为「实现有 bug」，实际是 DSP 层的问题。
"""

import numpy as np
import pytest

from nbse.dsp import HOP, N_FFT, istft, rfft_freqs, stft, stft_multi


def test_stft_roundtrip_is_lossless():
    """STFT → ISTFT 必须严格可逆（这是全部频域方法的地基）。"""
    rng = np.random.default_rng(0)
    x = rng.normal(size=16000)

    y = istft(stft(x), length=len(x))

    assert y.shape == x.shape
    np.testing.assert_allclose(y, x, atol=1e-10)


def test_stft_roundtrip_on_tone():
    """纯音的往返同样无损——避免宽带噪声掩盖特定频点的相位错误。"""
    t = np.arange(16000) / 16000
    for f0 in (200.0, 1000.0, 3000.0, 7000.0):
        x = np.sin(2 * np.pi * f0 * t)
        y = istft(stft(x), length=len(x))
        np.testing.assert_allclose(y, x, atol=1e-10)


def test_stft_shapes():
    x = np.random.default_rng(1).normal(size=16000)
    Z = stft(x)
    assert Z.shape[0] == N_FFT // 2 + 1
    assert Z.ndim == 2


def test_stft_multi_preserves_channel_dimension():
    x = np.random.default_rng(2).normal(size=(4, 8000))
    Z = stft_multi(x)

    assert Z.shape[0] == 4
    assert Z.shape[1] == N_FFT // 2 + 1


def test_rfft_freqs_matches_numpy():
    freqs = rfft_freqs(N_FFT, 16000)
    np.testing.assert_allclose(freqs, np.fft.rfftfreq(N_FFT, d=1 / 16000))
    assert freqs[0] == 0.0
    assert freqs[-1] == pytest.approx(8000.0)


def test_istft_pads_short_input_to_requested_length():
    """ISTFT 输出短于要求长度时应补零，而非报错。"""
    x = np.random.default_rng(3).normal(size=1000)
    Z = stft(x)
    y = istft(Z, length=5000)
    assert len(y) == 5000
    np.testing.assert_allclose(y[:1000], x, atol=1e-10)
    assert np.allclose(y[1000:], 0.0)
