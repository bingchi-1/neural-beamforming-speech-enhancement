"""可复现性测试：GPU 计算必须逐位可重复。

背景（实测，2026-10-10）：默认设置下，同一个输入连续 8 次前向得到 **8 个不同
结果**，最大差约 1.6e-05。原因不是 BatchNorm（已确认 running stats 不变、
模型在 eval 模式），而是 cuDNN 卷积算法选择的不确定性。

这个量级本身无害，但在波束形成链路里会被放大：掩蔽的微小扰动经协方差估计与
矩阵求逆后，权重可相差 0.78，SI-SDR 相差 2.3 dB——足以让「神经方法是否优于
基线」的结论完全反过来，且同一个 checkpoint 两次评测得到不同结果表。
"""

import numpy as np
import pytest

from nbse.determinism import determinism_status, enable_determinism


def test_enable_determinism_sets_all_flags():
    import torch

    enable_determinism()
    st = determinism_status()
    assert st["deterministic_algorithms"] is True
    assert st["cudnn_deterministic"] is True
    assert st["cudnn_benchmark"] is False
    # 幂等
    enable_determinism()
    assert determinism_status() == st


def test_enable_determinism_seeds_torch():
    import torch

    enable_determinism(seed=1234)
    a = torch.randn(4)
    enable_determinism(seed=1234)
    b = torch.randn(4)
    np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_crn_forward_is_bitwise_reproducible():
    """CRN 前向必须逐位可重复 —— 这是整条评测链路可复现的前提。"""
    torch = pytest.importorskip("torch")
    from nbse.models.crn import CRN_SMALL

    enable_determinism(seed=0)
    model = CRN_SMALL(n_mics=4, n_freqs=257).eval()

    x = torch.rand(1, 4, 257, 63)
    with torch.no_grad():
        outs = [model(x).numpy() for _ in range(6)]

    base = outs[0]
    for i, o in enumerate(outs[1:], start=2):
        assert np.array_equal(o, base), \
            f"第 {i} 次前向与第 1 次不同（最大差 {np.abs(o - base).max():.3e}）"


@pytest.mark.skipif(not __import__("torch").cuda.is_available(),
                    reason="需要 CUDA")
def test_crn_forward_is_reproducible_on_cuda():
    """CUDA 上同样必须逐位可重复。

    这条是本次修复的直接回归测试：未开确定性时，同一输入连续 8 次前向会得到
    8 个不同结果（cuDNN 卷积算法不确定）。若这条失败，说明确定性设置被绕过
    （例如某处重新设了 cudnn.benchmark），评测结论将不可复核。
    """
    torch = pytest.importorskip("torch")
    from nbse.models.crn import CRN_SMALL

    enable_determinism(seed=0)
    model = CRN_SMALL(n_mics=4, n_freqs=257).to("cuda").eval()
    x = torch.rand(1, 4, 257, 63, device="cuda")

    with torch.no_grad():
        outs = [model(x).cpu().numpy() for _ in range(6)]

    base = outs[0]
    for i, o in enumerate(outs[1:], start=2):
        assert np.array_equal(o, base), \
            f"CUDA 上第 {i} 次前向与第 1 次不同（最大差 {np.abs(o - base).max():.3e}）"
