"""T3.4 诊断：掩蔽预测任务的「信息天花板」估计。

目的：判断 CRN 残留的掩蔽 MSE（~0.0127）到底是
    (a) 模型容量/优化不足 —— 换更强或更直接的模型应能明显更低；还是
    (b) 该数据上软掩蔽目标的固有不确定性 —— 任何模型都到不了零。

方法：用**闭式最小二乘**拟合一个逐频点线性回归器
        mask[f, t] ≈ w_f · φ(mag[:, f, t]) + b_f
    其中 φ 是观测幅度的若干非线性基（自身、sqrt、log1p）。这是该特征下的
    最优线性预测，无需迭代训练，因此能给出一个干净、可复现的参考下界。

若这个线性模型的 MSE 与 CRN 同量级（甚至更高），说明 CRN 已接近该特征集
所能提供的极限，残留属情形 (b)；
若线性模型显著更低，则 CRN 存在容量或优化问题，判据应继续追。

用法：
    python scripts/diagnose_mask_ceiling.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nbse.models.dataset import NbseDataset  # noqa: E402

# CRN 过拟合测试的实测结果，供对照（scripts/verify_overfit.py 输出）
CRN_REFERENCE_MSE = 0.014150


def build_features(mag: np.ndarray) -> np.ndarray:
    """由观测幅度构造非线性基。

    取 [mag, sqrt(mag), log1p(mag)]——掩蔽本质是「幅度比」，而比值对幅度是
    接近 1/(1+1/r) 的非线性函数，因此这组基能覆盖主要形状。
    全 1 列作为偏置。
    """
    return np.stack([
        np.ones_like(mag),
        mag,
        np.sqrt(np.maximum(mag, 0.0)),
        np.log1p(np.maximum(mag, 0.0)),
    ], axis=0)                       # [n_basis, N]


def fit_per_bin_lstsq(mag: np.ndarray, mask: np.ndarray,
                      n_freqs: int) -> tuple[float, np.ndarray]:
    """逐频点闭式最小二乘。返回 (MSE, 每频点系数)。

    mag / mask 形状 [N, F, T]，N 为样本数。
    """
    n, f, t = mag.shape
    assert f == n_freqs

    pred = np.empty_like(mask)
    coefs = np.empty((n_freqs, 4))

    for k in range(n_freqs):
        X = build_features(mag[:, k, :].reshape(-1))        # [4, N*T]
        y = mask[:, k, :].reshape(-1)                       # [N*T]
        A = X @ X.T                                         # [4,4]
        b = X @ y                                           # [4]
        # 轻微岭正则，避免共线基（mag 与 sqrt(mag) 在高幅度区高度相关）导致病态
        A = A + 1e-8 * np.trace(A) / 4 * np.eye(4)
        w = np.linalg.solve(A, b)
        coefs[k] = w
        pred[:, k, :] = (w[:, None] * X).sum(axis=0).reshape(n, t)

    mse = float(np.mean((pred - mask) ** 2))
    return mse, coefs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/sim/train")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--segment-length", type=int, default=16000)
    args = ap.parse_args()

    ds = NbseDataset(args.data, segment_length=args.segment_length,
                     random_crop=False, limit=args.n, seed=20261009)

    mag = np.stack([ds[i]["mag"][0] for i in range(len(ds))])    # [N,F,T]
    mask = np.stack([ds[i]["mask"] for i in range(len(ds))])     # [N,F,T]
    n, f, t = mag.shape

    print("=" * 72)
    print("掩蔽预测的信息天花板估计")
    print("=" * 72)
    print(f"数据：{n} 条 × {f} 频点 × {t} 帧 = {mask.size:,} 个时频点")

    # ── 基线 1：常数预测（不看观测）
    const_mse = float(np.mean((mask - mask.mean()) ** 2))
    print(f"\n基线 1 · 常数预测（不利用观测）     MSE = {const_mse:.6f}")
    print(f"          解释方差比例 = 0.00%（定义）")

    # ── 基线 2：逐频点的「掩蔽均值」预测（用了频率信息，不用幅度信息）
    per_bin_mean = mask.mean(axis=(0, 2), keepdims=True)
    bin_mse = float(np.mean((mask - per_bin_mean) ** 2))
    print(f"基线 2 · 逐频点均值（用频率不用幅度） MSSE = {bin_mse:.6f}")
    print(f"          解释方差比例 = {(1 - bin_mse / const_mse) * 100:.2f}%")

    # ── 参考：闭式最小二乘（该特征集下的最优线性预测）
    lin_mse, _ = fit_per_bin_lstsq(mag, mask, f)
    print(f"\n参考 · 逐频点闭式最小二乘            MSE = {lin_mse:.6f}")
    print(f"          解释方差比例 = {(1 - lin_mse / const_mse) * 100:.2f}%")
    print(f"          （特征：1, mag, sqrt(mag), log1p(mag)）")

    # ── 对照：CRN 实测
    print(f"\nCRN 过拟合实测                       MSE = {CRN_REFERENCE_MSE:.6f}")
    print(f"          解释方差比例 = "
          f"{(1 - CRN_REFERENCE_MSE / const_mse) * 100:.2f}%")

    # ── 判读
    print("\n" + "=" * 72)
    ratio = CRN_REFERENCE_MSE / lin_mse
    print(f"CRN / 线性最优 = {ratio:.3f}")
    if ratio <= 1.05:
        verdict = ("CRN 已达到（甚至优于）该特征集下的闭式最优线性预测，"
                   "残留 MSE 属**数据固有不确定性**，不是模型缺陷。")
    elif ratio <= 2.0:
        verdict = ("CRN 与闭式最优线性预测同量级，残留主要来自固有不确定性；"
                   "模型侧仍有少量可改进空间。")
    else:
        verdict = ("CRN 明显劣于闭式最优线性预测，**存在容量或优化问题**，"
                   "应继续排查而非接受当前残留。")
    print("判读：" + verdict)
    print("=" * 72)

    # 线性模型的显式形式，便于判断掩蔽形状是否合理
    return 0


if __name__ == "__main__":
    sys.exit(main())
