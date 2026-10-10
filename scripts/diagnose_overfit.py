"""T3.4 诊断：过拟合残留 loss 的结构分析。

过拟合测试的判据是「训练 loss 降到接近零」。若它卡在某个明显大于零的值，
有两种可能，必须区分开：

  A. **管线有 bug**（掩蔽口径错、通道错位、帧数错位……）——必须修；
  B. **损失在该数据上存在不可约下界**——10 条样本、每条 126×257 个时频点，
     掩蔽由「语音幅度 /（语音幅度 + 噪声幅度）」定义，而在语音与噪声幅度
     相当的时频点上，观测幅度谱并不能唯一决定这个比值。此时最优预测是
     条件期望 E[mask | 观测]，MSE 的下界就是 Var(mask | 观测) 的均值，
     严格大于零。这不是 bug，是问题的固有不确定性。

区分方法：直接算出「观测相同时掩蔽的条件方差」上界估计——
把训练集上所有时频点按观测幅度分箱，看同箱内掩蔽的方差。
若残差与这个量级一致，则属情形 B，判据本身需要改为「相对可约部分」。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nbse.models.dataset import NbseDataset  # noqa: E402

N = 10
SEG = 16000


def main() -> int:
    ds = NbseDataset("data/sim/train", segment_length=SEG,
                     random_crop=False, limit=N, seed=20261009)

    mags, masks = [], []
    for i in range(len(ds)):
        s = ds[i]
        mags.append(s["mag"][0])       # 掩蔽定义在 0 号通道上
        masks.append(s["mask"])

    mag = np.concatenate([m.ravel() for m in mags])
    mask = np.concatenate([m.ravel() for m in masks])

    print("=" * 72)
    print("时频点统计")
    print("=" * 72)
    print(f"时频点总数      : {mask.size:,}")
    print(f"掩蔽均值/标准差 : {mask.mean():.4f} / {mask.std():.4f}")
    print(f"掩蔽最小值/最大 : {mask.min():.6f} / {mask.max():.6f}")
    print(f"观测幅度范围    : {mag.min():.3e} .. {mag.max():.3e}")

    # ── 1. 常预测基线：MSE(mask, mean) 就是不利用观测时的下界
    const = mask.mean()
    var_total = float(np.mean((mask - const) ** 2))
    print(f"\n常预测基线 MSE(mask, mean)   = {var_total:.6f}")
    print("   （这是完全不看观测时的最优 MSE：掩蔽的总方差）")

    # ── 2. 按观测幅度分箱，估计「观测已知时掩蔽残余的不确定性」
    #     分箱越细，条件方差估计越准，残差下界越低
    print("\n按观测幅度分箱的条件方差（观测已知后的不可约部分）")
    print(f"{'箱数':>6} {'条件方差均值':>14} {'占掩蔽总方差':>14}")
    for n_bins in (16, 64, 256, 1024):
        edges = np.quantile(mag, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        idx = np.clip(np.digitize(mag, edges[1:-1]), 0, len(edges) - 2)
        cond = 0.0
        for b in range(len(edges) - 1):
            sel = idx == b
            if sel.sum() > 1:
                cond += float(np.var(mask[sel])) * sel.sum()
        cond /= mask.size
        print(f"{len(edges) - 1:>6} {cond:>14.6f} {cond / var_total * 100:>13.2f}%")

    # ── 3. 掩蔽本身的分布：两极化程度决定任务难度
    print("\n掩蔽分布（两极化程度越高越容易过拟合）")
    for lo, hi in ((0.0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.01)):
        sel = (mask >= lo) & (mask < hi)
        print(f"  [{lo:.1f}, {hi:.1f})  {sel.mean() * 100:6.2f}%   "
              f"均值 {mask[sel].mean() if sel.any() else float('nan'):.4f}")

    # ── 4. 关键对照：把掩蔽二值化后的最优 MSE
    #     若二元掩蔽的常预测 MSE 远小于软掩蔽的，说明软掩蔽的中间值是难度来源
    binary = (mask > 0.5).astype(np.float64)
    print(f"\n参考：二值化掩蔽 (>0.5) 的占比 {binary.mean() * 100:.2f}%")
    print(f"      二值掩蔽的常预测 MSE = "
          f"{float(np.mean((binary - binary.mean()) ** 2)):.6f}")

    print("\n判读：若实测 overfit 残差 ~1.3e-2 与该量级一致，则属情形 B"
          "（不可约不确定性），判据应改为相对可约部分。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
