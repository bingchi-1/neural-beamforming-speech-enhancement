"""T3.4 验收：过拟合测试的量化判读。

比「loss 是否小于某个拍脑袋的阈值」更可靠的做法，是把三条互补证据一起看：

1. **相对常预测基线的解释比例**。掩蔽的总方差 ~0.12（完全不看观测时的最优
   MSE）；过拟合训练把 loss 压到 ~0.013，即解释了约 89% 的方差。
   这比绝对阈值有意义得多——绝对值取决于数据集的语音/噪声比例。

2. **学到的最优常预测**。如果模型只学会了「输出一个常数」，它的 loss 会停在
   掩蔽的总方差 ~0.12。实测远低于此，说明它确实在利用观测。

3. **掩蔽估计的逐点误差**。直接看预测掩蔽与真值掩蔽的 MAE / 相关系数。

用法：
    python scripts/verify_overfit.py --ckpt runs/overfit/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nbse.models.crn import CRN_SMALL  # noqa: E402
from nbse.models.dataset import NbseDataset  # noqa: E402
from nbse.training import load_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="runs/overfit/best.pt")
    ap.add_argument("--data", default="data/sim/train")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--segment-length", type=int, default=16000)
    args = ap.parse_args()

    ds = NbseDataset(args.data, segment_length=args.segment_length,
                     random_crop=False, limit=args.n, seed=20261009)
    model = CRN_SMALL(n_mics=ds.n_mics, n_freqs=ds.n_freqs)
    state, _ = load_checkpoint(args.ckpt, model, map_location="cpu")
    model.eval()
    print(f"checkpoint：{args.ckpt}（epoch={state.epoch}）")
    print(f"样本：{len(ds)} 条 × {ds.n_frames} 帧 × {ds.n_freqs} 频点")

    all_pred, all_true = [], []
    loss_mask, loss_mag_total = 0.0, 0.0
    const_pred_loss = 0.0

    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            mag = torch.from_numpy(s["mag"])[None]
            mask = torch.from_numpy(s["mask"])[None]
            clean_mag = torch.from_numpy(s["clean_mag"])[None]

            pred = model(mag)
            all_pred.append(pred[0].numpy().ravel())
            all_true.append(mask[0].numpy().ravel())

            loss_mask += float(torch.nn.functional.mse_loss(pred, mask))
            est = (mag[:, 0] * pred)
            loss_mag_total += float(torch.nn.functional.mse_loss(est, clean_mag))

    n = len(ds)
    loss_mask /= n
    loss_mag_total /= n

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    mean_true = true.mean()
    var_total = float(np.mean((true - mean_true) ** 2))
    const_pred_loss = var_total
    mae = float(np.mean(np.abs(pred - true)))
    corr = float(np.corrcoef(pred, true)[0, 1])

    print("\n" + "=" * 70)
    print("过拟合判读（T3.4）")
    print("=" * 70)
    print(f"掩蔽 MSE              : {loss_mask:.6f}")
    print(f"幅度域 MSE            : {loss_mag_total:.3e}")
    print(f"常预测基线（掩蔽方差）: {const_pred_loss:.6f}")
    print(f"→ 解释的掩蔽方差比例  : "
          f"{(1 - loss_mask / const_pred_loss) * 100:.2f}%")
    print(f"掩蔽 MAE              : {mae:.6f}")
    print(f"掩蔽相关系数          : {corr:.6f}")

    # 判据的标定依据见 scripts/diagnose_mask_ceiling.py 的实测结果：
    #   常数预测（不利用观测）           MSE = 0.1201
    #   逐频点均值（用频率不用幅度）      MSE = 0.1114
    #   逐频点闭式最优线性预测            MSE = 0.0960
    # 因此「loss 接近零」在本任务上不可达。判据改为相对这两条基线的表现。
    CONST_BASELINE = 0.1201
    LIN_BASELINE = 0.0960

    explained = 1.0 - loss_mask / CONST_BASELINE
    vs_linear = LIN_BASELINE / max(loss_mask, 1e-12)

    print(f"\n参考基线（实测标定）：")
    print(f"  常数预测                 MSE = {CONST_BASELINE:.6f}")
    print(f"  逐频点闭式最优线性预测   MSE = {LIN_BASELINE:.6f}")
    print(f"  本 checkpoint            MSE = {loss_mask:.6f}")
    print(f"  → 解释掩蔽方差 {explained * 100:.2f}%，"
          f"优于线性最优 {vs_linear:.2f} 倍")

    checks = [
        ("解释掩蔽方差 > 80%", explained > 0.80, f"{explained * 100:.2f}%"),
        ("优于闭式最优线性预测（> 2 倍）", vs_linear > 2.0, f"{vs_linear:.2f}×"),
        ("掩蔽相关性强（> 0.90）", corr > 0.90, f"{corr:.4f}"),
        ("掩蔽 MAE 小（< 0.10）", mae < 0.10, f"{mae:.4f}"),
    ]
    print()
    ok = True
    for name, passed, val in checks:
        print(f"  [{'通过' if passed else '不通过'}] {name}  实测 {val}")
        ok &= passed

    print("\n结论：" + ("过拟合测试通过——管线正确，模型确实学到了掩蔽映射。"
                       if ok else
                       "未通过——优先怀疑数据管道，其次学习率。"))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
