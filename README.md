# neural-beamforming-speech-enhancement

麦克风阵列语音增强基准：在仿真阵列数据上系统对比**传统波束形成**、**神经波束形成**与**单通道增强**三条路线。

> **状态**：开发中（W3/6）。保底交付点（W2）已达成——`results/w2_baseline.csv`
> 含 Noisy / DS / MVDR 三行结果；W3 的 CRN 模型、数据集封装与训练脚本已完成。
> 结果表与图表将在 W6 补全。

---

## 对比方法

| 方法 | 是否需要训练 | 说明 |
|---|---|---|
| Noisy | — | 未处理基线（取单通道） |
| Delay-and-Sum | 否 | 经典延时求和波束形成 |
| MVDR (Oracle DOA) | 否 | 最小方差无畸变响应，使用真值方位角 |
| MVDR (GCC-PHAT) | 否 | 使用 GCC-PHAT 估计的方位角 |
| mask-MVDR (Oracle DOA) | 是 | 神经网络估计掩蔽驱动 MVDR |
| mask-MVDR (GCC-PHAT) | 是 | 同上，使用估计方位角 |
| DCCRN-lite | 是 | 单通道增强对照 |

`Oracle vs GCC-PHAT` 的成对对比，量化「DOA 估计误差带来多少性能损失」。

## 评测指标

PESQ（宽带 16 kHz）、STOI、SI-SDR。评估条件：3 个 SNR 档（−5 / 0 / 5 dB）× 2 个 T60 档（0.3 s / 0.6 s）。

<!-- 结果表在此补全（W6） -->

---

## 为什么使用仿真数据？

本项目用 `pyroomacoustics`（镜像源法）仿真房间冲激响应，而非使用真实阵列录音。原因：

1. **可控性**：声源方位角、混响时间 T60、信噪比均可精确设定，因此能做严格的消融实验——真实录音做不到这一点。
2. **仿真 RIR 是波束形成领域的标准做法**，学术界在评估波束形成器性能时普遍采用。
3. **参考信号可知**：评估 PESQ / STOI 需要干净参考信号，真实录音中该参考难以获得。

（真实录音数据集如 CHiME、L3DAS 受限于数据规模与标注质量，不在本项目范围内。）

---

## 环境要求

**GPU**：RTX 50 系（Blackwell，sm_120）需要 CUDA 12.8+ 构建的 PyTorch，且 `torch >= 2.7`。

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

验证 GPU 可用：

```bash
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

应输出类似 `12.8 True`。若 `torch.version.cuda` 为 `None`，说明装成了 CPU 版。

---

## 快速开始

```bash
# 1) 下载数据（约 950 MB，幂等，可重复执行；也可按 docs/DATASETS.md 手动放入）
python scripts/download_data.py

# 2) 生成多通道样本。--jobs 为渲染进程数（房间仿真各作业独立，可并行）
python scripts/generate_dataset.py --split train --n 3000 --jobs 8
python scripts/generate_dataset.py --split val   --n 120  --jobs 8

# 3) 生成配对评估网格（3 档 SNR × 2 档 T60，同语料同房间同噪声）
python scripts/generate_dataset.py --split test --grid --n-utterances 20 \
    --snr -5 0 5 --t60 0.3 0.6 --jobs 8

# 4) 训练
python train.py --data data/sim/train --val data/sim/val

# 5) 评测，产出结果表
python eval.py --data data/sim/test --out results/w2_baseline.csv
```

### 数据生成的性能

生成是仿真是 CPU 瓶颈，实测（RTX 5070 Ti Laptop + Intel Ultra 7 255HX，20 逻辑核）：

| 配置 | 吞吐 | 说明 |
|---|---|---|
| 默认 | ~27 条/秒 | `max_order_cap: 30` + `--jobs 8` |
| 串行 | ~6–7 条/秒 | 同口径但 `--jobs 1`，慢 3.8× |
| 完整仿真 | — | `max_order_cap: null`，最差房间 2.1 s/条 |

两项优化互不冲突，合计比最初的完整仿真 + 串行快约 15–20 倍：

- **`--jobs N`**：房间仿真各作业独立，用进程池并行。实测 8 进程加速 3.81×
  （两侧同样冷启动、同样工作量）。作业列表仍在主进程按同一种子生成、结果按序
  回收，因此产出与串行**逐位一致**（有回归测试保证）。
- **`room.max_order_cap: 30`**：由 `inverse_sabine` 反推的镜像源阶数随「大房间 +
  长 T60」急剧上升（实测中位 52、最大 114），而计算量随阶数超线性增长（阶数与
  耗时相关系数 0.927）。但高阶层镜像源能量可忽略——上限 30 时 RIR 能量保留
  99.94%、相对误差 1.9%，且 SNR 标定、通道相关度、DRR 三项下游口径均无变化。

> 注意 `max_order_cap` 已计入 RIR 缓存文件名。改这个值会让旧缓存自动失效，
> 避免新旧口径的 RIR 混在同一批数据里而静默失真。

---

## 仓库结构

```
nbse/                 源码（入库）
  data/               房间仿真、语料读取、批量生成（不依赖 torch）
  beamformers/        传统波束形成（DS / MVDR）+ 几何
  models/             CRN 网络、数据集封装（PyTorch 依赖隔离在此）
  metrics/            PESQ / STOI / SI-SDR 封装
  dsp.py              STFT / ISTFT
  evaluate.py         评测逻辑
  training.py         训练循环、损失、checkpoint
configs/default.yaml  全部仿真与训练参数
scripts/              数据下载、样本生成、诊断脚本
tests/                单元测试
train.py              训练入口
eval.py               评测入口，产出结果表
results/              结果表（入库）
data/                 下载的数据集与生成样本（不入库）
rirs/                 RIR 缓存（不入库）
runs/                 训练产物（不入库）
```

---

## 设计文档

- 设计：`docs/superpowers/specs/2026-10-09-neural-beamforming-design.md`
- 实施计划：`docs/superpowers/plans/2026-10-09-neural-beamforming-plan.md`

## 许可

MIT
