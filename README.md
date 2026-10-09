# neural-beamforming-speech-enhancement

麦克风阵列语音增强基准：在仿真阵列数据上系统对比**传统波束形成**、**神经波束形成**与**单通道增强**三条路线。

> **状态**：开发中（W1/6）。结果表与图表将在 W6 补全。

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
# W1 完成后补全
```

---

## 仓库结构

```
nbse/                 源码（入库）
  data/               房间仿真与 Dataset
  beamformers/        传统波束形成 + DOA 估计
  models/             神经网络
  metrics/            评测指标封装
configs/default.yaml  全部仿真与训练参数
scripts/              数据下载等工具脚本
tests/                单元测试
data/                 下载的数据集（不入库）
rirs/                 RIR 缓存（不入库）
```

---

## 设计文档

- 设计：`docs/superpowers/specs/2026-10-09-neural-beamforming-design.md`
- 实施计划：`docs/superpowers/plans/2026-10-09-neural-beamforming-plan.md`

## 许可

MIT
