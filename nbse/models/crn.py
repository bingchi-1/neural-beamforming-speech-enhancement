"""T3.1 多通道 CRN（Convolutional Recurrent Network）掩蔽估计器。

设计文档 §5「系统架构」、实施计划 T3.1。

做什么
------
输入多通道观测的**线性幅度谱**，输出**单通道掩蔽**（取值 0~1）：

    mag  [B, n_mics, n_freqs, T]  ──→  mask  [B, n_freqs, T]

注意本模块**只估掩蔽，不重建波形**。把掩蔽与观测频谱相乘、再解 MVDR 权重
这件事由 T4.1（mask-MVDR）负责。这样切分是有意的：掩蔽网络与融合规则
可以分别测试、分别替换。

怎么用
------
    from nbse.models.crn import CRN, CRN_SMALL

    model = CRN(n_mics=4, n_freqs=257)        # 默认配置，见下
    mask = model(mag)                          # [B, 257, T]

    model = CRN_SMALL()                        # 过拟合测试/冒烟用的小模型

`n_freqs` 必须与 `dsp.N_FFT // 2 + 1` 一致（N_FFT=512 → 257）；
判据是 `mag` 各通道来自 `dsp.stft_multi` 的模。`T` 可以是任意长度——
网络内部会做形状裁剪，见 `_crop_or_pad`。

依赖
----
仅 torch（`torch.nn`）。不依赖 numpy、不依赖本仓库其他模块，
因此模型可以独立实例化与测试，不需要先跑仿真管道。

架构（设计文档 §T3.1：卷积编码器 → GRU 瓶颈 → 卷积解码器）
----------------------------------------------------------
下面以默认配置、输入 [B, 4, 257, T] 为例，括号内是实测尺寸（T=250）：

    输入      [B, 4, 257, 250]
    log1p     [B, 4, 257, 250]                    动态范围压缩
    编码器    4 层 (Conv2d 5×5 / stride 2, BN, LeakyReLU)
              每层把频率轴与时帧轴同时减半（取整规则 (s-1)//2 + 1）：
              [B, 16, 129, 125] → [B, 32, 65, 63]
              → [B, 64, 33, 32] → [B, 128, 17, 16]
    瓶颈      permute+reshape 成 [B, T', C·F']（此处 2176）
              → GRU（双向，batch_first）→ [B, T', 2·hidden]（此处 256）
              → Linear 升维回 [B, T', C·F']     ← 关键，见第 6 条
              → reshape/permute 回 [B, 128, 17, 16]
    解码器    4 层 (ConvTranspose2d 5×5 / stride 2, BN, LeakyReLU)
              逐层上采样并裁剪到目标尺寸，最终回到 [B, 16, 257, 250]
    输出      Conv2d 5×5 → 1 通道 → squeeze → [B, 257, 250] → 激活

注意编码器每层输出尺寸是**逐层实测**的，不是 `n_freqs // 2**k`：整数除法不
满足结合律，257 → 129 → 65 → 33 → 17，而 257 // 2⁴ = 16。

参数规模
--------
默认配置（n_mics=4, n_freqs=257, base_channels=32,
encoder_channels=(16,32,64,128), gru_hidden=128, gru_layers=1, bidirectional=True）
**实测 3,281,137 个可训练参数**（约 3.28 M，模型体积约 13 MB fp32），
满足「5 M 以内」；batch=2、T=250 时 CUDA 峰值显存仅 81.9 MB（含反传），
11.9 GB 的卡上余量充足。
`CRN_SMALL` 实测 601,089 个参数。

几处非显然的设计决策
--------------------
1. **为什么用 `log1p` 而不是 `log(x+eps)`**：幅度谱动态范围极大（静音帧可到
   1e-6，语音帧可到 1）。`log(x+eps)` 的结果依赖 eps 取值——eps 太小时静音帧
   变成很大的负数、训练不稳定，太大时又压掉低电平细节。`log1p(x)` 在 x→0 时
   → 0（有界），x 大时 ≈ log(x)，无需调参且天然把静音帧映射到 0 附近。
2. **为什么编码器逐层同时下采样频率轴与时帧轴**：频率轴的相邻频点高度相关，
   卷积本身就在利用这一点；时帧轴下采样则是为了缩短 GRU 的序列长度——这是
   显存的主要矛盾。两者用同一 stride 便于解码器精确还原。
3. **为什么下采样到频率轴约 4 就停**：4 元圆阵半径仅 4 cm，阵列孔径远小于
   波长，其空间分辨能力集中在中低频。频率轴继续下采样会把 257 个频点压成
   两三个，掩蔽将无法区分共振峰与谐波结构，反而损害增强质量。因此层数由
   `n_freqs` 动态决定，而不是固定写死。
4. **为什么 GRU 放在频率轴已充分压缩之后**：GRU 的输入维度是 `C·F'`，
   F' 若还有几十个频点，输入维度会达到数千，参数量与显存瞬间失控。
   先卷积压缩、再循环建模，是 CRN 这一类结构的关键。
5. **为什么 GRU 之后要有一层 Linear 升维**：GRU 的职责是**压缩**——它把
   `C·F'`（2176）压成 `hidden·方向数`（256）。因此进解码器前必须用
   `gru_expand` 显式升维回 `C·F'`，**不能靠 reshape**：两者维度不等会直接抛错，
   而报错信息（尺寸不匹配）极易被误读成 `gru_input_size` 算错了。
6. **为什么必须显式裁剪输出**：`Conv2d(stride=2)` 与
   `ConvTranspose2d(stride=2)` 的整除规则并不互为逆运算——`ConvTranspose2d`
   的输出边长是 `(s-1)//2 + 1` 的逆，实测每层比目标多 1 帧（258 vs 257、
   126 vs 125），必须裁掉。`n_freqs=257` 与任意 `T` 都不是 2 的幂，这个偏差必然
   出现。若放任不管，掩蔽与观测频谱在第一维上就对不齐、无法相乘，而且
   **报错会推迟到 T4.1 才出现**。故在 forward 中按逐层记录的目标尺寸还原。
7. **为什么解码器目标尺寸要包含输入尺寸本身**：`sizes` 只记了编码器的 4 个
   **输出**尺寸，而解码器第 1 层要做的是把 17 还原到 33（对应编码器第 2 层的
   输出），最后一层才回到 257。正确的目标序列是 `[输入尺寸] + sizes[:-1]`；
   若写成 `reversed(sizes)`，整体错位一层，输出会停在第 1 层编码器的尺寸
   （129×125），**且不会报任何错**——这是个静默错误，只能靠形状测试发现。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 编码器停止下采样的频率轴下界（理由见模块 docstring 第 3 条）
MIN_FREQ_BINS = 4

# 掩蔽激活函数可选项
MASK_ACTS = ("sigmoid", "relu")

# 默认配置下实测的可训练参数量，供测试与文档核对
DEFAULT_NUM_PARAMETERS = 3_281_137


def _crop_or_pad(x: torch.Tensor, f_target: int, t_target: int) -> torch.Tensor:
    """把 x 的空间维裁剪或补齐到 (f_target, t_target)。

    实测在本项目的配置下，`ConvTranspose2d(kernel=5, stride=2, padding=2,
    output_padding=1)` 每层的输出恰好比目标**各维多 1 帧**（258 vs 257、
    126 vs 125），因此正常路径是**纯裁剪**——已验证 257→129→65→33→17 的
    往返全程只需裁剪，不需要补零。

    保留补齐分支是为了 T 极小（如 T=1、3）时编码器会把时间轴压到 1 帧、
    解码器再放大回来，此时边界处可能短缺，补零比报错更合适。

    裁剪一律取靠前的窗口（而非居中）——因为解码器输出与目标帧的对应关系是
    从第 0 帧对齐的，截尾才不引入位移。
    """
    f_now, t_now = x.shape[-2], x.shape[-1]

    if f_now > f_target:
        x = x[..., :f_target, :]
    if t_now > t_target:
        x = x[..., :, :t_target]

    # 再补齐
    f_now, t_now = x.shape[-2], x.shape[-1]
    pad_f, pad_t = f_target - f_now, t_target - t_now
    if pad_f > 0 or pad_t > 0:
        # F.pad 的最后一维在前：先补时帧轴，再补频率轴
        x = F.pad(x, (0, max(pad_t, 0), 0, max(pad_f, 0)))

    return x


class CRN(nn.Module):
    """多通道 CRN 掩蔽估计器：卷积编码器 → GRU 瓶颈 → 卷积解码器。

    用法：
        model = CRN(n_mics=4, n_freqs=257)     # 默认配置
        mask = model(mag)                       # mag [B,4,257,T] → [B,257,T]

    Parameters
    ----------
    n_mics        : 输入麦克风通道数。多通道在**通道维拼接**后送入编码器，
                    这是「多通道」的体现——网络因此能看到阵元间的幅度/相位差异。
    n_freqs       : 频率轴长度，须为 `N_FFT // 2 + 1`（512 → 257）。
    base_channels : 解码器宽度基准。默认 32 使得解码器宽度与
                    `encoder_channels` 的镜像一致（128→64→32→16）；
                    取 16 则解码器宽度减半（64→32→16→8）。
                    默认配置下不改变参数量上限，仅供缩小模型用。
    encoder_channels : 编码器各层输出通道数。其长度决定**最大**下采样层数，
                    实际层数还会被 `n_freqs` 与 `MIN_FREQ_BINS` 限制（见 `depth`）。
    gru_hidden    : GRU 隐层维度。
    gru_layers    : GRU 层数。
    bidirectional : 是否双向。双向能看到未来帧，离线增强可接受；
                    实时场景应设 False。
    mask_act      : 掩蔽激活。"sigmoid" 给出 (0,1) 的软掩蔽（默认）；
                    "relu" 无上界，仅在需要幅度放大的实验中使用。
    """

    def __init__(self, n_mics: int = 4, n_freqs: int = 257, base_channels: int = 32,
                 encoder_channels: tuple = (16, 32, 64, 128), gru_hidden: int = 128,
                 gru_layers: int = 1, bidirectional: bool = True,
                 mask_act: str = "sigmoid"):
        super().__init__()

        if mask_act not in MASK_ACTS:
            raise ValueError(f"未知掩蔽激活：{mask_act!r}，可选 {MASK_ACTS}")
        if n_mics < 1 or n_freqs < 1:
            raise ValueError(f"n_mics / n_freqs 必须为正，收到 {n_mics} / {n_freqs}")
        if not encoder_channels:
            raise ValueError("encoder_channels 不能为空")

        self.n_mics = int(n_mics)
        self.n_freqs = int(n_freqs)
        self.base_channels = int(base_channels)
        self.encoder_channels = tuple(int(c) for c in encoder_channels)
        self.gru_hidden = int(gru_hidden)
        self.gru_layers = int(gru_layers)
        self.bidirectional = bool(bidirectional)
        self.mask_act = mask_act

        # ── 决定实际下采样层数 ──────────────────────────────────
        # 每层把频率轴减半（向下取整）。一直下采样到不足 2·MIN_FREQ_BINS
        # 为止，保证最深层仍有约 MIN_FREQ_BINS 个频点。
        depth = min(len(self.encoder_channels), self._max_depth_by_freqs())
        if depth < 1:
            raise ValueError(
                f"n_freqs={self.n_freqs} 太小，无法做任何频率轴下采样"
                f"（需 ≥ {2 * MIN_FREQ_BINS}）")
        self.depth = depth
        self._enc_channels = self.encoder_channels[:depth]

        # 解码器宽度：以 base_channels 相对 32 的比例缩放最后一层编码通道数。
        # 这样 base_channels=32 时解码器恰为编码器的镜像，取更小值则整体变窄。
        scale = self.base_channels / 32.0
        self._dec_channels = [max(1, int(round(c * scale)))
                              for c in reversed(self._enc_channels)]

        # ── 编码器 ──────────────────────────────────────────────
        # 第 1 层输入通道是 n_mics：多通道在此汇合
        enc_in = [self.n_mics, *self._enc_channels[:-1]]
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=(5, 5), stride=(2, 2),
                          padding=(2, 2)),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for cin, cout in zip(enc_in, self._enc_channels)
        ])

        # ── GRU 瓶颈 ────────────────────────────────────────────
        # 频率轴已被压缩到约 4，故输入维度 C·F' 是可控的（见 docstring 第 4 条）。
        # F' 必须逐层模拟取整，**不能**写成 n_freqs // 2**depth：
        # 整数除法不满足结合律，257 → 129 → 65 → 33 → 17，而 257 // 2**4 = 16。
        # 用后者会让 GRU 的 input_size 少算一层频点（2048 vs 实际 2176）而报错。
        self.enc_freq_sizes = self._simulate_freq_sizes()
        self.gru_input_size = self._enc_channels[-1] * self.enc_freq_sizes[-1]
        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        gru_out = self.gru_hidden * (2 if self.bidirectional else 1)

        # GRU 是**压缩**：它把 C·F' 维空间特征压到 gru_out 维（默认 2176 → 256）。
        # 因此进解码器前必须显式升维还原，**不能**靠 reshape——gru_out 与 C·F'
        # 一般不相等，reshape 会直接抛尺寸错误，且报错信息会误导人以为是
        # gru_input_size 算错了。这一层是瓶颈的逆变换。
        self.gru_expand = nn.Linear(gru_out, self.gru_input_size)
        self.gru_norm = nn.BatchNorm2d(self._enc_channels[-1])

        # ── 解码器 ──────────────────────────────────────────────
        dec_in = [self._enc_channels[-1], *self._dec_channels[:-1]]
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(cin, cout, kernel_size=(5, 5), stride=(2, 2),
                                   padding=(2, 2), output_padding=(1, 1)),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for cin, cout in zip(dec_in, self._dec_channels)
        ])

        # ── 输出层：压到 1 通道（掩蔽）再激活 ────────────────────
        self.out_conv = nn.Conv2d(self._dec_channels[-1], 1,
                                  kernel_size=(5, 5), padding=(2, 2))

    # ── 构造期辅助 ──────────────────────────────────────────────

    def _max_depth_by_freqs(self) -> int:
        """按频率轴能安全下采样的层数（每层减半，最深层保留 ≥ MIN_FREQ_BINS）。"""
        d, f = 0, self.n_freqs
        while d < len(self.encoder_channels) and f // 2 >= MIN_FREQ_BINS:
            f //= 2
            d += 1
        return d

    def _simulate_freq_sizes(self) -> list[int]:
        """逐层模拟 Conv2d(kernel=5, stride=2, padding=2) 后的频率轴长度。

             f_out = floor((f_in + 2·2 - 5) / 2) + 1 = floor((f_in - 1) / 2) + 1

        必须逐层算而不是一次除以 2**depth——整数除法不可结合（见 __init__ 注释）。
        返回值同时用于 GRU 的 input_size 与自检。
        """
        sizes, f = [], self.n_freqs
        for _ in range(self.depth):
            f = (f - 1) // 2 + 1
            sizes.append(f)
        return sizes

    # ── 前向 ────────────────────────────────────────────────────

    def forward(self, mag: torch.Tensor) -> torch.Tensor:
        """mag [B, n_mics, n_freqs, T] 线性幅度谱 → mask [B, n_freqs, T]，取值 (0,1)。

        `T` 可为任意长度（含非 2 的幂），输出严格回到输入的 (n_freqs, T)。
        """
        if mag.dim() != 4:
            raise ValueError(
                f"期望 [B, n_mics, n_freqs, T] 的 4 维幅度谱，收到 {tuple(mag.shape)}")
        if mag.shape[1] != self.n_mics:
            raise ValueError(
                f"通道数不符：期望 {self.n_mics}，收到 {mag.shape[1]}")
        if mag.shape[2] != self.n_freqs:
            raise ValueError(
                f"频率轴不符：期望 {self.n_freqs}（= N_FFT//2+1），收到 {mag.shape[2]}。"
                f"模型是在 __init__ 时按固定 n_freqs 建好的，不能中途改变。")

        # 1) 动态范围压缩。log1p 而非 log(x+eps)，理由见 docstring 第 1 条。
        #    mag 理论非负，clamp 兜住上游可能传来的极小负值（浮点误差）。
        x = torch.log1p(mag.clamp_min(0.0))

        # 2) 编码器。逐层记录真实尺寸，解码器据此精确还原（关键，见 docstring 第 5 条）
        #    注意张量轴序固定为 [B, C, 频率, 时间]：-2 是频率、-1 是时间。
        #    核对过 Conv2d(kernel=(5,5), stride=(2,2), padding=(2,2)) 的实际行为：
        #    257 频率 → 129，250 帧 → 125，即每轴 (s-1)//2 + 1。
        sizes: list[tuple[int, int]] = []
        for enc in self.encoder:
            x = enc(x)
            sizes.append((x.shape[-2], x.shape[-1]))

        # 解码器各层要还原到的尺寸。必须包含**编码器输入尺寸**本身：
        # 第 1 层解码是把 17 还原到 33（不是 17→17），最后一层才回到 257。
        # 只写 reversed(sizes) 会整体错位一层——实测输出只剩下 (129,125)，
        # 即第 1 层编码器的尺寸，而不会报任何错。
        dec_targets = [(mag.shape[2], mag.shape[3]), *sizes[:-1]]
        assert len(dec_targets) == self.depth

        # 3) GRU 瓶颈：把频率轴与通道维压平进特征维，时间轴作为序列维
        b, c_enc, f_enc, t_enc = x.shape
        seq = x.permute(0, 3, 1, 2).reshape(b, t_enc, c_enc * f_enc)   # [B, T', C·F']
        seq, _ = self.gru(seq)                                        # [B, T', gru_out]

        # 4) 升维还原回 [B, C, F', T']。GRU 把 C·F' 压到了 gru_out（默认 256），
        #    必须经 gru_expand 这一层 Linear 才能回到 C·F'，不能直接 reshape。
        seq = self.gru_expand(seq)                                    # [B, T', C·F']
        # reshape 后语义是 [B, T', C, F']（F' 落在最后一维），
        # 故换回 [B, C, F', T'] 要写 permute(0, 2, 3, 1)。
        # 写成 permute(0, 3, 1, 2) 会得到 [B, F', T', C]，通道维被错当成 F'，
        # 表现为 BatchNorm 报「running_mean should contain 17 elements not 128」。
        x = seq.reshape(b, t_enc, c_enc, f_enc).permute(0, 2, 3, 1)   # [B, C, F', T']
        x = self.gru_norm(x)
        assert x.shape[-2:] == (f_enc, t_enc), "GRU 还原后的频率/时间轴与编码器不一致"

        # 5) 解码器：逐层上采样，并按目标尺寸裁剪/补齐
        for dec, (f_tgt, t_tgt) in zip(self.decoder, reversed(dec_targets)):
            x = dec(x)
            x = _crop_or_pad(x, f_tgt, t_tgt)

        # 6) 输出掩蔽
        x = self.out_conv(x)                                   # [B, 1, F, T]
        mask = x.squeeze(1)                                    # [B, F, T]

        if self.mask_act == "sigmoid":
            return torch.sigmoid(mask)
        return F.relu(mask)

    # ── 规模查询 ────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        """可训练参数总量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:                                 # pragma: no cover
        return (f"{self.__class__.__name__}(n_mics={self.n_mics}, "
                f"n_freqs={self.n_freqs}, depth={self.depth}, "
                f"enc={self._enc_channels}, dec={self._dec_channels}, "
                f"gru={self.gru_hidden}x{self.gru_layers}"
                f"{'bi' if self.bidirectional else 'uni'}, "
                f"mask_act={self.mask_act!r}, "
                f"params={self.num_parameters:,})")


def CRN_SMALL(**kwargs) -> CRN:
    """小号 CRN，供 T3.4 过拟合测试与显存冒烟使用。

    参数量约为默认配置的 1/4，可在 10 条样本上快速训到接近零 loss——
    这是验证「模型与数据管道没有 bug」的关键手段（实施计划 T3.4）。

    默认 n_mics=4, n_freqs=257，与正式配置一致，避免因形状不一致而
    掩盖管道问题；只缩小通道数与 GRU 宽度。
    """
    params = dict(
        n_mics=4,
        n_freqs=257,
        base_channels=16,
        encoder_channels=(8, 16, 32),
        gru_hidden=64,
        gru_layers=1,
        bidirectional=True,
        mask_act="sigmoid",
    )
    params.update(kwargs)
    return CRN(**params)


if __name__ == "__main__":                                     # pragma: no cover
    # 直接运行本文件即可做形状与规模自检（不依赖 pytest）：
    #   D:\Anaconda_envs\envs\pytorch\python.exe -m nbse.models.crn
    torch.manual_seed(0)

    print("=" * 68)
    for name, model in (("CRN(默认)", CRN()), ("CRN_SMALL()", CRN_SMALL())):
        print(f"{name}")
        print(f"  {model}")
        model.eval()
        for b, t in ((2, 250), (1, 125), (3, 63)):
            mag = torch.rand(b, model.n_mics, model.n_freqs, t)
            with torch.no_grad():
                mask = model(mag)
            ok = tuple(mask.shape) == (b, model.n_freqs, t)
            print(f"  输入 [B={b}, {model.n_mics}, {model.n_freqs}, T={t}] "
                  f"→ 输出 {tuple(mask.shape)}  {'OK' if ok else '形状错误!'}")
            assert ok, f"形状不匹配：{tuple(mask.shape)} != {(b, model.n_freqs, t)}"
            assert torch.isfinite(mask).all(), "掩蔽出现 NaN/Inf"
            assert (mask > 0).all() and (mask < 1).all(), "掩蔽越界 (0,1)"
        print()

    # 显存冒烟：训练态 + 反传（BatchNorm 训练/推理两条路径都要走到）
    model = CRN()
    print(f"默认配置参数量：{model.num_parameters:,}（上限 5,000,000）")
    assert model.num_parameters < 5_000_000, "参数量超出 5M 预算"

    mag = torch.rand(2, 4, 257, 250)
    if torch.cuda.is_available():
        model = model.cuda()
        mag = mag.cuda()
    model.train()
    loss = model(mag).mean()
    loss.backward()
    if torch.cuda.is_available():
        print(f"训练态反传通过；CUDA 峰值显存 "
              f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    else:
        print("训练态反传通过（CPU）")
    print("=" * 68)
    print("自检全部通过")
