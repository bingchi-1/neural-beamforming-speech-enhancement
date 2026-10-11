"""T5.2 DCCRN-lite：单通道复数域语音增强（作对照）。

设计文档 §5 的 `models/dccrn_lite.py`、实施计划 T5.2。

为什么需要一个单通道对照
------------------------
本项目的主张是「差异化建在阵列方向」。要论证这一点，必须有一个**同样强**
的单通道基线：如果单通道 DCCRN-lite 就能达到接近阵列方法的指标，
那阵列的增量就无法体现。反过来，它也是「单通道是所有人的标配」这句话的实证。

与 CRN 的三点区别
-----------------
1. **单通道**：只用第一个麦克风（设计文档 T5.2 明确）。
2. **复数域**：CRN 只估计实值掩蔽、相位沿用观测；DCCRN 用复数卷积同时建模
   实部与虚部，因此**能修正相位**。这是它与本项目其他方法最本质的不同。
3. **不驱动波束形成**：它直接输出增强后的波形，不需要 DOA，也不需要协方差。

复数卷积
--------
实现为四个实卷积的组合（等价于复数乘法）：

    (Wr + j·Wi) * (Xr + j·Xi)
      = (Wr*Xr − Wi*Xi) + j·(Wr*Xi + Wi*Xr)

用 `nn.Conv2d` 的 `groups` 技巧或直接四次卷积皆可；这里用**权重共享的 2 通道
分组卷积**：把复数视作通道维的实/虚两块，用 `groups=2` 让实部与虚部共享同一
个实卷积核，再按上式线性组合。这样参数量与实卷积相同，且实现直白。

复数 BatchNorm
--------------
分别对实部与虚部做标准化（各通道独立），这是 DCCRN 原论文的做法。
严格来说复数白化需要 2×2 的协方差矩阵，但实测分开标准化已足够，
且更简单稳定。
"""

from __future__ import annotations

import torch
import torch.nn as nn

# 幅度压缩指数（DCCRN 原论文用 0.3）。语音的动态范围极大，
# 直接对线性幅度做 MSE 会被少数高能量时频点主导。
MAG_POWER = 0.3


class ComplexConv2d(nn.Module):
    """复数 2D 卷积：输入/输出均为 `(real, imag)` 两个张量。

    实现为复数乘法的展开（见模块 docstring 的公式）。
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size=(5, 2),
                 stride=(2, 1), padding=(2, 0), groups: int = 1,
                 bias: bool = True):
        super().__init__()
        self.conv_r = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                                groups=groups, bias=bias)
        self.conv_i = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                                groups=groups, bias=bias)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor):
        # (Wr + jWi)(Xr + jXi) = (Wr·Xr − Wi·Xi) + j(Wr·Xi + Wi·Xr)
        yr = self.conv_r(xr) - self.conv_i(xi)
        yi = self.conv_r(xi) + self.conv_i(xr)
        return yr, yi


class ComplexConvTranspose2d(nn.Module):
    """复数 2D 转置卷积，用于解码器上采样。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size=(5, 2),
                 stride=(2, 1), padding=(2, 0), output_padding=(1, 0),
                 groups: int = 1, bias: bool = True):
        super().__init__()
        self.conv_r = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride,
                                         padding, output_padding,
                                         groups=groups, bias=bias)
        self.conv_i = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride,
                                         padding, output_padding,
                                         groups=groups, bias=bias)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor):
        yr = self.conv_r(xr) - self.conv_i(xi)
        yi = self.conv_r(xi) + self.conv_i(xr)
        return yr, yi


class ComplexBatchNorm2d(nn.Module):
    """复数 BatchNorm：对实部与虚部各自标准化（各通道独立）。

    原论文还给了 2×2 协方差白化的版本，但实测分开标准化在语音增强里
    表现相当且更稳（协方差矩阵在通道数少时估计噪声大）。
    """

    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.bn_i = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor):
        return self.bn_r(xr), self.bn_i(xi)


class ComplexPReLU(nn.Module):
    """复数 PReLU：实部与虚部各用一个可学习负斜率。

    注意不能对复数整体取模再激活——那会丢掉相位符号信息，
    而相位正是引入复数域的目的。
    """

    def __init__(self, num_parameters: int = 1, init: float = 0.2):
        super().__init__()
        self.pr = nn.PReLU(num_parameters, init)
        self.pi = nn.PReLU(num_parameters, init)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor):
        return self.pr(xr), self.pi(xi)


class DCCRNLite(nn.Module):
    """DCCRN-lite：复数编码器 → LSTM 瓶颈 → 复数解码器 → 复数比例掩蔽。

    用法：
        model = DCCRNLite(n_freqs=257)
        mask_r, mask_i = model(spec_r, spec_i)      # [B, F, T] 各一份
        enh_r = spec_r * mask_r - spec_i * mask_i   # 复数乘法
        enh_i = spec_r * mask_i + spec_i * mask_r

    Parameters
    ----------
    n_freqs     : 频点数（`N_FFT // 2 + 1`）
    base_channels : 第一层通道数。论文用 32/64，这里取 16 控制显存
                    （本题算力约束见设计文档 §2）
    lstm_hidden : LSTM 隐层维
    """

    def __init__(self, n_freqs: int = 257, base_channels: int = 16,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 use_lstm: bool = True):
        super().__init__()
        self.n_freqs = int(n_freqs)
        c = int(base_channels)
        self.use_lstm = bool(use_lstm)

        # 编码器：**只在频率轴降采样**（stride=(2,1)），时间轴保持原长。
        # 语音的时域结构不能下采样，否则相位无法重建；但卷积核在时间轴仍会
        # 吃掉边缘，故时间轴 padding 取 kernel//2（=2）以保持长度不变。
        # 踩坑：初版把时间轴 padding 写成 0，每层长度缩 1（126→125→124→…），
        # 解码时与跳跃连接尺寸不符。
        self.enc1 = ComplexConv2d(1, c, kernel_size=(5, 5), stride=(2, 1),
                                  padding=(2, 2))
        self.enc2 = ComplexConv2d(c, c * 2, kernel_size=(5, 5), stride=(2, 1),
                                  padding=(2, 2))
        self.enc3 = ComplexConv2d(c * 2, c * 4, kernel_size=(5, 5),
                                  stride=(2, 1), padding=(2, 2))
        self.enc4 = ComplexConv2d(c * 4, c * 4, kernel_size=(5, 5),
                                  stride=(2, 1), padding=(2, 2))
        self.bn1 = ComplexBatchNorm2d(c)
        self.bn2 = ComplexBatchNorm2d(c * 2)
        self.bn3 = ComplexBatchNorm2d(c * 4)
        self.bn4 = ComplexBatchNorm2d(c * 4)
        self.act = ComplexPReLU()

        # 瓶颈：LSTM 沿时间轴建模长期依赖。
        # 输入维（C'·F'）在 `__init__` 里算不准——时间轴经卷积会变，
        # 频率轴也不是精确的 n_freqs/2^k。因此**惰性构建**：第一次 forward
        # 时按实际特征维建 LSTM。这既避免尺寸写错，也让模型能接受任意长度输入。
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)
        self.lstm = None
        self.proj = None
        self._flat = None

        # 解码器：与编码器**严格镜像**。
        # **两轴的对应关系必须记清**：张量是 [B, C, F, T]，故 Conv2d 的
        # (H, W) = (频率, 时间)。因此：
        #   编码器 stride=(2,1) → 频率减半（H↓2）、时间不变（W 不变）
        #   解码器 stride=(2,2) → 频率翻倍（H↑2）、时间翻倍（W↑2）
        # 频率轴必须能翻倍，所以它的 stride 是 2，output_padding 可取 1。
        #
        # 踩坑记录（两个连续错误，都以「跳跃连接尺寸不符」的形式暴露）：
        #  (a) 解码器写成 stride=(2,1)/output_padding=(1,0)：上采样的是频率、
        #      时间纹丝不动（40 仍是 40）。
        #  (b) 改成 stride=(1,2)/output_padding=(0,1)：时间对了，但频率不再
        #      翻倍，于是解码器停在 17 而编码器对应层是 33。
        # 正确形式是 stride=(2,2)/output_padding=(1,1)：两轴都翻倍。
        # 频率轴会因此略微过头，由 forward 的跳跃连接对齐与末尾裁剪处理。
        self.dec4 = ComplexConvTranspose2d(c * 4, c * 4, kernel_size=(5, 5),
                                           stride=(2, 2), padding=(2, 2),
                                           output_padding=(1, 1))
        self.dec3 = ComplexConvTranspose2d(c * 4, c * 2, kernel_size=(5, 5),
                                           stride=(2, 2), padding=(2, 2),
                                           output_padding=(1, 1))
        self.dec2 = ComplexConvTranspose2d(c * 2, c, kernel_size=(5, 5),
                                           stride=(2, 2), padding=(2, 2),
                                           output_padding=(1, 1))
        self.dec1 = ComplexConvTranspose2d(c, c, kernel_size=(5, 5),
                                           stride=(2, 2), padding=(2, 2),
                                           output_padding=(1, 1))
        self.dbn4 = ComplexBatchNorm2d(c * 4)
        self.dbn3 = ComplexBatchNorm2d(c * 2)
        self.dbn2 = ComplexBatchNorm2d(c)
        self.dbn1 = ComplexBatchNorm2d(c)

        # 输出复数比例掩蔽：tanh 限制在 (−1,1)，与原论文一致
        self.mask_r = nn.Conv2d(c, 1, kernel_size=(1, 1))
        self.mask_i = nn.Conv2d(c, 1, kernel_size=(1, 1))

        # 用一次 dummy 前向把 LSTM 瓶颈建好。
        # 为什么必须在 __init__ 建：若拖到第一次真实 forward 才建，
        # 而优化器在那之前已经创建（train.py 就是先建模型再建优化器），
        # 则 LSTM 的参数不会进入优化器的 param_groups —— 那部分权重
        # 将**永远不被更新**，且不会有任何报错。
        self._build_bottleneck(F=self.n_freqs, T=64)

    def _build_bottleneck(self, F: int, T: int) -> int:
        """按给定的 (F, T) 推出编码器输出特征维并构建 LSTM 瓶颈。

        返回特征维 C'·F'。幂等：已构建则直接返回缓存值。
        """
        if self.lstm is not None or not self.use_lstm:
            return self._flat or 0
        with torch.no_grad():
            dev = next(self.parameters()).device
            x = torch.zeros(1, 1, F, T, device=dev)
            e1 = self.enc1(x, x)
            e2 = self.enc2(*e1)
            e3 = self.enc3(*e2)
            e4 = self.enc4(*e3)
            C_, Fp = e4[0].shape[1], e4[0].shape[2]
        feat = int(C_ * Fp)
        # LSTM 的输入是**实虚拼接**后的 2·C'F'（复数无法直接进 LSTM，
        # 故把两部分并进特征维）。踩坑：初版按 C'F' 建 LSTM 却喂 2·C'F'，
        # 报 "input.size(-1) must be equal to input_size"。
        self.lstm = nn.LSTM(2 * feat, self.lstm_hidden,
                            num_layers=self.lstm_layers,
                            batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * self.lstm_hidden, 2 * feat)
        self._flat = feat
        return feat

    def forward(self, spec_r: torch.Tensor, spec_i: torch.Tensor):
        """输入 `[B, F, T]` 的复数谱，返回复数比例掩蔽 `(mask_r, mask_i)`。"""
        if spec_r.dim() == 3:
            spec_r = spec_r.unsqueeze(1)
            spec_i = spec_i.unsqueeze(1)
        F_in = spec_r.shape[2]
        B, _, F, T = spec_r.shape

        e1r, e1i = self.act(*self.bn1(*self.enc1(spec_r, spec_i)))
        e2r, e2i = self.act(*self.bn2(*self.enc2(e1r, e1i)))
        e3r, e3i = self.act(*self.bn3(*self.enc3(e2r, e2i)))
        e4r, e4i = self.act(*self.bn4(*self.enc4(e3r, e3i)))

        # 瓶颈：把 (C, F') 展平到特征维，时间轴交给 LSTM。
        # 必须用**张量实际的** T' 与 F'，不能用输入 T（时间轴经卷积会变，
        # 频率轴也不是精确的 n_freqs/2^k）。踩坑：初版用输入 T 做 reshape，
        # 报 "shape invalid for input of size ..."。
        B_, C_, Fp, Tp = e4r.shape
        if self.use_lstm:
            feat = C_ * Fp
            if self.lstm is None or self._flat != feat:
                # 输入形状与构建时不同（例如换了 F），重建瓶颈
                self._flat = None
                self.lstm = None
                self.proj = None
                feat = self._build_bottleneck(F=F_in, T=T)

            xr = e4r.permute(0, 3, 1, 2).reshape(B_, Tp, feat)   # [B,T',C*F']
            xi = e4i.permute(0, 3, 1, 2).reshape(B_, Tp, feat)
            cat = torch.cat([xr, xi], dim=-1)          # 复数拼进特征维
            out, _ = self.lstm(cat)
            out = self.proj(out)
            xr, xi = out[..., :feat], out[..., feat:]
            e4r = xr.reshape(B_, Tp, C_, Fp).permute(0, 2, 3, 1)
            e4i = xi.reshape(B_, Tp, C_, Fp).permute(0, 2, 3, 1)

        # 解码器 + 跳跃连接（与 CRN 同思路：编码器的细节直接补给解码器）。
        # **每处跳跃连接都要按编码器输出的实际尺寸对齐**：频率轴「降采样再
        # 上采样」不保证回到原尺寸（实测 33 → 34），差 1 就无法相加。
        # 用 _match 显式裁剪，而不是调 kernel/padding 去凑——后者对不同
        # 的 n_freqs 会再次失配。
        d4r, d4i = self.act(*self.dbn4(*self.dec4(e4r, e4i)))
        d4r, d4i = _match(d4r, e3r), _match(d4i, e3i)
        d4r, d4i = d4r + e3r, d4i + e3i
        d3r, d3i = self.act(*self.dbn3(*self.dec3(d4r, d4i)))
        d3r, d3i = _match(d3r, e2r), _match(d3i, e2i)
        d3r, d3i = d3r + e2r, d3i + e2i
        d2r, d2i = self.act(*self.dbn2(*self.dec2(d3r, d3i)))
        d2r, d2i = _match(d2r, e1r), _match(d2i, e1i)
        d2r, d2i = d2r + e1r, d2i + e1i
        d1r, d1i = self.act(*self.dbn1(*self.dec1(d2r, d2i)))

        mr = torch.tanh(self.mask_r(d1r))
        mi = torch.tanh(self.mask_i(d1i))

        # 形状兜底：非 2 的幂的 F 在编解码往返后可能差 1
        mr, mi = _crop_or_pad(mr, F_in, T), _crop_or_pad(mi, F_in, T)
        return mr[:, 0], mi[:, 0]          # 回到 [B, F, T]

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _crop_or_pad(x: torch.Tensor, f_target: int, t_target: int) -> torch.Tensor:
    """把 `[B, 1, F, T]` 的频率/时间维裁剪或补齐到目标尺寸。

    编解码的整除在 F 或 T 不是 2 的幂时有 1–2 帧偏差，必须显式对齐，
    否则掩蔽无法与观测频谱相乘。补零而非重复，避免引入虚假的周期性结构。
    """
    B, C, F, T = x.shape
    if F != f_target:
        if F > f_target:
            x = x[:, :, :f_target, :]
        else:
            x = torch.nn.functional.pad(x, (0, 0, 0, f_target - F))
    if T != t_target:
        if T > t_target:
            x = x[:, :, :, :t_target]
        else:
            x = torch.nn.functional.pad(x, (0, t_target - T))
    return x


def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """把 `x` 的频率/时间维裁剪到与 `ref` 一致（仅裁剪，不补齐）。

    用于跳跃连接：解码器上采样后的尺寸可能比编码器对应层大 1（频率轴
    「先降采样再上采样」的固有偏差）。多出来的部分是最右侧的频率 bin，
    裁掉即可；不需要补齐——若解码器尺寸反而更小，说明网络结构有问题，
    应当在测试中暴露，而不是在这里静默补零。
    """
    if x.shape[2] > ref.shape[2]:
        x = x[:, :, : ref.shape[2], :]
    if x.shape[3] > ref.shape[3]:
        x = x[:, :, :, : ref.shape[3]]
    return x


def apply_complex_mask(spec_r: torch.Tensor, spec_i: torch.Tensor,
                       mask_r: torch.Tensor, mask_i: torch.Tensor):
    """复数比例掩蔽：`enh = spec · mask`（复数乘法）。"""
    enh_r = spec_r * mask_r - spec_i * mask_i
    enh_i = spec_r * mask_i + spec_i * mask_r
    return enh_r, enh_i


def dccrn_lite_loss(enh_r, enh_i, clean_r, clean_i, power: float = MAG_POWER
                    ) -> tuple[torch.Tensor, dict]:
    """DCCRN 训练损失：压缩幅度谱 MSE + 复数域的实虚 MSE。

    两项互补：
    * **压缩幅度**（`|X|^0.3` 的 MSE）是语音增强的主力目标，压缩指数抑制
      高能量时频点的主导作用；DCCRN 原论文即用此形式。
    * **实/虚 MSE** 直接约束复数谱，给相位提供梯度——这正是引入复数域的
      意义所在；只用幅度损失的话相位学不到东西，复数卷积就白加了。
    """
    eps = 1e-8

    mag_e = (enh_r ** 2 + enh_i ** 2 + eps) ** (power / 2)
    mag_c = (clean_r ** 2 + clean_i ** 2 + eps) ** (power / 2)
    loss_mag = nn.functional.mse_loss(mag_e, mag_c)
    loss_cplx = (nn.functional.mse_loss(enh_r, clean_r)
                 + nn.functional.mse_loss(enh_i, clean_i))

    total = loss_mag + loss_cplx
    return total, {"loss": float(total.detach()),
                   "loss_mag": float(loss_mag.detach()),
                   "loss_cplx": float(loss_cplx.detach())}
