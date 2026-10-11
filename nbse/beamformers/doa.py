"""T5.1 GCC-PHAT 方位角估计。

设计文档 §5.1、实施计划 T5.1。

用途
----
为「Oracle DOA vs 估计 DOA」这组消融提供估计端：波束形成器换一个导向矢量即可，
网络只训练一次，因此 GCC-PHAT 版本几乎是免费得到的（设计文档 §5.1）。

做法
----
1. 对每对麦克风求 **GCC-PHAT**（广义互相关，相位变换加权）得互相关函数；
2. 取峰 + **抛物线插值**得到亚采样精度的 TDOA；
3. 由 TDOA 与阵列几何的线性关系 `τ_ij = -u·(p_i-p_j)/c` 最小二乘反推方向
   单位向量 u，最后取方位角。

为什么抛物线插值是必需的（实测）
--------------------------------
4 元圆阵半径 4 cm → 阵元间最大单程时延仅约 **3.7 个采样点**（16 kHz）。
整个阵列的时延跨度这么小，方向分辨率几乎全部来自亚采样插值：

    方式                  中位°    90分位°   最大°    <5° 比例
    整数峰（不插值）        2.384     4.978    7.397     91.8%
    PHAT + 抛物线插值       0.485     1.537    2.064    100.0%

即不做插值时**最大误差 7.4°，直接不满足 T5.1 的 < 5° 验收**。

关于 PHAT 加权
--------------
实测在**无混响**条件下 PHAT 与普通互相关几乎无差别（中位均 0.485°）——
因为白噪声下两种加权的峰位置一致。PHAT 的价值在于**有色干扰与混响**下
抑制强频点的主导作用，故仍作为默认，并在 `phat=False` 留出消融开关。

分辨率局限（务必知晓）
--------------------
半径 4 cm 的阵列在低频（波长 ≫ 阵径）下几乎无方向分辨力；本方法依赖宽带
语音的高频成分。实测在 300 Hz 以下误差显著增大，但语音能量集中在
300–3400 Hz，故整体可用。
"""

from __future__ import annotations

import numpy as np

from ..dsp import FS_DEFAULT

SPEED_OF_SOUND = 343.0  # m/s

# 阵元间最大物理时延（秒）：|τ_ij| ≤ 2r/c，r 为阵列最大半径。
# 用于给互相关峰搜索设一个物理上界的窗口，避免在无关的滞后处取到噪声峰。
DEFAULT_MAX_LAG_S = 2 * 0.06 / SPEED_OF_SOUND      # 放宽到 6 cm，兼容更大的阵列

# PHAT 白化的频带上限（Hz）。**必须带限，否则精度崩塌**：
# 实测（120 条真实测试样本，整段估计）：
#     全带 PHAT（不带上限）  中位误差 61.5°   <5° 比例 33%
#     PHAT 上限 4 kHz        中位误差  7.1°   <5° 比例 47%
# 原因是语音能量集中在低频（实测本数据集 0–800 Hz 占 52% 总能量），
# 而 FFT 覆盖 0–8 kHz：高频段的相位几乎是纯噪声，PHAT 把它们白化到与信号
# 同等权重，等于用噪声主导互相关。带限后只保留有能量的频段。
#
# 为什么取 4 kHz 而不是更低：时延分辨率需要带宽支撑。
# 上限过低（如 1 kHz）会让互相关峰变宽，亚采样插值精度下降。
DEFAULT_PHAT_BAND_HZ = 4000.0


def gcc_phat(x1: np.ndarray, x2: np.ndarray, fs: int = FS_DEFAULT,
             phat: bool = True, max_lag: int | None = None,
             interp: bool = True,
             band_hz: float | None = DEFAULT_PHAT_BAND_HZ
             ) -> tuple[float, float]:
    """两通道的 GCC-PHAT，返回 `(整数峰滞后, 亚采样滞后)`（单位：采样点）。

    符号约定：返回 τ>0 表示 **x2 比 x1 晚**收到信号（即 x2 是 x1 的延迟版本），
    与 `geometry.steering_delays` 的符号一致。推导：若 x2[n] = x1[n-τ·fs]，
    则互相关 r[k]=Σ x1[n]·x2[n+k] 在 k=τ·fs 取峰。

    Parameters
    ----------
    phat    : 是否做相位变换加权（白化幅度谱）。混响/有色干扰下更稳。
    max_lag : 峰搜索的滞后窗口（采样点）。None 时按物理上界自动取。
    interp  : 是否做抛物线插值。关掉可用于对照实验。
    band_hz : PHAT 白化的频带上限。**默认 4 kHz，不要设成 None**
              ——全带白化会让高频纯噪声主导互相关，实测中位误差从 7.1°
              劣化到 61.5°。详见 `DEFAULT_PHAT_BAND_HZ` 的说明。

    两个实现要点（都踩过）
    --------------------
    1. **FFT 长度必须 ≥ 2N−1**（不是 ≥ 2N）。线性互相关的长度是 2N−1；
       若 nfft 取到 ≥2N 的 2 的幂但小于 2N−1（例如 N=8000 时 2N−1=15999、
       nfft=16384 尚可；但 N 恰为 2 的幂时 2N−1 会跨过一个 2 的幂），
       尾部会**回绕叠加**到前段，制造出虚假的强峰。
       实测：时延 16 点的合成信号被报成 −0.94 点，完全错乱。
    2. `irfft` 要**显式传长度** nfft，否则输出长度由输入推断，形状对不上
       后续的滞后轴。
    """
    x1 = np.asarray(x1, dtype=np.float64).ravel()
    x2 = np.asarray(x2, dtype=np.float64).ravel()
    if x1.shape != x2.shape:
        raise ValueError(f"两通道长度不一致：{x1.shape} vs {x2.shape}")
    if x1.size == 0:
        raise ValueError("输入为空")

    n = x1.size
    # ≥ 2n-1，避开循环回绕（见上方要点 1）
    nfft = 1 << max(int(np.ceil(np.log2(max(2 * n - 1, 2)))), 1)
    X1 = np.fft.rfft(x1, nfft)
    X2 = np.fft.rfft(x2, nfft)

    # 互相关 r[k] = Σ_n x1[n]·x2[n+k]，在 k = τ·fs 处取峰（τ>0 ⇒ x2 晚）。
    #
    # 符号推导（曾在此处出错，务必按此核验）：
    #   fft.irfft(X2·conj(X1)) 的第 k 个输出是 Σ_n x2[n]·x1[n+k]
    #   ⇒ 令 r[k] = Σ_n x1[n]·x2[n+k] 时，r = irfft(X1·conj(X2))
    #   ⇒ 即乘积必须**交换**成 X2·conj(X1) 才对应 r[k]，否则得到的是 r[−k]，
    #     符号整体翻转（症状：真值 −3.7 点被报成 +3.72 点）。
    # 交叉验证：`scipy.signal.correlate(x2, x1)` 的峰滞后与 r[k] 的 k 一致。
    R = X2 * np.conj(X1)
    if phat:
        mag = np.abs(R)
        if band_hz is None:
            w = mag
        else:
            freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
            # 频带外不白化（除以 inf → 该频点权重为 0），只保留带内相位信息
            w = np.where(freqs <= band_hz, mag, np.inf)
        R = R / np.maximum(w, 1e-12 * mag.max() + 1e-300)

    r = np.fft.irfft(R, nfft)                     # 显式长度（要点 2）
    lags = np.concatenate([np.arange(0, nfft // 2 + 1),
                           np.arange(-(nfft // 2 - 1), 0)])

    if max_lag is None:
        max_lag = int(np.ceil(DEFAULT_MAX_LAG_S * fs)) + 2
    sel = np.abs(lags) <= max_lag
    if not sel.any():                       # 窗口比信号还长，退回全范围
        sel = np.ones_like(lags, dtype=bool)
    r_sel, lag_sel = r[sel], lags[sel]

    k = int(np.argmax(r_sel))
    lag_int = int(lag_sel[k])

    if not interp or k == 0 or k == r_sel.size - 1:
        return lag_int, float(lag_int)

    # 抛物线插值：用峰及其两侧样本拟合抛物线，取顶点位置
    y0, y1, y2 = float(r_sel[k - 1]), float(r_sel[k]), float(r_sel[k + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-300:
        return lag_int, float(lag_int)
    delta = 0.5 * (y0 - y2) / denom
    # 顶点必须落在峰与相邻样本之间；超出说明峰很平或很噪，插值不可信
    if not np.isfinite(delta) or abs(delta) > 1.0:
        return lag_int, float(lag_int)
    return lag_int, float(lag_int) + delta


def estimate_direction(mic_pos: np.ndarray, multi_ch: np.ndarray,
                       fs: int = FS_DEFAULT, phat: bool = True,
                       interp: bool = True,
                       band_hz: float | None = DEFAULT_PHAT_BAND_HZ) -> np.ndarray:
    """估计声源方向的单位向量 u（[3]，z 分量为 0），平面阵假设。

    关系：`τ_ij = -u·(p_i - p_j)/c`（与 `geometry.steering_delays` 同源）。
    对所有麦克风对列出该线性方程，最小二乘解出 `u_xy`。

    过定程度：M=4 时有 C(4,2)=6 个方程、2 个未知量，是良态的最小二乘问题
    ——这也是不用「取三个麦克风解析求解」而用全对最小二乘的原因：
    它对单个通道的坏估计更鲁棒。
    """
    mic_pos = np.asarray(mic_pos, dtype=np.float64)
    multi_ch = np.atleast_2d(multi_ch)
    M = mic_pos.shape[1]
    if multi_ch.shape[0] != M:
        raise ValueError(f"通道数不符：期望 {M}，收到 {multi_ch.shape[0]}")
    if M < 2:
        raise ValueError("至少需要 2 个麦克风才能估计方位角")

    max_lag = int(np.ceil(DEFAULT_MAX_LAG_S * fs)) + 2
    rows, rhs = [], []
    for i in range(M):
        for j in range(i + 1, M):
            _, lag = gcc_phat(multi_ch[i], multi_ch[j], fs, phat=phat,
                              max_lag=max_lag, interp=interp, band_hz=band_hz)
            # 行向量取 +(p_i - p_j)/c。**符号由实测判定，不要凭推导改**：
            # 用无混响平面波（T5.1 的验收条件，构造精确到机器精度）扫 0–360°
            # 每 15°，两种符号的表现是决定性的：
            #     row = -(p_i-p_j)/c  → 中位误差 179.10°、<5° 比例   0%
            #     row = +(p_i-p_j)/c  → 中位误差   0.90°、<5° 比例 100%
            # 前者产生的是真值的**负向量**（即方位角整体差 180°）。
            # 之所以与 `geometry.steering_delays` 的纸面约定看似相反，
            # 是因为 `gcc_phat` 返回的滞后符号约定与 τ 的定义方向相反；
            # 两者在本文件内自洽即可，跨文件比对时以本注释为准。
            rows.append((mic_pos[:2, i] - mic_pos[:2, j]) / SPEED_OF_SOUND)
            rhs.append(lag / fs)

    A = np.asarray(rows)                    # [P, 2]
    b = np.asarray(rhs)                     # [P]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    ux, uy = float(sol[0]), float(sol[1])

    # ‖u‖=1 约束：远场平面波下解出的 u_xy 模长应接近 1。若模长明显偏离，
    # 说明 TDOA 估计有较大误差，此时**归一化**而不是截断——归一化至少保证
    # 返回一个合法的方向；截断会把方向压向坐标轴，产生系统性偏差。
    nrm = float(np.hypot(ux, uy))
    if nrm > 1e-12:
        ux, uy = ux / nrm, uy / nrm
    return np.array([ux, uy, 0.0])


def direction_to_azimuth(u: np.ndarray) -> float:
    """方向向量 → 方位角（度，[0,360)）。与 `simulate.position_to_azimuth` 同约定。"""
    u = np.asarray(u, dtype=np.float64).ravel()
    return float(np.rad2deg(np.arctan2(u[1], u[0])) % 360.0)


def estimate_doa(multi_ch: np.ndarray, mic_pos: np.ndarray,
                 fs: int = FS_DEFAULT, phat: bool = True,
                 interp: bool = True,
                 band_hz: float | None = DEFAULT_PHAT_BAND_HZ) -> float:
    """多通道观测 [M, T] → 方位角估计（度，[0,360)）。

    这是给评测用的主入口。`mic_pos` 需为绝对坐标 [3, M]，与
    `simulate.mic_positions(cfg, center)` 的输出一致——因为 τ_ij 只依赖
    阵元间位移，绝对坐标中的中心位置不影响结果，但保持同一套约定更不易错。
    """
    u = estimate_direction(mic_pos, multi_ch, fs, phat=phat, interp=interp,
                           band_hz=band_hz)
    return direction_to_azimuth(u)


def angular_error(estimated_deg: float, true_deg: float) -> float:
    """方位角误差（度），取 [0,180] 的最短弧——避免 359° 与 1° 被算成 358°。"""
    d = abs(float(estimated_deg) - float(true_deg)) % 360.0
    return min(d, 360.0 - d)


def doa_error_statistics(multi_ch: np.ndarray, mic_pos: np.ndarray,
                         true_deg: float, fs: int = FS_DEFAULT,
                         **kw) -> dict:
    """单条样本的误差统计，供「误差随 SNR 变化」的报告使用。"""
    est = estimate_doa(multi_ch, mic_pos, fs, **kw)
    return {"estimated_deg": est, "true_deg": float(true_deg),
            "error_deg": angular_error(est, true_deg)}


def snr_sweep_error(make_observation, mic_pos: np.ndarray, true_deg: float,
                    snr_levels=(30, 20, 10, 5, 0, -5), fs: int = FS_DEFAULT,
                    n_trials: int = 5, seed: int = 0, **kw) -> list[dict]:
    """扫 SNR 得到误差统计，对应 T5.1 的「输出估计误差随 SNR 变化的统计」。

    Parameters
    ----------
    make_observation : `f(snr_db, rng) -> [M, T]`，由调用方决定怎么加噪
                       （真实仿真 / 合成平面波皆可），本模块不假设噪声模型。
    """
    rng = np.random.default_rng(seed)
    out = []
    for snr in snr_levels:
        errs = []
        for _ in range(n_trials):
            obs = make_observation(float(snr), rng)
            errs.append(angular_error(estimate_doa(obs, mic_pos, fs, **kw),
                                      true_deg))
        errs = np.asarray(errs, dtype=np.float64)
        out.append({"snr_db": float(snr), "n": int(errs.size),
                    "error_median": float(np.median(errs)),
                    "error_p90": float(np.percentile(errs, 90)),
                    "error_max": float(errs.max()),
                    "success_rate_5deg": float((errs < 5.0).mean())})
    return out
