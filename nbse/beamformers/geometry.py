"""阵列几何与导向矢量。

所有波束形成器共用此处定义的时延与导向矢量，保证各方法在
同一套几何约定下比较——否则对比结果没有意义。

方位角约定
----------
以 x 轴正方向为 0°，逆时针为正（与 simulate.mic_positions 的圆阵参数一致）。
声源方向单位向量 u = [cos θ, sin θ, 0]。
"""

from __future__ import annotations

import numpy as np

SPEED_OF_SOUND = 343.0  # m/s


def direction_vector(doa_deg: float) -> np.ndarray:
    """声源方向的单位向量 [3]（假设与阵列同高，故 z 分量为 0）。"""
    theta = np.deg2rad(doa_deg)
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def steering_delays(mic_pos: np.ndarray, doa_deg: float,
                    center_xy: tuple = (0.0, 0.0)) -> np.ndarray:
    """各阵元相对阵列中心的传播时延 τ_m（秒），形状 [n_mics]。

    若声源位于方向 u，阵元 m 到声源的距离比中心近 u·p_m，
    因此波前**提前**到达，即 τ_m = -(u·p_m)/c。

    符号约定：τ_m > 0 表示该阵元比中心晚收到信号。
    """
    mic_pos = np.asarray(mic_pos, dtype=np.float64)
    center = np.array([center_xy[0], center_xy[1], 0.0])

    rel = mic_pos - center[:, None]          # [3, n_mics]
    u = direction_vector(doa_deg)            # [3]

    # 阵元相对中心、沿声源方向投影的距离（米）
    proj = u @ rel                           # [n_mics]

    return -proj / SPEED_OF_SOUND


def steering_vector(mic_pos: np.ndarray, doa_deg: float, freqs: np.ndarray,
                    center_xy: tuple = (0.0, 0.0)) -> np.ndarray:
    """频域导向矢量 d(f) = exp(-j·2π·f·τ_m)，形状 [n_freqs, n_mics]。

    这是「声源在 doa 方向时，各阵元相对中心的相位」。
    若阵元上观测为 X_m(f) = S(f)·d_m(f)，用共轭补偿即可对齐。
    """
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    tau = steering_delays(mic_pos, doa_deg, center_xy)     # [n_mics]

    return np.exp(-2j * np.pi * freqs[:, None] * tau[None, :])


def propagation_delays(mic_pos: np.ndarray, source_pos: np.ndarray) -> np.ndarray:
    """由声源实际坐标算各阵元时延（秒），形状 [n_mics]。

    与 steering_delays 不同，这里不假设远场平面波，而是用真实球面波
    距离——用于生成仿真数据或在单元测试中构造精确对齐的信号。
    """
    mic_pos = np.asarray(mic_pos, dtype=np.float64)
    source_pos = np.asarray(source_pos, dtype=np.float64).ravel()

    dist = np.linalg.norm(mic_pos - source_pos[:, None], axis=0)

    # 减去到阵列中心的距离，得到相对时延
    center_dist = np.linalg.norm(source_pos - np.array([
        np.mean(mic_pos[0]), np.mean(mic_pos[1]), np.mean(mic_pos[2])]))
    return (dist - center_dist) / SPEED_OF_SOUND
