"""T1.3 房间仿真：由干净语音与噪声合成多通道麦克风阵列观测。

设计文档 §4.3 / §4.4。

核心约定
--------
* 采样率 16 kHz（PESQ 宽带模式要求）。
* 噪声**同样经房间冲激响应卷积**，而非直接相加——这样才能模拟
  空间相干噪声场，这是与单通道增强的本质区别。
* 信噪比定义在**参考麦克风（0 号）**上，即以该通道的干净能量与噪声能量之比为准。
* 干净参考信号取**无混响（dry）的原始语音**。这是语音增强文献的通行做法：
  评估的是「去噪 + 去混响」的综合效果。

输出
----
每个样本包含：
    multi_ch     [n_mics, T]  多通道混合观测
    clean_ref    [T]          单通道干净参考（dry）
    noise_multi  [n_mics, T]  纯噪声多通道信号（MVDR 估计噪声协方差需要）
    doa_true     float        声源方位角（度，相对阵列中心），供 Oracle DOA 使用
    snr_db       float        该样本的实际信噪比
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EPS = 1e-12
SPEED_OF_SOUND = 343.0  # m/s


# ── 配置 ────────────────────────────────────────────────────────

@dataclass
class ArrayConfig:
    """麦克风阵列几何。"""
    type: str = "circular"      # circular | linear
    n_mics: int = 4
    radius: float = 0.04        # 米，圆阵半径
    spacing: float = 0.04       # 米，线阵阵元间距（type=linear 时使用）
    center_xy: tuple = (0.0, 0.0)
    height: float = 1.0         # 米，离地高度


@dataclass
class RoomConfig:
    """单次仿真的房间与声源配置（全部为已采样确定值）。"""
    size: np.ndarray            # [3] 房间长宽高（米）
    source_pos: np.ndarray      # [3] 声源位置
    noise_pos: np.ndarray       # [3] 噪声源位置
    t60: float                  # 秒

    def key(self) -> str:
        """稳定哈希，用于 RIR 缓存文件名。"""
        payload = np.concatenate([self.size, self.source_pos,
                                  self.noise_pos, [self.t60]])
        h = hashlib.sha1(np.asarray(payload, dtype=np.float64).tobytes())
        return h.hexdigest()[:16]


@dataclass
class SimConfig:
    """仿真全局参数，字段与 configs/default.yaml 对应。"""
    sample_rate: int = 16000
    duration: float = 4.0

    array: ArrayConfig = field(default_factory=ArrayConfig)

    room_size_range: tuple = (3.0, 8.0)
    t60_range: tuple = (0.2, 0.8)
    source_distance_range: tuple = (1.0, 3.0)
    source_azimuth_range: tuple = (0.0, 360.0)

    spatial_coherent_noise: bool = True
    snr_db: float | None = None     # None 表示从 train_snr_range 随机采样
    train_snr_range: tuple = (-5.0, 10.0)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        arr = ArrayConfig(
            type=raw["array"]["type"],
            n_mics=raw["array"]["n_mics"],
            radius=raw["array"]["radius"],
        )
        # 用房间尺寸上界作为高度，保证声源/阵列在房间内
        return cls(
            sample_rate=raw["audio"]["sample_rate"],
            duration=raw["audio"]["duration"],
            array=arr,
            room_size_range=tuple(raw["room"]["size_range"]),
            t60_range=tuple(raw["room"]["t60_range"]),
            source_distance_range=tuple(raw["room"]["source_distance_range"]),
            source_azimuth_range=tuple(raw["room"]["source_azimuth_range"]),
            spatial_coherent_noise=bool(raw["noise"]["spatial_coherent"]),
            train_snr_range=tuple(raw["noise"]["train_snr_range"]),
        )


# ── 阵列几何 ────────────────────────────────────────────────────

def mic_positions(cfg: ArrayConfig, center_xy: tuple | None = None) -> np.ndarray:
    """返回阵列坐标 [3, n_mics]（米）。

    center_xy 为 None 时回退到 cfg.center_xy。

    自行构造而非依赖 pyroomacoustics 的数组辅助函数——几何是纯数学，
    自己实现可以让单元测试无需安装仿真库即可运行。
    """
    cx, cy = cfg.center_xy if center_xy is None else center_xy
    if cfg.type == "circular":
        phi = np.linspace(0.0, 2.0 * np.pi, cfg.n_mics, endpoint=False)
        xy = np.vstack([cx + cfg.radius * np.cos(phi),
                        cy + cfg.radius * np.sin(phi)])
    elif cfg.type == "linear":
        offsets = (np.arange(cfg.n_mics) - (cfg.n_mics - 1) / 2.0) * cfg.spacing
        xy = np.vstack([cx + offsets, np.full(cfg.n_mics, cy)])
    else:
        raise ValueError(f"未知阵列类型：{cfg.type}")

    return np.vstack([xy, np.full(cfg.n_mics, cfg.height)])


def azimuth_to_position(azimuth_deg: float, distance: float,
                        center_xy: tuple, height: float) -> np.ndarray:
    """由方位角与距离求声源坐标 [3]。

    方位角以 x 轴正方向为 0°，逆时针为正，与 mic_positions 的圆阵参数一致。
    """
    theta = np.deg2rad(azimuth_deg)
    cx, cy = center_xy
    return np.array([cx + distance * np.cos(theta),
                     cy + distance * np.sin(theta),
                     height])


def position_to_azimuth(pos: np.ndarray, center_xy: tuple) -> float:
    """由坐标反推方位角（度），取值范围 [0, 360)。"""
    dx = pos[0] - center_xy[0]
    dy = pos[1] - center_xy[1]
    return float(np.rad2deg(np.arctan2(dy, dx)) % 360.0)


# ── 房间采样 ────────────────────────────────────────────────────

def array_center(room_cfg: RoomConfig) -> tuple:
    """阵列所在位置（房间中心）。

    与 sample_room 的摆放约定一致。由房间配置推导而非存进 cfg，
    这样重跑既有房间配置时不会因残留状态而算错方位角。
    """
    return (float(room_cfg.size[0] / 2), float(room_cfg.size[1] / 2))


def sample_room(cfg: SimConfig, rng: np.random.Generator) -> RoomConfig:
    """随机采样一个房间配置。

    房间边长采样后，将阵列置于房间中心、声源按方位角与距离摆放，
    并强制把声源裁剪进房间内部，避免 pyroomacoustics 报错。
    """
    size = rng.uniform(*cfg.room_size_range, size=3)

    azimuth = rng.uniform(*cfg.source_azimuth_range)
    distance = rng.uniform(*cfg.source_distance_range)
    t60 = rng.uniform(*cfg.t60_range)

    # 阵列放在房间中心（不修改 cfg——阵列位置由房间配置唯一决定）
    center = (float(size[0] / 2), float(size[1] / 2))
    pos = azimuth_to_position(azimuth, distance, center, cfg.array.height)

    # 裁剪进房间（留 0.3 m 余量）
    margin = 0.3
    pos = np.clip(pos, margin, np.maximum(size - margin, margin + EPS))

    # 噪声源放在另一方位（相差 60°–180°），避免与语音源重合
    noise_az = (azimuth + rng.uniform(60.0, 180.0)) % 360.0
    n_pos = azimuth_to_position(noise_az, distance, center, cfg.array.height)
    n_pos = np.clip(n_pos, margin, np.maximum(size - margin, margin + EPS))

    return RoomConfig(size=size, source_pos=pos, noise_pos=n_pos, t60=t60)


# ── RIR 生成与缓存 ──────────────────────────────────────────────

def _room_and_rirs(room_cfg: RoomConfig, cfg: SimConfig,
                   n_sources: int) -> np.ndarray:
    """构建房间并返回 RIR，形状 [n_sources, n_mics, n_taps]。

    约定索引：0 号为语音源，1 号为噪声源。
    """
    import pyroomacoustics as pra

    # 由 T60 反推墙面吸收系数与镜像源阶数
    e_absorption, max_order = pra.inverse_sabine(room_cfg.t60, room_cfg.size)
    room = pra.ShoeBox(
        room_cfg.size,
        fs=cfg.sample_rate,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )

    sources = [room_cfg.source_pos, room_cfg.noise_pos][:n_sources]
    for i, pos in enumerate(sources):
        room.add_source(pos, signal=np.zeros(int(cfg.sample_rate * 1.0)))

    room.add_microphone_array(
        pra.MicrophoneArray(mic_positions(cfg.array, array_center(room_cfg)),
                            room.fs))
    room.compute_rir()

    # room.rir[mic][source] -> 转成 [source, mic, taps]
    n_mics = cfg.array.n_mics
    rirs = [[np.asarray(room.rir[m][s]) for m in range(n_mics)]
            for s in range(n_sources)]

    # 统一长度后堆叠
    max_len = max(len(r) for src in rirs for r in src)
    out = np.zeros((n_sources, n_mics, max_len))
    for s in range(n_sources):
        for m in range(n_mics):
            out[s, m, : len(rirs[s][m])] = rirs[s][m]
    return out


def array_signature(cfg: SimConfig) -> str:
    """阵列几何 + 采样率的短哈希，用于 RIR 缓存文件名。

    只按阵元数命名是不够的：半径、阵型、高度、采样率任一改变都会得到
    完全不同的 RIR，却会命中同一个缓存文件，静默复用错误的房间响应。
    """
    a = cfg.array
    payload = (f"{a.type}|{a.n_mics}|{a.radius}|{a.spacing}|{a.height}"
               f"|{cfg.sample_rate}")
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def get_rirs(room_cfg: RoomConfig, cfg: SimConfig, cache_dir: Path | None,
             n_sources: int = 2) -> np.ndarray:
    """取 RIR，优先读缓存。

    仿真是 CPU 瓶颈，训练时反复重算不可接受——设计文档 §4.5 要求缓存到磁盘。
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / (f"{room_cfg.key()}_m{cfg.array.n_mics}"
                            f"_a{array_signature(cfg)}.npz")
        if path.exists():
            with np.load(path) as z:
                return z["rir"]

    rirs = _room_and_rirs(room_cfg, cfg, n_sources)

    if cache_dir is not None:
        np.savez_compressed(path, rir=rirs)
    return rirs


# ── 卷积与信噪比 ────────────────────────────────────────────────

def convolve_rir(signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """用 FFT 做快速卷积，输出长度与输入一致（截断尾音）。"""
    n = len(signal) + len(rir) - 1
    nfft = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(signal, nfft) * np.fft.rfft(rir, nfft), nfft)
    return y[: len(signal)]


def measure_snr(clean_ref: np.ndarray, noise_ref: np.ndarray) -> float:
    """参考通道上的信噪比（dB）。"""
    c = float(np.linalg.norm(clean_ref))
    n = float(np.linalg.norm(noise_ref))
    return 20.0 * np.log10((c + EPS) / (n + EPS))


def scale_to_snr(clean_ref: np.ndarray, noise_ref: np.ndarray,
                 target_snr_db: float) -> float:
    """求噪声的缩放因子，使参考通道信噪比等于目标值。"""
    c = float(np.linalg.norm(clean_ref))
    n = float(np.linalg.norm(noise_ref))
    if n < EPS:
        return 1.0
    return c / (n * 10.0 ** (target_snr_db / 20.0))


# ── 主入口 ──────────────────────────────────────────────────────

def simulate_sample(clean: np.ndarray, noise: np.ndarray, cfg: SimConfig,
                    rng: np.random.Generator,
                    room_cfg: RoomConfig | None = None,
                    cache_dir: Path | None = None) -> dict:
    """合成一个多通道样本。

    Parameters
    ----------
    clean : [T] 干净语音（将被裁剪/补齐到 cfg.duration）
    noise : [T] 噪声，长度至少与 clean 相同
    room_cfg : 指定房间配置；None 则随机采样（评估时用固定配置复现）
    cache_dir : RIR 缓存目录；None 表示不缓存
    """
    n = int(cfg.sample_rate * cfg.duration)
    clean = _fit_length(np.asarray(clean, dtype=np.float64), n)
    noise = _fit_length(np.asarray(noise, dtype=np.float64), n)

    # 归一化，避免不同语料电平差异影响 SNR 控制
    clean = clean / (np.max(np.abs(clean)) + EPS)

    if room_cfg is None:
        room_cfg = sample_room(cfg, rng)

    rirs = get_rirs(room_cfg, cfg, cache_dir, n_sources=2)

    # 语音与噪声分别经各自的 RIR 卷积 → 空间相干
    clean_multi = np.stack([convolve_rir(clean, rirs[0, m])
                            for m in range(cfg.array.n_mics)])
    if cfg.spatial_coherent_noise:
        noise_multi = np.stack([convolve_rir(noise, rirs[1, m])
                                for m in range(cfg.array.n_mics)])
    else:
        noise_multi = np.tile(noise, (cfg.array.n_mics, 1))

    # 在参考通道上对齐到目标信噪比
    target = cfg.snr_db if cfg.snr_db is not None \
        else float(rng.uniform(*cfg.train_snr_range))
    scale = scale_to_snr(clean_multi[0], noise_multi[0], target)

    noise_multi = noise_multi * scale
    multi_ch = clean_multi + noise_multi

    return {
        "multi_ch": multi_ch,
        "clean_ref": clean,
        "noise_multi": noise_multi,
        "doa_true": position_to_azimuth(room_cfg.source_pos,
                                        array_center(room_cfg)),
        "snr_db": measure_snr(clean_multi[0], noise_multi[0]),
        "room": room_cfg,
    }


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """裁剪或循环补齐到长度 n。"""
    if len(x) >= n:
        return x[:n]
    reps = int(np.ceil(n / max(len(x), 1)))
    return np.tile(x, reps)[:n]
