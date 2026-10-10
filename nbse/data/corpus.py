"""T1.4 语料读取：LibriSpeech / DEMAND 的枚举、说话人切分与音频加载。

设计文档 §4.1（语料与噪声）、§4.2（说话人切分）。

本模块只负责「把磁盘上的语料变成 numpy 数组」——房间仿真在 `simulate.py`，
两者在 `generate.py` 里汇合成多通道样本。

解压后的目录结构：

    data/LibriSpeech/<subset>/<speaker>/<chapter>/<speaker>-<chapter>-<utt>.flac
    data/DEMAND/<ENV>/ch01.wav ... ch16.wav          （单通道 16 kHz）

切分约定（§4.2，三者按说话人互不重叠）：

    train : dev-clean 的前 30 人
    val   : dev-clean 的其余 10 人
    test  : test-clean 全部 40 人
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# LibriSpeech 是 flac；兼容 wav 便于自造小语料做测试
AUDIO_SUFFIXES = (".flac", ".wav")


@dataclass(frozen=True)
class Utterance:
    """一条干净语音。"""
    path: Path
    speaker: str
    subset: str
    chapter: str


def _speaker_key(name: str) -> tuple:
    """说话人 id 排序键。

    LibriSpeech 的 id 是数字字符串但位数不一（'1272' 与 '174' 并存），
    直接按字符串排会得到 '1272' < '174' 这种反直觉顺序，故按数值排。
    """
    return (0, int(name), "") if name.isdigit() else (1, 0, name)


def list_utterances(root: str | Path, subset: str) -> list[Utterance]:
    """枚举某个子集下的全部语音，按（说话人, 章节, 文件名）稳定排序。"""
    base = Path(root) / subset
    if not base.is_dir():
        raise FileNotFoundError(
            f"找不到子集目录：{base}\n"
            f"请先运行 scripts/download_data.py 下载并解压 LibriSpeech。"
        )

    out: list[Utterance] = []
    for p in base.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        rel = p.relative_to(base).parts
        if len(rel) != 3:      # 期望 <speaker>/<chapter>/<file>，其余（如说明文档）跳过
            continue
        out.append(Utterance(path=p, speaker=rel[0], subset=subset, chapter=rel[1]))

    if not out:
        raise FileNotFoundError(f"{base} 下没有找到任何音频文件（结构是否为 <说话人>/<章节>/<文件>？）")

    out.sort(key=lambda u: (_speaker_key(u.speaker), u.chapter, u.path.name))
    return out


def list_speakers(root: str | Path, subset: str) -> list[str]:
    """子集内的说话人 id 列表（有序、去重）。"""
    seen = {u.speaker for u in list_utterances(root, subset)}
    return sorted(seen, key=_speaker_key)


def split_speakers(root: str | Path, n_train: int = 30, n_val: int = 10,
                   dev_subset: str = "dev-clean",
                   test_subset: str = "test-clean") -> dict[str, list[str]]:
    """按 §4.2 切分说话人。返回 {'train': [...], 'val': [...], 'test': [...]}。"""
    dev = list_speakers(root, dev_subset)
    if len(dev) < n_train + n_val:
        raise ValueError(
            f"{dev_subset} 只有 {len(dev)} 个说话人，不够切出 "
            f"{n_train} 训练 + {n_val} 验证。"
        )

    split = {
        "train": dev[:n_train],
        "val": dev[n_train:n_train + n_val],
        "test": list_speakers(root, test_subset),
    }

    # 三者必须互不重叠，否则测试集泄漏进训练（§4.2 的硬要求）
    if set(split["train"]) & set(split["val"]):
        raise AssertionError("训练与验证说话人重叠")
    if (set(split["train"]) | set(split["val"])) & set(split["test"]):
        raise AssertionError("dev-clean 与 test-clean 说话人重叠")
    return split


def utterances_for_split(root: str | Path, split: str, n_train: int = 30,
                         n_val: int = 10, dev_subset: str = "dev-clean",
                         test_subset: str = "test-clean") -> list[Utterance]:
    """取某个切分的全部语音（按说话人过滤）。"""
    speakers = split_speakers(root, n_train, n_val, dev_subset, test_subset)
    if split not in speakers:
        raise ValueError(f"未知切分：{split}（可选 {sorted(speakers)}）")

    if split == "test":
        pool = list_utterances(root, test_subset)
    else:
        pool = list_utterances(root, dev_subset)

    wanted = set(speakers[split])
    return [u for u in pool if u.speaker in wanted]


def list_noise_files(demand_root: str | Path) -> list[Path]:
    """DEMAND 各环境下的噪声 wav（16 kHz 单通道），按路径稳定排序。"""
    base = Path(demand_root)
    if not base.is_dir():
        raise FileNotFoundError(
            f"找不到 DEMAND 目录：{base}\n"
            f"请先运行 scripts/download_data.py 下载并解压 DEMAND。"
        )

    files = sorted(p for p in base.rglob("*.wav") if p.is_file())
    if not files:
        raise FileNotFoundError(f"{base} 下没有找到任何 wav 噪声文件")
    return files


def load_audio(path: str | Path, dtype=np.float64) -> tuple[np.ndarray, int]:
    """读音频为单通道 numpy 数组，返回 (波形, 采样率)。

    多声道输入取均值降为单通道——DEMAND 与 LibriSpeech 本都是单通道，
    这里只是让接口对输入更宽容。
    """
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    return data[:, 0].astype(dtype), int(sr)
