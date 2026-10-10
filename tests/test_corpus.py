"""T1.4 验收测试之一：语料读取。

用临时目录自造一棵假的 LibriSpeech / DEMAND 树，因此**不需要下载真实数据**。
真实语料上的端到端验证由 scripts/generate_dataset.py 完成。
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nbse.data.corpus import (
    list_noise_files,
    list_speakers,
    list_utterances,
    load_audio,
    split_speakers,
    utterances_for_split,
)

SR = 16000


def _write(path: Path, seconds: float = 0.05, channels: int = 1, seed: int = 0) -> Path:
    """写一个短音频（flac 或 wav 由后缀决定）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = max(int(SR * seconds), 1)
    x = (0.1 * rng.normal(size=(n, channels))).astype(np.float32)
    sf.write(str(path), x, SR)
    return path


@pytest.fixture
def fake_librispeech(tmp_path):
    """假 LibriSpeech：dev-clean 3 人 / test-clean 2 人。

    说话人 id 故意位数不一（26 / 174 / 1272），用来验证排序按数值而非字符串。
    """
    root = tmp_path / "LibriSpeech"
    for spk, chapters in {"174": ("1", "2"), "1272": ("1",), "26": ("1",)}.items():
        for ch in chapters:
            for utt in ("0001", "0002"):
                _write(root / "dev-clean" / spk / ch / f"{spk}-{ch}-{utt}.flac")
    for spk in ("300", "400"):
        _write(root / "test-clean" / spk / "1" / f"{spk}-1-0001.flac")
    return root


@pytest.fixture
def fake_demand(tmp_path):
    root = tmp_path / "DEMAND"
    for env, n in {"DKITCHEN": 2, "DLIVING": 1, "OOFFICE": 1}.items():
        for i in range(1, n + 1):
            _write(root / env / f"ch{i:02d}.wav", seconds=0.2)
    return root


# ── 枚举 ────────────────────────────────────────────────────────

def test_list_utterances_parses_structure(fake_librispeech):
    utts = list_utterances(fake_librispeech, "dev-clean")

    assert len(utts) == 8          # 174 有 2 章 × 2 条，1272/26 各 1 章 × 2 条
    assert all(u.path.exists() for u in utts)
    assert all(u.subset == "dev-clean" for u in utts)
    assert {u.speaker for u in utts} == {"26", "174", "1272"}
    assert {u.chapter for u in utts} == {"1", "2"}


def test_list_utterances_is_stably_sorted(fake_librispeech):
    """两次枚举顺序必须一致——随机取样的可复现性依赖它。

    注意顺序是按**说话人数值**排的，不是按路径字符串：字典序会给出
    1272 < 174 < 26，与按数值的 26 < 174 < 1272 不同。
    """
    utts = list_utterances(fake_librispeech, "dev-clean")

    assert [u.path for u in utts] == [u.path for u in
                                      list_utterances(fake_librispeech, "dev-clean")]
    assert list(dict.fromkeys(u.speaker for u in utts)) == ["26", "174", "1272"]


def test_list_utterances_missing_subset_raises(fake_librispeech):
    with pytest.raises(FileNotFoundError, match="download_data"):
        list_utterances(fake_librispeech, "train-clean-100")


def test_speakers_sorted_numerically_not_lexically(fake_librispeech):
    assert list_speakers(fake_librispeech, "dev-clean") == ["26", "174", "1272"]


# ── 说话人切分（§4.2）──────────────────────────────────────────

def test_split_speakers_is_disjoint_and_deterministic(fake_librispeech):
    split = split_speakers(fake_librispeech, n_train=2, n_val=1)

    assert split["train"] == ["26", "174"]
    assert split["val"] == ["1272"]
    assert split["test"] == ["300", "400"]

    assert not set(split["train"]) & set(split["val"])
    assert not (set(split["train"]) | set(split["val"])) & set(split["test"])
    assert split_speakers(fake_librispeech, n_train=2, n_val=1) == split


def test_split_speakers_raises_when_too_few(fake_librispeech):
    with pytest.raises(ValueError, match="说话人"):
        split_speakers(fake_librispeech, n_train=5, n_val=5)


def test_utterances_for_split_filters_by_speaker(fake_librispeech):
    val = utterances_for_split(fake_librispeech, "val", n_train=2, n_val=1)
    test = utterances_for_split(fake_librispeech, "test", n_train=2, n_val=1)

    assert {u.speaker for u in val} == {"1272"}
    assert {u.speaker for u in test} == {"300", "400"}
    assert all(u.subset == "test-clean" for u in test)


def test_utterances_for_unknown_split_raises(fake_librispeech):
    with pytest.raises(ValueError, match="未知切分"):
        utterances_for_split(fake_librispeech, "dev", n_train=2, n_val=1)


# ── 噪声 ────────────────────────────────────────────────────────

def test_list_noise_files_spans_environments(fake_demand):
    files = list_noise_files(fake_demand)

    assert len(files) == 4
    assert {p.parent.name for p in files} == {"DKITCHEN", "DLIVING", "OOFFICE"}
    assert files == sorted(files)


def test_list_noise_files_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        list_noise_files(tmp_path / "nope")


# ── 音频加载 ────────────────────────────────────────────────────

def test_load_audio_returns_mono_and_samplerate(tmp_path):
    x, sr = load_audio(_write(tmp_path / "a.flac", seconds=0.1))

    assert sr == SR
    assert x.ndim == 1
    assert len(x) == int(SR * 0.1)
    assert x.dtype == np.float64        # 默认按 float64 交给仿真，内部再归一化


def test_load_audio_downmixes_stereo(tmp_path):
    x, sr = load_audio(_write(tmp_path / "st.wav", channels=2))

    assert sr == SR
    assert x.ndim == 1
