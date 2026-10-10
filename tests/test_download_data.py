"""T1.2 回归测试：数据脚本解压后的目录布局。

这些 bug 都是「脚本先于真实数据写成、从未真正跑过」造成的：
  1. LibriSpeech 的 tar 内层自带 `LibriSpeech/` 前缀，若解进 `data/LibriSpeech/`
     会多出一层嵌套（`data/LibriSpeech/LibriSpeech/dev-clean`），
     结果是 `report_stats()` 把已下载的数据一律报成「缺失」。
  2. `fetch_demand()` 的解压目标 `target_dir` 从未定义（NameError），
     且它查找的压缩包名漏了 `_16k` 后缀，与 docs/DATASETS.md 要求手动放置的文件名不一致——
     手动下载好的压缩包会被当成不存在而重新下载。

测试用假压缩包走真实解压路径，不联网、不碰真实 data/。
"""

import importlib.util
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dl(tmp_path, monkeypatch):
    """加载 download_data 模块，并把它的 DATA_ROOT 重定向到临时目录。"""
    spec = importlib.util.spec_from_file_location(
        "download_data", ROOT / "scripts" / "download_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "DATA_ROOT", tmp_path)
    return mod


def _make_tar(path: Path, member: str) -> None:
    """造一个 tar.gz，内部只有 member 这一个文件（路径即 member）。"""
    payload = path.parent / "_payload.txt"
    payload.write_bytes(b"x" * 16)
    with tarfile.open(path, "w:gz") as tf:
        tf.add(payload, arcname=member)
    payload.unlink()


def _make_zip(path: Path, member: str) -> None:
    payload = path.parent / "_payload.txt"
    payload.write_bytes(b"x" * 16)
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(payload, arcname=member)
    payload.unlink()


def test_librispeech_extracts_without_extra_nesting(dl, tmp_path):
    """压缩包内层已有 LibriSpeech/ 前缀，解压后不应再多出一层。"""
    # LibriSpeech 官方 tar 的结构就是 LibriSpeech/dev-clean/<说话人>/<章节>/<文件>.flac
    _make_tar(tmp_path / "dev-clean.tar.gz", "LibriSpeech/dev-clean/1234/5678/1.flac")

    dl.fetch_librispeech(["dev-clean"])

    assert (tmp_path / "LibriSpeech" / "dev-clean" / "1234" / "5678" / "1.flac").exists()
    assert not (tmp_path / "LibriSpeech" / "LibriSpeech").exists(), "多了一层嵌套"


def test_librispeech_stats_do_not_report_missing(dl, tmp_path, capsys):
    """解压后的数据在统计里必须被认出来，而不是报「缺失」。"""
    _make_tar(tmp_path / "dev-clean.tar.gz", "LibriSpeech/dev-clean/1234/5678/1.flac")
    dl.fetch_librispeech(["dev-clean"])

    dl.report_stats()
    dev_lines = [ln for ln in capsys.readouterr().out.splitlines() if "dev-clean" in ln]

    assert dev_lines, "统计输出里没有 dev-clean 一行"
    assert "缺失" not in dev_lines[0]


def test_demand_zip_with_16k_suffix_is_reused(dl, tmp_path):
    """手动放入的 DKITCHEN_16k.zip 应被直接使用（不重下），并解到 data/DEMAND/ 之下。

    `download()` 见到目标文件已存在就跳过下载，所以本测试不联网。
    旧代码在这里会先 NameError，其次会因文件名不匹配而重新下载。
    """
    _make_zip(tmp_path / "DKITCHEN_16k.zip", "DKITCHEN/ch01.wav")

    dl.fetch_demand(["DKITCHEN"])

    assert (tmp_path / "DEMAND" / "DKITCHEN" / "ch01.wav").exists()


def test_demand_skips_env_already_extracted(dl, tmp_path):
    """已解压的环境应被识别为已存在，不去碰网络。"""
    target = tmp_path / "DEMAND" / "DKITCHEN"
    target.mkdir(parents=True)
    (target / "ch01.wav").write_bytes(b"x" * 16)

    dl.fetch_demand(["DKITCHEN"])  # 不应抛异常，也不应下载

    assert (target / "ch01.wav").exists()
