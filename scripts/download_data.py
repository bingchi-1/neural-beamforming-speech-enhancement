"""T1.2 数据下载与校验。

下载 LibriSpeech（干净语音）与 DEMAND（噪声），解压到仓库根目录的 data/。
脚本幂等：已存在的文件会跳过，中断后重跑会从断点续传。

用法：
    python scripts/download_data.py                 # 下载全部
    python scripts/download_data.py --librispeech   # 只下 LibriSpeech
    python scripts/download_data.py --demand DKITCHEN DLIVING
    python scripts/download_data.py --stats         # 只统计已有数据
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

LIBRISPEECH_URLS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
}

# DEMAND 各环境以独立 zip 发布（Zenodo record 1227121）。
# 文件名必须带采样率后缀：16 kHz 版本约 80–105 MB，
# 而 48 kHz 版本是它的三倍，且本项目只需要 16 kHz。
DEMAND_URL = "https://zenodo.org/records/1227121/files/{env}_16k.zip"

# 设计文档 §4.1 指定的三个环境
DEFAULT_DEMAND_ENVS = ["DKITCHEN", "DLIVING", "OOFFICE"]

CHUNK = 1 << 20  # 1 MiB


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, dest: Path) -> Path:
    """下载 url 到 dest，支持断点续传。已完整下载则直接返回。"""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 临时文件用于中止后续传；完成后改名
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "nbse-downloader/0.1"})
    if resume_from:
        req.add_header("Range", f"bytes={resume_from}-")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            # 服务器不支持 Range 时返回 200，需从头写
            if resume_from and resp.status == 200:
                print(f"  服务器不支持续传，重新下载")
                resume_from = 0

            total = None
            if resp.headers.get("Content-Length"):
                total = int(resp.headers["Content-Length"]) + resume_from

            mode = "ab" if resume_from else "wb"
            done = resume_from
            with open(tmp, mode) as f:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        sys.stdout.write(
                            f"\r  {_human(done)} / {_human(total)}  ({pct:5.1f}%)"
                        )
                        sys.stdout.flush()
        print()
    except urllib.error.URLError as e:
        print(f"\n  下载失败：{e}")
        print(f"  可手动下载 {url} 并放到 {tmp}")
        raise

    tmp.replace(dest)
    return dest


def extract(archive: Path, dest_dir: Path) -> None:
    """解压 tar.gz 或 zip。解压完自动删除压缩包以省磁盘。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  解压 {archive.name} ...")

    if archive.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest_dir)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    else:
        raise ValueError(f"不认识的压缩格式：{archive.name}")

    archive.unlink()
    print(f"  解压完成，已删除 {archive.name}")


def fetch_librispeech(subsets: list[str]) -> None:
    for name in subsets:
        url = LIBRISPEECH_URLS[name]
        target_dir = DATA_ROOT / "LibriSpeech"
        marker = target_dir / name

        if marker.exists():
            print(f"[LibriSpeech/{name}] 已存在，跳过")
            continue

        print(f"[LibriSpeech/{name}] 下载中 ...")
        archive = download(url, DATA_ROOT / f"{name}.tar.gz")
        extract(archive, target_dir)


def _demand_already_extracted(env: str) -> bool:
    """判断某环境是否已解压。

    压缩包内层目录名可能是 DKITCHEN_16k 或 DKITCHEN，两种都认。
    """
    root = DATA_ROOT / "DEMAND"
    if not root.exists():
        return False
    return any(root.glob(f"*{env}*"))


def fetch_demand(envs: list[str]) -> None:
    for env in envs:
        if _demand_already_extracted(env):
            print(f"[DEMAND/{env}] 已存在，跳过")
            continue

        print(f"[DEMAND/{env}] 下载中 ...")
        try:
            archive = download(DEMAND_URL.format(env=env), DATA_ROOT / f"{env}.zip")
        except urllib.error.URLError:
            print(f"  跳过 {env}（网络问题，可稍后重跑本脚本）")
            continue
        extract(archive, target_dir)


def _wav_duration(paths) -> float:
    """用 soundfile 统计总时长；不可用时返回 0。"""
    try:
        import soundfile as sf
    except Exception:
        return 0.0
    total = 0.0
    for p in paths:
        try:
            info = sf.info(str(p))
            total += info.frames / info.samplerate
        except Exception:
            pass
    return total


def report_stats() -> None:
    print("\n" + "=" * 56)
    print("数据统计")
    print("=" * 56)

    ls_root = DATA_ROOT / "LibriSpeech"
    if ls_root.exists():
        for subset in ("dev-clean", "test-clean"):
            root = ls_root / subset
            if not root.exists():
                print(f"LibriSpeech/{subset:12} 缺失")
                continue
            spk = len([d for d in root.iterdir() if d.is_dir()])
            wavs = list(root.rglob("*.flac")) + list(root.rglob("*.wav"))
            hours = _wav_duration(wavs) / 3600
            unit = f"{hours:.1f} h" if hours else "（安装 soundfile 后可见时长）"
            print(f"LibriSpeech/{subset:12} {spk:3d} 说话人  {len(wavs):5d} 条  {unit}")
    else:
        print("LibriSpeech 缺失")

    dm_root = DATA_ROOT / "DEMAND"
    if dm_root.exists():
        envs = sorted(d for d in dm_root.iterdir() if d.is_dir())
        print(f"DEMAND         {len(envs)} 个环境：{', '.join(e.name for e in envs) or '（空）'}")
    else:
        print("DEMAND 缺失")

    print("=" * 56)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--librispeech", action="store_true", help="只下载 LibriSpeech")
    ap.add_argument("--demand", nargs="*", metavar="ENV", help="只下载指定 DEMAND 环境")
    ap.add_argument("--stats", action="store_true", help="只统计已有数据，不下载")
    args = ap.parse_args()

    if args.stats:
        report_stats()
        return 0

    do_ls = args.librispeech or not (args.librispeech or args.demand)
    do_dm = bool(args.demand) or not (args.librispeech or args.demand)

    if do_ls:
        fetch_librispeech(["dev-clean", "test-clean"])
    if do_dm:
        fetch_demand(args.demand if args.demand else DEFAULT_DEMAND_ENVS)

    report_stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())
