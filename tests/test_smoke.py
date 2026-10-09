"""T1.1 验收测试：脚手架是否可用。

对应实施计划 T1.1 的验收标准：
  - `pip install -r requirements.txt` 成功
  - 包与配置可正常加载
"""

import importlib
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_config_loads():
    """配置可解析，且关键字段与设计文档一致。"""
    cfg_path = ROOT / "configs" / "default.yaml"
    assert cfg_path.exists(), f"缺少配置文件：{cfg_path}"

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    assert cfg["audio"]["sample_rate"] == 16000, "PESQ 宽带模式要求 16 kHz"
    assert cfg["array"]["n_mics"] == 4
    assert cfg["room"]["source_azimuth_range"] == [0.0, 360.0], "Oracle DOA 依赖方位角真值"
    assert cfg["evaluation"]["t60_levels"] == [0.3, 0.6]
    assert cfg["data"]["split"]["train_speakers"] == 30


def test_package_imports():
    """所有子包可导入（此时内部模块可能还是空的，只验证包结构）。"""
    for name in ["nbse", "nbse.data", "nbse.beamformers", "nbse.models", "nbse.metrics"]:
        importlib.import_module(name)


def test_torch_cuda_available():
    """GPU 可用性 —— 这是 `pip install -r requirements.txt` 成功与否的真正判据。

    若此测试失败，几乎总是因为装了 CPU 版 PyTorch。
    请按 requirements.txt 顶部的说明改用 cu128 索引安装。
    """
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.fail(
            f"torch {torch.__version__} 无法使用 GPU "
            f"(torch.version.cuda={torch.version.cuda})。\n"
            "这通常是 CPU 版 wheel。请执行：\n"
            "  pip install torch torchaudio --index-url "
            "https://download.pytorch.org/whl/cu128"
        )

    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}  {props.total_memory / 1024**3:.1f} GB  "
          f"sm_{props.major}{props.minor}")
