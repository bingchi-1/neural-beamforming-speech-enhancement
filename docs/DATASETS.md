# 数据下载

总计约 **950 MB**。全部下载后放到仓库根目录的 `data/`（该目录已被 `.gitignore` 排除）。

下载完成后运行：

```bash
python scripts/download_data.py
```

脚本会**识别已存在的压缩包**（不重复下载）、自动解压、删除压缩包，最后打印数据统计。

---

## 1. LibriSpeech（干净语音）

| 文件名 | 大小 | 链接 |
|---|---|---|
| `dev-clean.tar.gz` | 337 MB | https://www.openslr.org/resources/12/dev-clean.tar.gz |
| `test-clean.tar.gz` | 346 MB | https://www.openslr.org/resources/12/test-clean.tar.gz |

资源页（若直链较慢，可从此页寻找就近镜像）：https://www.openslr.org/12/

解压后结构：

```
data/LibriSpeech/
  dev-clean/   40 说话人 5.4 h   ← 训练用前 30 人，验证用其余 10 人
  test-clean/  40 说话人 5.4 h   ← 全部用于测试
```

---

## 2. DEMAND（噪声）

**文件名必须带 `_16k` 后缀。**

DEMAND 同时提供 16 kHz 与 48 kHz 两个版本，48 kHz 版体积是前者约三倍，
而本项目只需要 16 kHz（PESQ 宽带模式要求该采样率）。

| 文件名 | 大小 | 链接 |
|---|---|---|
| `DKITCHEN_16k.zip` | 105 MB | https://zenodo.org/records/1227121/files/DKITCHEN_16k.zip |
| `DLIVING_16k.zip` | 77 MB | https://zenodo.org/records/1227121/files/DLIVING_16k.zip |
| `OOFFICE_16k.zip` | 85 MB | https://zenodo.org/records/1227121/files/OOFFICE_16k.zip |

数据集合集页：https://zenodo.org/records/1227121

---

## 放置方式

```
neural-beamforming-speech-enhancement/
  data/
    dev-clean.tar.gz
    test-clean.tar.gz
    DKITCHEN_16k.zip
    DLIVING_16k.zip
    OOFFICE_16k.zip
```

**直接放压缩包即可，不需要手动解压**——`download_data.py` 会处理。
若使用下载管理器，注意不要产生 `xxx.tar.gz.part` 之类的残留文件。

---

## 为什么噪声要选这三个环境

`DKITCHEN`（厨房）、`DLIVING`（客厅）、`OOFFICE`（办公室）都是**非平稳**的
家居/办公噪声，击中了本项目的核心问题设定：真实环境中风噪、交通噪声使信噪比
降至 −15 ~ −5 dB，静态噪声假设不成立。

选用 DEMAND 还有一个理由：VoiceBank-DEMAND 这一经典语音增强基准的噪声就取自该库，
属于领域标准做法，便于与既有文献直接对比。
