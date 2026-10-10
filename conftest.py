# 放在仓库根目录，使 pytest 把 rootdir 加入 sys.path，从而能 import nbse。
#
# ── OpenMP 冲突的变通（必须在此处，不能只在命令行设）────────────────
# 症状：按文件跑任意测试子集时**间歇性** `Fatal Python error: Aborted`（exit=3），
# 报错为
#     OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
#     already initialized.
# 原因：conda 环境里 torch（自带一份 OpenMP）与 scipy / pyroomacoustics
# （经 MKL 引入另一份 libiomp5md）在一个进程里各链了一份，加载顺序决定是否触发。
# 因此它表现为「跑子集崩、跑全量不崩」，且与任何单个测试文件无关。
#
# 这不是代码 bug，是原生库冲突。官方文档给出的变通就是下面这个环境变量，
# 必须在**导入 torch / scipy 之前**设置，所以放在 conftest 顶层而不是某个 fixture 里。
# 风险：该变通被 Intel 标注为「不安全、不受支持」，可能掩盖真实的多线程问题。
# 对本项目可接受——测试是单进程 CPU 计算，不依赖 OpenMP 的并行正确性；
# 但正式训练脚本不设它，以免掩盖训练期的真实异常。
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
