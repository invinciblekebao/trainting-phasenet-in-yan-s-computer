# PhaseNet++ 跨设备训练代码：NPZ / HDF5

本包按你确认的近期训练配置整理：**`phasenet + UNet`，训练 Noise/P/S 三类震相拾取**。包含数据读取、归一化、标签生成、模型与损失、训练、验证、学习率调度、日志、最佳模型和断点保存、断点恢复。

源代码来自 `phasenet_plus_4080_data9_11` 工作副本。模型、损失和 NPZ/HDF5 读取逻辑沿用现有实现。仅在本包副本中移除内存映射及其他模型的入口/导入，将默认模型改为 `phasenet`；原项目未修改。还保留 `phasenet_plus.py` 模型封装用于查阅，它的多任务训练与这里确认的 P/S 流程不同。

**包内没有训练数据、划分列表、权重、内存映射模块、数据制作/转换工具或历史源码。** 你使用自己准备的新训练数据即可。

## 文件组成

| 文件 / 目录 | 用途 |
|---|---|
| `train.py` | 实际训练、验证、checkpoint 保存与恢复入口 |
| `run_training.py` | 推荐启动入口，明确使用 phasenet + UNet，检查路径和续训参数 |
| `utils.py` | 日志统计、优化器参数分组、权重保存、分布式支持 |
| `eqnet/models/` | PhaseNet、PhaseNet+ 封装、UNet 与所需 ResNet 定义 |
| `eqnet/data/seismic_trace.py` | NPZ/HDF5 读取、归一化、软标签、裁窗、增强 |
| `eqnet/utils/` | 震相拾取指标、后处理、训练绘图 |
| `scripts/check_environment.py` | 环境、模型前向/反向/参数更新检查 |
| `scripts/check_data.py` | 使用实际读取器检查新数据 |
| `configs/train_npz.ps1`、`configs/train_hdf5.ps1` | Windows 启动示例 |
| `requirements-runtime.txt` | 本机已验证的辅助依赖版本，供 Python 3.10 环境参考 |
| `docs/` | 数据格式、环境快照、源码来源与验证记录 |

## 1. 准备另一台设备的环境

以下命令在**另一台设备的 Anaconda Prompt** 中执行，先进入这个包的根目录。如果已经有合适的 PyTorch 环境，直接激活它并检查即可；本次没有安装或修改本机依赖。

```text
conda create -n phasenet_train python=3.10
conda activate phasenet_train
```

然后按目标设备的操作系统、显卡和驱动选择并安装 **PyTorch 和配套 torchvision**：[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)。本机实际验证版本是 Python 3.10.20、torch 2.11.0+cu130、torchvision 0.26.0+cu130；这是环境快照，另一台设备不一定适合相同 CUDA 构建。

安装其他依赖并检查：

```text
python -m pip install -r requirements-runtime.txt
python scripts/check_environment.py --smoke --device cuda
```

如只做 CPU 检查，最后一个参数改为 `--device cpu`。GPU 正式训练前应确认输出中 `cuda_available` 为 `true`。

`requirements.txt` 是原项目的依赖声明；其中 torch/torchvision 没有锁定版本。推荐先安装适配设备的 torch/torchvision，再安装 `requirements-runtime.txt`。无需运行 `setup.py`，保持整个 `eqnet` 目录结构即可导入。W&B 仅在你主动启用 `--wandb` 时需要另配依赖与账号，本包默认关闭。

## 2. 使用新 NPZ 数据训练

推荐的新数据布局（文件由你准备，包内不含它们）：

```text
your_data/
  train.csv
  val.csv
  test.csv
  train/001.npz
  val/002.npz
  test/003.npz
```

CSV **没有表头**，每行第一列是一个 NPZ 路径。例如 `train.csv` 中的一行是 `train/001.npz`；相对路径按 CSV 所在目录解析。字段要求见 [数据格式](docs/data_format.md)。

在包根目录、已激活的 Python 环境中执行（以下单行命令兼容 Windows 和 Linux，改成你自己的路径）：

```text
python scripts/check_data.py --format npz --data-list "your_data/train.csv" --limit 0
python scripts/check_data.py --format npz --data-list "your_data/val.csv" --limit 0
python run_training.py --format npz --train-list "your_data/train.csv" --val-list "your_data/val.csv" --output-dir "runs/npz_01" --batch-size 8 --epochs 100 --workers 4
```

`--limit 0` 检查清单中的所有样本；先快速抽查可使用 `--limit 20`。NPZ 训练数据需要 P、S 采样点索引，只有波形而无标签无法监督训练。

## 3. 使用新 HDF5 数据训练

你准备的 HDF5 需要符合现有读取器的 **事件组 / 台站 dataset** 结构及属性要求，详见 [数据格式](docs/data_format.md)。文件扩展名是 `.hdf5` 或 `.h5` 都可以，`--format` 参数统一写 `h5`。

```text
python scripts/check_data.py --format h5 --data-list "your_data/train.txt" --hdf5-file "your_data/train.hdf5" --limit 20
python scripts/check_data.py --format h5 --data-list "your_data/val.txt" --hdf5-file "your_data/val.hdf5" --limit 20
python run_training.py --format h5 --train-list "your_data/train.txt" --val-list "your_data/val.txt" --train-hdf5 "your_data/train.hdf5" --val-hdf5 "your_data/val.hdf5" --output-dir "runs/hdf5_01" --batch-size 8 --epochs 100 --workers 4
```

每份 TXT 每行写一个 HDF5 内部路径，例如 `372/372_819`。这里 `--test-data-list`/`--test-hdf5-file` 在内部训练入口中代表**验证集**；不要把最终测试集交给训练时验证。

HDF5 文件名和目录可以任意改变，只需修改命令里的路径。若使用 PowerShell 模板，默认要求数据目录里有 `train.txt`、`val.txt`、`train.hdf5`、`val.hdf5`；其他命名用上面的 Python 命令启动。

## 4. 参数、输出与恢复

推荐入口固定沿用：100 Hz、phasenet + UNet、AdamW、初始 LR 0.001、weight decay 0.0001、1 轮线性 warmup、余弦学习率、关闭 SNR 筛选、关闭绘图、关闭早停。`--batch-size` 默认 8，可以按新设备显存调整。`--epochs` 是总轮数，至少 2；默认 100。`--workers` 默认 4，Windows 排查数据加载问题时可改成 0。

新实验选择新的输出目录；未指定时会在包内 `runs/` 创建带时间戳的目录。每轮保存：

- `training_log.csv` / `training_log.jsonl`：逐轮训练与验证日志。
- `model_<epoch>.pth`：该轮完整断点，编号从 0 开始。
- `checkpoint.pth`：最新完整断点。
- `model_best.pth`：验证损失最低的断点。

恢复未完成的训练，在原训练命令后加 `--resume "runs/hdf5_01/checkpoint.pth"`，并保持相同 batch size、总 epochs、LR、数据集与其他训练设置。入口会同时恢复模型、优化器和调度器。例如：

```text
python run_training.py --format h5 --train-list "your_data/train.txt" --val-list "your_data/val.txt" --train-hdf5 "your_data/train.hdf5" --val-hdf5 "your_data/val.hdf5" --output-dir "runs/hdf5_01" --batch-size 8 --epochs 100 --workers 4 --resume "runs/hdf5_01/checkpoint.pth"
```

旧实验已经完成第 150 轮；如果之后选择在新数据上从头训练，不需要迁移旧权重。完成全部轮数后追加训练，或更改 batch size 续训，需要另行调整学习率计划，不能仅加大 `--epochs` 并恢复旧调度器。本包按你的范围要求不附带历史续训改写工具。

学习率、采样率、增强、阈值等更细参数可直接用 `python train.py --help` 查看，然后从 `train.py` 启动。直接启动时显式加 `--model phasenet --backbone unet --format npz` 或 `--format h5`，保持你的实验设置；推荐入口的参数值见 `run_training.py`。

## 5. 单独测试保留集

训练结束后，用从未参与训练和验证的 test 清单评估。例如 NPZ：

```text
python train.py --model phasenet --backbone unet --format npz --test-only --test-data-list "your_data/test.csv" --resume "runs/npz_01/model_best.pth" --disable-snr-filter --disable-plot --batch-size 8 --workers 0 --output-dir "runs/npz_01_test"
```

HDF5 测试仍按入口的数据集构建方式同时传入两组参数，均指向保留测试集：

```text
python train.py --model phasenet --backbone unet --format h5 --test-only --data-list "your_data/test.txt" --hdf5-file "your_data/test.hdf5" --test-data-list "your_data/test.txt" --test-hdf5-file "your_data/test.hdf5" --resume "runs/hdf5_01/model_best.pth" --disable-snr-filter --disable-plot --batch-size 8 --workers 0 --output-dir "runs/hdf5_01_test"
```

这里直接调用 `train.py`；`run_training.py` 用于训练/恢复，不用于 test-only。

## 6. 本包的验证边界

实际执行结果记录在 `docs/verification_report.json`：使用少量真实 DiTing 波形分别验证 NPZ/HDF5 读取、两轮训练、验证、日志、权重保存和从第 1 轮断点恢复。测试数据、权重和缓存不放入交付包。

这证明本机精简副本的训练流程可运行。另一台设备的环境和你准备的新数据尚未验证，请先运行环境检查和数据检查。保持完整包目录一起复制，避免再次按功能打散 Python 导入路径。
