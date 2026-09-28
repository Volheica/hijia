# 计算机视觉招新题

本仓库为招新题作业代码。任务包括图像基础操作、PyTorch 手写数字识别,
以及两个自主选题方向(时序预测、图像分类)。

## 环境

Python 3.12,Windows 环境,依赖见 `requirements.txt`。

PyTorch 需要按显卡选择构建版本,RTX 50 系必须用 CUDA 12.8 及以上:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

其余依赖:

```bash
pip install -r requirements.txt
```

## 数据集

数据集统一放在 `D:\datasets`,不进仓库。目录约定:

| 数据集 | 路径 | 用途 |
|---|---|---|
| MNIST | `D:\datasets\MNIST` | task-3 的 MLP 识别 |
| Air Passengers | `D:\datasets\timeseries\airline-passengers.csv` | 方向 3 时序预测 |
| CIFAR-10(备用) | `D:\datasets\cifar-10-batches-py` | 方向 2 图像分类备选 |

MNIST 已转换成本地原始格式,直接这样调用即可,不会联网下载:

```python
from torchvision import datasets
train_set = datasets.MNIST(root=r"D:\datasets", train=True, download=True)
```

Air Passengers 的下载脚本见 `src/download_data.py`。

## 目录结构

```
├── src/          源码,每个任务一个脚本
├── results/      运行产生的图、曲线(视频走邮件提交,不入库)
├── data/         小体积示例文件(example1.jpg 等)
├── report/       实验报告
└── .vscode/      编辑器配置,已固定解释器到 D:\envs\cv
```

## 运行方式

1. 用 VSCode 打开本文件夹,解释器会自动选到 `D:\envs\cv`(见 `.vscode/settings.json`)
2. 终端里确认提示符前面有 `(cv)`
3. 运行对应脚本,例如:

```bash
python src/task2_01_basic.py
```

结果默认保存到 `results/`。

## 提交说明

代码在本仓库;视频、图片、报告打包发到指定邮箱,
并在题目仓库 Issues 按 `班别-姓名-学号-仓库链接` 格式提交。
