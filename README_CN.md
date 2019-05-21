# PyTorch风格指南和最佳实践

首先，[PyTorch 官方教程](https://pytorch.org/tutorials)是最好的入门文档。

## 推荐使用Python 3.6

由于兼容性和一些从3.6开始的新的特性支持，推荐大家使用Python 3.6。


## 文件组织

不要将所有的代码全部放在一个文件中！最好把将模型架构单独放在一个文件中（如：`models/unet.py`），
而损失函数定义(指不在PyTorch官方实现中的损失函数，如YOLO的损失函数)等其他的代码放在各自的文件中（如: `losses/yolo.py`）。

直接运行的代码，如训练和测试脚本，分别放在`train`和`test`文件夹中。

整个的文件树如下：

```bash
├── README.md
├── config
├── losses
├── models
├── requirements.txt
├── test
├── train
└── utils
```

- `README.md`: 项目描述
- `requirements.txt`: 项目需要的包
- `config`: 存放配置文件的文件夹.
- `models`: 包含基础模型结构.
- `losses`: PyTorch中没有的需要自己实现的损失函数.
- `utils`: 数据预处理、后处理和其他工具函数
- `train`: 训练脚本文件夹 (如 `cloud_train.py`, `road_train`.py ...)
- `test`: 测试脚本（模型训练好后，实际运行的脚本）文件夹. (如 `cloud_detection.py`, `road_detection.py` ...)

## 命名约定

推荐遵循[Google Python styleguide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)对变量，类等对象进行命名。

一些常用的规则总结如下：

| 类型       | 约定               | 示例                                                   |
| ---------- | ------------------ | ------------------------------------------------------ |
| 包和模块   | 带下划线的小写字母 | from **prefetch_generator** import BackgroundGenerator |
| 类         | 首字母大写         | class **DataLoader**                                   |
| 常量       | 带下划线的大写字母 | **BATCH_SIZE=16**                                      |
| 实例       | 带下划线的小写字母 | **dataset** = Dataset                                  |
| 方法和函数 | 带下划线的小写字母 | def **visualize_tensor()**                             |
| 变量       | 带下划线的小写字母 | **background_color**='Blue'                            |

**尽可能使每个变量或者函数名是一些有意义的单词。**


## 代码示例

- MNIST数字识别

```bash
$ python train/mnist_train.py --cfg config/mnist.yaml
```

## TODO

- U-Net语义分割