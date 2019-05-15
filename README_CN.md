# PyTorch风格指南和最佳实践

首先，[PyTorch 官方教程](https://pytorch.org/tutorials)是最好的入门文档。

## 推荐使用Python 3.6

由于兼容性和一些从3.6开始的新的特性支持，推荐大家使用Python 3.6。



## 文件组织

不要将所有的代码全部放在一个文件中！最好把不同的

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