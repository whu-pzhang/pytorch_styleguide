<div align="center">
📖 English
&emsp;&emsp; | &emsp;&emsp;
<a href="https://github.com/whu-pzhang/pytorch_styleguide/blob/master/README_CN.md">📖 中文</a>
</div> 
<br>

# A PyTorch style-guide & best practices

The offical [PyTorch Tutorials](https://pytorch.org/tutorials) is the best materials for beginers.

## recommend using Python3.6

From our experience, we recommend using Python 3.6 because of the compatibility and serval new features which
became very handy for clean and simple code.


## File Organization

Don't put all stuffs into the same file! A best practices is to separate the model into a separate
file, such as `models/unet.py`. And keep the layers, losses in respective files(`losses/unet_loss.py`).
The finally models contained multiple networkds should be reference in a file with its name(e.g, yolov3.py, GCGAN.py).

The main routine, such as the train and test scripts should only import from the file having the model's name.

```bash
├── README.md
├── cfg
├── losses
├── models
├── requirements.txt
├── test
├── train
└── utils
```

- `README.md`: descriptions of your project
- `requirements.txt`: the required packages of your project
- `cfg`: configuration file folder. such as your training parameters.
- `models`: contain the basic backbone architectures and final models.
- `losses`: losses which not contained in native PyTorch, such as yolov3_loss.
- `utils`: other tools for data prepare and post processing, such as split large remote sensing image into your network desired size.
- `train`: as your can known from the folder name. (cloud_train.py, road_train.py ...)
- `test`: same as above. (cloud_detection.py, road_detection.py ...)


## Naming Conventions

Follow the [Google Styleguide for Python](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).

Here is a summary of the most commonly used rules:

| Type                | Convention         | Example                                                |
| ------------------- | ------------------ | ------------------------------------------------------ |
| Packages & Modules  | lower_with_under   | from **prefetch_generator** import BackgroundGenerator |
| Classes             | CapWords           | class **DataLoader**                                   |
| Constants           | CAPS_WITH_UNDER    | **BATCH_SIZE=16**                                      |
| Instances           | lower_with_under   | **dataset** = Dataset                                  |
| Methods & Functions | lower_with_under() | def **visualize_tensor()**                             |
| Variables           | lower_with_under   | **background_color**='Blue'                            |

**Try your best to using meaningful words.**

