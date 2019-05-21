import torch
import shutil
from pathlib import Path
import yaml


def parse_cfg(cfg_path):
    with open(cfg_path, "r") as fp:
        content = fp.read()
        y = yaml.load(content)
    return y


def select_device(force_cpu=False):
    if force_cpu:
        cuda = False
        device = torch.device("cpu")
    else:
        cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if cuda else "cpu")

    print(f"Using {device.type} {torch.cuda.get_device_properties(0) if cuda else ''}")
    return device


def save_checkpoint(state, is_best, checkpoint_path):
    filepath = Path(checkpoint_path) / "last.pth.tar"
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint directory does not exist! Making directory {checkpoint_path}")
        Path(checkpoint_path).mkdir()
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, Path(checkpoint_path) / "best.pth.tar")


def load_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path) / "last.pth.tar"
    if not checkpoint_path.exists():
        raise(f"File doesn't exist {str(checkpoint_path)}")

    checkpoint = torch.load(str(checkpoint_path))
    return checkpoint
