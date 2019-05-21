import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import numpy as np
import time
from pathlib import Path

from tqdm import tqdm
from tensorboardX import SummaryWriter
from prefetch_generator import BackgroundGenerator

import sys
sys.path.insert(0, '.')

from losses import CustomLoss
from utils import utils
from models import SimpleConvNet


# set flags
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--cfg", help="config file path")
    opt = parser.parse_args()
    print(opt)

    config = utils.parse_cfg(opt.cfg)
    #
    version = config["output_version"]
    data_root = config["data_root_dir"]
    train_bs = config["train"]["batch_size"]
    test_bs = config["test"]["batch_size"]
    train_nworkers = config["train"]["nworkers"]
    test_nworkers = config["test"]["nworkers"]
    learning_rate = float(config["train"]["learning_rate"])
    log_dir = config["log_dir"]  # for tensorboard visualization
    num_epochs = config["train"]["num_epochs"]
    resume = config["train"]["resume"]
    ckpt_path = config["train"]["checkpoint"]
    test_interval = config["test"]["test_interval"]
    backup_interval = config["train"]["backup_interval"]  # save interval

    data_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.1307, ), std=(0.3081, ))
    ])

    train_dataset = torchvision.datasets.MNIST(root=data_root,
                                               train=True, download=True,
                                               transform=data_transform)
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=train_bs, pin_memory=True,
                              num_workers=train_nworkers)

    test_dataset = torchvision.datasets.MNIST(root=data_root,
                                              train=False, download=True,
                                              transform=data_transform)
    test_loader = DataLoader(test_dataset, shuffle=True,
                             batch_size=test_bs, pin_memory=True,
                             num_workers=test_nworkers)

    device = utils.select_device(force_cpu=False)

    # instantiate network
    net = SimpleConvNet(1, 10).to(device)

    # criterion
    loss_fn = CustomLoss()
    # optimer
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.5)

    # load checkpoint if needed
    start_epoch = 0
    if resume:
        ckpt = utils.load_checkpoint(ckpt_path)  # custom method for loading last checkpoint
        net.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt['epoch']
        start_niter = ckpt['niter']
        optimizer.load_state_dict(ckpt['optim_state'])
        print("last checkpoint restored")

    # if we want to run experiment on multiple GPUs we move the models there
    # net = torch.nn.DataParallel(net)

    writer = SummaryWriter(log_dir)
    niter = start_niter
    for epoch in range(start_epoch, num_epochs):
        train_pbar = tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader))

        start_time = time.time()
        net.train()
        for i, (img, label) in train_pbar:
            img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
            prepare_time = time.time() - start_time
            # forward
            optimizer.zero_grad()
            output = net(img)
            loss = loss_fn(output, label)
            # backward + update
            loss.backward()
            optimizer.step()

            process_time = time.time() - start_time - prepare_time
            train_pbar.set_description(
                f"Compute efficiency: {process_time/(process_time+prepare_time):.3f}, epoch: {epoch+1}/{num_epochs}")

            niter += 1
            writer.add_scalar("Train/Loss", loss.item(), niter)

            start_time = time.time()

        # do a test pass every x epochs
        if (epoch + 1) % test_interval == 0:
            net.eval()
            correct = 0
            test_loss = 0

            for i, (img, label) in enumerate(test_loader):
                img, label = img.to(device, non_blocking=True), label.to(device, non_blocking=True)
                output = net(img)
                test_loss += F.cross_entropy(output, label).item()
                pred = torch.argmax(output, dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

            total = len(test_loader.dataset)
            test_loss /= total

            writer.add_scalar("Test/loss", test_loss, niter)
            writer.add_scalar("Test/Accu", correct / total, niter)
            start_time = time.time()

        # save model state
        if (epoch + 1) % backup_interval == 0:
            state = {"epoch": epoch,
                     "niter": niter,
                     "model_state": net.state_dict(),
                     "optim_state": optimizer.state_dict()
                     }
            # filename = Path(ckpt_path) / f"mnist_{version}_{epoch+1}.pth"
            # torch.save(state, filename)
            utils.save_checkpoint(state, False, ckpt_path)
