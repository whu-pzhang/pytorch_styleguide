import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from prefetch_generator import BackgroundGenerator
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models import ConvNet
import utils
from losses import CustomLoss

# set flags
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="config file path")
    opt = parser.parse_args()
    print(opt)

    config = utils.parse_cfg(opt.cfg)
    data_root = config["data_root_dir"]
    train_bs = config["train"]["batch_size"]
    test_bs = config["test"]["batch_size"]
    train_nworkers = config["train"]["nworkers"]
    test_nworkers = config["test"]["nworkers"]
    learning_rate = config["train"]["learning_rate"]
    log_dir = config["log_dir"]  # for tensorboard visualization
    num_epochs = config["train"]["num_epochs"]
    resume = config["train"]["resume"]
    ckpt_path = config["train"]["checkpoint"]
    test_interval = config["test"]["test_interval"]
    backup_interval = config["train"]["backup_interval"]

    data_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.1308, ), std=(0.3083, ))
    ])

    train_dataset = torchvision.datasets.MNIST(root=data_root,
                                               train=True, download=True,
                                               transform=data_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                               batch_size=train_bs, pin_memory=True,
                                               num_workers=train_nworkers)

    test_dataset = torchvision.datasets.MNIST(root=data_root,
                                              train=False, download=True,
                                              transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True,
                                              batch_size=test_bs, pin_memory=True,
                                              num_workers=test_nworkers)

    device = utils.select_device(force_cpu=False)

    # instantiate network
    net = ConvNet(1, 10).to(device)

    # criterion
    loss_fn = CustomLoss()
    # optimer
    optimizer = torch.optim.Adam(net.parameters(), learning_rate)

    # load checkpoint if needed
    start_n_iter = 0
    start_epoch = 0
    # if resume:
    #     ckpt = utils.load_checkpoint(ckpt_path)  # custom method for loading last checkpoint
    #     net.load_state_dict(ckpt['model_state_dict'])
    #     start_epoch = ckpt['epoch']
    #     start_niter = ckpt['niter']
    #     optimizer.load_state_dict(ckpt['optim_state_dict'])
    #     print("last checkpoint restored")

    # # if we want to run experiment on multiple GPUs we move the models there
    # net = torch.nn.DataParallel(net)

    writer = SummaryWriter(log_dir)
    n_iter = start_n_iter
    for epoch in range(start_epoch, num_epochs):
        net.train()

        pbar = tqdm(enumerate(BackgroundGenerator(train_loader)))
        start_time = time.time()

        for i, (img, label) in pbar:
            img, label = img.to(device), label.to(device)
            prepare_time = time.time() - start_time
            # forward
            optimizer.zero_grad()
            output = net(img)
            loss = loss_fn(output, label)

            # backward + update
            loss.backward()
            optimizer.step()

            writer.add_scalar("Train/Loss", loss.item(), n_iter)

            process_time = time.time() - start_time - prepare_time

            pbar.set_description(
                f"Compute efficiency: {process_time/(process_time+prepare_time):.3f}, epoch: {epoch}/{num_epochs}:")
            start_time = time.time()

        # do a test pass every x epochs
        if epoch % test_interval == test_interval - 1:
            net.eval()
            correct = 0
            test_loss = 0

            pbar = tqdm(enumerate(BackgroundGenerator(test_loader)),
                        total=len(test_loader))
            for i, (img, label) in pbar:
                img, label = img.to(device), label.to(device)
                output = net(img)
                test_loss += F.cross_entropy(output, label).item()
                pred = torch.argmax(output, dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

            total = len(test_loader.dataset)
            test_loss /= total

            writer.add_scalar("Test/Accu", correct / total, n_iter)
