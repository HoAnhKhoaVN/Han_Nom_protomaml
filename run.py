## Standard libraries
import os
import numpy as np
import random
import json
from PIL import Image
from collections import defaultdict
from statistics import mean, stdev
from copy import deepcopy
from tqdm import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## Torchvision
import torchvision
from torchvision.datasets import CIFAR100, SVHN
from torchvision import transforms


from dataset import HanNomDatasetNShot
import argparse
from torch.utils.tensorboard import SummaryWriter
from pickle import dump, load
import pytorch_lightning as pl
from torchvision.datasets import ImageFolder
from dataset import ImageDataset
from train import train_model
from module.protomaml import ProtoMAML
from parser_utils import get_args
from prepare_ds import load_dataset

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((32, 32), antialias= False),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])


def main():
    # region get aguments
    args = get_args()
    print(f'args: {args}')
    # endregion get aguments
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # region Experiment placeholder
    exp_path = args.exp
    os.makedirs(exp_path, exist_ok = True)
    # endregion

    # region writer tensorboard
    writer = SummaryWriter(
        log_dir=exp_path
    )
    # endregion

    # region set seed to reproduce
    # Setting the seed
    pl.seed_everything(args.seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # endregion

    # region Load dataset
    (
        train_promaml_loader,
        val_promaml_loader,
        test_promaml_loader,
        classes
    )= load_dataset(args)
    # endregion

    # region Training
    protomaml_model = train_model(
        train_loader=train_promaml_loader,
        val_loader=val_promaml_loader,
        device= device,
        epoch = args.epoch,
        exp= args.exp,
        test_loader= test_promaml_loader,
        proto_dim=len(classes),
        lr=1e-3,
        lr_inner=0.1,
        lr_output=0.1,
        num_inner_steps=args.update_step,
    )

    # endregion


if __name__ == "__main__":
    main()