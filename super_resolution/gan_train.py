#!/usr/bin/env python3
"""Pretrain and adversarially train a lightweight SRGAN on Wikiart images."""

import argparse
import io
import math
import os
import random
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import functional as TF
from torchvision.transforms import v2 as tv2
from tqdm import tqdm
from generative_adversial_network import DiscriminatorNetwork, GeneratorNetwork
import torchvision
DATA_DIR = os.environ.get("TRAINING_DATA_DIR", "./wikiart-parquet")
IMAGES_DIR = Path(os.environ.get("IMAGES_DIR", DATA_DIR + "-images/train"))
SCALE = 4
VGG_FEATURE_SCALE = 12.75

CKPT_DIR = Path(os.environ.get("CKPT_DIR", "./checkpoints"))

class WikiartDataset(Dataset):
    def __init__(self, root: Path, hr_size: int, n_crops: int = 4):
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.files = sorted(p for p in root.rglob("*")
                            if p.is_file() and p.suffix.lower() in extensions)
        if not self.files:
            raise FileNotFoundError(f"No images under {root}")
        # per-crop transform; each call samples a new crop window + flip
        self.crop = tv2.Compose([
            tv2.RandomResizedCrop(hr_size, antialias=True),
            tv2.RandomHorizontalFlip(),
        ])
        self.n_crops = n_crops

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = torchvision.io.decode_image(str(self.files[index]), mode="RGB")  # uint8 CHW
        return torch.stack([self.crop(img) for _ in range(self.n_crops)])

def save_checkpoint(epoch, generator, discriminator, gen_opt):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_optimizer": gen_opt.state_dict(),
        },
        CKPT_DIR / "srgan.pt",
    )
    print(f"checkpoint saved at epoch {epoch + 1}", flush=True)

def main():
    N_CROPS = 1
    SAVE_EVERY = 10
    lr_size = 48
    hr_size = 2 * lr_size
    target_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    is_cuda = (target_device == "cuda")

    if is_cuda:
        torch.backends.cudnn.benchmark = True          # fixed 96px input shapes → autotune convs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_ds = WikiartDataset(IMAGES_DIR, hr_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=512 // N_CROPS,      # 512 files → 2048 samples per step
        shuffle=True,
        num_workers=8,
        prefetch_factor=4,             # each worker keeps 4 batches ready
        pin_memory=is_cuda,
        persistent_workers=True
    )

    random_transform = tv2.TrivialAugmentWide()

    lr_resize = tv2.Resize((lr_size,lr_size),antialias=True)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    normalize = tv2.Compose([
        tv2.ToDtype(torch.float32, scale=True),
        tv2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


    generator = GeneratorNetwork(
        3,
        16,
        16,
        32,
        n_residual_blocks=4,
        n_tail_blocks=1
    ).to(target_device)

    discriminator = DiscriminatorNetwork(
        3,
        16,
        [4,4],
        [4,4],
        img_size=(hr_size,hr_size),
        n_dense_1=128,
    ).to(target_device)

    N_SEEN_IMAGES = int(2e6)
    print(len(train_ds))
    n_epochs = int(N_SEEN_IMAGES//len(train_ds))
    lr = 1e-4
#https://github.com/NVIDIA/DALI this shit is too slow
    generator_optimizer = torch.optim.Adam(generator.parameters(),lr=lr, betas=(0.9, 0.999))
    print(
        f"generator_parameters={sum(p.numel() for p in generator.parameters()):,} "
        f"discriminator_parameters={sum(p.numel() for p in discriminator.parameters()):,}",
        flush=True,
    )
    generator.train()
    for epoch in tqdm(range(n_epochs)):
        for batch_idx, hr_img in enumerate(train_loader):
            hr_img = hr_img.to(target_device, non_blocking=is_cuda)
            hr_img = hr_img.flatten(0, 1)
            hr_img = random_transform(hr_img)

            lr_img = lr_resize(hr_img)

            hr_img = normalize(hr_img)
            lr_img = normalize(lr_img)

            # discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),lr=lr, betas=(0.9, 0.999))

            generator_optimizer.zero_grad(set_to_none=True)
            sr_img = generator(lr_img)
            loss = F.mse_loss(sr_img, hr_img)
            loss.backward()
            generator_optimizer.step()

            if (epoch + 1) % SAVE_EVERY == 0:
                save_checkpoint(epoch, generator, discriminator, generator_optimizer)

    save_checkpoint(int(n_epochs), generator, discriminator, generator_optimizer)

if __name__ == "__main__":
    main()
