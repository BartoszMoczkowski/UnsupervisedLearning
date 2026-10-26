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

def save_checkpoint(epoch, generator, discriminator, gen_opt, filename="srgan.pt"):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_optimizer": gen_opt.state_dict(),
        },
        CKPT_DIR / filename,
    )
    print(f"checkpoint saved at epoch {epoch + 1}: {CKPT_DIR / filename}", flush=True)

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

    # Keep SR targets photometrically faithful: random crops/flips are already in the dataset.
    lr_resize = tv2.Resize((lr_size,lr_size),antialias=True)
    normalize = tv2.Compose([
        tv2.ToDtype(torch.float32, scale=True),
        tv2.Normalize(mean=3*[0.5], std=3*[0.5])  # [0, 1] -> [-1, 1]
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

    N_SEEN_IMAGES = int(1e6)
    n_epochs = int(N_SEEN_IMAGES//len(train_ds))
    lr = 1e-4
    generator_optimizer = torch.optim.Adam(generator.parameters(),lr=lr, betas=(0.9, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),lr=lr, betas=(0.9, 0.999))
    print(
        f"generator_parameters={sum(p.numel() for p in generator.parameters()):,} "
        f"discriminator_parameters={sum(p.numel() for p in discriminator.parameters()):,}",
        flush=True,
    )
    generator.train()
    for epoch in tqdm(range(n_epochs), desc="MSE pretraining"):
        for batch_idx, hr_img in enumerate(train_loader):
            hr_img = hr_img.to(target_device, non_blocking=is_cuda)
            hr_img = hr_img.flatten(0, 1)

            lr_img = lr_resize(hr_img)

            hr_img = normalize(hr_img)
            lr_img = normalize(lr_img)

            generator_optimizer.zero_grad(set_to_none=True)
            sr_img = generator(lr_img)
            loss = F.mse_loss(sr_img, hr_img)
            loss.backward()
            generator_optimizer.step()


    # Keep a clean recovery point; never overwrite it during adversarial training.
    save_checkpoint(
        n_epochs - 1, generator, discriminator, generator_optimizer,
        filename="srresnet.pt",
    )


    # SRGAN alternates D and G from the start; do not pretrain D against a frozen G.
    generator.train()
    discriminator.train()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    for epoch in tqdm(range(n_epochs), desc="Adversarial training"):
        for batch_idx, hr_img in enumerate(train_loader):
            hr_img = hr_img.to(target_device, non_blocking=is_cuda)
            hr_img = hr_img.flatten(0, 1)

            lr_img = lr_resize(hr_img)

            hr_img = normalize(hr_img)
            lr_img = normalize(lr_img)

            discriminator.requires_grad_(True)
            with torch.no_grad():
                sr_img = generator(lr_img)
            discriminator_optimizer.zero_grad(set_to_none=True)

            d_fake = discriminator(sr_img)
            d_real = discriminator(hr_img)
            d_loss = bce_loss(d_fake,torch.zeros_like(d_fake)) + bce_loss(d_real,torch.ones_like(d_real))

            d_loss.backward()
            discriminator_optimizer.step()

            # Use the same D mode for fake batches in both updates; freeze only weights.
            discriminator.requires_grad_(False)
            sr_img = generator(lr_img)
            generator_optimizer.zero_grad(set_to_none=True)

            d_fake_adv = discriminator(sr_img)
            adv_loss = bce_loss(d_fake_adv,torch.ones_like(d_fake_adv))
            content_loss = F.mse_loss(sr_img, hr_img)

            loss = content_loss + adv_loss * 1e-3
            loss.backward()
            generator_optimizer.step()



        if (epoch + 1) % SAVE_EVERY == 0:
            save_checkpoint(epoch, generator, discriminator, generator_optimizer)

    save_checkpoint(n_epochs - 1, generator, discriminator, generator_optimizer)




if __name__ == "__main__":
    main()
