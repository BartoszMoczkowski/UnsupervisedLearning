import os
from collections.abc import Iterable
from pathlib import Path

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import v2 as tv2

DATA_DIR = os.environ.get("TRAINING_DATA_DIR", "./wikiart-parquet")
_image_dirs = os.environ.get(
    "IMAGES_DIRS",
    os.environ.get("IMAGES_DIR", DATA_DIR + "-images/300/train_98"),
)
IMAGES_DIR = [Path(root) for root in _image_dirs.split(os.pathsep) if root]


class WikiartDataset(Dataset):
    def __init__(
        self, root: Path | str | Iterable[Path | str], hr_size: int, n_crops: int = 4
    ):
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        roots = [Path(root)] if isinstance(root, (str, Path)) else [Path(p) for p in root]
        self.files = sorted(
            path
            for root_path in roots
            for path in root_path.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        if not self.files:
            raise FileNotFoundError(f"No images under {roots}")
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
