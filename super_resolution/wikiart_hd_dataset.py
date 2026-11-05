import os
from pathlib import Path
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2 as tv2
import torch

DATA_DIR = os.environ.get("TRAINING_DATA_DIR", "./wikiart-parquet")
IMAGES_DIR = Path(os.environ.get("IMAGES_DIR", DATA_DIR + "-images/train"))

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
