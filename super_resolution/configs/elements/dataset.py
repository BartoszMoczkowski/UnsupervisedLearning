import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2 as tv2


def _find_image_files(root: Path, extensions: set[str]) -> list[Path]:
    """os.walk-based scan: much faster than pathlib.rglob on large or network mounts."""
    files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in extensions:
                files.append(path)
    return sorted(files)


class WikiartDataset(Dataset):
    def __init__(self, root: Path, hr_size: int, n_crops: int = 1) -> None:
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.files = _find_image_files(root, extensions)
        if not self.files:
            raise FileNotFoundError(
                f"No images under {root}. This dataset expects decoded image files "
                f"({sorted(extensions)}), not parquet archives - extract them first "
                f"or point the config at the extracted directory."
            )
        print(f"WikiartDataset: found {len(self.files)} images under {root}")
        self.crop = tv2.Compose([
            tv2.RandomResizedCrop(hr_size, antialias=True),
            tv2.RandomHorizontalFlip(),
        ])
        self.n_crops = n_crops

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = torchvision.io.decode_image(str(self.files[index]), mode="RGB")
        if self.n_crops == 1:
            return self.crop(img)
        return torch.stack([self.crop(img) for _ in range(self.n_crops)])


class DatasetConfig(ABC):
    train_test_eval: tuple[float, float, float]
    seed: int
    batch_size: int

    @abstractmethod
    def load_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        pass

    def load_train_dataloader(self) -> DataLoader:
        train_dataset, _, _ = self.load_datasets()
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)


class WikiartDatasetConfig(DatasetConfig):
    def __init__(
        self,
        root: Path,
        hr_size: int,
        train_test_eval: tuple[float, float, float] = (0.98, 0.01, 0.01),
        seed: int = 0,
        batch_size: int = 16,
        n_crops: int = 1,
        num_workers: int = 0,
    ) -> None:
        self.root = Path(root)
        self.hr_size = hr_size
        self.train_test_eval = train_test_eval
        self.seed = seed
        self.batch_size = batch_size
        self.n_crops = n_crops
        self.num_workers = num_workers

    def load_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        dataset = WikiartDataset(self.root, self.hr_size, self.n_crops)
        n_total = len(dataset)
        counts = [int(n_total * fraction) for fraction in self.train_test_eval]
        counts = [
            max(count, 1) if fraction > 0 else 0
            for count, fraction in zip(counts, self.train_test_eval)
        ]
        if sum(counts) > n_total:  # tiny dataset: keep train at least 1
            counts = [max(n_total - 2, 1), 1, 1] if n_total >= 3 else [n_total, 0, 0]
        counts[0] = n_total - counts[1] - counts[2]  # exact remainder into train
        generator = torch.Generator().manual_seed(self.seed)
        return tuple(random_split(dataset, counts, generator=generator))  # type: ignore[return-value]

    def load_train_dataloader(self) -> DataLoader:
        train_dataset, _, _ = self.load_datasets()
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
