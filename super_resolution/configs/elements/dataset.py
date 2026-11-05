from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2 as tv2


class WikiartDataset(Dataset):
    def __init__(self, root: Path, hr_size: int, n_crops: int = 1) -> None:
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.files = sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        if not self.files:
            raise FileNotFoundError(f"No images under {root}")
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
        train_test_eval: tuple[float, float, float] = (1.0, 0.0, 0.0),
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
        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, test_dataset, eval_dataset = random_split(
            dataset,
            self.train_test_eval,
            generator=generator,
        )
        return train_dataset, test_dataset, eval_dataset

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
