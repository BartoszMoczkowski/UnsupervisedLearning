import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2 as tv2

PathInput = str | Path
RootInput = PathInput | Iterable[PathInput]


def _normalize_roots(roots: RootInput) -> list[Path]:
    if isinstance(roots, (str, Path)):
        roots = [roots]
    normalized = []
    for root in roots:
        path = Path(root)
        if path not in normalized:
            normalized.append(path)
    return normalized


def _find_image_files(root: Path, extensions: set[str]) -> list[Path]:
    """os.walk-based scan: much faster than pathlib.rglob on large or network mounts."""
    files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in extensions:
                files.append(path)
    return sorted(files)


def _split_name(path: Path) -> str | None:
    match = re.match(r"^(train|test|eval)(?:_|$)", path.name.lower())
    return match.group(1) if match else None


def _discover_split_roots(roots: list[Path]) -> dict[str, list[Path]] | None:
    """Find extractor split folders directly or one resolution level below each root."""
    split_roots: dict[str, list[Path]] = {name: [] for name in ("train", "test", "eval")}
    for root in roots:
        candidates = [root]
        if _split_name(root) is not None and root.parent.is_dir():
            # A config pointed at train_98 should still find sibling test/eval dirs.
            candidates.extend(sorted(path for path in root.parent.iterdir() if path.is_dir()))
        elif root.is_dir():
            children = sorted(path for path in root.iterdir() if path.is_dir())
            candidates.extend(children)
            for child in children:
                if _split_name(child) is None:
                    candidates.extend(
                        sorted(path for path in child.iterdir() if path.is_dir())
                    )
        for candidate in candidates:
            split = _split_name(candidate)
            if split is not None and candidate not in split_roots[split]:
                split_roots[split].append(candidate)
    return split_roots if any(split_roots.values()) else None


class WikiartDataset(Dataset):
    def __init__(
        self, root: RootInput, hr_size: int, n_crops: int = 1, allow_empty: bool = False
    ) -> None:
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.roots = _normalize_roots(root)
        self.files = sorted(
            path
            for root_path in self.roots
            for path in _find_image_files(root_path, extensions)
        )
        if not self.files and not allow_empty:
            roots_text = ", ".join(str(path) for path in self.roots) or "<no directories>"
            raise FileNotFoundError(
                f"No images under {roots_text}. This dataset expects decoded image files "
                f"({sorted(extensions)}), not parquet archives - extract them first "
                f"or point the config at the extracted directory."
            )
        if self.roots or not allow_empty:
            print(
                f"WikiartDataset: found {len(self.files)} images under "
                + ", ".join(str(path) for path in self.roots)
            )
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
        root: RootInput,
        hr_size: int,
        train_test_eval: tuple[float, float, float] = (0.98, 0.01, 0.01),
        seed: int = 43,
        batch_size: int = 16,
        n_crops: int = 1,
        num_workers: int = 4,
    ) -> None:
        self.roots = _normalize_roots(root)
        # Keep the original single-Path shape for backwards-compatible configs.
        self.root = self.roots[0] if isinstance(root, (str, Path)) else self.roots
        self.hr_size = hr_size
        self.train_test_eval = train_test_eval
        self.seed = seed
        self.batch_size = batch_size
        self.n_crops = n_crops
        self.num_workers = num_workers

    def load_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        # Older pickled stage configs only have root, not the normalized roots list.
        roots = getattr(self, "roots", _normalize_roots(self.root))
        split_roots = _discover_split_roots(roots)
        if split_roots is not None:
            if not split_roots["train"]:
                raise FileNotFoundError(
                    "Pre-split dataset directories were found, but no train directory "
                    f"was present under {roots}"
                )
            return tuple(
                WikiartDataset(
                    split_roots[split],
                    self.hr_size,
                    self.n_crops,
                    allow_empty=split != "train",
                )
                for split in ("train", "test", "eval")
            )  # type: ignore[return-value]

        # Backwards compatibility: a directory of unsplit images is still split
        # in memory using the original seeded random_split behavior.
        dataset = WikiartDataset(roots, self.hr_size, self.n_crops)
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
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
