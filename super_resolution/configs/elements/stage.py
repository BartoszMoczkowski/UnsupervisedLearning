import hashlib
import pickle
from pathlib import Path
from typing import NotRequired, TypedDict

import torch

from super_resolution.configs.elements.dataset import DatasetConfig
from super_resolution.configs.elements.loss import DiscriminatorLoss, GeneratorLoss
from super_resolution.configs.elements.model import DiscriminatorModel, GeneratorModel
from super_resolution.configs.elements.preprocessing import PreprocessingStep

EXPERIMENTS_PATH = Path("./experiments_data")


def serializable_config(value):
    """Convert a config object to plain Python data stable across processes.

    torch/torchvision modules are reduced to their class name: their internal pickle
    bytes are not stable across processes (and can embed hook state), which would
    otherwise make Stage.compute_id() non-deterministic.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable_config(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [serializable_config(item) for item in value]
    if isinstance(value, torch.nn.Module):
        return value.__class__.__name__
    if hasattr(value, "__dict__"):
        config = {key: serializable_config(item) for key, item in sorted(value.__dict__.items())}
        config["type"] = value.__class__.__name__
        return config
    return str(value)


class StageState(TypedDict):
    completed: bool
    n_epochs_completed: int
    n_seen_images: int
    generator: dict
    discriminator: NotRequired[dict]
    generator_optimizer: dict
    discriminator_optimizer: NotRequired[dict]


class Stage:
    def __init__(
        self,
        human_readable_name: str,
        dataset: DatasetConfig,
        generator_model: GeneratorModel,
        generator_loss: GeneratorLoss,
        n_seen_images: int,
        general_preprocessing: PreprocessingStep | None = None,
        hr_preprocessing: PreprocessingStep | None = None,
        lr_preprocessing: PreprocessingStep | None = None,
        discriminator_model: DiscriminatorModel | None = None,
        discriminator_loss: DiscriminatorLoss | None = None,
    ) -> None:
        self.human_readable_name = human_readable_name
        self.dataset = dataset
        self.general_preprocessing = general_preprocessing
        self.hr_preprocessing = hr_preprocessing
        self.lr_preprocessing = lr_preprocessing
        self.generator_model = generator_model
        self.generator_loss = generator_loss
        self.discriminator_model = discriminator_model
        self.discriminator_loss = discriminator_loss
        self.n_seen_images = n_seen_images

    def compute_id(self) -> str:
        config = self.__dict__.copy()
        del config["human_readable_name"]
        del config["n_seen_images"]
        config_hash = hashlib.sha256(pickle.dumps(serializable_config(config))).hexdigest()[:16]
        return f"{self.human_readable_name}_{config_hash}"

    def data_path(self) -> Path:
        return EXPERIMENTS_PATH / Path(self.compute_id())

    def config_data_path(self) -> Path:
        return self.data_path() / Path(f"{self.human_readable_name}_config.pkl")

    def state_data_path(self) -> Path:
        return self.data_path() / Path(f"{self.human_readable_name}_state.pt")

    def save_stage_config(self):
        self.data_path().mkdir(parents=True, exist_ok=True)
        with open(self.config_data_path(), "wb") as file:
            pickle.dump(self, file)

    def save_stage_results(self, stage_state: StageState):
        self.data_path().mkdir(parents=True, exist_ok=True)
        torch.save(stage_state, self.state_data_path())

    @classmethod
    def from_stage_config(cls, path: Path) -> "Stage":
        with open(path, "rb") as file:
            return pickle.load(file)

    def load_state(self) -> StageState:
        return torch.load(self.state_data_path(), map_location="cpu", weights_only=True)


class Experiment:
    def __init__(self, name: str, stages: list[Stage]) -> None:
        self.name = name
        self.stages = stages
