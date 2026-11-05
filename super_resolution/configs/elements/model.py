from abc import ABC, abstractmethod

import torch

from super_resolution.generative_adversial_network import DiscriminatorNetwork, GeneratorNetwork


class GeneratorModelConfig(ABC):
    @abstractmethod
    def build(self) -> torch.nn.Module:
        pass


class DiscriminatorModelConfig(ABC):
    @abstractmethod
    def build(self) -> torch.nn.Module:
        pass


class OptimizerConfig(ABC):
    @abstractmethod
    def build(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        pass


class SRGeneratorConfig(GeneratorModelConfig):
    def __init__(
        self,
        channels_input: int = 3,
        channels_pre: int = 16,
        channels_residual: int = 16,
        channels_tail: int = 32,
        n_residual_blocks: int = 8,
        n_tail_blocks: int = 2,
    ) -> None:
        self.channels_input = channels_input
        self.channels_pre = channels_pre
        self.channels_residual = channels_residual
        self.channels_tail = channels_tail
        self.n_residual_blocks = n_residual_blocks
        self.n_tail_blocks = n_tail_blocks

    def build(self) -> GeneratorNetwork:
        return GeneratorNetwork(
            self.channels_input,
            self.channels_pre,
            self.channels_residual,
            self.channels_tail,
            self.n_residual_blocks,
            self.n_tail_blocks,
        )


class SRDiscriminatorConfig(DiscriminatorModelConfig):
    def __init__(
        self,
        img_size: tuple[int, int],
        channels_input: int = 3,
        channels_pre: int = 16,
        channels_blocks: list[int] | None = None,
        strides_blocks: list[int] | None = None,
        n_dense_1: int = 512,
    ) -> None:
        self.img_size = img_size
        self.channels_input = channels_input
        self.channels_pre = channels_pre
        self.channels_blocks = channels_blocks if channels_blocks is not None else [4, 4]
        self.strides_blocks = strides_blocks if strides_blocks is not None else [4, 4]
        self.n_dense_1 = n_dense_1

    def build(self) -> DiscriminatorNetwork:
        return DiscriminatorNetwork(
            self.channels_input,
            self.channels_pre,
            self.channels_blocks,
            self.strides_blocks,
            self.img_size,
            self.n_dense_1,
        )


class AdamOptimizerConfig(OptimizerConfig):
    def __init__(self, lr: float = 1e-4, betas: tuple[float, float] = (0.9, 0.999)) -> None:
        self.lr = lr
        self.betas = betas

    def build(self, model: torch.nn.Module) -> torch.optim.Adam:
        return torch.optim.Adam(model.parameters(), lr=self.lr, betas=self.betas)


class GeneratorModel:
    def __init__(self, model_config: GeneratorModelConfig, optimizer_config: OptimizerConfig) -> None:
        self.model_config = model_config
        self.optimizer_config = optimizer_config


class DiscriminatorModel:
    def __init__(self, model_config: DiscriminatorModelConfig, optimizer_config: OptimizerConfig) -> None:
        self.model_config = model_config
        self.optimizer_config = optimizer_config
