from abc import ABC, abstractmethod

import torch

from super_resolution.generative_adversial_network import DiscriminatorNetwork, GeneratorNetwork
from super_resolution.models.dct_generator import DCTHybridGenerator, DCTResNetGenerator
from super_resolution.models.discriminator import SRGANDiscriminator
from super_resolution.models.packed import PackedBlockGenerator
from super_resolution.models.srresnet import LumaSRResNet, SRResNetGenerator, YCbCrSRResNet


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


class SRResNetModelConfig(GeneratorModelConfig):
    """A0 paper-faithful SRResNet, with RGB / YCbCr / luma-only input-output variants."""

    def __init__(
        self,
        width: int = 64,
        num_blocks: int = 16,
        scale: int = 4,
        normalize: bool = True,
        input_output: str = "rgb",
    ) -> None:
        if input_output not in ("rgb", "ycbcr", "y"):
            raise ValueError(f"input_output must be 'rgb', 'ycbcr' or 'y', got {input_output}")
        self.width = width
        self.num_blocks = num_blocks
        self.scale = scale
        self.normalize = normalize
        self.input_output = input_output

    def build(self) -> torch.nn.Module:
        if self.input_output == "ycbcr":
            return YCbCrSRResNet(self.width, self.num_blocks, self.scale)
        if self.input_output == "y":
            return LumaSRResNet(self.width, self.num_blocks, self.scale)
        return SRResNetGenerator(3, 3, self.width, self.num_blocks, self.scale, self.normalize)


class PackedBlockModelConfig(GeneratorModelConfig):
    """A2 packed-block control: pixel unshuffle, no DCT."""

    def __init__(
        self,
        width: int = 64,
        num_blocks: int = 16,
        scale: int = 4,
        block_size: int = 8,
    ) -> None:
        self.width = width
        self.num_blocks = num_blocks
        self.scale = scale
        self.block_size = block_size

    def build(self) -> PackedBlockGenerator:
        return PackedBlockGenerator(3, self.width, self.num_blocks, self.scale, self.block_size)


class DCTGeneratorModelConfig(GeneratorModelConfig):
    """A3 pure DCT-domain generator."""

    def __init__(
        self,
        width: int = 64,
        num_blocks: int = 16,
        scale: int = 4,
        block_size: int = 8,
    ) -> None:
        self.width = width
        self.num_blocks = num_blocks
        self.scale = scale
        self.block_size = block_size

    def build(self) -> DCTResNetGenerator:
        return DCTResNetGenerator(3, self.width, self.num_blocks, self.scale, self.block_size)


class DCTHybridModelConfig(GeneratorModelConfig):
    """A4 hybrid DCT generator with lightweight spatial refinement tail."""

    def __init__(
        self,
        dct_width: int = 64,
        dct_num_blocks: int = 16,
        spatial_width: int = 32,
        n_spatial_blocks: int = 2,
        scale: int = 4,
        block_size: int = 8,
    ) -> None:
        self.dct_width = dct_width
        self.dct_num_blocks = dct_num_blocks
        self.spatial_width = spatial_width
        self.n_spatial_blocks = n_spatial_blocks
        self.scale = scale
        self.block_size = block_size

    def build(self) -> DCTHybridGenerator:
        return DCTHybridGenerator(
            3, self.dct_width, self.dct_num_blocks, self.spatial_width,
            self.n_spatial_blocks, self.scale, self.block_size,
        )


class SRGANPaperDiscriminatorConfig(DiscriminatorModelConfig):
    """A5 paper-style discriminator (dense layers, 512x6x6 for 96x96 input)."""

    def __init__(self, img_size: tuple[int, int] = (96, 96), feature_width: int = 64) -> None:
        self.img_size = img_size
        self.feature_width = feature_width

    def build(self) -> SRGANDiscriminator:
        return SRGANDiscriminator(3, self.img_size, self.feature_width)


class GeneratorModel:
    def __init__(self, model_config: GeneratorModelConfig, optimizer_config: OptimizerConfig) -> None:
        self.model_config = model_config
        self.optimizer_config = optimizer_config


class DiscriminatorModel:
    def __init__(self, model_config: DiscriminatorModelConfig, optimizer_config: OptimizerConfig) -> None:
        self.model_config = model_config
        self.optimizer_config = optimizer_config
