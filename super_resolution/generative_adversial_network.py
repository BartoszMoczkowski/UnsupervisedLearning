import torch
import torch.nn as nn


class ResidualBock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, main_x: torch.Tensor):
        return self.layers(main_x)


class GeneratorNetwork(nn.Module):
    def __init__(
        self,
        channels_input: int,
        channels_pre: int,
        channels_residual: int,
        channels_tail: int,
        n_residual_blocks: int = 8,
        n_tail_blocks: int = 2,
    ):
        super().__init__()
        if channels_pre != channels_residual:
            raise ValueError("channels_pre and channels_residual must match")
        if channels_tail % 4:
            raise ValueError("channels_tail must be divisible by 4")

        self.pre_residual = nn.Sequential(
            nn.Conv2d(channels_input, channels_pre, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.ModuleList(
            ResidualBock(channels_residual, channels_residual)
            for _ in range(n_residual_blocks)
        )
        self.post_residual = nn.Sequential(
            nn.Conv2d(channels_residual, channels_residual, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_residual),
        )

        tail = []
        in_channels = channels_residual
        for _ in range(n_tail_blocks):
            tail.extend(
                (
                    nn.Conv2d(in_channels, channels_tail, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = channels_tail // 4
        self.tail = nn.Sequential(*tail)
        self.out_layer = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        pre_residual_x = self.pre_residual(x)
        partial_residual_x = pre_residual_x
        for residual_block in self.residual:
            partial_residual_x = residual_block(partial_residual_x) + partial_residual_x
        tail_x = self.post_residual(partial_residual_x) + pre_residual_x
        return self.out_layer(self.tail(tail_x))


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, main_x: torch.Tensor):
        return self.layers(main_x)


class DiscriminatorNetwork(nn.Module):
    def __init__(
        self,
        channels_input: int,
        channels_pre: int,
        channels_blocks: list[int],
        strides_blocks: list[int],
        img_size: tuple[int, int],
        n_dense_1: int = 512,
    ):
        super().__init__()
        if len(channels_blocks) != len(strides_blocks):
            raise ValueError("channels_blocks and strides_blocks must have equal lengths")

        self.pre_block = nn.Sequential(
            nn.Conv2d(channels_input, channels_pre, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        blocks = []
        prev_channels = channels_pre
        current_width, current_height = img_size
        channels = channels_pre
        for channels, stride in zip(channels_blocks, strides_blocks):
            current_width = (current_width - 1) // stride + 1
            current_height = (current_height - 1) // stride + 1
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        prev_channels,
                        channels,
                        kernel_size=3,
                        padding=1,
                        stride=stride,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            prev_channels = channels
        self.blocks = nn.ModuleList(blocks)

        flattened_dim_n = current_width * current_height * channels
        self.out_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim_n, n_dense_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_dense_1, 1),
        )

    def forward(self, x: torch.Tensor):
        block_x = self.pre_block(x)
        for block in self.blocks:
            block_x = block(block_x)
        return self.out_layers(block_x)
