from abc import ABC
from enum import Enum
import torch
from torchvision.transforms import v2 as tv2

class PreprocessingStep(ABC):

    def apply(self, img : torch.Tensor):
        pass


class RandomCropAugment(PreprocessingStep):
    """Paper-style HR crop + spatial augmentation before any DCT transform (plan 2.2):
    random crop, horizontal/vertical flips, and 90-degree rotations."""

    def __init__(self, hr_size: int) -> None:
        self.hr_size = hr_size
        self.crop = tv2.Compose([
            tv2.RandomCrop(hr_size),
            tv2.RandomHorizontalFlip(),
            tv2.RandomVerticalFlip(),
            tv2.RandomApply([tv2.RandomRotation((90, 90))], p=0.5),
        ])

    def apply(self, img: torch.Tensor):
        return self.crop(img)


class BicubicDownsample(PreprocessingStep):
    """LR generation from cropped HR by bicubic downsampling with antialiasing (plan 2.2).

    Emits float32 in [0, 1] so the LR tensor is model-ready regardless of the order
    in which the train loop applies the preprocessing steps.
    """

    def __init__(self, scale: int = 4) -> None:
        self.scale = scale
        self.to_float = tv2.ToDtype(torch.float32, scale=True)

    def apply(self, img: torch.Tensor):
        height, width = img.shape[-2:]
        if height % self.scale or width % self.scale:
            raise ValueError(
                f"HR crop {height}x{width} not divisible by scale {self.scale}"
            )
        resize = tv2.Resize(
            (height // self.scale, width // self.scale),
            interpolation=tv2.InterpolationMode.BICUBIC,
            antialias=True,
        )
        return self.to_float(resize(img))



class RandomCropFlip(PreprocessingStep):

    def __init__(self, hr_size : int) -> None:
        self.hr_size = hr_size
        self.crop = tv2.Compose([
            tv2.RandomResizedCrop(hr_size, antialias=True),
            tv2.RandomHorizontalFlip(),
        ])

    def apply(self, img: torch.Tensor):

        return self.crop(img)

class SquareResize(PreprocessingStep):

    def __init__(self, size : int) -> None:
        self.resize = tv2.Resize((size,size),antialias=True)


    def apply(self, img: torch.Tensor):

        return self.resize(img)

class Int8To01Float(PreprocessingStep):

    scale = tv2.Compose([
        tv2.ToDtype(torch.float32, scale=True),
    ])

    def apply(self, img: torch.Tensor):

        return self.scale(img)


class PreprocessingStepList(Enum):
    RANDOM_CROP_FLIP = RandomCropFlip
    SQUARE_RESIZE = SquareResize
    INT8_TO_01_FLOAT = Int8To01Float
    RANDOM_CROP_AUGMENT = RandomCropAugment
    BICUBIC_DOWNSAMPLE = BicubicDownsample
