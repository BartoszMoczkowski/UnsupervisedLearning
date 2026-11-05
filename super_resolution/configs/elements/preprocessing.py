from abc import ABC
from enum import Enum
import torch
from torchvision.transforms import v2 as tv2

class PreprocessingStep(ABC):

    def apply(self, img : torch.Tensor):
        pass



class RandomCropFlip(PreprocessingStep):

    def __init__(self, hr_size : int) -> None:
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
