import abc
import argparse
from typing import Optional, Callable, Tuple, Any

from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from driftrec.data_modules.degradations import RandomDegradations
from driftrec.util.registry import Registry

DataModuleRegistry = Registry("DataModule")


def ensure_transform(transform_or_none: Optional[Callable], default=None):
    """
    Ensures that the given object, which may be a Callable or None, is a valid transform,
    simply by replacing None values with a `default` transformation (no-op by default).
    """
    if transform_or_none is not None:
        return transform_or_none
    return default


class SplitViaLabelImageFolder(ImageFolder):
    """
    A subclass of torchvision.datasets.ImageFolder, implementing train/test/valid/... split via labels (subdirectories).
    Hacky but fairly convenient. Pass the argument `which` at construction to pick which subfolder it will see,
    e.g., `which="train"`, `which="test"`, `which="valid"`.
    """
    def __init__(self, root: str, which: str, *args, **kwargs):
        self.which = which
        super().__init__(root, *args, **kwargs)

    def find_classes(self, directory: str):
        # Limit found 'classes' to the one specified in `self.which`
        names, dicts = super().find_classes(directory)
        return ([self.which], {self.which: dicts[self.which]})


class Img2ImgDataset(Dataset):
    """
    A dataset that wraps an underlying dataset, replacing the paired values

        (data, label)

    with

        (left_transform(data), right_transform(data))

    Useful for turning (e.g.) a (image, label) dataset into a (f(image), g(image)) dataset.
    """
    def __init__(
        self, underlying_dataset: Dataset,
        left_transform: Optional[Callable] = None, right_transform: Optional[Callable] = None,
        common_pre_transform: Optional[Callable] = None, common_post_transform: Optional[Callable] = None,
        n_max: int = None,
        **kwargs
    ):
        self.underlying_dataset = underlying_dataset
        self.left_transform = ensure_transform(left_transform)
        self.right_transform = ensure_transform(right_transform)
        self.common_pre_transform = common_pre_transform
        self.common_post_transform = common_post_transform
        self.n_max = n_max

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        x, _ = self.underlying_dataset[index]   # ignore labels of underlying dataset
        x = self.common_pre_transform(x) if self.common_pre_transform is not None else x
        left = self.left_transform(x)
        right = self.right_transform(x)
        if self.common_post_transform is not None:
            left, right = self.common_post_transform(left, right)
        return left, right

    def __len__(self):
        n = len(self.underlying_dataset)
        if self.n_max is not None:
            n = min(n, self.n_max)
        return n

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser):
        parser.add_argument('--n-max', type=int, required=False,
            help="Pass to limit the number of data points in the dataset. Useful for quick debugging runs.")


class Img2DegradedImgDataset(Img2ImgDataset):
    """
    A special type of Img2ImgDataset that wraps an underlying dataset, replacing the paired values

        ( data, label )

    with

        ( post(pre(data)), post(random_degradation(pre(data)) )
    """
    def __init__(
        self,
        underlying_dataset: Dataset,
        pre_transform: Optional[Callable] = None,
        post_transform: Optional[Callable] = None,
        common_pre_transform: Optional[Callable] = None,
        common_post_transform: Optional[Callable] = None,
        **kwargs
    ):
        # intentionally no super().__init__() call here
        self.pre = ensure_transform(pre_transform)
        self.post = ensure_transform(post_transform)
        self.random_degradations = RandomDegradations(**kwargs)
        self.common_pre_transform = common_pre_transform
        left_transform = T.Compose([self.pre, self.post])
        right_transform = T.Compose([self.pre, self.random_degradations, self.post])
        super().__init__(underlying_dataset,
            left_transform=left_transform, right_transform=right_transform,
            common_pre_transform=common_pre_transform,
            common_post_transform=common_post_transform,
            **kwargs
        )

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser):
        Img2ImgDataset.add_argparse_args(parser)
        RandomDegradations.add_argparse_args(parser)  # Add args for the RandomDegradations object
        return parser


class JointRandomCrop(nn.Module):
    """
    Like torchvision.transforms.RandomCrop, but crops the same region from two images and returns both.
    """
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        super().__init__()
        self.crop = T.RandomCrop(size, padding, pad_if_needed, fill, padding_mode)

    def forward(self, left, right):
        i, j, h, w = self.crop.get_params(left, self.crop.size)
        return T.functional.crop(left, i, j, h, w), T.functional.crop(right, i, j, h, w)
