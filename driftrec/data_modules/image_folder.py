import abc
import warnings
from typing import Optional, Callable

from torch.utils.data import DataLoader
import torchvision.transforms as T
import pytorch_lightning as pl

from .shared import DataModuleRegistry, Img2DegradedImgDataset, SplitViaLabelImageFolder,\
    ensure_transform, JointRandomCrop


class MyDataModule(pl.LightningDataModule, abc.ABC):
    """
    The abstract class for a data module. Ensures that all data modules have a `get_num_channels` method.
    """
    @abc.abstractmethod
    def get_num_channels(self):
        pass


@DataModuleRegistry.register("imagefolder")
@DataModuleRegistry.register("celeba-hq")  # backwards compatibility for existing checkpoints
class ImageFolderDataModule(MyDataModule):
    @staticmethod
    def add_argparse_args(parser):
        group = parser.add_argument_group("ImageFolderDataModule")
        group.add_argument("--data_dir", type=str, help="Where is the root directory (subdirs train/, test/, valid/)?")
        group.add_argument("--batch_size", type=int, default=8, help="The batch size. Applies to all DataLoaders. 8 by default.")
        group.add_argument("--num_workers", type=int, default=4, help="The number of workers. Applies to all DataLoaders. 4 by default.")
        group.add_argument("--pin_memory", type=bool, default=True, help="Pin memory for GPU? True by default.")
        group.add_argument(
            "--random-resized-crop", type=int, required=False,
            help="Random-resize-crop to <val>x<val> square region *before* corruptions. Uses scale=(0.2, 1.0) [can be changed with --random-resized-crop-scale], ratio=(1.0, 1.0), antialias=True."
        )
        group.add_argument("--random-resized-crop-scale", type=float, nargs=2, default=(0.2, 1.0), help="Scale range for random-resized-crop. (0.2, 1.0) by default.")
        group.add_argument("--random-post-crop", type=int, required=False,
            help="Random-crop both x and y to <val>x<val> square region *after* corruptions. Same region is cropped from both images.")

        img2img_group = parser.add_argument_group("Img2DegradedImgDataset")
        Img2DegradedImgDataset.add_argparse_args(img2img_group)
        return parser

    def __init__(
        self, data_dir: str, batch_size: int = 8, num_workers: int = 4, pin_memory: bool = True,
        pre_transform: Optional[Callable] = None, post_transform: Optional[Callable] = None,
        random_resized_crop = None, random_resized_crop_scale = (0.2, 1.0), random_post_crop = None,
        **kwargs
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataloader_kwargs = dict(
            batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
        )
        self.kwargs = kwargs
        
        self.pre_transform = ensure_transform(pre_transform, default=T.ToTensor())
        self.post_transform = ensure_transform(post_transform, default=lambda x: x)
        self.random_resized_crop = random_resized_crop
        self.random_resized_crop_scale = random_resized_crop_scale
        self.random_post_crop = random_post_crop
        print(f"ImageFolderDataModule with random resized crop {random_resized_crop}, "
              f"random post crop {random_post_crop}")

        # pre- and post-cropping as common transforms for both x0 and y
        if random_resized_crop is not None and random_resized_crop > 0:
            common_pre_transform = T.RandomResizedCrop(
                (self.random_resized_crop, self.random_resized_crop),
                scale=self.random_resized_crop_scale, ratio=(1.0, 1.0), antialias=True
            )
        else:
            common_pre_transform = None

        if random_post_crop is not None and random_post_crop > 0:
            common_post_transform = JointRandomCrop((random_post_crop, random_post_crop))
        else:
            common_post_transform = None
        self.common_pre_transform = common_pre_transform
        self.common_post_transform = common_post_transform

        if post_transform is None and random_post_crop is None:
            warnings.warn("Using CenterCrop((256, 256)) as post_transform by default")
            post_transform = T.CenterCrop((256, 256))

    def setup(self, stage: str):
        shared_kwargs = dict(
            pre_transform=self.pre_transform, post_transform=self.post_transform,
            common_pre_transform=self.common_pre_transform, common_post_transform=self.common_post_transform,
            **self.kwargs
        )
        if stage == "fit":
            self.ds_train = Img2DegradedImgDataset(
                SplitViaLabelImageFolder(self.data_dir, which="train"), **shared_kwargs)
            self.ds_valid = Img2DegradedImgDataset(
                SplitViaLabelImageFolder(self.data_dir, which="valid"), **shared_kwargs)
        elif stage == "test":
            self.ds_test  = Img2DegradedImgDataset(
                SplitViaLabelImageFolder(self.data_dir, which="test"), **shared_kwargs)

    def train_dataloader(self):
        return DataLoader(self.ds_train, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.ds_valid, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.ds_test, shuffle=False, **self.dataloader_kwargs)

    def get_num_channels(self):
        return 3
