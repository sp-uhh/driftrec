from typing import Tuple, Union
import io
import warnings

import numpy as np
import torch
import torchvision.transforms as T
import PIL
from PIL import Image

from driftrec.util.params import required_length


class JPEGCompression:
    def __repr__(self):
        return f"JPEGCompression({self.quality})"

    def __init__(self, quality: int):
        assert 0 <= quality <= 100, "Invalid 'quality': outside of [0, 100]"
        self.quality = quality

    def __call__(self, img: Union[torch.Tensor, PIL.Image.Image]):
        # Convert tensor to PIL image, write a compressed JPEG file of it into a BytesIO buffer,
        # then load from this buffer back into a new PIL image, and turn that back into a tensor
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)
        else:
            assert isinstance(img, PIL.Image.Image), "Expected 'img' to be a PIL image or a tensor"

        out = io.BytesIO()
        img.save(out, format='jpeg', quality=self.quality)
        out.seek(0)
        img = PIL.Image.open(out)
        tensor = T.ToTensor()(img)
        return tensor


class RandomJPEGCompression:
    def __repr__(self):
        return f"RandomJPEGCompression(qualities=({self.qualities[0]},{self.qualities[1]-1}))"

    def __init__(self, qualities: Tuple[int, int]):
        assert 0 <= qualities[0] <= 100 and 0 <= qualities[1] <= 100, "Invalid 'qualities' outside of [0, 100]"
        self.qualities = (qualities[0], qualities[1]+1)
        self._quality_history = []

    def __call__(self, img: torch.Tensor):
        quality = np.random.randint(*self.qualities)
        self._quality_history.append(quality)
        return JPEGCompression(quality)(img)


class DownUpPIL:
    def __repr__(self):
        return f"DownUpPIL(factors={self.factors}, down={self.down}, up={self.up})"

    def __init__(self, factors, down, up, inbetween=None):
        self.factors = factors
        self.down = down
        self.up = up
        self.down_enum = getattr(Image.Resampling, self.down.upper())
        self.up_enum = getattr(Image.Resampling, self.up.upper())
        self.inbetween = inbetween

    @torch.no_grad()
    def __call__(self, tensor: torch.Tensor):
        factor = np.random.choice(self.factors)
        img = T.ToPILImage()(tensor)
        lo = img.resize((img.width//factor, img.height//factor), resample=self.down_enum)
        if self.inbetween is not None:
            lo = self.inbetween(lo)
            if isinstance(lo, torch.Tensor):
                lo = T.ToPILImage()(lo)
        hi = lo.resize((img.width, img.height), resample=self.up_enum)
        tensor = T.ToTensor()(hi)
        return tensor


class RandomDegradations:
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--D-jpeg", type=int, nargs=2,
                            help="An interval of random JPEG quality values (inclusive on both ends).")
        parser.add_argument("--D-downup-PIL", type=str, nargs='+', action=required_length(3),
            help="<down> <up> <factors> [<factors> ...] - Use Pillow to downsample by random <factor> from <factors> using <down> strategy, then upsample to original resolution using <up> strategy. "
                " Valid strategies: 'nearest', 'box', 'bilinear', 'hamming', 'bicubic', 'lanczos'. 'lanczos' for both resamplings is recommended."
        )
        return parser

    def __init__(self, D_jpeg=None, D_downup_PIL=None, **unused_kwargs):
        corruptions = []
        if D_jpeg:
            corruptions.append(RandomJPEGCompression(tuple(D_jpeg)))
        if D_downup_PIL is not None:
            down, up, *factors = D_downup_PIL
            factors = [int(factor) for factor in factors]
            corruptions.append(DownUpPIL(factors=factors, down=down, up=up))
        if len(corruptions) == 0:
            warnings.warn("Degradations object was constructed with no corruptions whatsoever!")

        self.corruptions = corruptions
        self.transform = T.RandomChoice(self.corruptions)

    def __call__(self, img: torch.Tensor):
        return self.transform(img)
