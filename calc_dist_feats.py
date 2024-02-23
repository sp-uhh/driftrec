import glob
import os
import warnings
import argparse

import tqdm
import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T

import PIL
from piq import FID, KID
from piq.feature_extractors import InceptionV3


@torch.no_grad()
def compute_feats(
        filenames,
        feature_extractor: torch.nn.Module = None,
        device: str = 'cuda') -> torch.Tensor:
    r"""Generate low-dimensional image descriptors
    Args:
        loader: Should return dict with key `images` in it
        feature_extractor: model used to generate image features, if None use `InceptionNetV3` model.
            Model should return a list with features from one of the network layers.
        out_features: size of `feature_extractor` output
        device: Device on which to compute inference of the model
    """
    if feature_extractor is None:
        print('WARNING: default feature extractor (InceptionNet V3) is used.')
        feature_extractor = InceptionV3()
    else:
        assert isinstance(feature_extractor, torch.nn.Module), \
            f"Feature extractor must be PyTorch module. Got {type(feature_extractor)}"
    feature_extractor.to(device)
    feature_extractor.eval()

    total_feats = []
    for filename in tqdm.tqdm(filenames):
        try:
            images = T.ToTensor()(np.asarray(PIL.Image.open(filename))).unsqueeze(0)
        except PIL.UnidentifiedImageError:
            warnings.warn(f"Ignoring image {filename}, could not be read")
            continue
        images = images.float().to(device)

        # Get features
        features = feature_extractor(images)
        assert len(features) == 1, \
            f"feature_encoder must return list with features from one layer. Got {len(features)}"
        total_feats.append(features[0].view(1, -1))

    return torch.cat(total_feats, dim=0)


def get_or_create_feats(filelist, output_file):
    if os.path.isfile(output_file):
        return np.load(output_file)
    else:
        print(f"Processing for {output_file}")

    extr = InceptionV3()
    feats = compute_feats(filelist, extr)
    saved_feats = feats.cpu().numpy()
    np.save(output_file, saved_feats)
    return saved_feats


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=str, help="Path to ground truth directory", required=True)
    parser.add_argument("--enh-dir", type=str, help="Path to corrupted/enhanced directory", required=True)
    parser.add_argument("--cuda-device", type=int, help="CUDA device to use", default=0)
    args = parser.parse_args()

    with torch.cuda.device(args.cuda_device):
        gt_dir, enh_dir = args.gt_dir, args.enh_dir
        gt_feats_file = os.path.join(gt_dir, "feats.npy")
        enh_feats_file = os.path.join(enh_dir, "feats.npy")
        gt_filelist = list(sorted(glob.glob(os.path.join(gt_dir, "*.png"))))
        enh_filelist = list(sorted(glob.glob(os.path.join(enh_dir, "*.png"))))

        gt_basenames = [os.path.basename(f) for f in gt_filelist]
        enh_basenames = [os.path.basename(f) for f in enh_filelist]
        if not gt_basenames == enh_basenames:
            diff = set(gt_basenames) ^ set(enh_basenames)
            raise ValueError(f"File list in {enh_dir} does not match file list in {gt_dir}, stopping! Diff (XOR): {diff}")

        print(f"Loading or calculating ground-truth features from dir {gt_dir}...")
        gt_feats = torch.from_numpy(get_or_create_feats(gt_filelist, gt_feats_file))

        print(f"Calculating features for dirs {enh_dir}, using {gt_dir} as ground truth dir. "
                f"Will save feats to <dir>/feats.npy, and save KID/FID scores to <dir>/scores.csv.")
        scores_file = os.path.join(enh_dir, "scores.csv")
        enh_feats = torch.from_numpy(get_or_create_feats(enh_filelist, enh_feats_file))
        fid = FID().compute_metric(gt_feats, enh_feats).item()
        kid = KID().compute_metric(gt_feats, enh_feats).item()
        print(f"FID: {fid}, KID: {kid}")
        pd.DataFrame({"FID": [fid], "KID": [kid]}).to_csv(scores_file, index=False)
        print("===================================== Done!")


if __name__ == "__main__":
    main()
