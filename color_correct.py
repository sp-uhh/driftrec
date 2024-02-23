import glob
import os
import warnings
import argparse

import tqdm
import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T
from PIL import Image

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt-dir", type=str, help="Path to target image directory (may be ground-truth, or corrupted images - this will just use the global image RGB means from each file to apply a shift)", required=True)
    parser.add_argument("--enh-dir", type=str, help="Path to enhanced image directory", required=True)
    parser.add_argument("--out-dir", type=str, help="Path to output image directory. Will be created if inexistent", required=True)
    args = parser.parse_args()

    tgt_dir, enh_dir = args.tgt_dir, args.enh_dir
    tgt_filelist = list(sorted(glob.glob(os.path.join(tgt_dir, "*.png"))))
    enh_filelist = list(sorted(glob.glob(os.path.join(enh_dir, "*.png"))))

    tgt_basenames = [os.path.basename(f) for f in tgt_filelist]
    enh_basenames = [os.path.basename(f) for f in enh_filelist]
    if not tgt_basenames == enh_basenames:
        diff = set(tgt_basenames) ^ set(enh_basenames)
        raise ValueError(f"File list in {enh_dir} does not match file list in {tgt_dir}, stopping! Diff (XOR): {diff}")

    os.makedirs(args.out_dir, exist_ok=True)

    for f in tqdm.tqdm(enh_filelist):
        f_tgt = os.path.join(tgt_dir, os.path.basename(f))

        img = Image.open(f)
        arr = np.array(img)
        rgb_means = arr.mean(axis=(0, 1))

        img_tgt = Image.open(f_tgt)
        arr_tgt = np.array(img_tgt)
        rgb_means_tgt = arr_tgt.mean(axis=(0, 1))

        rgb_shift = rgb_means_tgt - rgb_means
        print(f"File: {f}, RGB shift: {rgb_shift}")
        arr_corrected = np.clip(np.round(arr + rgb_shift), 0, 255).astype(np.uint8)
        img_corrected = Image.fromarray(arr_corrected)
        img_corrected.save(os.path.join(args.out_dir, os.path.basename(f)))

    print("===================================== Done!")


if __name__ == "__main__":
    main()
