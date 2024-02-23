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
    parser.add_argument("--enh-dirs", type=str, nargs='+', help="Path to enhanced image directories to average samples from.", required=True)
    parser.add_argument("--out-dir", type=str, help="Path to output image directory. Will be created if inexistent", required=True)
    args = parser.parse_args()

    enh_dirs = args.enh_dirs
    # sanity check: does each directory have the same filelist?
    enh_filelist_0 = list(sorted(os.path.basename(f) for f in glob.glob(os.path.join(enh_dirs[0], "*.png"))))
    for enh_dir in enh_dirs[1:]:
        enh_filelist_i = list(sorted(os.path.basename(f) for f in glob.glob(os.path.join(enh_dir, "*.png"))))
        if not enh_filelist_0 == enh_filelist_i:
            diff = set(enh_filelist_i) ^ set(enh_filelist_0)
            raise ValueError(f"File list in {enh_dirs[0]} does not match file list in {enh_dir}, stopping! Diff (XOR): {diff}")

    os.makedirs(args.out_dir, exist_ok=True)
    # for each image in the filelist, average the samples from all directories
    for filename in tqdm.tqdm(enh_filelist_0):
        filename = os.path.basename(filename)
        # load all images
        enh_imgs = []
        for enh_dir in enh_dirs:
            enh_img = Image.open(os.path.join(enh_dir, filename))
            enh_imgs.append(np.array(enh_img))
        enh_imgs = np.stack(enh_imgs, axis=0)
        enh_imgs = torch.from_numpy(enh_imgs).permute(0, 3, 1, 2).float()
        # average
        enh_img = torch.mean(enh_imgs, dim=0, keepdim=True)
        enh_img = enh_img.permute(0, 2, 3, 1).squeeze(0).numpy()
        enh_img = Image.fromarray(enh_img.astype(np.uint8))
        enh_img.save(os.path.join(args.out_dir, filename))

    print("===================================== Done!")


if __name__ == "__main__":
    main()
