import os
import sys
import glob
import argparse

import tqdm
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torchjpeg.metrics import blocking_effect_factor


def get_bef(img):
    return blocking_effect_factor(img).item()

def prep_torchjpeg_input(x):
    return torch.from_numpy(x).moveaxis(2, 0).unsqueeze(0).cuda() / 255


def get_df(dir):
    results = { "file": [], "bef": [] }
    x_files = list(sorted(glob.glob(os.path.join(dir, "*.png"))))
    for j, x_file in tqdm.tqdm(enumerate(x_files), total=len(x_files), unit="file"):
        x = np.asarray(Image.open(x_file))
        bef = get_bef(prep_torchjpeg_input(x))
        results["file"].append(os.path.basename(x_file))
        results["bef"].append(bef)
        print(f"{os.path.basename(x_file)}: BEF={bef:.5f}")

    df = pd.DataFrame(results)
    return df


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Path to directory", required=True)
    args = parser.parse_args()

    pickle_file = os.path.join(args.dir, "bef.pkl")
    if os.path.isfile(pickle_file):
        print(f"BEF already calculated and saved to {pickle_file}. Please remove the file if you want to recalculate")
        sys.exit(0)
        pass

    print(f"Calculating BEF for {args.dir} Will save to {pickle_file}.")
    df = get_df(args.dir)
    if df.empty:
        print("No files found in the ground truth directory!")
        sys.exit(1)
    df.to_pickle(pickle_file)

if __name__ == "__main__":
    main()