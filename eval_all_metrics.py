import os
import sys
import glob
import argparse

import tqdm
import numpy as np
import pandas as pd
import torch

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from lpips import LPIPS
from torchjpeg.metrics import psnrb


def get_ssim(true, estimate):
    return structural_similarity(
        true, estimate,
        data_range=255, channel_axis=2,
        win_size=9
    )

def get_psnr(true, estimate):
    return peak_signal_noise_ratio(true, estimate)

def get_logspec_mse(true, estimate):
    spectrum = np.fft.fft2(true)
    estimate_spectrum = np.fft.fft2(estimate)
    logdiff = np.log(np.abs(spectrum) + 1e-6) - np.log(np.abs(estimate_spectrum) + 1e-6)
    return np.mean(np.abs(logdiff) ** 2)

def prep_lpips_input(x):
    return (torch.from_numpy(x).moveaxis(2, 0).unsqueeze(0).cuda() / 255. - 0.5) * 2

def prep_torchjpeg_input(x):
    return torch.from_numpy(x).moveaxis(2, 0).unsqueeze(0).cuda() / 255


def get_df(gt_dir, enh_dir, lpips_instance):
    results = { "file": [], "psnr": [], "psnrb": [], "ssim": [], "lpips": [] } #, "logspec_mse_hat": [] }
    x_files = list(sorted(glob.glob(os.path.join(gt_dir, "*.png"))))
    assert all([os.path.isfile(os.path.join(enh_dir, os.path.basename(f))) for f in x_files]),\
        "Not all files exist in the enhanced dir!"
    for j, x_file in tqdm.tqdm(enumerate(x_files), total=len(x_files), unit="file"):
        
        xhat_file = os.path.join(enh_dir, os.path.basename(x_file))
        x, xhat = [np.asarray(Image.open(f)) for f in (x_file, xhat_file)]
        ssim_hat = get_ssim(x, xhat)
        psnr_hat = get_psnr(x, xhat)
        psnrb_hat = psnrb(prep_torchjpeg_input(xhat), prep_torchjpeg_input(x)).item()
        lpips_hat = lpips_instance(prep_lpips_input(x), prep_lpips_input(xhat)).item()
        results["file"].append(os.path.basename(x_file))
        results["psnr"].append(psnr_hat)
        results["psnrb"].append(psnrb_hat)
        results["ssim"].append(ssim_hat)
        results["lpips"].append(lpips_hat)
        print(f"{os.path.basename(x_file)}: PSNR={psnr_hat:.2f}, PSNRB={psnrb_hat:.2f}, SSIM={ssim_hat:.2f}, LPIPS={lpips_hat:.2f}")

    df = pd.DataFrame(results)
    return df


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=str, help="Path to ground truth directory", required=True)
    parser.add_argument("--enh-dir", type=str, help="Path to corrupted/enhanced directory", required=True)
    parser.add_argument("--cuda-device", type=int, help="CUDA device to use (for LPIPS)", default=0)
    args = parser.parse_args()

    pickle_file = os.path.join(args.enh_dir, "metrics.pkl")

    with torch.cuda.device(args.cuda_device):
        lpips_instance = LPIPS(net='alex').cuda()
        print(f"Calculating metrics (PSNR, PSNRB, SSIM, LPIPS) using {args.gt_dir} as ground truth dir, {args.enh_dir} as enhanced dir. Will save to {args.enh_dir}/metrics.pkl.")
        df = get_df(args.gt_dir, args.enh_dir, lpips_instance=lpips_instance)
        if df.empty:
            print("No files found in the ground truth directory!")
            sys.exit(1)
        df.to_pickle(pickle_file)

if __name__ == "__main__":
    main()
