import os, sys
import pathlib
import glob
import argparse

from PIL import Image

import tqdm
import torch
import torchvision.transforms as T, torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset

from driftrec.model import DiscriminativeModel, ScoreModel
from driftrec.sampling.correctors import CorrectorRegistry
from driftrec.sampling.predictors import PredictorRegistry


class ImageGlobsDataset(Dataset):
    def __init__(self, image_globs, transform=None):
        image_paths = []
        for image_glob in image_globs:
            image_paths.extend(list(sorted(glob.glob(image_glob))))
        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        if self.transform is not None:
            x = self.transform(x)
        return x, image_path

    def __len__(self):
        return len(self.image_paths)



@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="Path to model checkpoint file.", required=True)
    parser.add_argument("--indir", type=str, help="Path to input directory (will consider the files '{args.indir}/<pattern>').", required=True)
    parser.add_argument("--patterns", type=str, nargs='+', help='Patterns for image files. "*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG" by default.',
                        default=["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG"])
    parser.add_argument("--outdir", type=str, help="Path to output directory. Enhanced files will have the same basename.", required=True)
    parser.add_argument("--batch_size", type=int, help="Batch size for simultaneous sampling. 1 by default. >1 will not work for images with different resolutions.", default=1)
    parser.add_argument("--ode", type=float, nargs=2, help="Use ODE reverse integrator with <rtol> <atol>.")
    parser.add_argument("--N", type=int, help="Number of sampling steps (for score models). 100 by default.", default=100)
    parser.add_argument("--predictor", type=str, choices=PredictorRegistry.get_all_names(),
        help="Predictor to use. 'euler_maruyama' by default.", default="euler_maruyama")
    parser.add_argument("--corrector", type=str, choices=CorrectorRegistry.get_all_names(),
        help="Corrector to use. 'none' by default.", default="none")
    parser.add_argument("--snr", type=float, help="SNR for corrector. 0.15 by default.", default=0.15)
    parser.add_argument("--initial-corrector-steps", type=int, default=0, help="Run corrector for this many steps at t=T before the rest of the reverse process. 0 by default.")
    parser.add_argument("--initial-corrector-snr", type=float, default=0.15, help="SNR for corrector at t=T. 0.15 by default.")
    parser.add_argument("--corrector-steps", type=int, default=1, help="Run corrector for this many steps in each iteration. 1 by default. (only has an effect when --corrector is provided)")
    parser.add_argument("--t-eps", type=float, help="Minimum time for sampling. 0.03 by default.", default=0.03)
    parser.add_argument("--t-omega", type=float, help="Maximum time for sampling. 1.0 by default.", default=1.0)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=0)

    args = parser.parse_args()

    # Determine the model kwargs
    model_kwargs = dict(
        data_dir='/dev/null', random_crop_size=None,
        transform=T.ToTensor(),
        batch_size=args.batch_size
    )

    # Find out what type of model to instantiate
    ckpt = torch.load(args.ckpt, map_location='cpu')
    with torch.cuda.device(args.cuda_device):
        # Ouch! We forgot to save this in the checkpoints so need to do some duck-typing here :(
        if ckpt['hyper_parameters'].get("discriminative_mode", None):
            model = DiscriminativeModel.load_from_checkpoint(args.ckpt, **model_kwargs)
            model = model.cuda()
            sampling_fn = lambda ys: model(ys)
        else:
            model = ScoreModel.load_from_checkpoint(args.ckpt, **model_kwargs)
            model = model.cuda()
            if args.ode:
                sampling_fn = lambda ys: model.get_ode_sampler(y=ys, rtol=args.ode[0], atol=args.ode[1], eps=args.t_eps)()
            else:
                sampling_fn = lambda ys: model.get_pc_sampler(
                    args.predictor, args.corrector, y=ys, N=args.N, snr=args.snr, use_tqdm=True, eps=args.t_eps,
                    corrector_steps=args.corrector_steps, initial_corrector_steps=args.initial_corrector_steps,
                    initial_corrector_snr=args.initial_corrector_snr,
                    **({'T': args.t_omega} if args.t_omega != 1 else {}),
                    )()[0]
        del ckpt

        print("Using EMA" if args.ema else "NOT using EMA")
        model.eval(no_ema=not args.ema)

        # The following code loads the image files (*.png) from the indir and saves the enhanced versions
        # to the outdir using the same basename filenames. Enhancing happens via sampling_fn.

        # Create the output directory if it doesn't exist
        outdir = pathlib.Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Create the dataset
        image_globs = [os.path.join(args.indir, pattern) for pattern in args.patterns]
        dataset = ImageGlobsDataset(image_globs, transform=T.ToTensor())
        if not len(dataset) >= 1:
            print(f"Did not find any images in the folder {args.indir} according to patterns {args.patterns}!")
            sys.exit(1)

        def collate_fn(batch):
            # Pad all images to the same size in the last two dimensions, using the maximum size across all images in the batch
            max_width = max(img.shape[-1] for img, _ in batch)
            max_height = max(img.shape[-2] for img, _ in batch)
            filenames = []
            padded_imgs = []
            cropslices = []  # The crop slices to use to crop the images back to their original size
            for img, filename in batch:
                # pad the last two dimensions so the original image is approximately centered
                pad_left = (max_width - img.shape[-1]) // 2
                pad_right = max_width - img.shape[-1] - pad_left
                pad_top = (max_height - img.shape[-2]) // 2
                pad_bottom = max_height - img.shape[-2] - pad_top
                padexpr = (pad_left, pad_top, pad_right, pad_bottom)
                cropslice = (slice(pad_top, pad_top + img.shape[-2]), slice(pad_left, pad_left + img.shape[-1]))
                padded_img = F.pad(img, padexpr, fill=0)
                padded_imgs.append(padded_img)
                filenames.append(filename)
                cropslices.append(cropslice)
            return torch.stack(padded_imgs, dim=0), filenames, cropslices

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

        nfes = []  # unused for now, but could save this as well

        # Iterate over the dataset
        i = 0
        with torch.no_grad():
            for k, batch in tqdm.tqdm(enumerate(dataloader), unit="batch", total=len(dataloader)):
                imgbatch, filenames, cropslices = batch
                # Filter out files that already exist
                filenames_out = [outdir / os.path.basename(filename) for filename in filenames]
                filtered_idxs = [i for i, filename in enumerate(filenames_out) if not os.path.isfile(filename)]
                if len(filtered_idxs) == 0:
                    continue
                imgbatch = imgbatch[filtered_idxs]
                filenames = [filenames[i] for i in filtered_idxs]
                cropslices = [cropslices[i] for i in filtered_idxs]
                filenames_out = [filenames_out[i] for i in filtered_idxs]

                # Enhance the images
                ys = imgbatch.cuda()
                enhanced = sampling_fn(ys)

                # Save the enhanced images
                for e, fout, cropslice in zip(enhanced, filenames_out, cropslices):
                    e = e[..., cropslice[0], cropslice[1]]  # crop back to original size
                    e = F.to_pil_image(e.clamp(min=0.0, max=1.0))
                    # Save the enhanced image
                    e.save(fout)
                    i += 1


if __name__ == '__main__':
    main()
