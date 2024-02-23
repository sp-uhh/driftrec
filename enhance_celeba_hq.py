import os, sys
import pathlib
import math
import argparse

import numpy as np
import tqdm
import torch
from torchvision.transforms import functional as F

from driftrec.data_modules.degradations import RandomDegradations
from driftrec.model import DiscriminativeModel, ScoreModel
from driftrec.sampling.correctors import CorrectorRegistry
from driftrec.sampling.predictors import PredictorRegistry
from driftrec.util.params import get_argparse_groups


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="Path to model checkpoint file.", required=True)
    parser.add_argument("--indir", type=str, help="Path to input directory (will consider the files '{args.indir}/test/*.jpg').", required=True)
    parser.add_argument("--outdir", type=str, help="Path to output directory.", required=True)
    parser.add_argument("--batch_size", type=int, help="Batch size for simultaneous sampling. 6 by default.", default=6)
    parser.add_argument("--seed", type=int, help="Random seed to use (for image corruptions). 0 by default", default=0)
    parser.add_argument("--ode", type=float, nargs=2, help="Use ODE reverse integrator with <rtol> <atol>.")
    parser.add_argument("--N", type=int, help="Number of sampling steps (for score models). 100 by default.", default=100)
    parser.add_argument("--predictor", type=str, choices=PredictorRegistry.get_all_names(),
        help="Predictor to use. 'euler_maruyama' by default.", default="euler_maruyama")
    parser.add_argument("--corrector", type=str, choices=CorrectorRegistry.get_all_names(),
        help="Corrector to use. 'none' by default.", default="none")
    parser.add_argument("--snr", type=float, help="SNR for corrector. 0.15 by default.", default=0.15)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--max-k", type=int, help="When passed, will only enhance first `max_k` images.", required=False)
    parser.add_argument("--mu_sampler", action="store_true", help="Use the custom mu-sampler.")

    degradations_group = parser.add_argument_group("RandomDegradations",
        description="When passing at least one of these, the random degradations from the model checkpoint "
            "will be ignored, and the settings passed here will be used instead.")
    RandomDegradations.add_argparse_args(degradations_group)
    args = parser.parse_args()

    # Seed numpy globally to affect the image corruptions - not optimal, but should hopefull work
    # Note that the file order is fixed by a sorted() call inside the underlying ImageFolder dataset class
    np.random.seed(args.seed)

    # Determine the model kwargs
    model_kwargs = dict(data_dir=args.indir)
    # If any RandomDegradations args were passed, use them to override the ones from the checkpoint,
    # otherwise load the ones from the checkpoint.
    arg_groups = get_argparse_groups(parser, args)
    degr_kwargs = arg_groups["RandomDegradations"]
    if any(degr_kwargs.values()):
        print("Overriding random degradations, ignoring those stored in the checkpoint...")
        model_kwargs = {**model_kwargs, **degr_kwargs}

    # Find out what type of model to instantiate
    ckpt = torch.load(args.ckpt, map_location='cuda:0')
    # Ouch! We forgot to save this in the checkpoints so need to do some duck-typing here :(
    if ckpt['hyper_parameters'].get("discriminative_mode", None):
        model = DiscriminativeModel.load_from_checkpoint(args.ckpt, **model_kwargs)
        sampling_fn = lambda ys: model(ys)
    else:
        model = ScoreModel.load_from_checkpoint(args.ckpt, **model_kwargs)
        if args.ode is not None:
            sampling_fn = lambda ys: model.get_ode_sampler(y=ys, rtol=args.ode[0], atol=args.ode[1])()[0]
        elif args.mu_sampler:
            sampling_fn = lambda ys: model.get_mu_sampler(y=ys, N=args.N)()[0]
        elif model.soft:
            sampling_fn = lambda ys: model.get_soft_sampler(y=ys, N=args.N)()[0]
        else:
            sampling_fn = lambda ys: model.get_pc_sampler(
                args.predictor, args.corrector, y=ys, N=args.N, snr=args.snr, use_tqdm=True)()[0]

    model = model.cuda()
    print("Using EMA" if args.ema else "NOT using EMA")
    model.eval(no_ema=not args.ema)
    data_module = model.data_module
    data_module.setup("test")
    corruptions = data_module.ds_test.random_degradations.corruptions
    print("Used random degradations:", corruptions)
    data_loader = data_module.test_dataloader()

    k = 0  # globally unique increasing index of each data point
    nitems = len(data_loader)
    width = int(math.ceil(math.log10(nitems)))  # width of zero-padded filename

    # create the output directory
    pathlib.Path(args.outdir).mkdir(parents=False, exist_ok=True)

    max_k = args.max_k if args.max_k else 99999999999999
    for ibatch, batch in enumerate(tqdm.tqdm(iter(data_loader), unit="batch")):
        xs, ys = batch
        if os.path.exists(os.path.join(args.outdir, f"{ibatch*xs.shape[0]:0{width}}_x.png")):
            k += xs.shape[0]
            print(f"Skipped over batch {ibatch}, to image index {k}")
            continue

        xs, ys = xs.cuda(), ys.cuda()
        xhats = sampling_fn(ys)
        for x, y, xhat in zip(xs, ys, xhats):
            xhat = xhat.clamp(min=0.0, max=1.0)
            ix, iy, ixhat = (F.to_pil_image(tensor) for tensor in (x, y, xhat))
            ix.save(os.path.join(args.outdir, f"{k:0{width}}_x.png"))
            iy.save(os.path.join(args.outdir, f"{k:0{width}}_y.png"))
            ixhat.save(os.path.join(args.outdir, f"{k:0{width}}_xhat.png"))
            k += 1
            if k == max_k:
                print(f"Reached max_k {max_k}, quitting.")
                sys.exit(0)  # quit all loops


if __name__ == '__main__':
    main()