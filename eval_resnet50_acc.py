import os
import glob
import scipy as sp, scipy.io as spio
import pandas as pd
from PIL import Image
import tqdm

import torch
from torchvision.models import resnet50 as ResNet50, ResNet50_Weights


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, default='../ILSVRC2012_img_val/')
    parser.add_argument('--devkit-dir', type=str, default='../ILSVRC2012_devkit_t12/')
    parser.add_argument('--use-cuda', action='store_true', default=False)
    parser.add_argument('--no-save', action='store_true', default=False)
    args = parser.parse_args()

    print("Loading images from {}".format(args.img_dir))
    print("Loading devkit from {}".format(args.devkit_dir))

    meta = spio.loadmat(os.path.join(args.devkit_dir, 'data/meta.mat'))
    name_to_orig_id = {
        meta['synsets'][i]['words'][0][0].split(',')[0]: i
        for i in range(1000)
    }
    with open("imagenet_classes.txt", "r") as f:
        lines = [s.strip() for s in f.readlines()]
        categories_tv = {
            i+1: lines[i]
            for i in range(len(lines))
        }
    assert len(categories_tv) == 1000

    classes = pd.read_csv(
        os.path.join(args.devkit_dir, 'data/ILSVRC2012_validation_ground_truth.txt'),
        header=None,
        names=('class',),
    )
    classes.index = list(range(1, 50000+1))
    tv_to_ilsvrc2012 = lambda cls: name_to_orig_id[categories_tv[cls+1]]+1

    print("Loading ResNet50 model...")
    model = ResNet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = model.eval()
    if args.use_cuda:
        model = model.cuda()
    tf = ResNet50_Weights.IMAGENET1K_V2.transforms()

    print("Running evaluation...")

    results = []

    found_img_paths = glob.glob(os.path.join(args.img_dir, "ILSVRC2012_val_*"))
    if len(found_img_paths) == 0:
        raise ValueError("No images found in {}".format(args.img_dir))

    for found_img_path in tqdm.tqdm(found_img_paths):
        # parse i from path
        i = int(os.path.basename(found_img_path).split(".")[0].split("_")[-1])
        image = Image.open(found_img_path).convert('RGB')
        model_input = tf(image).unsqueeze(0)
        y_gt = classes.loc[i].iloc[0]
        if args.use_cuda:
            model_input = model_input.cuda()

        pred = model(model_input)
        probabilities = torch.nn.functional.softmax(pred[0], dim=0)
        _, top_classes = torch.topk(probabilities, 5, sorted=True)
        top_classes = [tv_to_ilsvrc2012(cls.cpu().item()) for cls in top_classes]

        results.append({
            "i": i,
            "y_gt": y_gt,
            "top_classes": top_classes,
            "top1_correct": y_gt == top_classes[0],
            "top5_correct": y_gt in top_classes,
        })

    results = pd.DataFrame(results)
    if not args.no_save:
        results.to_csv(os.path.join(args.img_dir, "resnet50_acc.csv"))
    
    print("Evaluated on {} images in total.".format(len(results)))
    print("Top-1 accuracy: {:.2f}%".format(100 * results.top1_correct.mean()))
    print("Top-5 accuracy: {:.2f}%".format(100 * results.top5_correct.mean()))


if __name__ == '__main__':
    main()