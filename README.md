# Setup

Create and activate a new Python 3.10 virtual environment, for example via

```bash
conda create -n driftrec python=3.10
conda activate driftrec
```

Install the requirements via:

```bash
pip install -r requirements.txt
```
and to ensure that PyTorch uses the CUDA version you have on your system, you can here optionally pass an `--extra-index-url` parameter such as `--extra-index-url https://download.pytorch.org/whl/cu118`, see [https://pytorch.org/](https://pytorch.org/).


# Dataset preparation

* Our scripts expect each dataset to have a `train/`, `valid/` and `test/` subfolder, each containing image files with no additional subfolders or labels.

* For CelebA-HQ 256, please download the CelebA-HQ dataset and resize it to 256x256. Then you can generate the same train/validation/test split as we used with:
```python
original_root = "<your_celeba_hq_256_folder>"
original_image_glob = original_root + "*.jpg"
imgfiles = np.array(list(sorted(glob.glob(original_image_glob))))
N = len(imgfiles)
print(f"Found {N} image files.")  # for CelebA-HQ, should be 30,000.

seed = 102405
np.random.seed(seed)
np.random.shuffle(imgfiles)

idxs = np.arange(N)
id_train = int(len(idxs) * 0.8)              # Train 80%
id_valid = int(len(idxs) * (0.8 + 0.05))     # Valid 5%, Test 15%
train, valid, test = np.split(idxs, (id_train, id_valid))
print(len(train), len(valid), len(test))

datasets = { "train": imgfiles[train], "valid": imgfiles[valid], "test": imgfiles[test] }
```
and copy/move all files in each respective subset to a matching subfolder (`train/`, `valid/` and `test/`).

* For D2KF2K, please download the [DIV2K_HR](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) datasets.
    * For Flickr2K, we generated the train/validation/test split with exactly the same script and random split as for CelebA-HQ above.
    * For DIV2K_HR, the original images for the test set are not available, so we used the images assigned by the authors as the 'validation set' as the test set instead. As a validation set, we randomly chose a small subset of the original training data (also removing them from the training set). These files are listed in `div2k_own_validation_files.txt`.
    * We then respectively merged the train/, valid/ and test/ subfolders from both generated datasets to get the overall D2KF2K dataset.


# Training

Our training runs, as presented in our journal submission, can be performed by the following commands:

* Best D2KF2K run (OUVE SDE):
```bash
python train.py --gpus 1 --D-jpeg 0 100 --max_epochs 300 --model ScoreModel \
    --lr 0.0001 --batch_size 8 --optimizer adamw --loss_type mse \
    --t-eps 0.01 --sde ouve --sigma-min 0.01 --sigma-max 1 --gamma 1 --sde-n 100 \
    --backbone ncsnpp --ema_decay 0.999 \
    --data_module imagefolder --data_dir <your_d2kf2k_folder> \
    --random-resized-crop 256
```

* Best CelebA-HQ256 run (CosVE SDE):
```bash
python train.py --gpus 1 --D-jpeg 0 100 --max_epochs 300 --model ScoreModel \
    --lr 0.0001 --batch_size 8 --optimizer adamw --loss_type mse \
    --t-eps 0.01 --sde cosve --sigma-min 0.01 --sigma-max 1 --gamma 1 --sde-n 100 \
    --backbone ncsnpp --ema_decay 0.999 \
    --data_module imagefolder --data_dir <your_celebahq256_folder>
```

## Regression baselines

* Regression Baseline training for D2KF2K:
```bash
python train.py --gpus 1 --D-jpeg 0 100 --max_epochs 3000 \
    --model DiscriminativeModel --discriminatively --discriminative_mode direct \
    --lr 0.0001 --batch_size 8 --optimizer adamw --loss_type mse \
    --backbone ncsnpp --ema_decay 0.999 \
    --data_module imagefolder --data_dir <your_image_dataset_folder> \
    --random-resized-crop 256
```

* Regression Baseline training for CelebA-HQ256:
```bash
python train.py --gpus 1 --D-jpeg 0 100 --max_epochs 300 \
    --model DiscriminativeModel --discriminatively --discriminative_mode direct \
    --lr 0.0001 --batch_size 8 --optimizer adamw --loss_type mse \
    --backbone ncsnpp --ema_decay 0.999 \
    --data_module imagefolder --data_dir <your_celebahq256_folder>
```

where the D2KF2K trainings run for 3000 epochs since there are much fewer images in the dataset (and the model will only see a single random resized crop for each one in each epoch).


# Sampling / JPEG Artifact Removal

For enhancement, we used the provided `enhance_folder.py` script. For the default Euler-Maruyama sampling with 100 steps, as described in our manuscript, you can use:

```bash
python enhance_folder.py --ckpt <path_to_ckpt_file> --indir <path_to_corrupted_jpeg_folder> --outdir <path_to_output_folder> --ema --N 100 --batch_size 1
```

where the `--batch_size` can be increased for parallel processing, but only if all images are of the same resolution (e.g. 256x256).

**Outputs will be different each time due to our method being stochastic!** If you want reproducibility, feel free to set a seed with `torch.random.manual_seed`.


# Evaluation

We have provided the following scripts for evaluation:

* `avg_samples.py`: Averages multiple image samples from a set of enhanced image folders into a single estimate.
* `color_correct.py`: Applies a per-channel global color correction to all images in a folder, see the "Limitations" section.
* `calc_dist_feats.py`: Calculates KID and FID between a folder containing image estimates and a ground-truth folder.
* `eval_all_metrics.py`: Evaluates per-image PSNR/PSNRB/SSIM/LPIPS between a folder containing image estimates and a ground-truth folder. Stores results in a .pkl pickled DataFrame.
* `eval_blockiness.py`: Calculates the blockiness measure (BEF) for a folder of images. Stores results in a .pkl pickled DataFrame.
* `eval_resnet50_acc.py`: Evaluates classifier accuracy using the pretrained ResNet-50 from torchvision. Written for ImageNet256 evaluation.
