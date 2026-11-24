# Batch Image Reconstruction Pipeline

This script prepares image pairs for future CNN or autoencoder training and evaluates classical interpolation baselines across multiple degradation levels.

## Quick test with the sample image

The script expects a directory, so place the existing sample image in a temporary folder or use the repository's `data` directory:

```powershell
python batch_pipeline.py `
  --input-dir data `
  --split-name personal `
  --scales 2 4 8 16 `
  --max-images 1 `
  --save-restorations
```

The `--max-images 1` option is only for a quick test. Remove it when processing a complete dataset.

## Output layout

```text
data/processed/personal/
├── targets/
│   └── image_name.png
├── scale_x2/
│   ├── inputs/
│   └── low_resolution/
├── scale_x4/
│   ├── inputs/
│   └── low_resolution/
└── ...

results/batch_pipeline/
├── metrics_per_image.csv
├── summary_metrics.csv
├── summary_metrics.json
├── plots/
│   ├── mae_by_scale.png
│   ├── psnr_db_by_scale.png
│   └── ssim_by_scale.png
└── restorations/                 # only with --save-restorations
```

For each image, the matching input and target have the same filename. For example:

```text
data/processed/train/scale_x4/inputs/0001.png
data/processed/train/targets/0001.png
```

The input is a same-size pixelated image. The target is the original high-resolution image. This pair is suitable for a future neural network that maps an image of size `H x W` to an image of the same size `H x W`.

## DIV2K

After downloading and extracting DIV2K, use the high-resolution folders as inputs. Process each split separately so that training, validation, and testing images remain separated:

```powershell
python batch_pipeline.py `
  --input-dir data/DIV2K_train_HR `
  --split-name train `
  --scales 2 4 8 16

python batch_pipeline.py `
  --input-dir data/DIV2K_valid_HR `
  --split-name valid `
  --scales 2 4 8 16
```

Do not place augmented versions of the same source image in different splits. Split the original images first, then create degraded versions.

The official DIV2K test set does not provide public high-resolution ground-truth images for ordinary metric evaluation, so use the training/validation high-resolution images for supervised development and reserve a held-out subset for final testing when appropriate.
