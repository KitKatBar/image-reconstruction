# CNN Reconstruction Experiment

This is the first learned model for the project. It trains a small residual CNN to map the same-size pixelated input to the original high-resolution target.

## Install

Install the baseline requirements first, then PyTorch:

```powershell
python -m pip install -r requirements-pipeline.txt
python -m pip install -r requirements-cnn.txt
```

## Recommended first experiment

Start with the 4x DIV2K experiment and a small test subset:

```powershell
python scripts\train_cnn.py `
  --dataset-name DIV2K `
  --scale 4 `
  --epochs 2 `
  --max-train-images 20 `
  --max-valid-images 10 `
  --device cpu
```

After verifying that the script works, run the full experiment:

```powershell
python scripts\train_cnn.py `
  --dataset-name DIV2K `
  --scale 4 `
  --epochs 20 `
  --batch-size 8 `
  --patch-size 128 `
  --patches-per-image 8 `
  --device cpu
```

The script trains on random 128x128 patches and evaluates the validation images using tiled full-image inference.

## Outputs

```text
checkpoints/DIV2K/CNN/scale_x4/
├── best.pt
└── last.pt

results/DIV2K/CNN/scale_x4/
├── training_history.csv
├── training_history.json
└── validation_metrics.json
```

The first CNN should be compared against the interpolation baseline at the same scale. For scale 4, the baseline values are in the interpolation validation summary file under the `scale=4` rows.
