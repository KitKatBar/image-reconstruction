# Image Reconstruction Pipeline

This first milestone creates a controlled degraded image, restores it with classical interpolation, displays the results, and calculates evaluation metrics.

## Install

```bash
python -m pip install -r requirements-pipeline.txt
```

## Run

Place an image anywhere in the project, then run:

```bash
python image_reconstruction_pipeline.py --image path/to/image.jpg --scale 4
```

`--scale 4` reduces the width and height to approximately one-fourth of their original values. The script then enlarges the result back to the original dimensions using nearest-neighbor interpolation for the visibly pixelated input.

To use LPIPS as an additional metric, install the optional packages and add `--lpips`:

```bash
python -m pip install torch lpips
python image_reconstruction_pipeline.py --image path/to/image.jpg --scale 4 --lpips
```

## Generated files

The default output directory is `results/pipeline_demo/` and contains:

- `low_resolution_input.png`: the actual low-resolution input
- `pixelated_input.png`: nearest-neighbor visualization of the degraded image
- `restored_bicubic.png`: bicubic interpolation result
- `restored_lanczos.png`: Lanczos interpolation result
- `comparison.png`: side-by-side visual comparison
- `metrics.json`: numerical results

The original image is the ground truth. Metrics compare the original image against each restored result.
