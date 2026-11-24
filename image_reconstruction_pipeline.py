"""Basic image reconstruction pipeline.

This script:
1. Loads an image.
2. Creates a low-resolution/pixelated degradation.
3. Restores the low-resolution image with bicubic and Lanczos interpolation.
4. Displays and optionally saves a comparison figure.
5. Calculates MAE, PSNR, and SSIM for each restoration method.

Example:
    python image_reconstruction_pipeline.py --image data/example.jpg --scale 4

Optional LPIPS support:
    python image_reconstruction_pipeline.py --image data/example.jpg \
        --scale 4 --lpips

Install dependencies with:
    python -m pip install -r requirements-pipeline.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(image_path: Path) -> Image.Image:
    """Load an image as an RGB PIL image."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        return Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not read image: {image_path}") from exc


def create_degraded_image(
    image: Image.Image, scale: int
) -> tuple[Image.Image, Image.Image]:
    """Create a low-resolution image and a nearest-neighbor pixelated image.

    The low-resolution image is the actual input used for interpolation.
    The pixelated image is an enlarged nearest-neighbor visualization of that
    low-resolution image, useful for seeing the information loss.
    """
    if scale < 2:
        raise ValueError("scale must be an integer greater than or equal to 2")

    width, height = image.size
    low_size = (max(1, width // scale), max(1, height // scale))
    low_resolution = image.resize(low_size, Image.Resampling.BOX)
    pixelated = low_resolution.resize((width, height), Image.Resampling.NEAREST)
    return low_resolution, pixelated


def restore_with_interpolation(
    low_resolution: Image.Image, output_size: tuple[int, int]
) -> Dict[str, Image.Image]:
    """Restore a low-resolution image using classical interpolation methods."""
    return {
        "bicubic": low_resolution.resize(output_size, Image.Resampling.BICUBIC),
        "lanczos": low_resolution.resize(output_size, Image.Resampling.LANCZOS),
    }


def image_to_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to float32 RGB values in [0, 1]."""
    return np.asarray(image, dtype=np.float32) / 255.0


def calculate_metrics(
    target: Image.Image, prediction: Image.Image
) -> Dict[str, float]:
    """Calculate MAE, PSNR, and SSIM for one reconstruction."""
    target_array = image_to_array(target)
    prediction_array = image_to_array(prediction)

    if target_array.shape != prediction_array.shape:
        raise ValueError(
            "Target and prediction must have the same dimensions: "
            f"{target_array.shape} versus {prediction_array.shape}"
        )

    difference = target_array - prediction_array
    mae = float(np.mean(np.abs(difference)))
    mse = float(np.mean(np.square(difference)))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))

    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError(
            "SSIM requires scikit-image. Install it with: "
            "python -m pip install scikit-image"
        ) from exc

    ssim = float(
        structural_similarity(
            target_array,
            prediction_array,
            channel_axis=2,
            data_range=1.0,
        )
    )

    return {"mae": mae, "psnr_db": psnr, "ssim": ssim}


def calculate_lpips(
    target: Image.Image, prediction: Image.Image
) -> float:
    """Calculate LPIPS if the optional lpips and torch packages are installed."""
    try:
        import lpips
        import torch
    except ImportError as exc:
        raise ImportError(
            "LPIPS requires torch and lpips. Install them with: "
            "python -m pip install torch lpips"
        ) from exc

    target_array = np.asarray(target, dtype=np.float32) / 127.5 - 1.0
    prediction_array = np.asarray(prediction, dtype=np.float32) / 127.5 - 1.0

    target_tensor = torch.from_numpy(target_array).permute(2, 0, 1).unsqueeze(0)
    prediction_tensor = (
        torch.from_numpy(prediction_array).permute(2, 0, 1).unsqueeze(0)
    )

    model = lpips.LPIPS(net="alex")
    with torch.no_grad():
        score = model(target_tensor, prediction_tensor).item()
    return float(score)


def save_comparison(
    original: Image.Image,
    pixelated: Image.Image,
    restorations: Dict[str, Image.Image],
    output_path: Path,
) -> None:
    """Save a side-by-side comparison figure."""
    images = [("Original", original), ("Pixelated input", pixelated)]
    images.extend((name.title(), image) for name, image in restorations.items())

    figure, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    if len(images) == 1:
        axes = [axes]

    for axis, (title, image) in zip(axes, images):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Downscaling factor; 4 means approximately 1/4 width and height",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pipeline_demo"),
        help="Directory for generated images and metrics",
    )
    parser.add_argument(
        "--lpips",
        action="store_true",
        help="Also calculate LPIPS; requires torch and lpips",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original = load_image(args.image)
    low_resolution, pixelated = create_degraded_image(original, args.scale)
    restorations = restore_with_interpolation(low_resolution, original.size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    low_resolution.save(args.output_dir / "low_resolution_input.png")
    pixelated.save(args.output_dir / "pixelated_input.png")
    for name, image in restorations.items():
        image.save(args.output_dir / f"restored_{name}.png")

    metrics = {}
    for name, prediction in restorations.items():
        metrics[name] = calculate_metrics(original, prediction)
        if args.lpips:
            metrics[name]["lpips"] = calculate_lpips(original, prediction)

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    save_comparison(
        original,
        pixelated,
        restorations,
        args.output_dir / "comparison.png",
    )

    print(f"Original image: {original.size[0]}x{original.size[1]}")
    print(f"Low-resolution image: {low_resolution.size[0]}x{low_resolution.size[1]}")
    print(f"Results saved to: {args.output_dir}")
    print("\nMetrics:")
    for method, values in metrics.items():
        formatted = ", ".join(
            f"{metric}={value:.6f}" if np.isfinite(value) else f"{metric}=inf"
            for metric, value in values.items()
        )
        print(f"  {method}: {formatted}")


if __name__ == "__main__":
    main()
