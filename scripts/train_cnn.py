"""Train and evaluate a small residual CNN on prepared image pairs.

The expected pair layout is produced by batch_pipeline.py:

    data/processed/DIV2K/train/scale_x4/inputs/0001.png
    data/processed/DIV2K/train/targets/0001.png
    data/processed/DIV2K/valid/scale_x4/inputs/0801.png
    data/processed/DIV2K/valid/targets/0801.png

The CNN is trained on random patches, then evaluated on complete validation
images using tiled inference. This avoids trying to place an entire 2K image
and all intermediate feature maps in memory at once.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn import ResidualCNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="DIV2K")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-valid-images", type=int, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU was requested but is not available")
        return torch.device("xpu")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def image_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def pad_if_needed(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, height, width = image.shape
    pad_height = max(0, patch_size - height)
    pad_width = max(0, patch_size - width)
    if pad_height == 0 and pad_width == 0:
        return image
    return torch.nn.functional.pad(image, (0, pad_width, 0, pad_height), mode="reflect")


def crop_pair(
    input_image: torch.Tensor,
    target_image: torch.Tensor,
    patch_size: int,
    training: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_image = pad_if_needed(input_image, patch_size)
    target_image = pad_if_needed(target_image, patch_size)
    _, height, width = input_image.shape

    if training:
        top = random.randint(0, height - patch_size)
        left = random.randint(0, width - patch_size)
    else:
        top = (height - patch_size) // 2
        left = (width - patch_size) // 2

    return (
        input_image[:, top : top + patch_size, left : left + patch_size],
        target_image[:, top : top + patch_size, left : left + patch_size],
    )


class PairDataset(Dataset):
    def __init__(
        self,
        input_dir: Path,
        target_dir: Path,
        patch_size: int,
        training: bool,
        patches_per_image: int = 1,
        max_images: int | None = None,
    ) -> None:
        self.input_paths = sorted(input_dir.glob("*.png"))
        if max_images is not None:
            self.input_paths = self.input_paths[:max_images]
        self.target_dir = target_dir
        self.patch_size = patch_size
        self.training = training
        self.patches_per_image = patches_per_image if training else 1

        if not self.input_paths:
            raise FileNotFoundError(f"No PNG inputs found in {input_dir}")
        missing = [path.name for path in self.input_paths if not (target_dir / path.name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing target files, first example: {missing[0]}")

    def __len__(self) -> int:
        return len(self.input_paths) * self.patches_per_image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.input_paths[index % len(self.input_paths)]
        input_image = image_tensor(path)
        target_image = image_tensor(self.target_dir / path.name)
        return crop_pair(input_image, target_image, self.patch_size, self.training)


def predict_tiled(
    model: nn.Module,
    input_image: torch.Tensor,
    device: torch.device,
    tile_size: int,
    overlap: int,
) -> torch.Tensor:
    """Run same-size inference over a full image without excessive memory use."""
    _, height, width = input_image.shape
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile-overlap must be smaller than tile-size")

    output = torch.zeros_like(input_image)
    weights = torch.zeros((1, height, width), dtype=input_image.dtype)

    for top in range(0, height, stride):
        for left in range(0, width, stride):
            bottom = min(top + tile_size, height)
            right = min(left + tile_size, width)
            actual_top = max(0, bottom - tile_size)
            actual_left = max(0, right - tile_size)
            tile = input_image[:, actual_top:bottom, actual_left:right]
            tile_height, tile_width = tile.shape[-2:]
            if tile_height < tile_size or tile_width < tile_size:
                tile = torch.nn.functional.pad(
                    tile,
                    (0, tile_size - tile_width, 0, tile_size - tile_height),
                    mode="reflect",
                )

            prediction = model(tile.unsqueeze(0).to(device)).squeeze(0).cpu()
            prediction = prediction[:, :tile_height, :tile_width]
            output[:, actual_top:bottom, actual_left:right] += prediction
            weights[:, actual_top:bottom, actual_left:right] += 1.0

    return output / weights.clamp_min(1.0)


def evaluate(
    model: nn.Module,
    dataset: PairDataset,
    device: torch.device,
    tile_size: int,
    overlap: int,
) -> dict:
    model.eval()
    mae_values: List[float] = []
    psnr_values: List[float] = []
    ssim_values: List[float] = []

    with torch.no_grad():
        for input_path in dataset.input_paths:
            input_image = image_tensor(input_path)
            target_image = image_tensor(dataset.target_dir / input_path.name)
            prediction = predict_tiled(model, input_image, device, tile_size, overlap)
            target_array = target_image.permute(1, 2, 0).numpy()
            prediction_array = prediction.permute(1, 2, 0).numpy()
            mae_values.append(float(np.mean(np.abs(target_array - prediction_array))))
            psnr_values.append(float(peak_signal_noise_ratio(target_array, prediction_array, data_range=1.0)))
            ssim_values.append(
                float(
                    structural_similarity(
                        target_array,
                        prediction_array,
                        channel_axis=2,
                        data_range=1.0,
                    )
                )
            )

    return {
        "image_count": len(dataset.input_paths),
        "mae_mean": float(np.mean(mae_values)),
        "psnr_db_mean": float(np.mean(psnr_values)),
        "ssim_mean": float(np.mean(ssim_values)),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    dataset_name = args.dataset_name
    scale_name = f"scale_x{args.scale}"

    processed_root = args.processed_dir / dataset_name
    train_input_dir = processed_root / "train" / scale_name / "inputs"
    valid_input_dir = processed_root / "valid" / scale_name / "inputs"
    train_target_dir = processed_root / "train" / "targets"
    valid_target_dir = processed_root / "valid" / "targets"

    train_dataset = PairDataset(
        train_input_dir,
        train_target_dir,
        args.patch_size,
        training=True,
        patches_per_image=args.patches_per_image,
        max_images=args.max_train_images,
    )
    valid_dataset = PairDataset(
        valid_input_dir,
        valid_target_dir,
        args.patch_size,
        training=False,
        max_images=args.max_valid_images,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type in {"cuda", "xpu"},
    )

    model = ResidualCNN(args.channels, args.blocks).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.L1Loss()
    checkpoint_dir = args.checkpoints_dir / dataset_name / "CNN" / scale_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir = args.results_dir / dataset_name / "CNN" / scale_name
    result_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    history = []
    print(f"Device: {device}")
    print(f"Training pairs: {len(train_dataset.input_paths)}")
    print(f"Validation pairs: {len(valid_dataset.input_paths)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        training_losses = []
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()
            training_losses.append(loss.item())

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for inputs, targets in DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers):
                predictions = model(inputs.to(device))
                validation_losses.append(loss_function(predictions, targets.to(device)).item())

        train_loss = float(np.mean(training_losses))
        valid_loss = float(np.mean(validation_losses))
        record = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
        history.append(record)
        print(f"Epoch {epoch:03d}/{args.epochs}: train={train_loss:.6f}, valid={valid_loss:.6f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # Store paths as strings so the checkpoint remains portable across
            # Windows, macOS, and Linux.
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(checkpoint, checkpoint_dir / "best.pt")

    with (result_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    # This checkpoint was created locally by this trusted project. PyTorch
    # 2.6 changed torch.load's default to weights_only=True, which rejects
    # older checkpoints containing pathlib objects in their metadata.
    best_checkpoint = torch.load(
        checkpoint_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    evaluation = evaluate(model, valid_dataset, device, args.tile_size, args.tile_overlap)
    evaluation.update(
        {
            "dataset": dataset_name,
            "model": "CNN",
            "scale": args.scale,
            "checkpoint": str(checkpoint_dir / "best.pt"),
        }
    )
    with (result_dir / "validation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(evaluation, file, indent=2)

    with (result_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "valid_loss"])
        writer.writeheader()
        writer.writerows(history)

    print("Final validation metrics:")
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
