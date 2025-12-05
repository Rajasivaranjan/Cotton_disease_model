import argparse
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with trained YOLOv8 classifier.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/classify/cotton_cls/weights/best.pt"),
        help="Path to the trained weights file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("yolo_dataset/test"),
        help="Path to the test dataset root.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional path to a specific image. If omitted, a random test image is used.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Inference image size.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of top predictions to display.",
    )
    return parser.parse_args()


def pick_test_image(data_dir: Path) -> Path:
    if not data_dir.exists():
        raise FileNotFoundError(f"Test data not found at {data_dir}. Run data_prep.py first.")
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for img_path in class_dir.iterdir():
            if img_path.is_file():
                return img_path
    raise FileNotFoundError(f"No images found in test data at {data_dir}.")


def format_predictions(result, topk: int) -> str:
    names = result.names
    probs = result.probs
    # Ultralytics Probs exposes top1/top5 and top1conf/top5conf
    top_indices = probs.top5 if topk > 1 else [probs.top1]
    top_indices = top_indices[: min(topk, len(names))]
    lines = []
    for idx in top_indices:
        # probs.data is a tensor/ndarray; cast to float for printing
        conf = float(probs.data[idx])
        lines.append(f"{names[idx]}: {conf:.3f}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found at {args.weights}. Train the model first.")

    image_path: Optional[Path] = args.image
    if image_path is None:
        image_path = pick_test_image(args.data_dir)
    elif not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(str(args.weights))
    results = model.predict(source=str(image_path), imgsz=args.imgsz, verbose=False)

    if not results:
        raise RuntimeError("No results returned by the model.")

    summary = format_predictions(results[0], args.topk)
    print(f"Image: {image_path.resolve()}")
    print("Top predictions:")
    print(summary)


if __name__ == "__main__":
    main()

