import io
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from PIL import Image
from ultralytics import YOLO


DEFAULT_WEIGHTS = Path("runs/classify/train3/weights/best.pt")
DEFAULT_TEST_DIR = Path("yolo_dataset/test")


@st.cache_resource(show_spinner=False)
def load_model(weights_path: Path) -> YOLO:
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    return YOLO(str(weights_path))


def list_sample_images(root: Path, limit: int = 10) -> List[Path]:
    if not root.exists():
        return []
    samples: List[Path] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(class_dir.iterdir()):
            if img.is_file():
                samples.append(img)
                break
        if len(samples) >= limit:
            break
    return samples


def format_predictions(result, topk: int) -> List[Tuple[str, float]]:
    names = result.names
    probs = result.probs
    top_indices = probs.top5 if topk > 1 else [probs.top1]
    top_indices = top_indices[: min(topk, len(names))]
    formatted = []
    for idx in top_indices:
        conf = float(probs.data[idx])
        formatted.append((names[idx], conf))
    return formatted


def main() -> None:
    st.title("Cotton Disease Classification (YOLOv8-cls)")

    weights_str = st.text_input(
        "Weights path", value=str(DEFAULT_WEIGHTS), help="Path to YOLOv8 classification weights (.pt)"
    )
    weights_path = Path(weights_str)

    test_dir_str = st.text_input(
        "Test images directory",
        value=str(DEFAULT_TEST_DIR),
        help="Used to provide sample images if you do not upload one.",
    )
    test_dir = Path(test_dir_str)

    topk = st.slider("Top-K predictions", min_value=1, max_value=5, value=3, step=1)
    imgsz = st.number_input("Image size", min_value=64, max_value=640, value=224, step=32)

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    sample_images = list_sample_images(test_dir)
    sample_choice = None
    if uploaded_file is None and sample_images:
        sample_labels = [f"{p.parent.name}/{p.name}" for p in sample_images]
        selected = st.selectbox("Or pick a sample image", sample_labels)
        idx = sample_labels.index(selected)
        sample_choice = sample_images[idx]

    run_inference = st.button("Run Inference", type="primary")

    if not run_inference:
        return

    try:
        model = load_model(weights_path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    if uploaded_file:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        source_desc = uploaded_file.name
    elif sample_choice:
        image = Image.open(sample_choice).convert("RGB")
        source_desc = str(sample_choice)
    else:
        st.error("Please upload an image or ensure the test directory has samples.")
        return

    with st.spinner("Running inference..."):
        results = model.predict(source=image, imgsz=int(imgsz), verbose=False)

    if not results:
        st.error("No results returned by the model.")
        return

    st.subheader("Input Image")
    st.image(image, caption=source_desc, use_column_width=True)

    st.subheader("Predictions")
    preds = format_predictions(results[0], topk)
    for cls, conf in preds:
        st.write(f"{cls}: {conf:.3f}")


if __name__ == "__main__":
    main()

