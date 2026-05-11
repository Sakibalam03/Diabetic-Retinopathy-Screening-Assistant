"""
Step 4 — DR Screening Inference Pipeline
Runs EfficientNet-B3 grading + Grad-CAM heatmap on a fundus image.

Stage 1: EfficientNet-B3 → DR grade (0-4) + confidence
Stage 2: Grad-CAM → lesion heatmap (replaces YOLO; no mask data required)

Usage:
  python predict.py --image path/to/fundus.jpg

Output (CLI only):
  outputs/<stem>_annotated.jpg  — grade banner overlay
  outputs/<stem>_heatmap.jpg    — Grad-CAM lesion heatmap overlay
"""

import cv2
import io
import numpy as np
import torch
import torch.nn as nn
import argparse
import streamlit as st
from pathlib import Path
from PIL import Image
from torchvision import transforms, models

# ── Config ────────────────────────────────────────────────────────────────────
GRADER_WEIGHTS = "runs/grader/best.pt"
OUTPUT_DIR     = Path("outputs")

CLASS_NAMES  = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
GRADE_COLORS = [
    (46, 204, 113),    # green    — No DR
    (241, 196, 15),    # yellow   — Mild
    (230, 126, 34),    # orange   — Moderate
    (231, 76, 60),     # red      — Severe
    (142, 68, 173),    # purple   — Proliferative DR
]

UNCERTAINTY_THRESHOLD = 0.60   # flag if top-class probability is below this
MARGIN_THRESHOLD      = 0.15   # flag if gap between top-2 probabilities is below this
UNCERTAINTY_MSG = (
    "Low confidence prediction. Image quality or presentation may be atypical. "
    "Clinical review recommended regardless of grade."
)
MIN_IMAGE_DIMENSION = 224
DARK_GREEN_MEAN_THRESHOLD = 30
NON_FUNDUS_MSG = (
    "Image does not resemble a standard fundus photograph. "
    "Result is shown as a screening aid only; verify image type and quality before clinical use."
)


# ── Uncertainty detection ─────────────────────────────────────────────────────
def _flag_uncertainty(probs: list) -> tuple:
    """
    Return (uncertain: bool, message: str).
    Flag uncertain when either:
      - max(probs) < UNCERTAINTY_THRESHOLD  (low absolute confidence)
      - top-1 minus top-2 probability < MARGIN_THRESHOLD  (ambiguous between grades)

    `probs` is a list of 5 softmax probabilities summing to ~1.
    Uses UNCERTAINTY_THRESHOLD and MARGIN_THRESHOLD defined above.
    Returns (True, UNCERTAINTY_MSG) when uncertain, (False, "") otherwise.
    """
    sorted_probs = sorted(probs, reverse=True)
    low_confidence = sorted_probs[0] < UNCERTAINTY_THRESHOLD
    low_margin     = (sorted_probs[0] - sorted_probs[1]) < MARGIN_THRESHOLD
    uncertain      = low_confidence or low_margin
    return (True, UNCERTAINTY_MSG) if uncertain else (False, "")


# ── Device detection ──────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_grader(device: torch.device) -> nn.Module:
    if not Path(GRADER_WEIGHTS).exists():
        raise FileNotFoundError(
            f"Weights not found at {GRADER_WEIGHTS}. Run train_grader.py first."
        )
    ckpt = torch.load(GRADER_WEIGHTS, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "efficientnet_b0")
    if arch == "efficientnet_b0":
        m    = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 5)
        )
    else:
        m    = models.efficientnet_b3(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, 256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 5)
        )
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m.to(device)


# ── CLAHE enhancement ─────────────────────────────────────────────────────────
def enhance(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b, g, r = cv2.split(img)
    return cv2.merge([b, clahe.apply(g), r])


def validate_image_quality(img: np.ndarray) -> tuple[bool, str]:
    """
    Validate basic pre-inference image quality.

    Returns (False, message) for hard failures and (True, message) for soft
    warnings that should be shown before continuing with inference.
    """
    if img is None or not isinstance(img, np.ndarray) or img.ndim < 2:
        return False, "Cannot read image. Please upload a valid image file."

    h, w = img.shape[:2]
    if h < MIN_IMAGE_DIMENSION or w < MIN_IMAGE_DIMENSION:
        return (
            False,
            "Image resolution too low for reliable grading. Minimum 224x224 pixels required.",
        )

    enhanced = enhance(img)
    mean_green = float(enhanced[:, :, 1].mean())
    if mean_green < DARK_GREEN_MEAN_THRESHOLD:
        return (
            False,
            "Image appears predominantly dark. Please upload a properly lit fundus photograph.",
        )

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True, NON_FUNDUS_MSG

    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    bbox_area = float(bw * bh)
    contour_area = float(cv2.contourArea(largest))
    circular_region_area = np.pi * (min(bw, bh) / 2.0) ** 2
    circular_region_ratio = circular_region_area / bbox_area if bbox_area else 0.0
    fill_ratio = contour_area / bbox_area if bbox_area else 0.0

    border = max(1, min(h, w) // 20)
    border_pixels = np.concatenate([
        gray[:border, :].ravel(),
        gray[-border:, :].ravel(),
        gray[:, :border].ravel(),
        gray[:, -border:].ravel(),
    ])
    dark_periphery = float(np.mean(border_pixels < 30)) > 0.55
    poor_circular_fit = circular_region_ratio < 0.5 or fill_ratio < 0.5

    if dark_periphery and poor_circular_fit:
        return True, NON_FUNDUS_MSG

    return True, ""


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: nn.Module):
        self.model    = model
        self.gradients = None
        self.activations = None

        # Hook into the last conv block (works for both B0 and B3)
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Returns a (H, W) float32 heatmap in [0, 1]."""
        self.model.zero_grad()
        output = self.model(tensor)
        score  = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1).squeeze()  # (h, w)
        cam     = torch.relu(cam).cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


# ── Grading ───────────────────────────────────────────────────────────────────
def run_inference(model: nn.Module, image_source, device: torch.device):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = tfm(Image.open(image_source).convert("RGB")).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    with torch.enable_grad():
        gcam = GradCAM(model)
        output = model(tensor)
        probs  = torch.softmax(output, 1)[0].tolist()
        grade  = int(np.argmax(probs))
        cam    = gcam.generate(tensor, grade)

    return grade, probs, cam


def _encode_jpg(img_bgr: np.ndarray) -> bytes:
    """Encode a BGR numpy array to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", img_bgr)
    return buf.tobytes()


# ── Main ─────────────────────────────────────────────────────────────────────
def predict(image_input: bytes | str) -> dict | None:
    """
    Run DR screening inference.

    Returns a dict with grade, probs, confidence, referral, and image bytes.
    Images are returned as JPEG bytes — no files are written to disk.
    """
    if isinstance(image_input, bytes):
        arr        = np.frombuffer(image_input, np.uint8)
        img        = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        pil_source = io.BytesIO(image_input)
    else:
        img        = cv2.imread(image_input)
        pil_source = image_input

    if img is None:
        print("ERROR: Cannot read image")
        return None
    valid_quality, quality_msg = validate_image_quality(img)
    if not valid_quality:
        print(f"ERROR: {quality_msg}")
        return None
    if quality_msg:
        print(f"WARNING: {quality_msg}")

    h, w     = img.shape[:2]
    enhanced = enhance(img)

    device                    = get_device()
    model                     = load_grader(device)
    grade, probs, cam         = run_inference(model, pil_source, device)
    referral                  = grade >= 2
    uncertain, uncertain_msg  = _flag_uncertainty(probs)

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("    DIABETIC RETINOPATHY SCREENING RESULT")
    print("=" * 52)
    print(f"  Image      : {image_input if isinstance(image_input, str) else '<bytes>'}")
    print(f"  Grade      : {grade} - {CLASS_NAMES[grade]}")
    print(f"  Confidence : {probs[grade]:.1%}")
    print(f"  Referral   : {'YES - See ophthalmologist' if referral else 'No'}")
    print("=" * 52)

    # ── Annotated image (grade banner) ────────────────────────────────────────
    ann     = enhanced.copy()
    overlay = ann.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (30, 30, 100), -1)
    cv2.addWeighted(overlay, 0.7, ann, 0.3, 0, ann)
    cv2.putText(ann,
                f"Grade {grade}: {CLASS_NAMES[grade]}  |  {probs[grade]:.0%} confidence",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(ann,
                f"Referral: {'REQUIRED' if referral else 'Not required'}",
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── Grad-CAM heatmap overlay ──────────────────────────────────────────────
    cam_resized = cv2.resize(cam, (w, h))
    cam_uint8   = (cam_resized * 255).astype(np.uint8)
    colormap    = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    hm_overlay  = cv2.addWeighted(enhanced.copy(), 0.55, colormap, 0.45, 0)
    cv2.putText(hm_overlay,
                f"Grad-CAM | Grade {grade}: {CLASS_NAMES[grade]}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return {
        "grade":               grade,
        "grade_name":          CLASS_NAMES[grade],
        "confidence":          probs[grade],
        "probs":               probs,
        "referral":            referral,
        "uncertain":           uncertain,
        "uncertainty_message": uncertain_msg,
        "quality_warning":     quality_msg,
        "annotated_bytes":     _encode_jpg(ann),
        "heatmap_bytes":       _encode_jpg(hm_overlay),
        "counts":              {},   # YOLO lesion counts not available in Grad-CAM mode
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR Screening Inference")
    parser.add_argument("--image", required=True, help="Path to fundus image")
    args = parser.parse_args()
    with open(args.image, "rb") as f:
        result = predict(f.read())
    if result:
        OUTPUT_DIR.mkdir(exist_ok=True)
        stem = Path(args.image).stem
        ann_path = str(OUTPUT_DIR / f"{stem}_annotated.jpg")
        hm_path  = str(OUTPUT_DIR / f"{stem}_heatmap.jpg")
        with open(ann_path, "wb") as f:
            f.write(result["annotated_bytes"])
        with open(hm_path, "wb") as f:
            f.write(result["heatmap_bytes"])
        print(f"\n  Annotated image -> {ann_path}")
        print(f"  Grad-CAM heatmap -> {hm_path}\n")
