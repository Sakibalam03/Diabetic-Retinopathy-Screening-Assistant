#!/usr/bin/env python3
"""
evaluate.py — Evaluate the DR grader on a held-out image set.

Computes per-class accuracy, sensitivity (recall), specificity,
macro-averaged AUC-ROC, weighted F1-score, and a 5×5 confusion matrix.
Saves all results to evaluation/metrics.json.

Usage:
    python evaluate.py --data-dir data/raw/idrid --dataset idrid
    python evaluate.py --data-dir data/raw/aptos --dataset aptos
    python evaluate.py --data-dir path/to/class-folders --dataset folder

Dataset folder conventions:
    idrid  : expects IDRiD directory layout with CSV label files
    aptos  : expects class-named subfolders (No_DR, Mild, Moderate, Severe, Proliferative_DR)
    folder : expects subfolders named 0-4 (one per DR grade)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm

GRADER_WEIGHTS = "runs/grader/best.pt"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
OUTPUT_DIR = Path("evaluation")

EVAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model ────────────────────────────────────────────────────────────────────

def load_model(weights_path: str, device: torch.device) -> nn.Module:
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch", "efficientnet_b0")
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 5),
        )
    else:
        m = models.efficientnet_b3(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, 256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 5),
        )
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m.to(device)


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_idrid(data_dir: Path, split: str):
    """Load IDRiD images + labels from official CSV groundtruth files."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for IDRiD loading: pip install pandas")

    split_folder = {"train": "a. Training Set", "test": "b. Testing Set"}.get(split, "b. Testing Set")
    csv_name = {
        "train": "IDRiD_Disease_Grading_Training_Labels.csv",
        "test":  "IDRiD_Disease_Grading_Testing_Labels.csv",
    }.get(split, "IDRiD_Disease_Grading_Testing_Labels.csv")

    img_dir    = data_dir / "B. Disease Grading" / "1. Original Images" / split_folder
    label_path = data_dir / "B. Disease Grading" / "2. Groundtruths" / csv_name

    if not label_path.exists():
        raise FileNotFoundError(f"Label CSV not found: {label_path}")

    df = pd.read_csv(label_path)
    img_col, grade_col = df.columns[0], "Retinopathy grade"

    samples = []
    for _, row in df.iterrows():
        for ext in (".jpg", ".JPG", ".jpeg", ".png"):
            p = img_dir / f"{row[img_col]}{ext}"
            if p.exists():
                samples.append((p, int(row[grade_col])))
                break
    return samples


def load_aptos(data_dir: Path):
    """Load APTOS 2019 from class-named subfolders."""
    folder_to_grade = {
        "No_DR": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Proliferative_DR": 4,
    }
    samples = []
    for folder, grade in folder_to_grade.items():
        d = data_dir / folder
        if d.exists():
            for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):
                for p in d.glob(ext):
                    samples.append((p, grade))
    return samples


def load_folder(data_dir: Path):
    """Load from generic subfolders named 0–4 (one per DR grade)."""
    samples = []
    for grade in range(5):
        d = data_dir / str(grade)
        if d.exists():
            for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png"):
                for p in d.glob(ext):
                    samples.append((p, grade))
    return samples


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_per_class_specificity(cm: np.ndarray) -> list:
    """
    Compute per-class specificity from a confusion matrix using one-vs-rest.
    Specificity (True Negative Rate) = TN / (TN + FP) for each class.

    Args:
        cm: (num_classes × num_classes) confusion matrix where cm[i][j]
            is the count of samples with true label i predicted as label j.

    Returns:
        List of floats (one specificity per class). Return 0.0 where undefined.

    TODO(human): Implement this function body.
    For each class i, think in one-vs-rest terms:
      - Positive  = class i,  Negative = all other classes
      - FP = predicted as i but truly something else  →  column i sum minus diagonal
      - TN = not class i and not predicted as i       →  derive from cm.sum()
    Handle divide-by-zero gracefully and return a list of length num_classes.
    """
    pass


def run_evaluation(samples, model, device):
    y_true, y_pred, y_prob = [], [], []
    skipped = 0
    for img_path, label in tqdm(samples, desc="Evaluating", unit="img"):
        try:
            tensor = EVAL_TRANSFORM(
                Image.open(img_path).convert("RGB")
            ).unsqueeze(0).to(device)
        except Exception as e:
            print(f"  Skip {img_path.name}: {e}")
            skipped += 1
            continue
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
        y_true.append(label)
        y_pred.append(int(probs.argmax()))
        y_prob.append(probs)

    if skipped:
        print(f"  Warning: {skipped} images skipped due to read errors.")
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))

    sensitivities = []
    for i in range(5):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        sensitivities.append(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0)

    specificities = compute_per_class_specificity(cm)

    total = cm.sum()
    per_class_acc = []
    for i in range(5):
        tp = cm[i, i]
        tn = total - cm[i, :].sum() - cm[:, i].sum() + tp
        per_class_acc.append(float((tp + tn) / total) if total > 0 else 0.0)

    try:
        auc = float(roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro", labels=list(range(5))
        ))
    except Exception:
        auc = None

    weighted_f1  = float(f1_score(y_true, y_pred, average="weighted",
                                   labels=list(range(5)), zero_division=0))
    overall_acc  = float((y_true == y_pred).mean())

    return {
        "overall_accuracy": overall_acc,
        "weighted_f1":      weighted_f1,
        "macro_auc_roc":    auc,
        "per_class": {
            CLASS_NAMES[i]: {
                "accuracy":           per_class_acc[i],
                "sensitivity_recall": sensitivities[i],
                "specificity":        specificities[i] if specificities else None,
            }
            for i in range(5)
        },
        "confusion_matrix": cm.tolist(),
    }


# ── Display ───────────────────────────────────────────────────────────────────

def print_confusion_matrix(cm: np.ndarray):
    short = [n[:7] for n in CLASS_NAMES]
    header = f"{'True \\ Pred':<14}" + "".join(f"{s:>10}" for s in short)
    print(header)
    print("─" * len(header))
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name[:14]:<14}" + "".join(f"{int(cm[i, j]):>10}" for j in range(5)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate DR grader on held-out images")
    parser.add_argument("--data-dir", required=True, help="Path to dataset root directory")
    parser.add_argument(
        "--dataset", choices=["idrid", "aptos", "folder"], default="idrid",
        help="Dataset format — idrid|aptos|folder (default: idrid)",
    )
    parser.add_argument(
        "--split", choices=["train", "test"], default="test",
        help="IDRiD split to evaluate (default: test)",
    )
    parser.add_argument(
        "--weights", default=GRADER_WEIGHTS,
        help=f"Path to model weights (default: {GRADER_WEIGHTS})",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Weights : {args.weights}")
    print(f"Dataset : {args.dataset}  split={args.split}\n")

    model = load_model(args.weights, device)

    loaders = {
        "idrid":  lambda: load_idrid(data_dir, args.split),
        "aptos":  lambda: load_aptos(data_dir),
        "folder": lambda: load_folder(data_dir),
    }
    samples = loaders[args.dataset]()

    if not samples:
        print("Error: no samples found. Check --data-dir and --dataset format.")
        sys.exit(1)

    grade_counts = [sum(1 for _, g in samples if g == i) for i in range(5)]
    print(f"Samples : {len(samples)}")
    for i, (name, cnt) in enumerate(zip(CLASS_NAMES, grade_counts)):
        print(f"  Grade {i} ({name}): {cnt}")
    print()

    y_true, y_pred, y_prob = run_evaluation(samples, model, device)

    print("\n── Classification Report ─────────────────────────────────────")
    print(classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, labels=list(range(5)), zero_division=0
    ))

    metrics = compute_metrics(y_true, y_pred, y_prob)

    print(f"Overall Accuracy : {metrics['overall_accuracy']:.4f}")
    print(f"Weighted F1      : {metrics['weighted_f1']:.4f}")
    if metrics["macro_auc_roc"] is not None:
        print(f"Macro AUC-ROC    : {metrics['macro_auc_roc']:.4f}")

    print("\n── Per-Class Metrics ─────────────────────────────────────────")
    hdr = f"{'Class':<22}{'Accuracy':>10}{'Sensitivity':>13}{'Specificity':>13}"
    print(hdr)
    print("─" * len(hdr))
    for cls, m in metrics["per_class"].items():
        spec = f"{m['specificity']:.4f}" if m["specificity"] is not None else "N/A"
        print(f"{cls:<22}{m['accuracy']:>10.4f}{m['sensitivity_recall']:>13.4f}{spec:>13}")

    print("\n── Confusion Matrix (rows=True label, cols=Predicted) ────────")
    print_confusion_matrix(np.array(metrics["confusion_matrix"]))

    OUTPUT_DIR.mkdir(exist_ok=True)
    result = {
        "timestamp":          datetime.now().isoformat(),
        "dataset":            args.dataset,
        "split":              args.split,
        "data_dir":           str(data_dir.resolve()),
        "weights":            args.weights,
        "n_samples":          int(len(y_true)),
        "class_distribution": {CLASS_NAMES[i]: int(grade_counts[i]) for i in range(5)},
        "metrics":            metrics,
    }
    out_path = OUTPUT_DIR / "metrics.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
