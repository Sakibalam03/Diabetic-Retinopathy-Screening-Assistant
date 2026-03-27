"""
Dataset Downloader — Task 1 helper
Downloads IDRiD and APTOS 2019 datasets from Kaggle.

Prerequisites:
  - Add KAGGLE_API_TOKEN=<your_token> to .env in the project root

Usage:
  python download_data.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root before importing kaggle
load_dotenv(Path(__file__).parent / ".env")

try:
    import kaggle
except ImportError:
    print("ERROR: kaggle package not installed. Run: pip install kaggle")
    sys.exit(1)


IDRID_DEST = Path("data/raw/idrid")
APTOS_DEST = Path("data/raw/aptos")


def check_credentials():
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        print("ERROR: KAGGLE_API_TOKEN not found in .env")
        print("Add this line to your .env file:")
        print("  KAGGLE_API_TOKEN=<your_token>")
        sys.exit(1)
    print(f"Kaggle credentials loaded from .env (token: {token[:8]}...)")


def download_and_extract(dataset: str, dest: Path, description: str):
    dest.mkdir(parents=True, exist_ok=True)
    tmp = Path("data/_tmp")
    tmp.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {description}...")
    print(f"  Dataset : {dataset}")
    print(f"  Dest    : {dest}")

    os.system(f'kaggle datasets download -d {dataset} -p "{tmp}" --unzip')

    # Move contents into dest
    downloaded = list(tmp.iterdir())
    if not downloaded:
        print(f"WARNING: Nothing downloaded for {dataset}")
        return

    for item in downloaded:
        target = dest / item.name
        if not target.exists():
            item.rename(target)
        else:
            print(f"  Skipping (already exists): {item.name}")

    print(f"  Done -> {dest}")


def verify_structure():
    print("\nVerifying dataset structure...")
    idrid_train = IDRID_DEST / "B. Disease Grading" / "1. Original Images" / "a. Training Set"
    idrid_masks = IDRID_DEST / "B. Disease Grading" / "2. Groundtruths" / "a. Training Set"
    aptos_no_dr = APTOS_DEST / "No_DR"

    ok = True
    for p, name in [
        (idrid_train, "IDRiD training images"),
        (idrid_masks, "IDRiD training masks"),
    ]:
        if p.exists():
            count = len(list(p.glob("*.jpg")) + list(p.glob("*.JPG")))
            print(f"  {name}: {count} files")
        else:
            print(f"  MISSING: {p}")
            ok = False

    if aptos_no_dr.exists():
        count = len(list(aptos_no_dr.glob("*.jpg")))
        print(f"  APTOS No_DR: {count} files")
    else:
        print(f"  WARNING: APTOS No_DR folder not found at {aptos_no_dr}")
        print("  Check that APTOS 2019 has subfolders: No_DR, Mild, Moderate, Severe, Proliferative_DR")

    if ok:
        print("\nAll required datasets verified. Run prepare_data.py next.")
    else:
        print("\nSome datasets missing. Check the paths above.")


if __name__ == "__main__":
    check_credentials()
    download_and_extract(
        "mariaherrerot/idrid-dataset",
        IDRID_DEST,
        "IDRiD (Indian Diabetic Retinopathy Image Dataset)"
    )
    download_and_extract(
        "subhajeetdas/aptos-2019-jpg",
        APTOS_DEST,
        "APTOS 2019 DR Grading Dataset"
    )
    verify_structure()
