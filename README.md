# Diabetic Retinopathy Screening Assistant

An end-to-end AI-powered screening system that analyses retinal fundus photographs, grades diabetic retinopathy severity, and generates a downloadable clinical PDF report — all from a browser.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What It Does

Upload a colour fundus photograph. The system will:

- **Grade** the image on the 5-level ETDRS diabetic retinopathy scale (No DR → Proliferative DR)
- **Explain** the prediction with a Grad-CAM heatmap showing which retinal regions drove the result
- **Recommend** whether the patient requires ophthalmology referral
- **Generate** a structured clinical PDF report with patient details, diagnosis, clinical inferences, fundus image, and heatmap

---

## Screenshots

| Screening UI | PDF Report |
|---|---|
| ![UI](https://raw.githubusercontent.com/Sakibalam03/Diabetic-Retinopathy-Screening-Assistant/refs/heads/main/Sample-Outputs/Screenshot%202026-03-27%20224738.png) | ![PDF](https://raw.githubusercontent.com/Sakibalam03/Diabetic-Retinopathy-Screening-Assistant/refs/heads/main/Sample-Outputs/Screenshot%202026-03-27%20225223.png) |

> Upload a fundus image on the right, fill in patient details on the left, click **Save Patient Details**, then download the report.

---

## Grading Scale

| Grade | Label | Referral |
|---|---|---|
| 0 | No DR | No |
| 1 | Mild NPDR | No |
| 2 | Moderate NPDR | **Yes** |
| 3 | Severe NPDR | **Yes** |
| 4 | Proliferative DR | **Yes** |

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/diabetic-retinopathy.git
cd diabetic-retinopathy
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Python 3.10+ recommended. A virtual environment is advised:
> ```bash
> python -m venv venv
> source venv/bin/activate        # Linux / Mac
> venv\Scripts\activate           # Windows
> pip install -r requirements.txt
> ```

### 3. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

The model weights (`runs/grader/best.pt`) are included in the repo — no training required.

---

## How to Use

1. **Fill in patient details** in the left sidebar (Patient ID, Name, Age, Sex, Eye, Referring Physician)
2. Click **Save Patient Details**
3. **Upload** a fundus image (JPG or PNG) using the file uploader
4. Wait for the pipeline to run (~5–15 seconds on first load, instant after that)
5. Review the **DR grade**, **confidence**, **referral decision**, and **clinical inferences**
6. Inspect the **annotated image** and **Grad-CAM heatmap**
7. Click **Download PDF Report** to save the full clinical report

---

## How It Works

```
Fundus Image
     │
     ▼
CLAHE Enhancement          ← improves lesion visibility (green channel only)
     │
     ▼
EfficientNet-B0            ← fine-tuned on IDRiD + APTOS 2019 (combined ~4,000 images)
     │
     ├──► DR Grade (0–4) + Confidence score + Referral flag
     │
     └──► Grad-CAM Heatmap ← spatial attention map from last conv block
               │
               ▼
         PDF Report        ← patient details, diagnosis, clinical inferences, images
```

**Model:** EfficientNet-B0 fine-tuned for 5-class DR severity classification
**Explainability:** Grad-CAM (Gradient-weighted Class Activation Mapping) — highlights retinal regions that influenced the prediction
**Training data:** IDRiD (Indian fundus images) + APTOS 2019 (Kaggle competition dataset)

---

## Project Structure

```
diabetic-retinopathy/
├── app.py              # Streamlit web UI + PDF report generation
├── predict.py          # Inference pipeline + Grad-CAM implementation
├── download_data.py    # Kaggle dataset downloader
├── requirements.txt    # Python dependencies
└── runs/
    └── grader/
        └── best.pt     # Trained EfficientNet-B0 weights (16 MB)
```


## Tech Stack

| Component | Library |
|---|---|
| Deep Learning | PyTorch 2.0, TorchVision |
| Model Architecture | EfficientNet-B0 |
| Explainability | Grad-CAM (custom PyTorch hooks) |
| Image Processing | OpenCV, Pillow |
| Web UI | Streamlit |
| PDF Generation | fpdf2 |
| Training Utilities | scikit-learn, tqdm, NumPy |

---


## Model Performance

Evaluated on the **IDRiD held-out test set** (78 images; 0 No DR, 1 Mild, 33 Moderate, 19 Severe, 25 Proliferative DR).

| Metric | Value |
|---|---|
| Validation accuracy (training split) | **80.7 %** |
| Weighted F1-score (test set) | 0.24 |
| Referral sensitivity (Grade ≥ 2) | **64.9 %** |
| Referral specificity | 100 % |

> **Note on the test set:** IDRiD's official test split contains no Grade-0 and only one Grade-1 image, making aggregate accuracy an unreliable headline number. Per-class sensitivity — especially **Grade 4 (Proliferative DR)** — is the clinically critical figure; a missed PDR case is a preventable blindness event.

Run `evaluate.py` on your own data split to obtain the full per-class breakdown (see below).

---

## CLI Inference

Run the model on a single image without the web UI:

```bash
python predict.py --image path/to/fundus.jpg
```

Outputs grade, confidence, and referral decision to the console. Saves annotated image and Grad-CAM heatmap to `outputs/`.

---

## Model Evaluation

Run the evaluation script against a labelled image set to reproduce or update the metrics above:

```bash
# IDRiD held-out test set
python evaluate.py --data-dir data/raw/idrid --dataset idrid --split test

# APTOS 2019 (class-folder layout)
python evaluate.py --data-dir data/raw/aptos --dataset aptos

# Custom folder layout (subfolders named 0–4)
python evaluate.py --data-dir path/to/images --dataset folder
```

Results are saved to `evaluation/metrics.json` with a timestamp. The script reports:

- Per-class accuracy, sensitivity (recall), and specificity
- Macro-averaged AUC-ROC (one-vs-rest)
- Weighted F1-score
- Full 5 × 5 confusion matrix

Download IDRiD or APTOS via:

```bash
python download_data.py   # requires a Kaggle API key in .env
```

---

## Disclaimer

This tool is an AI screening aid and **does not replace clinical diagnosis** by a qualified ophthalmologist. All outputs should be reviewed by a licensed clinician before any medical decision is made.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
