import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch
from torchvision import models

from predict import enhance, GradCAM, predict, _encode_jpg, validate_image_quality


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def synthetic_bgr():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_image_bytes():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(img, (112, 112), 100, (70, 150, 110), -1)
    cv2.circle(img, (85, 95), 12, (30, 80, 70), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture
def mock_model():
    m = models.efficientnet_b0(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_f, 128), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(128, 5),
    )
    m.eval()
    return m


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_enhance(synthetic_bgr):
    out = enhance(synthetic_bgr)
    assert out.shape == synthetic_bgr.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, synthetic_bgr)


def test_gradcam_output_shape_and_range(mock_model):
    gcam = GradCAM(mock_model)
    tensor = torch.rand(1, 3, 224, 224, requires_grad=True)
    with torch.enable_grad():
        cam = gcam.generate(tensor, class_idx=0)
    assert cam.ndim == 2
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0


def test_predict_returns_valid_result(synthetic_image_bytes, mock_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with patch("predict.load_grader", return_value=mock_model.to(device)):
        result = predict(synthetic_image_bytes)

    assert result is not None


def test_encode_jpg_returns_nonempty_bytes(synthetic_bgr):
    out = _encode_jpg(synthetic_bgr)
    assert isinstance(out, bytes)
    assert len(out) > 0


def test_predict_result_schema_after_quality_validation(synthetic_image_bytes, mock_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with patch("predict.load_grader", return_value=mock_model.to(device)):
        result = predict(synthetic_image_bytes)

    expected_keys = {
        "grade",
        "grade_name",
        "confidence",
        "probs",
        "referral",
        "uncertain",
        "uncertainty_message",
        "quality_warning",
        "annotated_bytes",
        "heatmap_bytes",
        "counts",
    }
    assert result is not None
    assert expected_keys.issubset(result.keys())
    assert result["grade"] in range(5)
    assert 0.0 <= result["confidence"] <= 1.0


def test_validate_image_quality_rejects_blank_black_image():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    valid, message = validate_image_quality(img)
    assert valid is False
    assert message == "Image appears predominantly dark. Please upload a properly lit fundus photograph."


def test_validate_image_quality_rejects_too_small_image():
    img = np.full((200, 224, 3), 160, dtype=np.uint8)
    valid, message = validate_image_quality(img)
    assert valid is False
    assert message == "Image resolution too low for reliable grading. Minimum 224x224 pixels required."


def test_validate_image_quality_accepts_normal_synthetic_fundus():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(img, (112, 112), 100, (70, 150, 110), -1)
    valid, message = validate_image_quality(img)
    assert valid is True
    assert message == ""


def test_validate_image_quality_warns_on_non_fundus_shape():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 70), (205, 155), (70, 180, 110), -1)
    valid, message = validate_image_quality(img)
    assert valid is True
    assert "does not resemble a standard fundus photograph" in message
