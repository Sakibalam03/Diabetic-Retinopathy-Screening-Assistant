import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch
from torchvision import models

from predict import enhance, GradCAM, predict, _encode_jpg


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def synthetic_bgr():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_image_bytes():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
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
    with patch("predict.load_grader", return_value=mock_model):
        result = predict(synthetic_image_bytes)

    # TODO(human): assert the result is valid — check it's a non-None dict,
    # all expected keys are present, grade is in 0-4, and confidence is in [0.0, 1.0]


def test_encode_jpg_returns_nonempty_bytes(synthetic_bgr):
    out = _encode_jpg(synthetic_bgr)
    assert isinstance(out, bytes)
    assert len(out) > 0
