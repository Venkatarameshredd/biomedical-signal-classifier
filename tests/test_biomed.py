"""Comprehensive test suite for PulseGuard AI (Biomedical Signal Classifier)."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from biomed_classifier import config
from biomed_classifier.dsp import (
    butter_bandpass_filter,
    notch_filter,
    clean_ecg_signal,
    pan_tompkins_detector,
    segment_beats,
    extract_beat_features,
)
from biomed_classifier.data import (
    synthesize_single_beat,
    generate_continuous_rhythm,
    generate_benchmark_cohort,
)
from biomed_classifier.service import ClinicalECGService, get_service
from biomed_classifier.api import app


# -----------------------------------------------------------------------------
# Configuration Tests
# -----------------------------------------------------------------------------
def test_config_constants():
    assert config.SAMPLING_RATE == 250
    assert len(config.CLASSES) == 5
    assert len(config.CLASS_SHORT_NAMES) == 5
    assert len(config.CLASS_SEVERITY) == 5
    assert config.N_QUBITS == 4
    assert config.BEAT_WINDOW_LEN == config.PRE_R_SAMPLES + config.POST_R_SAMPLES


# -----------------------------------------------------------------------------
# Digital Signal Processing (DSP) Tests
# -----------------------------------------------------------------------------
def test_dsp_filtering():
    t = np.linspace(0, 4.0, int(4.0 * config.SAMPLING_RATE))
    # Synthetic signal: 1 Hz baseline wander + 1.2 Hz ECG + 50 Hz powerline interference
    raw_signal = (
        np.sin(2 * np.pi * 1.2 * t)
        + 0.5 * np.sin(2 * np.pi * 0.1 * t)
        + 0.8 * np.sin(2 * np.pi * 50.0 * t)
    )

    filtered = clean_ecg_signal(raw_signal, fs=config.SAMPLING_RATE)
    assert filtered.shape == raw_signal.shape
    assert not np.isnan(filtered).any()
    assert not np.isinf(filtered).any()

    # The 50 Hz powerline ripple should be suppressed
    assert np.std(filtered) < np.std(raw_signal)


def test_pan_tompkins_detector():
    # Generate 6 seconds of normal ECG
    _, raw_signal, _ = generate_continuous_rhythm(condition="normal", duration_seconds=6.0)
    cleaned = clean_ecg_signal(raw_signal, fs=config.SAMPLING_RATE)

    pt_res = pan_tompkins_detector(cleaned, fs=config.SAMPLING_RATE)
    assert "r_peaks" in pt_res
    r_peaks = pt_res["r_peaks"]

    # In 6 seconds at ~72 BPM, there should be roughly 6 to 9 peaks
    assert len(r_peaks) >= 4
    assert (r_peaks >= 0).all()
    assert (r_peaks < len(cleaned)).all()


def test_beat_segmentation():
    _, raw_signal, _ = generate_continuous_rhythm(condition="normal", duration_seconds=6.0)
    cleaned = clean_ecg_signal(raw_signal, fs=config.SAMPLING_RATE)
    pt_res = pan_tompkins_detector(cleaned, fs=config.SAMPLING_RATE)

    beats, valid_peaks = segment_beats(cleaned, pt_res["r_peaks"])
    assert len(beats) == len(valid_peaks)
    assert beats.shape[1] == config.BEAT_WINDOW_LEN
    assert beats.ndim == 2


def test_dwt_feature_extraction():
    beat = synthesize_single_beat(beat_type=0)
    feat = extract_beat_features(beat, prev_rr=0.8, post_rr=0.8, mean_rr=0.8)

    assert len(feat) == 36
    assert not np.isnan(feat).any()
    assert not np.isinf(feat).any()
    # Check that energy and variance features are non-negative
    assert feat[0] >= 0.0


# -----------------------------------------------------------------------------
# Synthetic Data Generation Tests
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("class_id", [0, 1, 2, 3, 4])
def test_synthetic_beat_classes(class_id):
    beat = synthesize_single_beat(beat_type=class_id)
    assert len(beat) == config.BEAT_WINDOW_LEN
    assert not np.isnan(beat).any()


@pytest.mark.parametrize("condition", ["normal", "afib", "pvc", "lbbb", "ischemia"])
def test_continuous_rhythm_generation(condition):
    t, sig, ann = generate_continuous_rhythm(condition=condition, duration_seconds=5.0)
    assert len(t) == len(sig)
    assert len(sig) == int(5.0 * config.SAMPLING_RATE)
    assert len(ann) > 0
    assert not np.isnan(sig).any()


def test_benchmark_cohort():
    cohort = generate_benchmark_cohort(n_samples_per_class=10, force_rebuild=True)
    assert cohort.X_features.shape == (50, 36)
    assert cohort.raw_beats.shape == (50, config.BEAT_WINDOW_LEN)
    assert cohort.y_labels.shape == (50,)
    assert set(np.unique(cohort.y_labels)) == {0, 1, 2, 3, 4}


# -----------------------------------------------------------------------------
# Clinical Service and Triage Tests
# -----------------------------------------------------------------------------
def test_clinical_service_inference():
    svc = get_service()
    patient = svc.get_benchmark_patient(condition="pvc", duration=8.0)
    res = svc.analyze_rhythm(patient["raw_signal"], selected_model="Random Forest (200 Trees)")

    assert res["status"] == "success"
    assert res["total_beats"] > 0
    assert res["heart_rate_bpm"] > 30.0
    assert "triage" in res
    assert res["triage"]["level"] in ["CRITICAL EMERGENCY", "EMERGENCY", "URGENT", "ALERT", "NORMAL"]
    assert len(res["beat_diagnoses"]) == res["total_beats"]


def test_clinical_service_multi_model():
    svc = get_service()
    patient = svc.get_benchmark_patient(condition="normal", duration=6.0)

    for m_name in svc.models.keys():
        res = svc.analyze_rhythm(patient["raw_signal"], selected_model=m_name)
        assert res["status"] == "success"
        assert len(res["beat_diagnoses"]) > 0


# -----------------------------------------------------------------------------
# FastAPI Endpoint Integration Tests
# -----------------------------------------------------------------------------
client = TestClient(app)


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["sampling_rate_hz"] == config.SAMPLING_RATE
    assert len(data["available_models"]) >= 4


def test_api_metrics():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert "Random Forest (200 Trees)" in metrics
    assert "Hybrid QNN (4 Qubits + PQC)" in metrics


def test_api_sample():
    response = client.get("/api/sample?condition=pvc&duration=5.0")
    assert response.status_code == 200
    data = response.json()
    assert data["condition"] == "pvc"
    assert len(data["signal"]) == 5 * config.SAMPLING_RATE
    assert len(data["time_axis"]) == len(data["signal"])


def test_api_circuit():
    response = client.get("/api/circuit")
    assert response.status_code == 200
    data = response.json()
    assert data["n_qubits"] == 4
    assert "circuit" in data


def test_api_analyze():
    # Fetch sample signal
    sample_resp = client.get("/api/sample?condition=normal&duration=5.0")
    sig = sample_resp.json()["signal"]

    payload = {
        "signal": sig,
        "sampling_rate": config.SAMPLING_RATE,
        "model": "Random Forest (200 Trees)",
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_beats"] > 0
    assert "triage" in data
    assert "beat_diagnoses" in data
