"""Configuration for Biomedical Signal Classifier (ECG Arrhythmia Triage AI)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.getenv("BIOMED_ARTIFACTS_DIR", ROOT_DIR / "artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Sampling parameters (Standard MIT-BIH / Holter telemetry: 360 Hz or resampled 250 Hz)
SAMPLING_RATE = 250  # Hz

# Digital Signal Processing (DSP) parameters
BANDPASS_LOWCUT = 0.5   # Hz (removes baseline wander & respiration artifacts)
BANDPASS_HIGHCUT = 45.0  # Hz (removes high-frequency EMG muscle noise & interference)
NOTCH_FREQ = 50.0       # Hz (powerline interference: 50 Hz or 60 Hz)
NOTCH_Q = 30.0          # Quality factor for notch filter

# Beat Segmentation parameters (centered on R-peak)
PRE_R_SAMPLES = int(0.25 * SAMPLING_RATE)   # 250 ms before R-peak (P-wave & PR segment)
POST_R_SAMPLES = int(0.45 * SAMPLING_RATE)  # 450 ms after R-peak (ST segment & T-wave)
BEAT_WINDOW_LEN = PRE_R_SAMPLES + POST_R_SAMPLES  # 175 samples at 250 Hz

# AAMI EC57 / PhysioNet Beat Diagnostic Classes
CLASSES = {
    0: "Normal Sinus Rhythm (NSR)",
    1: "Atrial Fibrillation / SVEB (AFIB)",
    2: "Premature Ventricular Contraction (PVC)",
    3: "Left Bundle Branch Block (LBBB)",
    4: "Ischemia / ST-T Abnormality (ST)",
}

CLASS_SHORT_NAMES = {
    0: "Normal",
    1: "AFIB/SVEB",
    2: "PVC/VEB",
    3: "LBBB",
    4: "Ischemia",
}

CLASS_SEVERITY = {
    0: "NORMAL",
    1: "URGENT",     # Thromboembolism / stroke risk
    2: "CRITICAL",   # Risk of Ventricular Tachycardia / Sudden Cardiac Death
    3: "ALERT",      # Intraventricular conduction delay
    4: "EMERGENCY",  # Myocardial perfusion defect / Acute Coronary Syndrome
}

CLASS_COLORS = {
    0: "#2ECC71",  # Green
    1: "#F39C12",  # Orange
    2: "#E74C3C",  # Red
    3: "#3498DB",  # Blue
    4: "#9B59B6",  # Purple
}

# Machine Learning & Quantum Architecture settings
RANDOM_STATE = 42
N_QUBITS = 4
QNN_LAYERS = 2
QNN_EPOCHS = 35
CNN_EPOCHS = 20
TEST_SIZE = 0.20

API_HOST = os.getenv("BIOMED_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("BIOMED_API_PORT", "8502"))
DASHBOARD_PORT = int(os.getenv("BIOMED_APP_PORT", "8501"))
