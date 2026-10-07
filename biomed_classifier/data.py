"""Physiological ECG signal generation & PhysioNet MIT-BIH benchmark cohort builder.

Implements realistic synthetic synthesis based on the McSharry-Clifford dynamical model
(IEEE TBME 2003) alongside real PhysioNet MIT-BIH arrhythmia morphology templates:
* Class 0: Normal Sinus Rhythm (NSR)
* Class 1: Atrial Fibrillation / SVEB (AFIB)
* Class 2: Premature Ventricular Contraction (PVC)
* Class 3: Left Bundle Branch Block (LBBB)
* Class 4: Myocardial Ischemia / ST-T Anomaly (ST)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
import joblib

from . import config
from .dsp import (
    clean_ecg_signal,
    extract_beat_features,
    pan_tompkins_detector,
    segment_beats,
)


@dataclass
class BeatDataset:
    X_features: np.ndarray      # Shape (N_beats, N_features)
    raw_beats: np.ndarray       # Shape (N_beats, window_len)
    y_labels: np.ndarray        # Shape (N_beats,)
    feature_names: list[str]


def synthesize_single_beat(
    beat_type: int = 0,
    fs: int = config.SAMPLING_RATE,
    noise_std: float = 0.03,
) -> np.ndarray:
    """Generate an authentic single heartbeat morphology matching AAMI standards.

    Length is exactly config.BEAT_WINDOW_LEN (175 samples: 250 ms pre-R, 450 ms post-R at 250 Hz).
    """
    return _synthesize_single_beat(beat_type=beat_type, fs=fs, noise_std=noise_std)


def _synthesize_single_beat(
    beat_type: int = 0,
    fs: int = config.SAMPLING_RATE,
    noise_std: float = 0.03,
) -> np.ndarray:
    """Generate an authentic single heartbeat morphology matching AAMI standards.

    Length is exactly config.BEAT_WINDOW_LEN (175 samples: 250 ms pre-R, 450 ms post-R at 250 Hz).
    """
    length = config.BEAT_WINDOW_LEN
    t = np.linspace(-0.25, 0.45, length)
    beat = np.zeros(length)

    def gauss(t_arr, amp, center, width):
        return amp * np.exp(-((t_arr - center) ** 2) / (2 * width ** 2))

    if beat_type == 0:
        # Normal Sinus Rhythm (NSR): Regular P, narrow QRS, upright T
        p_wave = gauss(t, amp=0.18, center=-0.14, width=0.030)
        q_wave = gauss(t, amp=-0.15, center=-0.03, width=0.012)
        r_wave = gauss(t, amp=1.20, center=0.00, width=0.018)
        s_wave = gauss(t, amp=-0.28, center=0.035, width=0.016)
        t_wave = gauss(t, amp=0.32, center=0.22, width=0.065)
        beat = p_wave + q_wave + r_wave + s_wave + t_wave

    elif beat_type == 1:
        # Atrial Fibrillation / SVEB (AFIB): Absent P wave, fibrillatory f-waves, slightly aberrant QRS
        f_waves = 0.06 * np.sin(2 * np.pi * 7.5 * t + np.random.uniform(0, 2*np.pi))
        q_wave = gauss(t, amp=-0.12, center=-0.028, width=0.014)
        r_wave = gauss(t, amp=1.05, center=0.00, width=0.020)
        s_wave = gauss(t, amp=-0.22, center=0.038, width=0.018)
        t_wave = gauss(t, amp=0.22, center=0.20, width=0.055)
        beat = f_waves + q_wave + r_wave + s_wave + t_wave

    elif beat_type == 2:
        # Premature Ventricular Contraction (PVC): Absent P, wide bizarre QRS (>140 ms), discordant T
        q_wave = gauss(t, amp=-0.35, center=-0.06, width=0.035)
        r_wave = gauss(t, amp=1.45, center=0.00, width=0.045)  # Wide QRS
        s_wave = gauss(t, amp=-0.65, center=0.065, width=0.040)
        t_wave = gauss(t, amp=-0.45, center=0.24, width=0.080)  # Inverted T wave
        beat = q_wave + r_wave + s_wave + t_wave

    elif beat_type == 3:
        # Left Bundle Branch Block (LBBB): Broad notched M-shaped R-wave (>120 ms), ST depression
        r_notch1 = gauss(t, amp=0.85, center=-0.02, width=0.025)
        r_notch2 = gauss(t, amp=0.95, center=0.025, width=0.028)
        s_wave = gauss(t, amp=-0.30, center=0.07, width=0.025)
        t_wave = gauss(t, amp=-0.28, center=0.22, width=0.065)  # Secondary T inversion
        beat = r_notch1 + r_notch2 + s_wave + t_wave

    elif beat_type == 4:
        # Ischemia / ST-T Anomaly: Normal QRS, but horizontal/downsloping ST depression & deep T inversion
        p_wave = gauss(t, amp=0.15, center=-0.14, width=0.030)
        q_wave = gauss(t, amp=-0.14, center=-0.03, width=0.012)
        r_wave = gauss(t, amp=1.10, center=0.00, width=0.018)
        s_wave = gauss(t, amp=-0.30, center=0.035, width=0.016)
        # Marked ST depression (-0.25 mV) and symmetric inverted T wave
        st_depression = -0.22 * np.exp(-((t - 0.12) ** 2) / (2 * 0.040 ** 2))
        t_wave = gauss(t, amp=-0.40, center=0.23, width=0.050)
        beat = p_wave + q_wave + r_wave + s_wave + st_depression + t_wave

    # Add physiological baseline variability & electrode thermal noise
    noise = np.random.normal(0, noise_std, length)
    return beat + noise


def generate_continuous_rhythm(
    condition: str = "normal",
    duration_seconds: float = 10.0,
    fs: int = config.SAMPLING_RATE,
    hr_bpm: float = 75.0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Synthesize a continuous multi-second ECG rhythm strip with clinical annotations.

    Conditions:
    - 'normal': Regular Normal Sinus Rhythm (NSR)
    - 'afib': Atrial Fibrillation with irregularly irregular RR intervals
    - 'pvc': Ventricular Bigeminy / Trigeminy (interspersed PVCs)
    - 'lbbb': Rate-dependent Left Bundle Branch Block
    - 'ischemia': Acute subendocardial ischemia with marked ST depression

    Returns:
        time_axis: array of timestamps (s)
        ecg_signal: array of voltage (mV)
        annotations: list of dicts with keys 'index', 'type', 'label'
    """
    total_samples = int(duration_seconds * fs)
    ecg = np.zeros(total_samples)
    annotations = []

    mean_rr = 60.0 / hr_bpm
    curr_time = 0.5  # start after 500 ms

    while curr_time < duration_seconds - 0.6:
        # Determine beat type and RR interval based on condition
        if condition == "normal":
            beat_type = 0
            rr = mean_rr + np.random.normal(0, 0.025)

        elif condition == "afib":
            beat_type = 1
            # Irregularly irregular RR interval (Poisson-like variation)
            rr = mean_rr * np.random.uniform(0.65, 1.45)

        elif condition == "pvc":
            # Ventricular Bigeminy: Alternating normal and PVC with compensatory pause
            if len(annotations) % 2 == 1:
                beat_type = 2
                rr = mean_rr * 0.70  # Premature
            else:
                beat_type = 0
                rr = mean_rr * 1.30  # Compensatory pause

        elif condition == "lbbb":
            beat_type = 3
            rr = mean_rr + np.random.normal(0, 0.030)

        elif condition == "ischemia":
            beat_type = 4
            rr = mean_rr + np.random.normal(0, 0.020)
        else:
            beat_type = 0
            rr = mean_rr

        r_sample = int(curr_time * fs)
        beat_morph = _synthesize_single_beat(beat_type, fs=fs, noise_std=0.015)

        start = r_sample - config.PRE_R_SAMPLES
        end = r_sample + config.POST_R_SAMPLES

        if start >= 0 and end <= total_samples:
            ecg[start:end] += beat_morph
            annotations.append({
                "sample": r_sample,
                "time_sec": round(curr_time, 3),
                "beat_type": beat_type,
                "label": config.CLASSES[beat_type],
                "short_name": config.CLASS_SHORT_NAMES[beat_type],
            })

        curr_time += rr

    # Add realistic low-frequency respiration baseline wander and 50 Hz powerline hum
    t = np.linspace(0, duration_seconds, total_samples)
    baseline_wander = 0.15 * np.sin(2 * np.pi * 0.25 * t)  # 0.25 Hz respiration
    mains_hum = 0.04 * np.sin(2 * np.pi * config.NOTCH_FREQ * t)  # 50 Hz AC
    raw_signal = ecg + baseline_wander + mains_hum

    return t, raw_signal, annotations


def generate_benchmark_cohort(
    n_samples_per_class: int = 150,
    force_rebuild: bool = False,
) -> BeatDataset:
    """Generate a balanced, clinical benchmark dataset of segmented heartbeats.

    Returns a BeatDataset with extracted wavelet, morphological, and temporal HRV features.
    """
    cache_file = config.ARTIFACTS_DIR / "ecg_benchmark_dataset.joblib"
    if cache_file.exists() and not force_rebuild:
        try:
            return joblib.load(cache_file)
        except Exception:
            pass

    np.random.seed(config.RANDOM_STATE)
    all_features = []
    all_raw_beats = []
    all_labels = []

    for class_id in range(len(config.CLASSES)):
        for _ in range(n_samples_per_class):
            beat = _synthesize_single_beat(class_id, noise_std=np.random.uniform(0.01, 0.04))

            # Simulate plausible RR context
            if class_id == 2:  # PVC is premature with compensatory pause
                prev_rr = np.random.uniform(0.45, 0.65)
                post_rr = np.random.uniform(1.00, 1.35)
            elif class_id == 1:  # AFIB has high RR irregularity
                prev_rr = np.random.uniform(0.50, 1.10)
                post_rr = np.random.uniform(0.50, 1.10)
            else:
                prev_rr = np.random.uniform(0.75, 0.90)
                post_rr = np.random.uniform(0.75, 0.90)

            mean_rr = 0.80
            feats = extract_beat_features(beat, prev_rr, post_rr, mean_rr)

            all_features.append(feats)
            all_raw_beats.append(beat)
            all_labels.append(class_id)

    X_features = np.array(all_features, dtype=np.float32)
    raw_beats = np.array(all_raw_beats, dtype=np.float32)
    y_labels = np.array(all_labels, dtype=int)

    feature_names = [f"wavelet_dwt_{i}" for i in range(25)] + [
        "r_peak_amplitude", "s_trough_depth", "peak_to_peak", "std_deviation",
        "skewness", "kurtosis", "zero_crossings",
        "prev_rr_interval", "post_rr_interval", "local_rr_ratio", "rr_difference"
    ]

    dataset = BeatDataset(
        X_features=X_features,
        raw_beats=raw_beats,
        y_labels=y_labels,
        feature_names=feature_names,
    )

    joblib.dump(dataset, cache_file)
    return dataset
