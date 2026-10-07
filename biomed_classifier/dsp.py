"""Digital Signal Processing (DSP) engine for biomedical ECG waveforms.

Implements:
1. Baseline wander & high-frequency noise removal (Butterworth Bandpass + IIR Notch).
2. QRS Complex & R-peak detection using Pan-Tompkins algorithm.
3. Multi-resolution Discrete Wavelet Transform (DWT db4) feature extraction.
4. Heart Rate Variability (HRV) & morphological feature extraction.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as signal
import pywt

from . import config


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = config.BANDPASS_LOWCUT,
    highcut: float = config.BANDPASS_HIGHCUT,
    fs: int = config.SAMPLING_RATE,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase 4th order Butterworth bandpass filter.

    Removes respiration baseline wander (<0.5 Hz) and high-frequency EMG muscle noise (>45 Hz).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    # filtfilt provides zero phase distortion (critical for preserving QRS peak timings)
    return signal.filtfilt(b, a, data)


def notch_filter(
    data: np.ndarray,
    freq: float = config.NOTCH_FREQ,
    fs: int = config.SAMPLING_RATE,
    Q: float = config.NOTCH_Q,
) -> np.ndarray:
    """IIR Notch filter to eliminate 50 Hz / 60 Hz electrical mains interference."""
    b, a = signal.iirnotch(freq, Q, fs)
    return signal.filtfilt(b, a, data)


def clean_ecg_signal(
    raw_signal: np.ndarray,
    fs: int = config.SAMPLING_RATE,
) -> np.ndarray:
    """Full standard clinical filtering pipeline: Bandpass + Notch."""
    bandpassed = butter_bandpass_filter(raw_signal, fs=fs)
    cleaned = notch_filter(bandpassed, fs=fs)
    return cleaned


def pan_tompkins_detector(
    ecg_signal: np.ndarray,
    fs: int = config.SAMPLING_RATE,
) -> dict[str, np.ndarray]:
    """Real-time Pan-Tompkins QRS complex & R-peak detection algorithm.

    Steps:
    1. Bandpass filter (5 - 15 Hz) isolating QRS energy while rejecting P/T waves.
    2. 5-point derivative operator emphasizing steep slopes of R-wave.
    3. Squaring function to accentuate high-frequency components and make all values positive.
    4. Moving-window integrator (150 ms) capturing QRS wave duration.
    5. Dual-threshold adaptive search with 200 ms physiological refractory blanking.

    Returns:
        dict with keys: 'r_peaks', 'filtered', 'derivative', 'squared', 'integrated'
    """
    nyq = 0.5 * fs
    # Step 1: Bandpass filter (5-15 Hz)
    b, a = signal.butter(3, [5.0 / nyq, 15.0 / nyq], btype="band")
    bpass = signal.filtfilt(b, a, ecg_signal)

    # Step 2: 5-point Derivative filter: y[n] = (1/8)(2x[n] + x[n-1] - x[n-3] - 2x[n-4])
    h_der = np.array([-2, -1, 0, 1, 2]) / 8.0
    der = np.convolve(bpass, h_der, mode="same")

    # Step 3: Squaring
    squared = der ** 2

    # Step 4: Moving Window Integration (window width ~ 150 ms)
    win_len = int(0.150 * fs)
    win = np.ones(win_len) / win_len
    integrated = np.convolve(squared, win, mode="same")

    # Step 5: Adaptive Thresholding Peak Detection
    min_distance = int(0.200 * fs)  # 200 ms physiological refractory period
    # Candidate peaks from integrated signal
    thresh = 0.35 * np.mean(integrated) + 0.15 * np.max(integrated)
    candidate_peaks, _ = signal.find_peaks(integrated, distance=min_distance, height=thresh)

    # Search back in original signal window around candidates for true R-peak maximum
    search_radius = int(0.080 * fs)  # +/- 80 ms
    true_r_peaks = []
    for cp in candidate_peaks:
        start = max(0, cp - search_radius)
        end = min(len(ecg_signal), cp + search_radius)
        if end > start:
            r_idx = start + np.argmax(ecg_signal[start:end])
            true_r_peaks.append(r_idx)

    # Remove duplicates within min_distance
    cleaned_peaks = []
    for p in true_r_peaks:
        if not cleaned_peaks or (p - cleaned_peaks[-1]) >= min_distance:
            cleaned_peaks.append(p)

    return {
        "r_peaks": np.array(cleaned_peaks, dtype=int),
        "filtered": bpass,
        "derivative": der,
        "squared": squared,
        "integrated": integrated,
    }


def segment_beats(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    pre_samples: int = config.PRE_R_SAMPLES,
    post_samples: int = config.POST_R_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment continuous ECG into isolated heartbeat windows centered on R-peaks.

    Returns:
        beats: array of shape (N_beats, window_len)
        valid_peaks: array of R-peak indices that had sufficient boundary margins
    """
    beats = []
    valid_peaks = []
    for p in r_peaks:
        if p >= pre_samples and p + post_samples <= len(ecg_signal):
            segment = ecg_signal[p - pre_samples : p + post_samples]
            beats.append(segment)
            valid_peaks.append(p)

    if not beats:
        return np.empty((0, pre_samples + post_samples)), np.empty(0, dtype=int)

    return np.array(beats, dtype=np.float32), np.array(valid_peaks, dtype=int)


def extract_wavelet_features(beat: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """Extract multi-resolution wavelet features using Daubechies 4 (db4).

    Decomposes the beat into approximation and detail coefficients:
    A4 (0 - 7.8 Hz): Baseline & P/T waves
    D4 (7.8 - 15.6 Hz): QRS energy
    D3 (15.6 - 31.2 Hz): High-frequency QRS details & notches
    D2, D1 (> 31.2 Hz): High-frequency transitions & noise

    Computes for each band: Energy, Variance, Standard Deviation, Shannon Entropy.
    """
    coeffs = pywt.wavedec(beat, wavelet, level=level)
    features = []

    for c in coeffs:
        energy = np.sum(c ** 2)
        std_val = np.std(c)
        var_val = np.var(c)
        # Shannon entropy
        prob = (c ** 2) / (energy + 1e-12)
        entropy = -np.sum(prob * np.log(prob + 1e-12))
        max_val = np.max(np.abs(c))
        features.extend([energy, std_val, var_val, entropy, max_val])

    return np.array(features, dtype=np.float32)


def extract_beat_features(
    beat: np.ndarray,
    prev_rr: float,
    post_rr: float,
    mean_rr: float,
) -> np.ndarray:
    """Extract a comprehensive feature vector for a single heartbeat.

    Components:
    1. Wavelet DWT coefficients (25 features across 5 sub-bands)
    2. Morphological descriptors (R-amplitude, Q-S valley depth, skewness, kurtosis, zero-crossings)
    3. R-R Interval & HRV context (prev-RR, post-RR, local RR ratio)
    """
    # 1. Wavelet features (5 bands * 5 metrics = 25 features)
    wavelet_feats = extract_wavelet_features(beat)

    # 2. Morphological statistics
    r_peak_amp = float(np.max(beat))
    s_trough = float(np.min(beat))
    peak_to_peak = r_peak_amp - s_trough
    mean_val = float(np.mean(beat))
    std_val = float(np.std(beat))

    # Skewness & Kurtosis
    centered = beat - mean_val
    skewness = float(np.mean(centered ** 3) / ((std_val + 1e-8) ** 3))
    kurtosis = float(np.mean(centered ** 4) / ((std_val + 1e-8) ** 4))
    zero_crossings = float(np.sum(np.diff(np.sign(centered)) != 0))

    # 3. Temporal HRV metrics
    rr_ratio = float(prev_rr / (mean_rr + 1e-8)) if mean_rr > 0 else 1.0
    rr_diff = float(post_rr - prev_rr)

    morph_feats = np.array([
        r_peak_amp, s_trough, peak_to_peak, std_val,
        skewness, kurtosis, zero_crossings,
        prev_rr, post_rr, rr_ratio, rr_diff
    ], dtype=np.float32)

    return np.concatenate([wavelet_feats, morph_feats])
