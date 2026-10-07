"""Clinical inference & telemetry triage service for ECG signal classification."""
from __future__ import annotations

import threading
from typing import Any
import numpy as np
import joblib

from . import config
from .dsp import (
    clean_ecg_signal,
    extract_beat_features,
    pan_tompkins_detector,
    segment_beats,
)
from .data import generate_continuous_rhythm
from .models import train_and_benchmark_all_models


class ClinicalECGService:
    """Clinical telemetry AI service for real-time arrhythmia triage."""

    def __init__(self):
        self._lock = threading.Lock()
        self.models: dict[str, Any] = {}
        self.scaler = None
        self.metrics: dict = {}
        self.loaded = False

    def load(self, force_retrain: bool = False) -> ClinicalECGService:
        with self._lock:
            if self.loaded and not force_retrain:
                return self

            pkg = train_and_benchmark_all_models(quick=False, force_retrain=force_retrain)
            self.models = pkg["models"]
            self.scaler = pkg["scaler"]
            self.metrics = pkg["metrics"]
            self.loaded = True
        return self

    def analyze_rhythm(
        self,
        raw_signal: np.ndarray,
        fs: int = config.SAMPLING_RATE,
        selected_model: str = "Hybrid QNN (4 Qubits + PQC)",
    ) -> dict[str, Any]:
        """Process continuous ECG signal, detect beats, and perform clinical triage.

        Returns comprehensive diagnostic metrics, rhythm classification, and beat-level timeline.
        """
        self.load()
        raw_signal = np.asarray(raw_signal, dtype=np.float32)

        # 1. DSP Filtering Pipeline (Zero-phase Bandpass + Notch)
        filtered = clean_ecg_signal(raw_signal, fs=fs)

        # 2. Pan-Tompkins QRS Detection
        pt_res = pan_tompkins_detector(filtered, fs=fs)
        r_peaks = pt_res["r_peaks"]

        if len(r_peaks) == 0:
            return {
                "status": "warning",
                "message": "No QRS complexes detected in signal.",
                "r_peaks": [],
                "heart_rate_bpm": 0,
                "diagnoses": [],
            }

        # 3. Beat Segmentation
        beats, valid_peaks = segment_beats(filtered, r_peaks)
        if len(beats) == 0:
            return {
                "status": "warning",
                "message": "Signal length insufficient around detected peaks.",
                "r_peaks": r_peaks.tolist(),
                "heart_rate_bpm": 0,
                "diagnoses": [],
            }

        # 4. R-R Intervals and Heart Rate (BPM)
        rr_intervals_sec = np.diff(r_peaks) / float(fs)
        mean_rr = float(np.mean(rr_intervals_sec)) if len(rr_intervals_sec) > 0 else 0.8
        heart_rate_bpm = float(60.0 / mean_rr) if mean_rr > 0 else 75.0

        # 5. Beat-by-Beat Feature Extraction
        beat_features = []
        for i, p in enumerate(valid_peaks):
            p_idx = np.where(r_peaks == p)[0][0]
            prev_rr = float(rr_intervals_sec[p_idx - 1]) if p_idx > 0 else mean_rr
            post_rr = float(rr_intervals_sec[p_idx]) if p_idx < len(rr_intervals_sec) else mean_rr
            feat = extract_beat_features(beats[i], prev_rr, post_rr, mean_rr)
            beat_features.append(feat)

        X_feat = np.array(beat_features, dtype=np.float32)
        X_scaled = self.scaler.transform(X_feat)

        # 6. Multi-Model Inference
        model_name = selected_model if selected_model in self.models else "Random Forest (200 Trees)"
        clf = self.models[model_name]

        if "1D-CNN" in model_name:
            probs = clf.predict_proba(beats)
        else:
            probs = clf.predict_proba(X_scaled)

        preds = np.argmax(probs, axis=1)

        # 7. Beat Annotations & Timeline
        beat_diagnoses = []
        for i, (peak_idx, pred_class) in enumerate(zip(valid_peaks, preds)):
            time_sec = float(peak_idx / fs)
            p_class = int(pred_class)
            beat_diagnoses.append({
                "beat_index": i + 1,
                "r_peak_sample": int(peak_idx),
                "timestamp_sec": round(time_sec, 3),
                "predicted_class": p_class,
                "class_name": config.CLASSES[p_class],
                "short_name": config.CLASS_SHORT_NAMES[p_class],
                "severity": config.CLASS_SEVERITY[p_class],
                "confidence": round(float(probs[i, p_class]), 4),
                "probabilities": {config.CLASS_SHORT_NAMES[k]: round(float(probs[i, k]), 4) for k in range(5)},
            })

        # 8. Overall Rhythm Clinical Triage Assessment
        class_counts = {k: int(np.sum(preds == k)) for k in range(5)}
        total_beats = len(preds)

        if class_counts[2] >= 2 or class_counts[2] / total_beats >= 0.20:
            triage_level = "CRITICAL EMERGENCY"
            triage_code = "critical"
            triage_recommendation = (
                "Frequent Ventricular Ectopic Beats (PVCs / Bigeminy) detected. High risk of "
                "Ventricular Tachycardia (VT) / Ventricular Fibrillation (VF). Immediate cardiac telemetry alert!"
            )
        elif class_counts[4] >= 1:
            triage_level = "EMERGENCY"
            triage_code = "emergency"
            triage_recommendation = (
                "Myocardial Ischemia / ST-T Anomaly detected. Significant ST segment deviation indicates "
                "myocardial perfusion defect. Urgent 12-lead ECG and troponin assessment recommended."
            )
        elif class_counts[1] / total_beats >= 0.25:
            triage_level = "URGENT"
            triage_code = "urgent"
            triage_recommendation = (
                "Atrial Fibrillation / Supraventricular Arrhythmia detected. High risk of thromboembolism. "
                "Rate/rhythm control and CHA2DS2-VASc stroke risk assessment indicated."
            )
        elif class_counts[3] / total_beats >= 0.30:
            triage_level = "ALERT"
            triage_code = "alert"
            triage_recommendation = (
                "Left Bundle Branch Block (LBBB) detected. Intraventricular conduction delay; assess for underlying "
                "structural cardiomyopathy or new-onset coronary artery disease."
            )
        else:
            triage_level = "NORMAL"
            triage_code = "normal"
            triage_recommendation = (
                "Normal Sinus Rhythm (NSR) predominant. No acute life-threatening ectopy detected. Routine monitoring."
            )

        return {
            "status": "success",
            "model_used": model_name,
            "total_beats": total_beats,
            "heart_rate_bpm": round(heart_rate_bpm, 1),
            "mean_rr_interval_ms": round(mean_rr * 1000, 1),
            "r_peaks": valid_peaks.tolist(),
            "filtered_signal": filtered,
            "pt_debug": pt_res,
            "class_counts": {config.CLASS_SHORT_NAMES[k]: v for k, v in class_counts.items()},
            "triage": {
                "level": triage_level,
                "code": triage_code,
                "recommendation": triage_recommendation,
            },
            "beat_diagnoses": beat_diagnoses,
        }

    def get_benchmark_patient(self, condition: str = "pvc", duration: float = 10.0) -> dict[str, Any]:
        """Generate a realistic continuous benchmark ECG strip with ground-truth annotations."""
        t, raw_sig, ann = generate_continuous_rhythm(condition=condition, duration_seconds=duration)
        return {
            "condition": condition,
            "time_axis": t,
            "raw_signal": raw_sig,
            "annotations": ann,
        }


_service: ClinicalECGService | None = None


def get_service() -> ClinicalECGService:
    global _service
    if _service is None:
        _service = ClinicalECGService()
    return _service.load()
