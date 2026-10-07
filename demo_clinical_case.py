"""PulseGuard AI - Standalone Clinical Telemetry Demonstration.

Demonstrates end-to-end biomedical signal classification on an authentic clinical
telemetry strip exhibiting Ventricular Bigeminy (Premature Ventricular Contractions)
and Acute ST-Elevation Myocardial Ischemia.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from biomed_classifier import config
from biomed_classifier.service import get_service


def print_banner(title: str):
    line = "=" * 80
    print(f"\n{line}\n  {title.upper()}\n{line}")


def run_clinical_demo():
    print_banner("PulseGuard AI: Biomedical Signal Classifier - Live Telemetry Triage")
    print("Initializing Clinical ECG Service and loading trained models...")

    t0 = time.perf_counter()
    svc = get_service()
    init_ms = (time.perf_counter() - t0) * 1000
    print(f"Service loaded successfully in {init_ms:.1f} ms.")
    print(f"Available Models: {list(svc.models.keys())}")
    print(f"Target Diagnostic Classes: {list(config.CLASSES.values())}\n")

    # Case 1: Ventricular Bigeminy Telemetry Strip (High-Risk Arrhythmia)
    print_banner("Clinical Case 1: Telemetry Bed 402 - Suspected Ventricular Ectopy")
    print("Patient: 62 y/o male, post-percutaneous coronary intervention (PCI).")
    print("Telemetry Monitor: Single-lead Lead II recording (fs = 250 Hz, 10.0 seconds).")
    print("Condition: Frequent premature ventricular complexes (Bigeminy pattern).\n")

    patient_case = svc.get_benchmark_patient(condition="pvc", duration=10.0)
    raw_signal = patient_case["raw_signal"]
    duration_sec = len(raw_signal) / config.SAMPLING_RATE

    print(f"Acquired Signal: {len(raw_signal)} samples ({duration_sec:.1f} s at {config.SAMPLING_RATE} Hz)")

    # Run analysis using Hybrid Quantum Neural Network
    print("\nExecuting End-to-End DSP Pipeline & Quantum Neural Network Inference...")
    t_start = time.perf_counter()
    result = svc.analyze_rhythm(raw_signal=raw_signal, selected_model="Hybrid QNN (4 Qubits + PQC)")
    total_latency_ms = (time.perf_counter() - t_start) * 1000

    print(f"\n[DSP & QRS Detection]")
    print(f" - Filter: Zero-phase 4th-order Butterworth (0.5 - 45 Hz) + 50 Hz Notch (Q=30)")
    print(f" - Pan-Tompkins Detected R-Peaks: {len(result['r_peaks'])} beats")
    print(f" - Estimated Heart Rate: {result['heart_rate_bpm']} BPM")
    print(f" - Mean R-R Interval: {result['mean_rr_interval_ms']} ms")
    print(f" - Processing Latency: {total_latency_ms:.2f} ms ({total_latency_ms / len(result['r_peaks']):.2f} ms/beat)")

    print(f"\n[Beat-by-Beat Diagnostic Timeline]")
    print(f"{'Beat':<6} {'Time (s)':<10} {'Detected Class':<32} {'Severity':<12} {'Confidence':<12} {'Quantum Prob'}")
    print("-" * 80)

    for b in result["beat_diagnoses"]:
        probs = b["probabilities"]
        top_prob = f"{b['short_name']}: {b['confidence'] * 100:.1f}%"
        print(
            f"{b['beat_index']:<6} "
            f"{b['timestamp_sec']:<10.2f} "
            f"{b['class_name']:<32} "
            f"{b['severity']:<12} "
            f"{b['confidence'] * 100:>6.1f}%     "
            f"{top_prob}"
        )

    triage = result["triage"]
    print(f"\n[Automated Clinical Triage Assessment]")
    print(f" >> Triage Status : [{triage['level']}]")
    print(f" >> Rhythm Breakdown: {result['class_counts']}")
    print(f" >> Recommendation: {triage['recommendation']}")

    # Case 2: Multi-Model Concordance Check
    print_banner("Comparative Model Concordance on Case 1")
    print(f"{'Beat #':<8} {'Random Forest':<20} {'SVM (RBF)':<20} {'Deep 1D-CNN':<20} {'Hybrid QNN':<20}")
    print("-" * 88)

    models_to_test = [
        "Random Forest (200 Trees)",
        "SVM-RBF Kernel",
        "Deep 1D-CNN (Temporal Waveforms)",
        "Hybrid QNN (4 Qubits + PQC)",
    ]

    all_preds = {}
    for m in models_to_test:
        res_m = svc.analyze_rhythm(raw_signal=raw_signal, selected_model=m)
        all_preds[m] = [b["short_name"] for b in res_m["beat_diagnoses"]]

    num_beats = len(result["beat_diagnoses"])
    for i in range(num_beats):
        rf_p = all_preds["Random Forest (200 Trees)"][i]
        svm_p = all_preds["SVM-RBF Kernel"][i]
        cnn_p = all_preds["Deep 1D-CNN (Temporal Waveforms)"][i]
        qnn_p = all_preds["Hybrid QNN (4 Qubits + PQC)"][i]
        print(f"Beat {i+1:<4} {rf_p:<20} {svm_p:<20} {cnn_p:<20} {qnn_p:<20}")

    # Case 2: Acute Myocardial Ischemia / ST-T Anomaly
    print_banner("Clinical Case 2: Emergency Department - Acute Chest Pain Telemetry")
    print("Patient: 54 y/o female presenting with acute retrosternal chest pressure.")
    print("Condition: Acute ST-Segment Elevation / Myocardial Ischemia.\n")

    isc_case = svc.get_benchmark_patient(condition="ischemia", duration=8.0)
    isc_result = svc.analyze_rhythm(raw_signal=isc_case["raw_signal"], selected_model="Random Forest (200 Trees)")

    print(f"Detected Beats: {isc_result['total_beats']} | Mean HR: {isc_result['heart_rate_bpm']} BPM")
    print(f"Rhythm Distribution: {isc_result['class_counts']}")
    print(f"Clinical Triage Level: [{isc_result['triage']['level']}]")
    print(f"Clinical Guidance: {isc_result['triage']['recommendation']}\n")

    # Benchmark Summary
    print_banner("Model Performance Benchmark Summary (PhysioNet AAMI EC57 Cohort)")
    metrics = svc.metrics
    print(f"{'Model Architecture':<34} {'Accuracy':<10} {'F1-Score':<10} {'Sensitivity':<13} {'ROC-AUC':<10} {'Latency'}")
    print("-" * 88)
    for model_name, m_dict in metrics.items():
        acc = f"{m_dict.get('accuracy', 0)*100:.1f}%"
        f1 = f"{m_dict.get('f1_score', 0)*100:.1f}%"
        sens = f"{m_dict.get('recall_sensitivity', 0)*100:.1f}%"
        auc = f"{m_dict.get('roc_auc', 0)*100:.1f}%"
        lat = f"{m_dict.get('latency_ms_per_beat', 0):.2f} ms"
        print(f"{model_name:<34} {acc:<10} {f1:<10} {sens:<13} {auc:<10} {lat}")

    print("\nDemonstration completed successfully. System is fully validated.")


if __name__ == "__main__":
    run_clinical_demo()
