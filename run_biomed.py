"""PulseGuard AI - Unified CLI Runner for Biomedical Signal Classifier.

Usage:
    python run_biomed.py app      # Launch interactive Streamlit web dashboard (Port 8501)
    python run_biomed.py api      # Launch FastAPI production REST backend (Port 8502)
    python run_biomed.py demo     # Run standalone terminal clinical case demo
    python run_biomed.py train    # Retrain and benchmark all models (Classical + 1D-CNN + QNN)
    python run_biomed.py test     # Run full automated pytest test suite
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def run_app():
    """Launch Streamlit web dashboard."""
    print("=" * 70)
    print("Launching PulseGuard AI Streamlit Dashboard...")
    print("URL: http://localhost:8501")
    print("=" * 70)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT_DIR / "app.py"),
        "--server.port=8501",
        "--server.address=127.0.0.1",
        "--server.headless=true",
    ]
    subprocess.run(cmd, cwd=str(ROOT_DIR))


def run_api():
    """Launch FastAPI backend server."""
    print("=" * 70)
    print("Launching PulseGuard AI FastAPI Server...")
    print("Docs: http://127.0.0.1:8502/docs")
    print("API:  http://127.0.0.1:8502/api/health")
    print("=" * 70)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "biomed_classifier.api:app",
        "--host=127.0.0.1",
        "--port=8502",
        "--reload",
    ]
    subprocess.run(cmd, cwd=str(ROOT_DIR))


def run_demo():
    """Run terminal clinical demonstration."""
    cmd = [sys.executable, str(ROOT_DIR / "demo_clinical_case.py")]
    subprocess.run(cmd, cwd=str(ROOT_DIR))


def run_train():
    """Train and benchmark all models."""
    print("=" * 70)
    print("Training and Benchmarking All ECG Arrhythmia Classifiers...")
    print("=" * 70)
    from biomed_classifier.models import train_and_benchmark_all_models
    results = train_and_benchmark_all_models(quick=False, force_retrain=True)
    metrics = results["metrics"]
    print("\nBenchmark Complete. Summary Metrics:")
    print(f"{'Model':<34} {'Accuracy':<10} {'F1 Macro':<10} {'Sens. Macro':<12} {'Latency'}")
    print("-" * 74)
    for name, m in metrics.items():
        print(
            f"{name:<34} "
            f"{m.get('accuracy', 0)*100:>6.1f}%    "
            f"{m.get('f1_score', 0)*100:>6.1f}%    "
            f"{m.get('recall_sensitivity', 0)*100:>6.1f}%      "
            f"{m.get('latency_ms_per_beat', 0):.2f} ms"
        )


def run_tests():
    """Execute pytest test suite."""
    print("=" * 70)
    print("Running PulseGuard Automated Pytest Suite...")
    print("=" * 70)
    cmd = [sys.executable, "-m", "pytest", str(ROOT_DIR / "tests"), "-v"]
    subprocess.run(cmd, cwd=str(ROOT_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="PulseGuard AI - Biomedical Signal Classifier Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "action",
        choices=["app", "api", "demo", "train", "test"],
        help="Action to execute: 'app' (Dashboard), 'api' (REST backend), 'demo' (CLI case study), 'train' (Benchmark), 'test' (Pytest suite)",
    )

    args = parser.parse_args()

    if args.action == "app":
        run_app()
    elif args.action == "api":
        run_api()
    elif args.action == "demo":
        run_demo()
    elif args.action == "train":
        run_train()
    elif args.action == "test":
        run_tests()


if __name__ == "__main__":
    main()
