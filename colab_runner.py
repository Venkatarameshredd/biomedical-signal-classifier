"""Google Colab 1-Click Deployment Runner for PulseGuard AI.

Run this script inside a single Google Colab cell:
    !python colab_runner.py
or paste the contents directly into a Colab cell.

This script:
1. Automatically installs required dependencies (PennyLane, PyTorch, Streamlit, PyWavelets, etc.).
2. Downloads and configures the Cloudflare Tunnel (`cloudflared`) binary.
3. Launches the PulseGuard AI Streamlit Dashboard on port 8501 in the background.
4. Exposes a live, secure, zero-config public HTTPS URL for hackathon judges.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path


def is_colab() -> bool:
    """Check if execution environment is Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"[*] Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def setup_colab_environment():
    print("=" * 80)
    print("  PULSEGUARD AI - BIOMEDICAL SIGNAL CLASSIFIER")
    print("  Google Colab 1-Click Deployment & Public Tunnel Setup")
    print("=" * 80)

    # 1. Install dependencies
    print("\n[Step 1/4] Installing Required Python Packages...")
    pkgs = [
        "streamlit",
        "pennylane",
        "torch",
        "pywavelets",
        "scikit-learn",
        "scipy",
        "plotly",
        "fastapi",
        "uvicorn",
        "httpx",
        "joblib",
        "pandas",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=False)
    print("Dependencies installed successfully.")

    # 2. Setup Cloudflared tunnel binary on Linux/Colab
    print("\n[Step 2/4] Setting up Cloudflare Public Tunnel...")
    cloudflared_path = Path("./cloudflared")
    if not cloudflared_path.exists():
        if sys.platform.startswith("linux"):
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            subprocess.run(["wget", "-q", "-O", "cloudflared", url], check=False)
            subprocess.run(["chmod", "+x", "cloudflared"], check=False)
        else:
            print("Note: Running on Windows/macOS. Cloudflared binary download skipped.")

    # 3. Launch Streamlit Dashboard in the background
    print("\n[Step 3/4] Starting Streamlit App (Port 8501)...")
    app_path = Path(__file__).resolve().parent / "app.py"
    streamlit_log = open("streamlit.log", "w")
    st_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--server.headless=true",
        ],
        stdout=streamlit_log,
        stderr=streamlit_log,
    )
    print(f"Streamlit running in background (PID: {st_proc.pid})")
    time.sleep(3)

    # 4. Launch Cloudflare Tunnel and parse public URL
    print("\n[Step 4/4] Establishing Public HTTPS Tunnel...")
    tunnel_url = None
    if Path("./cloudflared").exists():
        tunnel_log = open("tunnel.log", "w")
        tunnel_proc = subprocess.Popen(
            ["./cloudflared", "tunnel", "--url", "http://127.0.0.1:8501"],
            stdout=tunnel_log,
            stderr=tunnel_log,
        )
        print(f"Cloudflare Tunnel active (PID: {tunnel_proc.pid})")

        # Poll tunnel log for the generated URL
        for _ in range(25):
            time.sleep(1)
            if Path("tunnel.log").exists():
                content = Path("tunnel.log").read_text()
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                if match:
                    tunnel_url = match.group(0)
                    break

    print("\n" + "=" * 80)
    print("  PULSEGUARD AI IS LIVE AND READY FOR HACKATHON EVALUATION!")
    print("=" * 80)
    if tunnel_url:
        print(f"\n >>> PUBLIC LIVE DASHBOARD URL: {tunnel_url} <<<\n")
        print("Share this link with judges to interact with the live telemetry classifier.")
    else:
        print("\n >>> LOCAL DASHBOARD URL: http://localhost:8501 <<<")
        print("Tunnel log not yet populated or running locally. Access via localhost:8501")

    print("\nFeatures available on dashboard:")
    print("  * Page 1: Live Telemetry & Classifier (ECG strip monitor & beat-level timeline)")
    print("  * Page 2: Clinical Case Studies (Bigeminy, AFIB, Acute Ischemia)")
    print("  * Page 3: Signal Processing (DSP) Lab (Pan-Tompkins stages & DWT db4 sub-bands)")
    print("  * Page 4: Model Benchmark & Metrics (Multi-metric comparison & ROC curves)")
    print("  * Page 5: Quantum Neural Architecture (4-qubit circuit & PQC Hilbert space)")
    print("  * Page 6: Clinical Report Generator (Exportable cardiology triage report)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    setup_colab_environment()
