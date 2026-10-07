"""FastAPI REST API for real-time biomedical signal telemetry and arrhythmia classification."""
from __future__ import annotations

from typing import Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import config
from .service import get_service

app = FastAPI(
    title="Biomedical Signal Classifier API (PulseGuard AI)",
    version="1.0.0",
    description="Clinical arrhythmia triage & biomedical signal classifier using Classical ML, 1D-CNN, and Quantum Neural Networks.",
)


class ECGAnalysisRequest(BaseModel):
    signal: list[float] = Field(..., description="Raw continuous ECG voltage array (mV)")
    sampling_rate: int = Field(default=config.SAMPLING_RATE, description="Sampling frequency in Hz")
    model: str = Field(default="Hybrid QNN (4 Qubits + PQC)", description="Classifier model to use")


@app.get("/api/health")
def health():
    svc = get_service()
    return {
        "status": "healthy",
        "service": "PulseGuard Biomedical Signal Classifier",
        "sampling_rate_hz": config.SAMPLING_RATE,
        "available_models": list(svc.models.keys()),
        "classes": config.CLASSES,
    }


@app.get("/api/metrics")
def get_metrics():
    svc = get_service()
    return svc.metrics


@app.get("/api/sample")
def get_sample(
    condition: str = Query("pvc", description="normal | afib | pvc | lbbb | ischemia"),
    duration: float = Query(10.0, ge=3.0, le=30.0, description="Duration in seconds"),
):
    svc = get_service()
    res = svc.get_benchmark_patient(condition=condition, duration=duration)
    return {
        "condition": res["condition"],
        "sampling_rate": config.SAMPLING_RATE,
        "time_axis": res["time_axis"].tolist(),
        "signal": res["raw_signal"].tolist(),
        "annotations": res["annotations"],
    }


@app.post("/api/analyze")
def analyze_ecg(req: ECGAnalysisRequest):
    svc = get_service()
    if len(req.signal) < config.SAMPLING_RATE * 2:
        raise HTTPException(
            status_code=422,
            detail=f"Signal too short. Minimum duration is 2.0 seconds ({config.SAMPLING_RATE * 2} samples)."
        )

    res = svc.analyze_rhythm(
        raw_signal=req.signal,
        fs=req.sampling_rate,
        selected_model=req.model,
    )
    if res["status"] != "success":
        raise HTTPException(status_code=400, detail=res["message"])

    # Prepare response without raw numpy objects
    return {
        "status": res["status"],
        "model_used": res["model_used"],
        "heart_rate_bpm": res["heart_rate_bpm"],
        "mean_rr_interval_ms": res["mean_rr_interval_ms"],
        "total_beats": res["total_beats"],
        "r_peaks_samples": res["r_peaks"],
        "class_counts": res["class_counts"],
        "triage": res["triage"],
        "beat_diagnoses": res["beat_diagnoses"],
    }


@app.get("/api/circuit")
def get_quantum_circuit():
    svc = get_service()
    qnn = svc.models.get("Hybrid QNN (4 Qubits + PQC)")
    if qnn and hasattr(qnn, "circuit_diagram"):
        return {"circuit": qnn.circuit_diagram(), "n_qubits": config.N_QUBITS}
    return {"circuit": "Quantum circuit unavailable", "n_qubits": config.N_QUBITS}
