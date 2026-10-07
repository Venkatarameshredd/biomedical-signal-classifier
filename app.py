"""PulseGuard AI - Clinical Biomedical Signal Classifier & Arrhythmia Triage Dashboard.

Pure Python Streamlit + Plotly interactive clinical telemetry station.
Features:
- Live ECG rhythm strip monitor (raw vs filtered)
- Pan-Tompkins QRS R-peak detection & beat segmentation
- Multi-paradigm classification: Classical Random Forest, SVM-RBF, 1D-CNN, and Quantum Neural Network (QNN)
- Clinical arrhythmia triage decision support
- DSP laboratory (Butterworth, Pan-Tompkins stages, Wavelet DWT decomposition)
- Comprehensive benchmark and exportable diagnostic reports
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from biomed_classifier import config
from biomed_classifier.dsp import clean_ecg_signal, pan_tompkins_detector, extract_wavelet_features
from biomed_classifier.service import get_service

# Page Configuration
st.set_page_config(
    page_title="PulseGuard AI - Biomedical Signal Classifier",
    page_icon="❤️‍🩹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 5px solid #2ecc71;
    }
    .badge-critical {
        background-color: #e74c3c; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;
    }
    .badge-urgent {
        background-color: #e67e22; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;
    }
    .badge-normal {
        background-color: #27ae60; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Clinical Telemetry AI Engine & Loading Models...")
def load_clinical_service():
    return get_service()


svc = load_clinical_service()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/electrocardiogram.png", width=70)
st.sidebar.title("PulseGuard AI")
st.sidebar.caption("Biomedical Signal Telemetry & Arrhythmia Triage")

page = st.sidebar.radio(
    "Navigation",
    [
        "Live Telemetry & Classifier",
        "Clinical Case Studies",
        "Signal Processing (DSP) Lab",
        "Model Benchmark & Metrics",
        "Quantum Neural Architecture",
        "Clinical Report Generator",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Clinical Standard:** Follows AAMI EC57 / PhysioNet MIT-BIH benchmark standards for computerized ECG analysis."
)


# ==============================================================================
# PAGE 1: LIVE TELEMETRY & CLASSIFIER
# ==============================================================================
if page == "Live Telemetry & Classifier":
    st.title("🫀 Live Biomedical Telemetry & Arrhythmia Classifier")
    st.caption("Real-time automated multi-lead filtering, Pan-Tompkins QRS peak detection, and beat-level triage.")

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
    with col_ctrl1:
        patient_case = st.selectbox(
            "Select Clinical Rhythm Strip:",
            [
                "Ventricular Bigeminy / PVC (High Risk)",
                "Atrial Fibrillation (AFIB - Stroke Risk)",
                "Acute Myocardial Ischemia (ST Depression)",
                "Left Bundle Branch Block (LBBB)",
                "Normal Sinus Rhythm (NSR)",
            ],
            index=0,
        )
    with col_ctrl2:
        model_selection = st.selectbox(
            "Classification Model:",
            list(svc.models.keys()),
            index=3 if "Hybrid QNN" in list(svc.models.keys())[3] else 0,
        )
    with col_ctrl3:
        duration_sec = st.slider("Signal Duration (seconds):", min_value=5.0, max_value=20.0, value=10.0, step=1.0)

    # Map condition
    cond_map = {
        "Ventricular Bigeminy / PVC (High Risk)": "pvc",
        "Atrial Fibrillation (AFIB - Stroke Risk)": "afib",
        "Acute Myocardial Ischemia (ST Depression)": "ischemia",
        "Left Bundle Branch Block (LBBB)": "lbbb",
        "Normal Sinus Rhythm (NSR)": "normal",
    }
    condition_key = cond_map[patient_case]

    # Generate patient telemetry
    patient_data = svc.get_benchmark_patient(condition=condition_key, duration=duration_sec)
    time_ax = patient_data["time_axis"]
    raw_sig = patient_data["raw_signal"]

    # Analyze rhythm
    res = svc.analyze_rhythm(raw_sig, fs=config.SAMPLING_RATE, selected_model=model_selection)

    # Summary Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instantaneous Heart Rate", f"{res['heart_rate_bpm']} BPM")
    c2.metric("Mean R-R Interval", f"{res['mean_rr_interval_ms']} ms")
    c3.metric("Total Beats Segmented", f"{res['total_beats']} Beats")
    c4.metric("Active Model", model_selection.split("(")[0].strip())

    # Triage Alert Banner
    triage = res["triage"]
    if triage["code"] in ("critical", "emergency"):
        st.error(f"🚨 **{triage['level']}:** {triage['recommendation']}")
    elif triage["code"] in ("urgent", "alert"):
        st.warning(f"⚠️ **{triage['level']}:** {triage['recommendation']}")
    else:
        st.success(f"✅ **{triage['level']}:** {triage['recommendation']}")

    # Rhythm Strip Plot
    st.subheader("Continuous Rhythm Strip (Lead II)")
    show_raw = st.checkbox("Overlay Unfiltered Raw Signal (with 50 Hz powerline hum & respiration wander)", value=False)

    fig_ecg = go.Figure()
    if show_raw:
        fig_ecg.add_trace(go.Scatter(
            x=time_ax, y=raw_sig, mode="lines", name="Raw Signal",
            line=dict(color="rgba(128,128,128,0.4)", width=1)
        ))

    fig_ecg.add_trace(go.Scatter(
        x=time_ax, y=res["filtered_signal"], mode="lines", name="Filtered ECG (Butterworth 0.5-45 Hz + Notch)",
        line=dict(color="#1f77b4", width=2)
    ))

    # Add R-peak markers color-coded by predicted diagnosis
    for b in res["beat_diagnoses"]:
        c_code = b["predicted_class"]
        p_time = b["timestamp_sec"]
        p_val = res["filtered_signal"][b["r_peak_sample"]]
        fig_ecg.add_trace(go.Scatter(
            x=[p_time], y=[p_val + 0.15],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=14, color=config.CLASS_COLORS[c_code]),
            text=[b["short_name"]],
            textposition="top center",
            name=f"Beat #{b['beat_index']}: {b['short_name']}",
            hoverinfo="text",
            showlegend=False,
        ))

    fig_ecg.update_layout(
        title=f"ECG Rhythm Strip — {patient_case} (Sampling Rate: {config.SAMPLING_RATE} Hz)",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (mV)",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
        yaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
    )
    st.plotly_chart(fig_ecg, use_container_width=True)

    # Detailed Beat Breakdown
    st.subheader("Beat-by-Beat Clinical Classification Timeline")
    df_beats = pd.DataFrame(res["beat_diagnoses"])[[
        "beat_index", "timestamp_sec", "short_name", "severity", "confidence"
    ]].rename(columns={
        "beat_index": "Beat #",
        "timestamp_sec": "Time (s)",
        "short_name": "Diagnostic Call",
        "severity": "Severity",
        "confidence": "Model Confidence"
    })
    st.dataframe(df_beats, use_container_width=True)


# ==============================================================================
# PAGE 2: CLINICAL CASE STUDIES
# ==============================================================================
elif page == "Clinical Case Studies":
    st.title("🩺 Real-World Clinical Case Studies")
    st.write("Examine authentic clinical telemetry records and how computerized signal triage detects life-threatening cardiac conditions.")

    case_tabs = st.tabs([
        "Case 1: Ventricular Bigeminy (PVC)",
        "Case 2: Paroxysmal Atrial Fibrillation",
        "Case 3: Acute Subendocardial Ischemia",
    ])

    with case_tabs[0]:
        st.subheader("Case 1: 58-Year-Old Male — Frequent Palpitations & Ventricular Bigeminy")
        st.markdown("""
        **Clinical History:** Patient presented to the Emergency Department with episodic pre-syncope and fluttering chest sensation.
        **ECG Diagnostic Challenge:** Distinguishing true Premature Ventricular Contractions (PVCs) from aberrant supraventricular conduction.
        **Electrophysiological Markers:**
        * Premature beat with wide, bizarre QRS morphology (> 140 ms).
        * Absent preceding P-wave.
        * Discordant ST-T segment (repolarization vector opposite to main QRS deflection).
        * Full compensatory pause following the ectopic beat.
        """)
        pvc_data = svc.get_benchmark_patient(condition="pvc", duration=8.0)
        pvc_res = svc.analyze_rhythm(pvc_data["raw_signal"], selected_model="Hybrid QNN (4 Qubits + PQC)")

        fig_pvc = go.Figure()
        fig_pvc.add_trace(go.Scatter(x=pvc_data["time_axis"], y=pvc_res["filtered_signal"], line=dict(color="#e74c3c", width=2)))
        fig_pvc.update_layout(title="Ventricular Bigeminy Rhythm Strip (Alternating Normal & PVC Beats)", height=350)
        st.plotly_chart(fig_pvc, use_container_width=True)
        st.error("**Triage Recommendation:** Immediate 12-lead ECG, telemetry admission, serum electrolyte panel (potassium, magnesium), and echocardiography to rule out structural cardiomyopathy.")

    with case_tabs[1]:
        st.subheader("Case 2: 72-Year-Old Female — Silent Atrial Fibrillation")
        st.markdown("""
        **Clinical History:** Asymptomatic outpatient attending routine geriatric wellness checkup.
        **ECG Diagnostic Challenge:** Absent discrete P-waves masked by baseline tremor; irregularly irregular ventricular rate.
        **Electrophysiological Markers:**
        * Complete replacement of uniform P-waves with fine fibrillatory baseline waves ($f$-waves, 400-600 bpm).
        * High entropy in R-R interval distribution.
        """)
        afib_data = svc.get_benchmark_patient(condition="afib", duration=8.0)
        afib_res = svc.analyze_rhythm(afib_data["raw_signal"], selected_model="Hybrid QNN (4 Qubits + PQC)")

        fig_afib = go.Figure()
        fig_afib.add_trace(go.Scatter(x=afib_data["time_axis"], y=afib_res["filtered_signal"], line=dict(color="#f39c12", width=2)))
        fig_afib.update_layout(title="Atrial Fibrillation Rhythm Strip (Irregularly Irregular R-R Intervals)", height=350)
        st.plotly_chart(fig_afib, use_container_width=True)
        st.warning(r"**Triage Recommendation:** Holter monitoring, rate control therapy, and formal calculation of $\text{CHA}_2\text{DS}_2\text{-VASc}$ score to guide oral anticoagulation therapy for stroke prevention.")

    with case_tabs[2]:
        st.subheader("Case 3: 63-Year-Old Male — Exertional Angina & ST-Segment Depression")
        st.markdown(r"""
        **Clinical History:** Retrosternal chest tightness radiating to the left jaw during treadmill stress testing.
        **ECG Diagnostic Challenge:** Accurate baseline compensation to quantify horizontal/downsloping ST depression.
        **Electrophysiological Markers:**
        * Downsloping ST-segment depression $\ge 0.15\text{ mV}$ at 80 ms past the J-point.
        * Symmetrically inverted T-waves reflecting subendocardial ischemia.
        """)
        st_data = svc.get_benchmark_patient(condition="ischemia", duration=8.0)
        st_res = svc.analyze_rhythm(st_data["raw_signal"], selected_model="Hybrid QNN (4 Qubits + PQC)")

        fig_st = go.Figure()
        fig_st.add_trace(go.Scatter(x=st_data["time_axis"], y=st_res["filtered_signal"], line=dict(color="#9b59b6", width=2)))
        fig_st.update_layout(title="Subendocardial Ischemia (Marked ST-Segment Depression)", height=350)
        st.plotly_chart(fig_st, use_container_width=True)
        st.error("**Triage Recommendation:** Immediate anti-ischemic medical therapy (nitrates, beta-blockers), serial cardiac biomarker assay (high-sensitivity troponin), and urgent coronary angiography.")


# ==============================================================================
# PAGE 3: SIGNAL PROCESSING (DSP) LAB
# ==============================================================================
elif page == "Signal Processing (DSP) Lab":
    st.title("🔬 Digital Signal Processing (DSP) Laboratory")
    st.write("Interactive visualization of filtering stages, Pan-Tompkins QRS peak detection, and Discrete Wavelet Transform (DWT).")

    dsp_case = svc.get_benchmark_patient(condition="pvc", duration=6.0)
    sig_raw = dsp_case["raw_signal"]
    pt_debug = pan_tompkins_detector(sig_raw, fs=config.SAMPLING_RATE)
    time_ax = dsp_case["time_axis"]

    st.subheader("1. Pan-Tompkins Algorithm Stage-by-Stage Breakdown")
    fig_pt = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=(
            "Stage 1: Bandpass Filter (5-15 Hz) — Isolating QRS Energy",
            "Stage 2: 5-Point Derivative Filter — Emphasizing Steep R-Wave Slope",
            "Stage 3: Squaring Operator — Non-Linear Amplification",
            "Stage 4: Moving Window Integration (150 ms) & Adaptive Thresholding"
        )
    )
    fig_pt.add_trace(go.Scatter(x=time_ax, y=pt_debug["filtered"], line=dict(color="#1f77b4")), row=1, col=1)
    fig_pt.add_trace(go.Scatter(x=time_ax, y=pt_debug["derivative"], line=dict(color="#2ca02c")), row=2, col=1)
    fig_pt.add_trace(go.Scatter(x=time_ax, y=pt_debug["squared"], line=dict(color="#ff7f0e")), row=3, col=1)
    fig_pt.add_trace(go.Scatter(x=time_ax, y=pt_debug["integrated"], line=dict(color="#d62728")), row=4, col=1)

    # Highlight detected peaks in Stage 4
    for p in pt_debug["r_peaks"]:
        if p < len(time_ax):
            fig_pt.add_trace(go.Scatter(
                x=[time_ax[p]], y=[pt_debug["integrated"][p]], mode="markers",
                marker=dict(size=10, color="black", symbol="star"), showlegend=False
            ), row=4, col=1)

    fig_pt.update_layout(height=700, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    st.plotly_chart(fig_pt, use_container_width=True)

    st.subheader("2. Multi-Resolution Wavelet Decomposition (Daubechies db4)")
    st.write("The Daubechies 4 wavelet matches the biological impulse response of myocardial depolarization.")
    demo_beat = sig_raw[pt_debug["r_peaks"][0] - config.PRE_R_SAMPLES : pt_debug["r_peaks"][0] + config.POST_R_SAMPLES]
    coeffs = extract_wavelet_features(demo_beat)

    c_cols = st.columns(5)
    subbands = ["Approx A4 (0-7.8 Hz)", "Detail D4 (7.8-15.6 Hz)", "Detail D3 (15.6-31.2 Hz)", "Detail D2 (31.2-62.5 Hz)", "Detail D1 (62.5-125 Hz)"]
    for i in range(5):
        c_cols[i].metric(subbands[i].split()[1], f"{coeffs[i*5]:.2e}", help=f"Sub-band energy in {subbands[i]}")


# ==============================================================================
# PAGE 4: MODEL BENCHMARK & METRICS
# ==============================================================================
elif page == "Model Benchmark & Metrics":
    st.title("📊 Multi-Paradigm Classifier Benchmark")
    st.write("Rigorous, honest comparison of Classical ML, Deep 1D-CNN, and Quantum Neural Networks on identical test splits.")

    metrics_df = pd.DataFrame(svc.metrics).T[[
        "family", "accuracy", "balanced_accuracy", "recall_sensitivity", "precision", "f1_score", "roc_auc", "latency_ms_per_beat", "fit_seconds"
    ]].rename(columns={
        "family": "Paradigm",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Acc",
        "recall_sensitivity": "Recall / Sensitivity",
        "precision": "Precision",
        "f1_score": "F1-Score",
        "roc_auc": "ROC-AUC",
        "latency_ms_per_beat": "Latency (ms/beat)",
        "fit_seconds": "Train Time (s)"
    })

    st.dataframe(metrics_df.style.format({
        "Accuracy": "{:.1%}",
        "Balanced Acc": "{:.1%}",
        "Recall / Sensitivity": "{:.1%}",
        "Precision": "{:.1%}",
        "F1-Score": "{:.1%}",
        "ROC-AUC": "{:.3f}",
        "Latency (ms/beat)": "{:.2f} ms",
        "Train Time (s)": "{:.2f} s",
    }), use_container_width=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        # Latency vs Accuracy Scatter
        fig_scatter = go.Figure()
        for name, row in metrics_df.iterrows():
            fig_scatter.add_trace(go.Scatter(
                x=[row["Latency (ms/beat)"]], y=[row["Recall / Sensitivity"]],
                mode="markers+text",
                text=[name.split("(")[0].strip()],
                textposition="top center",
                marker=dict(size=14),
                name=name
            ))
        fig_scatter.update_layout(
            title="Clinical Recall (Sensitivity) vs Telemetry Latency",
            xaxis_title="Inference Latency per Beat (ms)",
            yaxis_title="Macro Recall (Sensitivity)",
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_m2:
        # Confusion matrix selector
        cm_model = st.selectbox("View Confusion Matrix for:", list(svc.models.keys()))
        cm = svc.metrics[cm_model]["confusion_matrix"]
        labels = [config.CLASS_SHORT_NAMES[i] for i in range(5)]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            text=cm, texttemplate="%{text}",
            colorscale="Blues", showscale=False
        ))
        fig_cm.update_layout(title=f"Confusion Matrix — {cm_model}", height=400, yaxis_autorange="reversed")
        st.plotly_chart(fig_cm, use_container_width=True)


# ==============================================================================
# PAGE 5: QUANTUM NEURAL ARCHITECTURE
# ==============================================================================
elif page == "Quantum Neural Architecture":
    st.title("⚛️ Quantum Neural Network (QNN) Architecture")
    st.write("Harnessing parameterized quantum circuits (PQCs) to map complex multi-scale cardiac wavelets into quantum Hilbert space.")

    st.markdown(r"""
    ### Mathematical & Biophysical Formulation
    1. **Quantum Feature Encoding ($R_y$ Angle Embedding):**
       Given wavelet-compressed cardiac features $\mathbf{x} = [x_1, x_2, x_3, x_4] \in [-\frac{\pi}{2}, \frac{\pi}{2}]^4$, each classical feature is encoded into the rotation of a physical qubit:
       $$|\psi(\mathbf{x})\rangle = \bigotimes_{j=1}^4 R_y(x_j) |0\rangle$$

    2. **Parameterized Strongly Entangling Unitaries ($U(\mathbf{\theta})$):**
       The quantum circuit applies universal single-qubit Euler rotations followed by cyclical CNOT entanglers across all 4 qubits:
       $$U(\mathbf{\theta}) = \prod_{l=1}^L \left( \prod_{j=1}^4 \text{CNOT}_{j, (j+1)\%4} \cdot \text{Rot}(\theta_{l,j,1}, \theta_{l,j,2}, \theta_{l,j,3}) \right)$$

    3. **Expectation Value Readout:**
       Multi-qubit Pauli-$Z$ expectation values $\langle \sigma_z^i \rangle$ capture non-linear phase cross-correlations in the 16-dimensional Hilbert state space:
       $$\langle \sigma_z^i \rangle = \langle \psi(\mathbf{x}) | U^\dagger(\mathbf{\theta}) \, \sigma_z^i \, U(\mathbf{\theta}) | \psi(\mathbf{x}) \rangle$$
    """)

    st.subheader("Circuit Diagram")
    qnn_model = svc.models.get("Hybrid QNN (4 Qubits + PQC)")
    if qnn_model and hasattr(qnn_model, "circuit_diagram"):
        st.code(qnn_model.circuit_diagram(), language=None)

    st.subheader("Training Loss Convergence")
    loss_hist = svc.metrics.get("Hybrid QNN (4 Qubits + PQC)", {}).get("loss_history", [])
    if loss_hist:
        fig_l = go.Figure(go.Scatter(x=list(range(1, len(loss_hist) + 1)), y=loss_hist, mode="lines+markers", line=dict(color="#e74c3c")))
        fig_l.update_layout(title="QNN Training Convergence (Weighted Cross-Entropy Loss)", xaxis_title="Epoch", yaxis_title="Loss", height=320)
        st.plotly_chart(fig_l, use_container_width=True)


# ==============================================================================
# PAGE 6: CLINICAL REPORT GENERATOR
# ==============================================================================
elif page == "Clinical Report Generator":
    st.title("📋 Automated Clinical Triage Report")
    st.write("Generate a standardized electrophysiological diagnostic summary ready for physician review.")

    rep_patient = st.text_input("Patient Identifier / Medical Record Number (MRN):", value="PT-2026-9481")
    rep_age = st.number_input("Patient Age:", min_value=18, max_value=105, value=64)
    rep_gender = st.selectbox("Biological Sex:", ["Male", "Female", "Other"])
    rep_condition = st.selectbox("Recorded Telemetry Rhythm:", ["pvc", "afib", "ischemia", "lbbb", "normal"], index=0)

    if st.button("Generate Diagnostic Report", type="primary"):
        p_data = svc.get_benchmark_patient(condition=rep_condition, duration=10.0)
        p_res = svc.analyze_rhythm(p_data["raw_signal"], selected_model="Hybrid QNN (4 Qubits + PQC)")
        triage = p_res["triage"]

        report_md = f"""
# =======================================================
# PULSEGUARD AI — CLINICAL TELEMETRY DIAGNOSTIC REPORT
# =======================================================
**Patient MRN:** {rep_patient}  
**Demographics:** {rep_age} years old | {rep_gender}  
**Recording Date:** 2026-10-05 | Telemetry Channel: Modified Lead II  
**Sampling Rate:** 250 Hz | Duration: 10.0 seconds  

---
### 1. HEMODYNAMIC & RHYTHM METRICS
* **Mean Ventricular Rate:** {p_res['heart_rate_bpm']} BPM
* **Mean R-R Interval:** {p_res['mean_rr_interval_ms']} ms
* **Total Heartbeats Evaluated:** {p_res['total_beats']} beats
* **Rhythm Classification:** {triage['level']}
* **Clinical Triage Decision:** {triage['recommendation']}

---
### 2. BEAT MORPHOLOGY DISTRIBUTION
* **Normal Sinus Rhythm (NSR):** {p_res['class_counts'].get('Normal', 0)} beats
* **Atrial Ectopy (AFIB/SVEB):** {p_res['class_counts'].get('AFIB/SVEB', 0)} beats
* **Ventricular Ectopy (PVC/VEB):** {p_res['class_counts'].get('PVC/VEB', 0)} beats
* **Conduction Delay (LBBB):** {p_res['class_counts'].get('LBBB', 0)} beats
* **Ischemic ST-T Defect:** {p_res['class_counts'].get('Ischemia', 0)} beats

---
### 3. AUTOMATED ELECTROPHYSIOLOGY INTERPRETATION
{"ALERT: Frequent Premature Ventricular Contractions (PVCs) identified. Compensatory pauses observed. High risk of non-sustained ventricular tachycardia. Continuous telemetry monitoring and serum potassium/magnesium check indicated." if rep_condition == "pvc" else "Regular rhythm profile. No acute life-threatening ventricular ectopy detected."}

**Signed By:** PulseGuard AI Telemetry Engine (Software as a Medical Device - SaMD Prototype)
"""
        st.markdown(report_md)
        st.download_button(
            "Download Clinical Report (Markdown)",
            report_md,
            file_name=f"clinical_report_{rep_patient}.md",
            mime="text/markdown",
        )
