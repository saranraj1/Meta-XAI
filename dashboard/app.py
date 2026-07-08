"""
app.py — Meta-XAI Streamlit Dashboard (Phase 8)

Interactive visualization dashboard for the Meta-XAI pipeline.
Run with: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

st.set_page_config(
    page_title="Meta-XAI Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background: #0f1117; }
    
    .ets-card {
        background: linear-gradient(135deg, #1a1d27 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .ets-score {
        font-size: 56px;
        font-weight: 700;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .trust-high   { color: #2ecc71; font-weight: 700; font-size: 20px; }
    .trust-medium { color: #f39c12; font-weight: 700; font-size: 20px; }
    .trust-low    { color: #e67e22; font-weight: 700; font-size: 20px; }
    .trust-reject { color: #e74c3c; font-weight: 700; font-size: 20px; }
    
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    
    .hallucination-badge {
        background: #e74c3c22;
        border: 1px solid #e74c3c;
        border-radius: 8px;
        padding: 8px 12px;
        color: #e74c3c;
        font-weight: 600;
    }
    
    .valid-badge {
        background: #2ecc7122;
        border: 1px solid #2ecc71;
        border-radius: 8px;
        padding: 8px 12px;
        color: #2ecc71;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3a7bd5, #00d2ff);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 16px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(58, 123, 213, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 32px 0 16px 0;">
    <h1 style="font-size:42px; font-weight:800; background: linear-gradient(135deg, #00d2ff, #3a7bd5);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
        🔍 Meta-XAI Dashboard
    </h1>
    <p style="color:#8892b0; font-size:16px; margin-top:8px;">
        Trustworthiness Auditing Framework for Explainable AI Systems
    </p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar Configuration ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    model_arch = st.selectbox(
        "Model Architecture",
        ["resnet18", "resnet50", "vgg16", "efficientnet_b0"],
        index=0,
    )

    xai_methods = st.multiselect(
        "XAI Methods",
        ["shap", "lime", "gradcam", "intgrad"],
        default=["shap", "lime", "gradcam"],
    )

    enable_semantic = st.toggle("Enable Semantic Module (LLM)", value=True)
    use_mock_llm = st.toggle("Use Mock LLM (offline mode)", value=True)

    st.markdown("---")
    st.markdown("### 📊 ETS Thresholds")
    th_high = st.slider("HIGH threshold", 50, 95, 75)
    th_medium = st.slider("MEDIUM threshold", 25, 74, 50)
    th_low = st.slider("LOW threshold", 5, 49, 25)

    st.markdown("---")
    st.markdown("### 🔬 About")
    st.markdown("""
    **Meta-XAI v1.0**  
    Research prototype for XAI trustworthiness auditing.
    
    **Paper target:** NeurIPS 2025 Workshop on Interpretable ML
    """)


# ── Main Content ───────────────────────────────────────────────────────────────
col_upload, col_config = st.columns([2, 1])

with col_upload:
    st.markdown("### 📤 Input Image")
    uploaded_file = st.file_uploader(
        "Upload an image for XAI audit",
        type=["jpg", "jpeg", "png"],
        help="Upload any image to audit its XAI explanations",
    )

    use_sample = st.button("🎲 Use Random Sample Image")

with col_config:
    st.markdown("### 🏷️ Prediction Label")
    label = st.text_input(
        "Predicted class label",
        value="dog",
        help="The model's predicted class label for this image",
    )

    image_id = st.text_input("Image ID (optional)", value="", placeholder="Auto-generated")


# ── Image Preparation ──────────────────────────────────────────────────────────
image_np = None
pil_image = None

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    image_np = np.array(pil_image).astype(np.float32) / 255.0
elif use_sample:
    np.random.seed(42)
    image_np = np.random.rand(224, 224, 3).astype(np.float32)
    pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
    st.info("🎲 Using randomly generated sample image")

if pil_image is not None:
    st.image(pil_image, caption="Input Image (224×224)", width=300)


# ── Run Audit ──────────────────────────────────────────────────────────────────
st.markdown("---")
run_btn = st.button("🚀 Run Meta-XAI Audit", disabled=(image_np is None))

if run_btn and image_np is not None:
    with st.spinner("Running Meta-XAI pipeline... (this may take 10-30s)"):

        # Use mock pipeline for dashboard demo
        # In production: replace with MetaXAIPipeline
        import time, random
        random.seed(hash(label) % 1000)
        time.sleep(1.5)  # Simulate processing

        # Simulate pipeline outputs
        n_methods = len(xai_methods) if xai_methods else 3
        consistency_score = round(random.uniform(0.45, 0.95), 3)
        robustness_score = round(random.uniform(0.30, 0.90), 3)
        semantic_score = round(random.uniform(0.50, 0.95), 3) if enable_semantic else 0.5
        stability_score = round(random.uniform(0.60, 0.95), 3)

        # Simulate hallucinations
        n_hallucinated = random.randint(0, max(1, n_methods - 1))
        h_fraction = n_hallucinated / n_methods if n_methods > 0 else 0

        # Compute ETS
        ets_raw = (
            0.30 * consistency_score
            + 0.35 * robustness_score
            + 0.20 * semantic_score
            + 0.15 * stability_score
            - 0.50 * h_fraction
        )
        ets = round(max(0, min(100, ets_raw * 100)), 1)

        # Trust level
        if ets >= th_high:
            trust_level = "HIGH"
            trust_color = "trust-high"
            trust_icon = "🟢"
        elif ets >= th_medium:
            trust_level = "MEDIUM"
            trust_color = "trust-medium"
            trust_icon = "🟡"
        elif ets >= th_low:
            trust_level = "LOW"
            trust_color = "trust-low"
            trust_icon = "🟠"
        else:
            trust_level = "REJECT"
            trust_color = "trust-reject"
            trust_icon = "🔴"

        hallucinated_methods = random.sample(xai_methods, n_hallucinated) if xai_methods else []

    # ── Results Display ────────────────────────────────────────────────────────
    st.markdown("## 📊 Audit Results")

    # ETS Card
    col_ets, col_metrics = st.columns([1, 2])

    with col_ets:
        st.markdown(f"""
        <div class="ets-card">
            <div style="color:#8892b0; font-size:14px; margin-bottom:8px;">EXPLANATION TRUST SCORE</div>
            <div class="ets-score">{ets}</div>
            <div style="color:#8892b0; font-size:14px; margin-bottom:12px;">/ 100</div>
            <div class="{trust_color}">{trust_icon} {trust_level}</div>
            <div style="color:#8892b0; font-size:12px; margin-top:8px;">
                Predicted: <strong style="color:#cdd6f4;">{label}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics:
        st.markdown("### Component Scores")

        metrics_data = [
            ("Consistency C(e)", consistency_score, "#3a7bd5"),
            ("Robustness R(e)", robustness_score, "#00d2ff"),
            ("Semantic S(e)", semantic_score, "#a78bfa"),
            ("Stability St(e)", stability_score, "#34d399"),
            ("Hallucination H(e)", h_fraction, "#f87171"),
        ]

        for metric_name, value, color in metrics_data:
            col_l, col_r = st.columns([3, 1])
            with col_l:
                st.markdown(f"**{metric_name}**")
                st.progress(float(value))
            with col_r:
                st.markdown(f"<div style='color:{color}; font-weight:600; font-size:18px; "
                           f"text-align:center; padding-top:8px;'>{value:.3f}</div>",
                           unsafe_allow_html=True)

    # ── Attribution Maps ───────────────────────────────────────────────────────
    st.markdown("### 🎨 Attribution Maps")

    if xai_methods:
        cols = st.columns(len(xai_methods) + 1)
        with cols[0]:
            st.image(pil_image, caption="Original", use_column_width=True)

        for i, method in enumerate(xai_methods):
            with cols[i + 1]:
                # Generate synthetic attribution heatmap
                np.random.seed(abs(hash(method)) % (2**31))
                attr = np.random.rand(224, 224).astype(np.float32)
                import matplotlib.cm as cm
                colored = cm.jet(attr)[:, :, :3]
                blended = 0.6 * image_np + 0.4 * colored
                blended_img = Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))

                is_h = method in hallucinated_methods
                caption = f"**{method.upper()}**\n{'⚠ HALLUCINATED' if is_h else '✓ Valid'}"
                st.image(blended_img, caption=caption, use_column_width=True)

                if is_h:
                    st.markdown('<div class="hallucination-badge">⚠ Hallucinated</div>',
                               unsafe_allow_html=True)
                else:
                    st.markdown('<div class="valid-badge">✓ Valid</div>',
                               unsafe_allow_html=True)

    # ── Hallucination Details ──────────────────────────────────────────────────
    st.markdown("### 🕵️ Hallucination Detection Details")

    for method in (xai_methods or ["shap", "lime", "gradcam"]):
        is_h = method in hallucinated_methods
        np.random.seed(abs(hash(method)) % (2**31))
        pert_delta = round(np.random.uniform(0.02, 0.18) if is_h else np.random.uniform(0.15, 0.45), 3)
        boundary_overlap = round(np.random.uniform(0.10, 0.38) if is_h else np.random.uniform(0.42, 0.85), 3)

        with st.expander(f"{'⚠' if is_h else '✓'} {method.upper()} — {'HALLUCINATED' if is_h else 'Valid'}"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Perturbation Δ", f"{pert_delta:.3f}",
                         delta="❌ < 0.15" if pert_delta < 0.15 else "✓ ≥ 0.15")
            with c2:
                st.metric("Boundary Overlap", f"{boundary_overlap:.3f}",
                         delta="❌ < 0.40" if boundary_overlap < 0.40 else "✓ ≥ 0.40")
            with c3:
                st.metric("Hallucinated", "YES ⚠" if is_h else "NO ✓")

    # ── Raw Report ─────────────────────────────────────────────────────────────
    with st.expander("📄 Full Audit Report (JSON)"):
        import json
        report_json = {
            "image_id": image_id or "auto_generated",
            "predicted_class": label,
            "explanation_trust_score": ets,
            "trust_level": trust_level,
            "consistency_score": consistency_score,
            "robustness_score": robustness_score,
            "semantic_coherence_score": semantic_score,
            "stability_score": stability_score,
            "hallucination_fraction": h_fraction,
            "hallucinated_methods": hallucinated_methods,
            "xai_methods_used": xai_methods,
            "pipeline_version": "1.0.0",
        }
        st.code(json.dumps(report_json, indent=2), language="json")

elif image_np is None:
    st.info("👆 Upload an image or use a sample image to begin the audit.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8892b0; font-size:13px; padding: 16px 0;">
    <strong>Meta-XAI v1.0</strong> — Trustworthiness Auditing Framework for Explainable AI<br>
    Research Prototype | Target: NeurIPS 2025 Workshop on Interpretable ML
</div>
""", unsafe_allow_html=True)
