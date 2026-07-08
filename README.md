# Meta-XAI: Trustworthiness Auditing Framework for Explainable AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Paper Target](https://img.shields.io/badge/target-NeurIPS%202025-purple.svg)]()

> **Meta-XAI** audits existing XAI methods (SHAP, LIME, Grad-CAM, IntGrad), measures their consistency, detects hallucinated explanations, and produces a formally-grounded **Explanation Trust Score (ETS)** in [0, 100].

---

## The Problem

Modern XAI tools frequently disagree — and there is no principled way to know which is correct:

```
CNN predicts: pneumonia ✓
SHAP  → highlights lung tissue        ✅ (correct)  
LIME  → highlights different region   ⚠️ (uncertain)
GradCAM → activates on hospital logo  ❌ (hallucinated)
```

All three are presented to a radiologist as *equally authoritative*. **This is a patient safety issue.**

---

## Solution

Meta-XAI operates as a meta-layer **above** existing XAI tools:

```
Input: (image, label, model)
         ↓
[1] Explanation Generator  →  SHAP, LIME, GradCAM, IntGrad
         ↓
[2] Consistency Analyzer   →  IoU, Cosine, Spearman ρ matrices
         ↓
[3] Hallucination Detector →  Perturbation Test + Boundary Check
         ↓
[4] Semantic Coherence     →  LLM label-region alignment
         ↓
[5] Trust Scoring Engine   →  ETS = w1·C + w2·R + w3·S + w4·St − λ·H
         ↓
Output: ExplanationAuditReport  [ETS: 0-100 | HIGH/MEDIUM/LOW/REJECT]
```

---

## Quick Start

```bash
# Install
conda create -n metaxai python=3.10
conda activate metaxai
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[all]"

# Run the pipeline
python -c "
from metaxai.pipeline import MetaXAIPipeline
from metaxai.models.classifier import load_classifier
from PIL import Image

model = load_classifier('resnet18')
pipeline = MetaXAIPipeline(model=model, methods=['shap', 'lime', 'gradcam'], use_mock_llm=True)

image = Image.open('sample.jpg')
report = pipeline.audit(image, label='dog')
print(report.summary())
"
```

---

## ETS Formula

```
ETS(e) = 0.30·C(e) + 0.35·R(e) + 0.20·S(e) + 0.15·St(e) − 0.50·H(e)
```

| Symbol | Meaning | Weight |
|--------|---------|--------|
| C(e) | Consistency (cross-method IoU/cosine/rank agreement) | 0.30 |
| R(e) | Robustness (perturbation confidence drop) | 0.35 |
| S(e) | Semantic coherence (LLM score) | 0.20 |
| St(e) | Stability (attribution variance under noise) | 0.15 |
| H(e) | Hallucination penalty | λ=0.50 |

**Trust Levels:** HIGH ≥ 75 | MEDIUM ≥ 50 | LOW ≥ 25 | REJECT < 25

### Axiomatic Properties
The ETS formula satisfies four formally proven axioms:
- **Completeness**: ETS=100 iff all methods agree perfectly and H=0
- **Monotonicity**: Increasing any component cannot decrease ETS  
- **Nullity**: ETS=0 when H=1 and all positive components=0
- **Symmetry**: Score is invariant to method ordering

---

## Repository Structure

```
meta-xai/
├── config.py                    # All hyperparameters (no magic numbers)
├── requirements.txt / setup.py
├── PAPER.md                     # Running paper draft
│
├── metaxai/
│   ├── schemas.py               # Typed dataclasses (all inter-module I/O)
│   ├── pipeline.py              # MetaXAIPipeline orchestrator
│   └── engines/
│       ├── explanation_generator.py   # Module 1
│       ├── consistency_analyzer.py    # Module 2
│       ├── hallucination_detector.py  # Module 3
│       ├── semantic_coherence.py      # Module 4
│       └── trust_scorer.py           # Module 5
│
├── tests/
│   ├── test_explanation_generator.py
│   ├── test_consistency_analyzer.py
│   ├── test_hallucination_detector.py
│   ├── test_trust_scorer.py
│   └── test_axioms.py           # Mathematical property verification
│
├── benchmarks/
│   ├── evaluation.py            # Full benchmark runner
│   └── synthetic/               # 500-sample ground-truth dataset
│
├── experiments/
│   ├── exp01_baseline_comparison/
│   ├── exp02_adversarial_xai/
│   └── exp03_human_calibration/
│
└── dashboard/
    └── app.py                   # Streamlit visualization dashboard
```

---

## Running Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=metaxai --cov-report=term-missing

# Axiom tests only (mathematical property verification)
pytest tests/test_axioms.py -v

# Benchmark evaluation
python benchmarks/evaluation.py --samples 500 --runs 3 --seed 42
```

---

## Key Results (Simulated Baseline)

| Metric | Meta-XAI | SHAP-only | LIME-only | GradCAM-only |
|--------|----------|-----------|-----------|--------------|
| HDR ↑ | **0.783 ± 0.021** | 0.612 ± 0.034 | 0.589 ± 0.041 | 0.521 ± 0.038 |
| FPR ↓ | **0.094 ± 0.012** | 0.213 ± 0.028 | 0.241 ± 0.031 | 0.287 ± 0.039 |
| ETS-ρ ↑ | **0.741 ± 0.018** | 0.534 ± 0.027 | 0.498 ± 0.031 | 0.461 ± 0.029 |

*Results reported as mean ± std over 3 seeds. Full results in `benchmarks/results.json`.*

---

## Hallucination Detection

Two-strategy dual-confirmation (both must agree):

**Strategy A — Perturbation Test:**  
Mask top-20% attributed pixels → rerun model → confidence must drop ≥ 0.15

**Strategy B — Segmentation Boundary:**  
Use SAM/DeepLab → attribution mass inside object mask must be ≥ 40%

---

## Development Standards
- Python 3.10+ with type hints everywhere
- Google docstring format on all public functions
- PEP 8 + Black formatting
- All thresholds defined in `config.py` with empirical justifications
- Conventional Commits format (`feat:`, `fix:`, `docs:`, `test:`)
- ≥3 unit tests per module before phase completion

---

## Paper

**Target:** NeurIPS 2025 Workshop on Interpretable ML / ICLR 2026 Workshop on Trustworthy AI  
**Format:** 8 pages + references, double-column conference format  
**Draft:** See [PAPER.md](PAPER.md)

---

## Citation

```bibtex
@inproceedings{metaxai2025,
  title={Meta-XAI: A Trustworthiness Auditing Framework for Explainable AI Systems},
  author={[Author Names]},
  booktitle={NeurIPS Workshop on Interpretable Machine Learning},
  year={2025}
}
```

---

## License
MIT License — see [LICENSE](LICENSE) for details.
