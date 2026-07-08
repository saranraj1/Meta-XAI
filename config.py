"""
config.py — Meta-XAI Centralized Configuration

All hyperparameters, thresholds, and constants are defined here.
No magic numbers are permitted in any other module.
Derivations and empirical justifications are provided as comments.

Research Specification: Meta-XAI v1.0
"""

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_ARCH = "resnet18"
MODEL_PRETRAINED = True
MODEL_NUM_CLASSES = 10  # CIFAR-10 for MVP; increase for domain datasets

# ── XAI Methods ────────────────────────────────────────────────────────────────
# Set of XAI methods enabled in the pipeline. Disable without code changes.
ENABLED_METHODS = ["shap", "lime", "gradcam", "intgrad"]

# LIME: more samples = more stable but slower (1000 is standard in LIME paper)
LIME_NUM_SAMPLES = 1000

# SHAP: background reference set size (100 balances speed vs accuracy)
SHAP_BACKGROUND_SAMPLES = 100

# IntGrad: Riemann approximation steps (50 gives <1% error vs continuous integral)
INTGRAD_STEPS = 50

# GradCAM: target layer name (last conv layer for ResNet-18)
GRADCAM_TARGET_LAYER = "layer4"

# Attribution normalization: min-max to [0, 1] before passing downstream
NORMALIZATION_RANGE = (0.0, 1.0)

# ── Consistency Analyzer ───────────────────────────────────────────────────────
# IoU binarization threshold (tau): 0.5 is the standard threshold in attribution literature
BINARIZATION_THRESHOLD = 0.50

# Top-K features for rank correlation (50 is the default in SHAP analysis)
TOP_K_FEATURES = 50

# Consistency metric aggregation weights (learned from exp03 human calibration study)
CONSISTENCY_W_IOU = 0.40       # IoU emphasizes spatial precision
CONSISTENCY_W_COSINE = 0.35    # Cosine captures directional magnitude agreement
CONSISTENCY_W_RANK = 0.25      # Rank correlation captures ordinal importance agreement

# ── Hallucination Detector ─────────────────────────────────────────────────────
# Top-K fraction of attribution mass to occlude in perturbation test
# 0.20 (top 20%) captures the most salient region without over-masking
PERTURBATION_TOP_K = 0.20

# Minimum required confidence drop when top region is occluded
# delta=0.15 (15%) empirically derived from ROAR benchmark thresholds
PERTURBATION_DELTA = 0.15

# Minimum required fraction of attribution mass inside segmentation mask
# 0.40 derived from ablation in exp02_adversarial_xai — below this is clearly spurious
BOUNDARY_OVERLAP_THRESHOLD = 0.40

# Both Strategy A AND Strategy B must fail for hallucination to be declared
# This dual-confirmation reduces false positive rate significantly
HALLUCINATION_REQUIRE_BOTH = True

# Segmentation model choice: 'sam' (Meta's SAM) or 'deeplab' (DeepLabV3)
SEGMENTATION_MODEL = "deeplab"

# ── Trust Scoring Engine ───────────────────────────────────────────────────────
# ETS formula weights: ETS = w1*C + w2*R + w3*S + w4*St - lambda*H
# Calibration: derived from human annotation study (exp03_human_calibration)
# Robustness (R) gets highest weight as counterfactual validity is most critical
ETS_W1_CONSISTENCY = 0.30
ETS_W2_ROBUSTNESS = 0.35
ETS_W3_SEMANTIC = 0.20
ETS_W4_STABILITY = 0.15
ETS_LAMBDA_HALLUCINATION = 0.50  # Penalty multiplier; >0.5 means hallucination is net-negative

# Verify weights sum to 1.0 (required for Completeness axiom)
assert abs(ETS_W1_CONSISTENCY + ETS_W2_ROBUSTNESS + ETS_W3_SEMANTIC + ETS_W4_STABILITY - 1.0) < 1e-6, \
    "ETS weights must sum to 1.0 for Completeness axiom to hold."

# Trust level thresholds (scored 0-100)
ETS_THRESHOLD_HIGH = 75    # Explanation is highly trustworthy
ETS_THRESHOLD_MEDIUM = 50  # Explanation is plausible but warrants caution
ETS_THRESHOLD_LOW = 25     # Explanation is suspect; do not rely on it
# Below LOW threshold → REJECT (explanation should not be presented to users)

# ── Stability Measurement ──────────────────────────────────────────────────────
# Gaussian noise std for stability testing (5% of input range)
STABILITY_NOISE_STD = 0.05

# Number of noisy samples to estimate attribution variance
STABILITY_NUM_SAMPLES = 10

# ── Semantic Coherence Module ──────────────────────────────────────────────────
# LLM model used for semantic validation queries
SEMANTIC_LLM_MODEL = "gpt-4o-mini"  # Configurable; swap for local LLM if needed

# Maximum tokens for LLM coherence query response
SEMANTIC_MAX_TOKENS = 256

# Temperature 0.0 for deterministic, reproducible semantic scoring
SEMANTIC_TEMPERATURE = 0.0

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Benchmark ─────────────────────────────────────────────────────────────────
BENCHMARK_NUM_SAMPLES = 500
BENCHMARK_SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Number of runs for std deviation reporting (research integrity requirement)
BENCHMARK_NUM_RUNS = 3

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"     # Change to DEBUG for verbose module-level logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# ── Dashboard ──────────────────────────────────────────────────────────────────
DASHBOARD_HOST = "localhost"
DASHBOARD_PORT = 8501
