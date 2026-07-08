"""
setup.py — Meta-XAI Package Setup
"""
from setuptools import setup, find_packages

setup(
    name="metaxai",
    version="1.0.0",
    author="Meta-XAI Research Team",
    description="Trustworthiness Auditing Framework for Explainable AI Systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "benchmarks*", "experiments*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "xai": ["shap>=0.42.0", "lime>=0.2.0.1", "captum>=0.6.0"],
        "llm": ["openai>=1.0.0"],
        "dashboard": ["streamlit>=1.28.0", "plotly>=5.15.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.7.0", "mypy>=1.5.0"],
        "all": [
            "shap>=0.42.0", "lime>=0.2.0.1", "captum>=0.6.0",
            "openai>=1.0.0", "streamlit>=1.28.0", "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "metaxai-benchmark=benchmarks.evaluation:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
