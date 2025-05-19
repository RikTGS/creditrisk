# creditrisk
Master thesis: the impact of rating definition methods and calibrating probabilities of default on capital

# Credit Risk Rating & PD Calibration Framework

This repository provides a modular framework for credit rating assignment and probability of default (PD) calibration. It supports multiple binning and calibration strategies and evaluates their impact on expected loss (EL), unexpected loss (UL), and overall rating performance.

## 📊 Key Features

- Multiple rating assignment techniques:
  - Equal-width, equal-count, and equal-risk binning
  - K-means clustering
  - Regression trees
  - Optimized threshold search
  - Stepwise objective threshold tuning

- PD calibration methods:
  - Historical default rate
  - Through-the-cycle (TTC)
  - Quasi-moment matching (QMM)
  - Scaled likelihood ratio (SLR)
  - Isotonic regression

- Evaluation metrics:
  - TEL, TUL, Brier score
  - Rating sensitivity to granularity
  - Statistical comparisons (ANOVA, Kruskal-Wallis, Tukey HSD)

## 📁 Structure Overview

- `gmsc_preprocessing.py`: Preprocessing pipeline for the GMSC dataset
- `logreg.py`: Logistic regression model setup and evaluation
- `scales.py`, `threshold_assignment.py`: Rating scale creation and assignment
- `regression_tree.py`, `clustering.py`, `search_scale.py`, `objective_funct.py`: Rating assignment strategies
- `pd_calib.py`: PD calibration implementations
- `EL_UL_K_Calc.py`: Expected and unexpected loss calculations
- `plot_*.py`: Analysis and visualization scripts

## ⚙️ Getting Started

1. Prepare and preprocess your dataset using `gmsc_preprocessing.py`
2. Train or load your predictive model (e.g. logistic regression)
3. Apply a rating assignment method (e.g. regression tree or clustering)
4. Calibrate PDs using one of the available techniques
5. Evaluate model performance using Brier score, TEL, TUL
6. Visualize outcomes via the plotting scripts

## 📦 Dependencies

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `statsmodels`, `scikit-posthocs`, `imblearn`

Install all requirements via:

```bash
pip install -r requirements.txt
