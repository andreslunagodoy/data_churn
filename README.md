# Telco Customer Churn — Predictive Analysis

Predicting which telecom customers are likely to churn using the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset (7,043 customers, 21 features). The project progresses through three modeling phases, from simple baselines to state-of-the-art gradient boosting with Bayesian hyperparameter optimization.

## Results

| Phase | Best Model | ROC-AUC | F1 | Recall | Runtime |
|-------|-----------|---------|-----|--------|---------|
| 1 — Baselines | Gradient Boosting | 0.843 | 0.575 | 0.519 | ~2s |
| 2 — Tuned sklearn | HGB (balanced + tuned) | 0.846 | 0.636 | 0.794 | ~2min |
| 3 — XGB/LGB/CatBoost | Stacking ensemble | 0.849 | 0.632 | 0.800 | ~7min |

The biggest gain is Phase 1 to Phase 2 (adding `class_weight='balanced'`). Phase 3 confirms the performance ceiling with diminishing returns.

## Project Structure

```
churn_2602/
  telco_churn.csv              # Raw dataset
  Churn_DA.ipynb               # Original analysis (sklearn baselines, no tuning)
  Churn_DA_advanced.ipynb      # All three phases: baselines → tuned sklearn → XGB/LGB/CatBoost
README.md                      # This file
README_learners.md             # Detailed walkthrough of tools and techniques
```

## Setup

```bash
# Create and activate the environment
conda create -n data_churn python=3.11 -y
conda activate data_churn

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn \
    xgboost lightgbm catboost optuna imbalanced-learn \
    ipykernel jupyter shap scipy

# Register Jupyter kernel
python -m ipykernel install --user --name data_churn --display-name "Python (data_churn)"
```

The original notebook (`Churn_DA.ipynb`) only requires `pandas numpy scikit-learn matplotlib seaborn`. The full stack is needed for the advanced notebook.

## Quick Start

Open `Churn_DA_advanced.ipynb` — it contains all three phases with timing instrumentation and a complete comparison. Run all cells; expect ~9 minutes total.

## Key Findings

**Top churn drivers** (consistent across all model families):
1. **Contract type** — month-to-month customers churn at ~3x the rate of annual contracts
2. **Tenure** — customers in their first 6 months are the highest risk group
3. **Internet service** — fiber optic customers churn more (higher charges, possible quality gap)
4. **Payment method** — electronic check correlates with higher churn
5. **Protection services** — customers without security/backup/support services leave more often

**The performance ceiling (~0.849 ROC-AUC) is a data limitation**, not a model limitation. All algorithms converge to the same band. Further improvement requires richer features (usage logs, complaint history, time-series behavior).
