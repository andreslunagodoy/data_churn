# Telco Customer Churn — Learner's Guide

A detailed walkthrough of every tool, technique, and design decision used in this project. If you're learning data science or ML engineering, this document explains not just *what* was done but *why*, and what alternatives exist.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [The Dataset](#2-the-dataset)
3. [Data Cleaning Techniques](#3-data-cleaning-techniques)
4. [Feature Engineering](#4-feature-engineering)
5. [Preprocessing Pipelines](#5-preprocessing-pipelines)
6. [Class Imbalance — Why It Matters and How to Fix It](#6-class-imbalance--why-it-matters-and-how-to-fix-it)
7. [Phase 1 Models — Sklearn Baselines](#7-phase-1-models--sklearn-baselines)
8. [Phase 2 — Hyperparameter Tuning](#8-phase-2--hyperparameter-tuning)
9. [Phase 3 Models — XGBoost, LightGBM, CatBoost](#9-phase-3-models--xgboost-lightgbm-catboost)
10. [Optuna — Bayesian Hyperparameter Optimization](#10-optuna--bayesian-hyperparameter-optimization)
11. [Ensemble Methods — Voting and Stacking](#11-ensemble-methods--voting-and-stacking)
12. [Evaluation Metrics — Which One to Trust](#12-evaluation-metrics--which-one-to-trust)
13. [Threshold Optimization](#13-threshold-optimization)
14. [Interpretability — SHAP Values](#14-interpretability--shap-values)
15. [PCA — Dimensionality Reduction and Visualization](#15-pca--dimensionality-reduction-and-visualization)
16. [SMOTE — Synthetic Oversampling](#16-smote--synthetic-oversampling)
17. [Calibration — Can You Trust the Probabilities?](#17-calibration--can-you-trust-the-probabilities)
18. [What Didn't Work and Why](#18-what-didnt-work-and-why)
19. [When to Stop Tuning](#19-when-to-stop-tuning)

---

## 1. The Problem

**Churn prediction** is a binary classification task: given a customer's profile, predict whether they will leave (churn = 1) or stay (churn = 0). It's one of the most common business ML problems because acquiring a new customer costs 5-25x more than retaining an existing one.

The goal isn't just high accuracy — it's catching as many churners as possible (high recall) while not wasting retention budget on customers who would have stayed anyway (reasonable precision).

---

## 2. The Dataset

The [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset has **7,043 rows and 21 columns**. It's small by industry standards but widely used for learning.

**Feature types:**
- **Numeric:** `tenure`, `MonthlyCharges`, `TotalCharges`
- **Binary categorical:** `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`
- **Multi-class categorical:** `Contract`, `InternetService`, `PaymentMethod`
- **Internet-dependent services:** `OnlineSecurity`, `OnlineBackup`, etc. — these have three values: `Yes`, `No`, and `No internet service`

**Key characteristics:**
- **Class imbalance:** 73.5% No Churn / 26.5% Churn — this is moderate imbalance but enough to bias models
- **No missing values** at first glance, but `TotalCharges` has 11 whitespace strings that `pd.to_numeric` converts to NaN — all are tenure=0 customers who haven't been charged yet
- **No duplicates**

---

## 3. Data Cleaning Techniques

### TotalCharges Conversion
`TotalCharges` is stored as a string in the CSV. We use `pd.to_numeric(df['TotalCharges'], errors='coerce')` which converts valid numbers and turns unparseable strings into `NaN`. The 11 NaN rows are all tenure=0 customers — filling with 0 is the correct semantic choice (they haven't paid anything yet).

**Why `errors='coerce'` instead of `errors='raise'`?** Because we want to identify *which* rows fail rather than crashing. In production, you'd log these for investigation.

### Binary Encoding
Columns like `Partner` have values `Yes`/`No`. We map these to 1/0 using `(df[col] == 'Yes').astype(int)`. This is simpler and more explicit than `LabelEncoder`, which assigns arbitrary numeric codes alphabetically (making `No` = 0, `Yes` = 1 by coincidence, but `Female` = 0, `Male` = 1 also by coincidence — easy to confuse).

### Internet-Dependent Services
Columns like `OnlineSecurity` have three values: `Yes`, `No`, `No internet service`. Since `No internet service` and `No` both mean "doesn't have this service," we collapse them: `(df[col] == 'Yes').astype(int)`. The internet service type itself is captured separately in `InternetService`.

---

## 4. Feature Engineering

Feature engineering is the process of creating new columns from existing data to help models find patterns. This is often more impactful than choosing a fancier algorithm.

### Count Features
- **`num_services`**: Total subscribed services (0-8). Customers with more services have higher switching costs and tend to churn less.
- **`num_protection`**: Count of security/backup/support services. A proxy for customer engagement with "value-add" features.
- **`num_streaming`**: Streaming services specifically. These customers use the internet heavily but may not be "sticky" in the same way protection services make them.

### Revenue Features
- **`avg_monthly_charge`**: `TotalCharges / (tenure + 1)`. Approximates what the customer has historically paid per month. We add 1 to tenure to avoid division by zero for brand-new customers.
- **`charge_ratio`**: `avg_monthly_charge / MonthlyCharges`. Values below 1 suggest the customer's plan got more expensive over time — a potential churn driver.
- **`charge_per_service`**: `MonthlyCharges / (num_services + 1)`. How much each service "costs." Helps distinguish customers who pay a lot for few services (bad value) vs. many services (good value).
- **`tenure_x_monthly`**: Interaction between tenure and monthly charges. Captures the idea that a high-paying long-term customer is very different from a high-paying new customer.

### Lifecycle Features
- **`is_new`**: Tenure ≤ 6 months. New customers are the highest-risk group — they haven't built loyalty or switching costs yet.
- **`is_mid`**: 6 < tenure ≤ 24 months. The "settling in" period.
- **`is_loyal`**: Tenure > 48 months. These customers rarely churn.
- **`tenure_sq`**: Squared tenure. Captures the non-linear relationship — churn risk drops steeply in the first year then flattens out. Tree models can learn this from raw tenure, but it helps linear models.

### Interaction Features
- **`new_echeck`**: `is_new * electronic_check`. New customers paying via electronic check churn at an exceptionally high rate. This interaction captures a specific high-risk subgroup.
- **`mtm_no_protect`**: Month-to-month contract AND no protection services. These customers have no contractual or service-based "stickiness."
- **`fiber_highcharge`**: Fiber optic AND above-median monthly charges. Fiber customers already churn more; those paying the most have the highest dissatisfaction signal.
- **`mtm_new_fiber`**: Three-way interaction — month-to-month, new, fiber optic. The highest-risk subgroup in the entire dataset.

**Why create interactions manually when tree models can learn them?** Two reasons: (1) linear models like Logistic Regression *cannot* learn interactions on their own — they need explicit feature crosses, and (2) even tree models benefit from explicit interactions because they reduce the depth needed to capture the pattern, leading to better generalization.

---

## 5. Preprocessing Pipelines

### Why Two Pipelines?

Different model families have different preprocessing needs:

**Linear models (Logistic Regression):**
- **Scaling** is mandatory — features on different scales (tenure: 0-72, MonthlyCharges: 18-119, tenure_sq: 0-5184) would dominate the loss function based on magnitude alone
- `StandardScaler` centers each feature to mean=0, std=1
- **One-hot encoding with `drop='first'`** — dropping one dummy column avoids the "dummy variable trap" (perfect multicollinearity) which breaks linear models

**Tree models (Random Forest, XGBoost, etc.):**
- **No scaling needed** — trees split on thresholds, so scale doesn't matter
- **OHE without dropping** — trees can handle the full set of dummies, and keeping all levels helps when one specific category (e.g., `Month-to-month`) is important

### ColumnTransformer
`sklearn.compose.ColumnTransformer` lets you apply different transformations to different columns in one step. This is cleaner than manually subsetting dataframes and avoids column alignment bugs.

### Pipeline
`sklearn.pipeline.Pipeline` chains steps (impute → scale → encode) so you can fit/transform in one call. More importantly, it prevents **data leakage**: the scaler learns its mean/std only from training data, and the same transform is applied to test data.

---

## 6. Class Imbalance — Why It Matters and How to Fix It

With 73.5% non-churn and 26.5% churn, a model that predicts "No Churn" for every customer gets 73.5% accuracy. That's the **majority class baseline** — and it catches zero churners.

Standard ML algorithms minimize overall error, which means they favor the majority class. The result: high accuracy, terrible recall.

### Solutions Used in This Project

**`class_weight='balanced'` (Phase 2):** This is the simplest and most effective fix. It tells the algorithm to weight misclassifications of the minority class higher. Internally, sklearn multiplies the loss for class `c` by `n_samples / (n_classes * n_samples_c)`. For our data, churn samples get weighted ~2.77x higher than non-churn.

**`scale_pos_weight` (XGBoost, Phase 3):** XGBoost's equivalent — it multiplies the gradient for positive (churn) samples. Setting it to `neg_count / pos_count ≈ 2.77` is the balanced equivalent.

**`is_unbalance=True` (LightGBM):** LightGBM's built-in flag that does the same reweighting.

**`auto_class_weights='Balanced'` (CatBoost):** CatBoost's version. `'SqrtBalanced'` is a gentler variant that uses the square root of the ratio.

**SMOTE (Phase 3):** Synthetic Minority Over-sampling Technique generates synthetic training samples for the minority class by interpolating between existing minority samples and their k-nearest neighbors. Unlike simple oversampling (duplicating existing rows), SMOTE creates novel data points in feature space. We use it on the *training* set only — never on the test set, which would be data leakage.

**Which is better?** In our experiments, model-native reweighting (`class_weight`, `scale_pos_weight`) slightly outperformed SMOTE. This is common for tabular data. SMOTE shines more when the minority class is severely underrepresented (<5%) or when the decision boundary is complex.

---

## 7. Phase 1 Models — Sklearn Baselines

### Logistic Regression
The simplest classification model. It fits a linear function `w1*x1 + w2*x2 + ... + b` and passes it through a sigmoid function to get a probability. Fast, interpretable, and a strong baseline for problems with roughly linear decision boundaries.

**When to use:** First model you should always try. If it gets close to tree models, the problem is mostly linear and you don't need complexity.

### Random Forest
An ensemble of decision trees, each trained on a random subset of the data and features. Predictions are averaged (regression) or majority-voted (classification). The randomness reduces overfitting compared to a single deep tree.

**Key parameters:**
- `n_estimators`: Number of trees. More is generally better until returns diminish (~100-500).
- `max_depth`: Maximum tree depth. Limits complexity; `None` means trees grow until leaves are pure or minimum samples are reached.
- `max_features`: Number of features considered at each split. `'sqrt'` is common — it decorrelates the trees.

### Gradient Boosting
Builds trees sequentially, where each new tree corrects the errors of the previous ensemble. Unlike Random Forest (parallel, averaging), Gradient Boosting is sequential and additive.

**Why it usually beats Random Forest:** Each tree is specifically targeted at the current weaknesses of the model, making it a more efficient learner.

---

## 8. Phase 2 — Hyperparameter Tuning

### RandomizedSearchCV
Instead of trying every combination of parameters (GridSearchCV, which is exponentially expensive), `RandomizedSearchCV` samples random combinations from specified distributions. Research shows that random search finds good hyperparameters faster than grid search because not all parameters are equally important — random search explores the important dimensions more efficiently.

**Key settings:**
- `n_iter`: Number of random combinations to try. 40-60 is a reasonable budget for most problems.
- `scoring='roc_auc'`: The metric to optimize during CV. We chose ROC-AUC because it evaluates the entire ranking quality, not just a single threshold.
- `cv=StratifiedKFold(5)`: 5-fold cross-validation that preserves the class ratio in each fold.
- `n_jobs=-1`: Use all CPU cores.

### HistGradientBoostingClassifier
Sklearn's histogram-based gradient boosting, inspired by LightGBM. It bins continuous features into 256 histogram buckets, making split finding O(n_bins) instead of O(n_samples). This makes it 10-100x faster than the older `GradientBoostingClassifier` on datasets with >10k rows.

**Why we switched to it in Phase 2:** Same accuracy, dramatically faster training, supports `class_weight='balanced'` natively.

---

## 9. Phase 3 Models — XGBoost, LightGBM, CatBoost

These are the three dominant gradient boosting libraries. They share the same core idea (sequential tree building to correct errors) but differ in implementation details.

### XGBoost (eXtreme Gradient Boosting)
The original "fast gradient boosting" library. Key innovations:
- **Regularization:** L1 and L2 penalties on leaf weights (`reg_alpha`, `reg_lambda`) reduce overfitting
- **`tree_method='hist'`**: Histogram-based splitting (same idea as sklearn's HistGBM)
- **`colsample_bytree`**: Randomly selects a fraction of features per tree (like Random Forest's `max_features`)
- **`subsample`**: Uses a random fraction of training rows per tree (bagging within boosting)
- **`gamma`**: Minimum loss reduction required to make a further split — acts as a complexity penalty

### LightGBM (Light Gradient Boosting Machine)
Microsoft's entry. Key differences from XGBoost:
- **Leaf-wise tree growth** (vs. XGBoost's level-wise). LightGBM grows the leaf with the highest loss reduction, producing deeper, more asymmetric trees. This is more efficient but can overfit on small datasets — `num_leaves` controls this.
- **Exclusive Feature Bundling (EFB)**: Automatically bundles mutually exclusive features (e.g., one-hot columns) to reduce dimensionality
- **`is_unbalance`**: Built-in class reweighting
- Generally the fastest of the three on CPU

### CatBoost (Categorical Boosting)
Yandex's entry. Key innovation:
- **Ordered target encoding**: Handles categorical features natively without OHE. Instead of encoding before training, CatBoost converts categoricals to numbers using running target statistics (with randomization to prevent leakage). This is why CatBoost can take raw categorical columns.
- **Ordered boosting**: Uses a permutation-based scheme to reduce prediction shift (a form of target leakage in standard gradient boosting)
- **`auto_class_weights`**: `'Balanced'` or `'SqrtBalanced'` for class imbalance
- Tends to work well out-of-the-box with less tuning

**In our case, we used CatBoost on the OHE-preprocessed data** (same as XGBoost/LightGBM) rather than passing raw categoricals. This was a pragmatic choice to keep preprocessing consistent. In practice, CatBoost's native handling shines more when you have high-cardinality categoricals (e.g., zip codes, product IDs).

### Why Three Libraries for the Same Algorithm?

Each has slightly different inductive biases (tree growth strategy, regularization defaults, handling of categoricals). In competitive ML, the best single model varies by dataset. Ensembling all three captures their complementary strengths.

---

## 10. Optuna — Bayesian Hyperparameter Optimization

### Why Not RandomizedSearchCV?

`RandomizedSearchCV` samples parameter combinations **independently** — each trial ignores the results of previous trials. Optuna uses **Bayesian optimization** (specifically, the Tree-structured Parzen Estimator / TPE): it builds a probabilistic model of which parameter regions yield good results and samples more from those regions over time.

**Analogy:** RandomizedSearchCV is like throwing darts blindfolded. Optuna is like throwing a dart, seeing where it landed, and adjusting your aim for the next throw.

### How TPE Works (Simplified)
1. Run a few random trials (warm-up)
2. Split all completed trials into "good" (above-median performance) and "bad" groups
3. Fit a density estimator to the parameter values of each group
4. Sample the next trial's parameters from regions where "good" trials are dense and "bad" trials are sparse
5. Repeat

### Key Optuna Concepts
- **Study:** The optimization session. `direction='maximize'` because we want high ROC-AUC.
- **Trial:** A single parameter configuration being evaluated.
- **`suggest_*` methods:** `suggest_int`, `suggest_float`, `suggest_categorical` define the search space. `log=True` for learning rate and regularization means the search is uniform in log-space (trying 0.01, 0.03, 0.1, 0.3 instead of 0.01, 0.08, 0.15, 0.22).
- **`n_trials`:** Budget. We used 50 for XGBoost/LightGBM and 30 for CatBoost (slower). More trials = better parameters, but with diminishing returns.

### CatBoost and `manual_cv_auc`
CatBoost models don't serialize well with Python's `pickle`/`joblib`, which sklearn's `cross_val_score` uses internally for parallelism (`n_jobs=-1`). Our workaround: a manual CV loop that trains and evaluates sequentially. This makes CatBoost tuning slower but avoids crashes.

---

## 11. Ensemble Methods — Voting and Stacking

### Soft Voting
The simplest ensemble: average the predicted probabilities from multiple models, then apply a threshold.

```
P(churn) = (P_lr + P_xgb + P_lgb + P_catboost + P_xgb_smote) / 5
```

This works because different models make different errors. Averaging tends to cancel out individual model noise, producing smoother, more robust predictions. No extra training needed.

### Stacking
A more sophisticated ensemble that *learns* how to combine base models:

1. **Generate out-of-fold (OOF) predictions:** For each base model, use 5-fold CV on the training set. Each training sample gets a prediction from a model that *didn't* see it during training. This prevents data leakage.
2. **Build a meta-feature matrix:** Stack the OOF predictions into a new dataset where each column is one base model's predictions.
3. **Train a meta-learner:** A simple model (Logistic Regression in our case) that learns the optimal weighting of base model predictions.
4. **Predict on test:** Pass each base model's test predictions through the meta-learner.

**Why OOF instead of just training predictions?** If you used the base models' predictions on the *training* set, the meta-learner would learn to trust whichever model memorizes the training data best (overfitting). OOF predictions simulate "unseen data" performance for each model.

**Meta-learner weights** in our case were approximately `LR: 2.9, XGB: 0.4, LGB: 0.4, CB: 1.7`. This means the meta-learner leans heavily on Logistic Regression and CatBoost — suggesting they contribute the most unique signal. XGBoost and LightGBM are largely redundant with each other (which makes sense — they're similar algorithms).

---

## 12. Evaluation Metrics — Which One to Trust

### Accuracy
`(TP + TN) / total`. Misleading with imbalanced classes — 73.5% accuracy is free by predicting "No Churn" for everyone.

### Precision
`TP / (TP + FP)`. Of customers the model flags as churners, how many actually are? High precision means fewer wasted retention offers. **Business cost of low precision:** spending money on customers who would have stayed anyway.

### Recall (Sensitivity)
`TP / (TP + FN)`. Of all actual churners, how many did the model catch? High recall means fewer missed churners. **Business cost of low recall:** losing customers you could have saved.

### F1 Score
`2 * Precision * Recall / (Precision + Recall)`. The harmonic mean — it penalizes imbalance between precision and recall. A model with 90% precision but 10% recall gets F1 = 0.18, not 0.50.

### ROC-AUC
Area Under the Receiver Operating Characteristic curve. Measures the model's ability to **rank** positive cases above negative cases, across *all* possible thresholds. Unlike F1, it's threshold-independent — it tells you "how good are the raw probabilities?" rather than "how good is the yes/no decision?"

**Why we optimize ROC-AUC during CV but report F1 on the test set:** ROC-AUC is a more stable optimization target (not threshold-dependent), but business decisions need F1/precision/recall at a specific operating point.

### Which Metric Matters Most for Churn?
Depends on the business context:
- **If retention offers are cheap** (e.g., a "thank you" email): optimize for **recall** — catch every churner
- **If retention offers are expensive** (e.g., a 50% discount): optimize for **precision** — only target likely churners
- **In general:** F1 or a threshold-optimized tradeoff is the pragmatic choice

---

## 13. Threshold Optimization

By default, classifiers predict class 1 when P(class 1) > 0.5. But 0.5 is arbitrary — it assumes equal misclassification costs and balanced classes. Neither is true for churn.

### How It Works
1. Generate predicted probabilities on the test set
2. Sweep through every possible threshold (0.01, 0.02, ..., 0.99)
3. At each threshold, compute precision, recall, and F1
4. Pick the threshold that maximizes your chosen metric (F1 in our case)

In our experiments, optimal thresholds ranged from 0.30 to 0.60 depending on the model. The Gradient Boosting baseline (Phase 1) has its optimal threshold at 0.30 — far below the default 0.50 — because without class weighting, its probabilities are skewed toward the majority class.

**Important caveat:** Threshold optimization should be done on a validation set, not the test set. We did it on the test set for simplicity, but in production you'd use a held-out validation set or cross-validation to avoid overfitting the threshold to test data.

---

## 14. Interpretability — SHAP Values

SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explaining individual predictions. For each prediction, SHAP assigns an importance value to each feature that represents how much it pushed the prediction toward churn (positive SHAP) or away from churn (negative SHAP).

### Why SHAP Over Feature Importance?

**Built-in feature importance** (e.g., Gini importance in Random Forest) measures how often a feature is used for splitting, weighted by the improvement it provides. This has known biases: it favors high-cardinality features and features correlated with others.

**Permutation importance** shuffles a feature's values and measures the drop in performance. It's model-agnostic but gives a single global number per feature.

**SHAP values** provide:
- **Per-sample explanations:** Why did this *specific* customer get a high churn score?
- **Direction:** Not just "tenure is important" but "low tenure pushes toward churn, high tenure pushes against"
- **Interaction effects:** SHAP can decompose feature interactions (though we didn't use this in the project)
- **Consistency guarantees:** Based on Shapley values from cooperative game theory, which have provable fairness properties

### Reading the SHAP Summary Plot
- Each row is a feature, each dot is a test sample
- Dot color = feature value (red = high, blue = low)
- Dot position on x-axis = SHAP value (right = pushes toward churn)
- If "tenure" shows blue dots on the right and red dots on the left: low tenure pushes toward churn, high tenure pushes against churn

---

## 15. PCA — Dimensionality Reduction and Visualization

Principal Component Analysis finds orthogonal linear combinations of the original features that capture the most variance. We used it for two purposes:

### 2D Visualization
Project the high-dimensional feature space onto 2 components to see if churn/non-churn clusters are visually separable. In our case, the two classes overlap significantly — confirming that the decision boundary is complex (not linearly separable in 2D).

### Variance Analysis
The cumulative variance plot shows how many components are needed to capture 95% of the total variance (20 out of 45 features in our case). This tells us roughly half the features are redundant or highly correlated.

**Did we use PCA as a preprocessing step for modeling?** No. Tree-based models handle redundant features naturally (they just ignore irrelevant ones). PCA can help linear models, but it destroys interpretability — the components are abstract mixtures of original features. Since interpretability matters for churn (we want to explain *why* a customer is at risk), we kept the original features.

---

## 16. SMOTE — Synthetic Oversampling

SMOTE (Synthetic Minority Over-sampling Technique) generates new minority-class training samples by interpolating between existing ones:

1. For each minority sample, find its k nearest neighbors (in feature space)
2. Randomly pick one of those neighbors
3. Create a new synthetic sample at a random point along the line between the original and the neighbor

This produces a balanced training set (50/50 in our case: 4,132 per class, up from 1,494 churn samples).

### Why SMOTE Over Random Oversampling?
Random oversampling duplicates existing samples, which gives the model multiple copies of the same point. This can cause overfitting — the model memorizes the minority examples. SMOTE creates *new* points, filling in the minority class region of feature space.

### SMOTE Limitations
- **Operates in feature space** — the synthetic samples are linear interpolations, which may not reflect real-world data distributions
- **Sensitive to noise** — if a minority sample is an outlier, SMOTE will create synthetic points in a misleading region
- **Must only be applied to training data** — applying SMOTE to the full dataset before splitting is a common and serious data leakage mistake

### Our Results
SMOTE-trained XGBoost performed comparably to XGBoost with `scale_pos_weight` but not better. Model-native reweighting is simpler and avoids the risks of synthesizing potentially unrealistic data points.

---

## 17. Calibration — Can You Trust the Probabilities?

A model is **well-calibrated** if its predicted probability matches the true event rate. If the model says "30% chance of churn" for 1,000 customers, roughly 300 should actually churn.

### Why It Matters
Many business applications need reliable probabilities, not just rankings:
- Expected revenue loss calculations
- Cost-benefit analysis for retention offers
- Risk tiers (high/medium/low risk)

### Calibration Plot
The calibration curve plots predicted probability (x-axis) vs. observed frequency (y-axis) in binned groups. A perfectly calibrated model follows the diagonal. Curves above the diagonal are *under-confident* (predict lower probabilities than the actual churn rate), curves below are *over-confident*.

**Our findings:** Tree models (especially with class weighting) tend to be slightly miscalibrated in the extremes. If reliable probabilities matter, consider `CalibratedClassifierCV` (Platt scaling or isotonic regression) as a post-processing step.

---

## 18. What Didn't Work and Why

**Phase 3 didn't meaningfully improve over Phase 2.** XGBoost, LightGBM, and CatBoost all converge to ROC-AUC ~0.849, the same as sklearn's `HistGradientBoostingClassifier`. F1 improved by 0.004 (within statistical noise for 1,409 test samples).

**Why?** The performance ceiling is determined by the **information content of the features**, not the model's capacity. All 20 original features are simple categorical/numeric fields. The engineered features extract more signal, but there's a finite amount of predictive information in demographics and subscription details alone.

**What would actually help:**
- **Behavioral data:** Call/data usage patterns, browsing history, app usage frequency
- **Temporal data:** Trend in monthly charges, number of support tickets over time, declining usage
- **External data:** Competitive offers in the market, coverage quality by geography
- **More data:** 7,043 samples is small. Models would generalize better with 100k+ customers.

---

## 19. When to Stop Tuning

A practical question every ML practitioner faces. Our three phases illustrate the diminishing returns:

| Phase | Effort | ROC-AUC | F1 | Marginal Gain |
|-------|--------|---------|-----|---------------|
| 1 (baselines) | 2 seconds | 0.843 | 0.575 | — |
| 2 (tuning + balancing) | 2 minutes | 0.846 | 0.636 | +0.061 F1 |
| 3 (advanced GBMs + Optuna) | 7 minutes | 0.849 | 0.637 | +0.001 F1 |

**Phase 2 was the sweet spot:** 90% of the achievable performance at 3% of the total compute. The lesson: invest your time in data quality, feature engineering, and handling class imbalance before reaching for exotic algorithms. The marginal gain from XGBoost over sklearn's HistGBM was essentially zero on this dataset.

**Signs you've hit the ceiling:**
- All model families converge to the same ROC-AUC
- Hyperparameter tuning moves the 4th decimal place
- Cross-validation scores have higher variance than the gap between models
- Feature importance shows the same top features regardless of the algorithm

At that point, **more models won't help — more data will.**
