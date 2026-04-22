# Wine Quality Prediction
**MSIM — Introduction to Data Science | Final Project**

A machine learning pipeline that predicts whether a red wine is **High quality** (score ≥ 6) or **Low quality** (score ≤ 5) based on its physicochemical properties, using the WineQT dataset.

---

## Research Questions

| # | Research Question |
|---|---|
| RQ1 | Can wine quality be accurately predicted from physicochemical properties alone? |
| RQ2 | Which physicochemical features are the most important predictors of wine quality? |
| RQ3 | Which classification model performs best? |
| RQ4 | Does applying StandardScaler improve model performance, and for which models? |

---

## Dataset

**File:** `WineQT.csv`

| Property | Value |
|---|---|
| Samples | 1,143 red wines |
| Features | 11 physicochemical properties |
| Target | `quality` score (3–8), binarised to Low (0) / High (1) |
| Missing values | None |
| Duplicates | None |

**Features:** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol

**Class distribution after binarisation:**
- Low quality (score ≤ 5): 522 samples (45.7%)
- High quality (score ≥ 6): 621 samples (54.3%)

---

## Project Structure

```
Project/
├── Wine_Quality_Prediction.ipynb   # Main notebook (EDA + modelling + results)
├── wine_quality_prediction.py      # Standalone Python script (same pipeline)
├── WineQT.csv                      # Dataset
├── proposal.pdf                    # Project proposal
├── outputs/                        # Saved plots (generated on run)
│   ├── 01_quality_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_boxplots_alcohol_volatility.png
│   ├── 04_rf_feature_importance.png
│   ├── 05_lr_feature_coefficients.png
│   ├── 06_model_accuracy_comparison.png
│   └── 07_model_f1_comparison.png
└── README.md
```

---

## Pipeline Overview

### 1. Exploratory Data Analysis
- Shape, data types, missing values, duplicates
- Summary statistics with coefficient of variation
- Quality score distribution (bar chart)
- Feature correlation heatmap
- Boxplots of alcohol and volatile acidity vs quality

### 2. Preprocessing
- Drop non-informative `Id` column
- Binarise `quality` target (threshold ≥ 6)
- 80/20 train-test split (`random_state=42`)
- StandardScaler fitted on training set only (no data leakage)

### 3. Model Training
Four classifiers trained on scaled data:

| Model | Accuracy | Precision (High) | Recall (High) | F1 Macro |
|---|---|---|---|---|
| **Random Forest** | **77.29%** | **0.8049** | 0.7795 | **0.7710** |
| Logistic Regression | 76.86% | 0.7937 | 0.7874 | 0.7660 |
| KNN | 72.49% | 0.7424 | 0.7717 | 0.7201 |
| Decision Tree | 68.12% | 0.7143 | 0.7087 | 0.6777 |

### 4. Feature Scaling Experiment (RQ4)
Models retrained on unscaled data for direct comparison. KNN gained +5.24% from scaling; tree-based models were unaffected.

### 5. Feature Importance (RQ2)
- Random Forest: mean decrease in impurity
- Logistic Regression: coefficient magnitudes (on scaled features)

**Top features (both models agree):**
1. Alcohol — strongest positive predictor
2. Sulphates — positive effect
3. Volatile acidity — strongest negative predictor

### 6. Model Comparison Visualisations (RQ3)
- Accuracy bar chart with winner highlighted
- Grouped F1-score chart (Low / High / Macro per model)

---

## Key Results

| Research Question | Answer |
|---|---|
| **RQ1** | YES — all models beat the 50% random baseline by 18–27 percentage points |
| **RQ2** | Alcohol (#1), sulphates (#2), volatile acidity (#3) — consistent across both models |
| **RQ3** | Random Forest wins (77.29% accuracy, macro F1 = 0.771) |
| **RQ4** | Scaling significantly helped only KNN (+5.24%); tree-based models are invariant |

### Hypothesis Validation

| Hypothesis | Result |
|---|---|
| H1: All models beat 50% baseline | PASS |
| H2: Alcohol & volatile acidity are top-2 features | FAIL — sulphates displaced volatile acidity at #2 in Random Forest |
| H3: Random Forest achieves highest accuracy | PASS |
| H4: Scaling boosts KNN and Logistic Regression | PASS (partial — LR showed no material gain) |

**3 out of 4 hypotheses supported.**

---

## How to Run

### Jupyter Notebook
```bash
jupyter notebook Wine_Quality_Prediction.ipynb
```
Run all cells top to bottom. All plots render inline and are also saved to `outputs/`.

### Python Script
```bash
python wine_quality_prediction.py
```
Runs the full pipeline and saves all plots to `outputs/`.

### Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plot generation |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | Preprocessing, models, evaluation metrics |

---

## Limitations & Future Work

- The ~23% error rate suggests unmeasured factors (grape variety, vintage, taster subjectivity) also influence quality scores.
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) and k-fold cross-validation would yield more robust performance estimates.
- Class imbalance techniques (SMOTE, class weights) may improve recall for the minority class.
- Extending to multi-class prediction (scores 3–8 directly) is a natural next step.
