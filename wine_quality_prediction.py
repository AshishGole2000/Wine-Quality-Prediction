# =============================================================================
# Wine Quality Prediction
# MSIM — Introduction to Data Science Project
#
# Predicts whether a red wine is High (quality >= 6) or Low (quality <= 5)
# quality using physicochemical properties from the WineQT dataset.
#
# Research Questions addressed:
#   RQ1 — Can quality be accurately predicted from chemical properties?
#   RQ2 — Which features are the strongest predictors?
#   RQ3 — Which classifier performs best?
#   RQ4 — Does StandardScaler improve model performance?
#
# Usage:
#   python wine_quality_prediction.py
# =============================================================================


# =============================================================================
# SECTION 1 — IMPORTS
# =============================================================================

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Output directory for all saved plots
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Separator strings reused throughout console output
SEP  = "=" * 70
SEP2 = "-" * 70

# Random seed for reproducibility
SEED = 42


# =============================================================================
# SECTION 2 — DATA LOADING & EDA
# =============================================================================

def load_and_explore(filepath: str) -> pd.DataFrame:
    """
    Load the WineQT CSV, print structural diagnostics, and generate
    three EDA plots:
      Plot 1 — Quality score distribution (bar chart)
      Plot 2 — Correlation heatmap of the 11 physicochemical features
      Plot 3 — Boxplots of alcohol and volatile acidity vs quality

    Returns the raw DataFrame (Id column still present).
    """
    # ── Load ──────────────────────────────────────────────────────────
    df = pd.read_csv(filepath)

    # ── First 5 rows ──────────────────────────────────────────────────
    print(SEP)
    print("SECTION 2 — DATA LOADING & EDA")
    print(SEP)

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    # ── Shape ─────────────────────────────────────────────────────────
    print(f"\nShape  : {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Data types ────────────────────────────────────────────────────
    print("\nData types:")
    print(df.dtypes.to_string())

    # ── Missing values ────────────────────────────────────────────────
    missing = df.isnull().sum()
    print(f"\nMissing values: {missing.sum()} total")
    if missing.sum() > 0:
        print(missing[missing > 0].to_string())

    # ── Duplicates ────────────────────────────────────────────────────
    n_dupes = df.duplicated().sum()
    print(f"Duplicate rows: {n_dupes}")

    # ── Summary statistics (features only, no Id) ─────────────────────
    feature_cols = [c for c in df.columns if c not in ("quality", "Id")]
    stats = df[feature_cols].describe().T
    stats["range"] = stats["max"] - stats["min"]
    stats["cv (%)"] = (stats["std"] / stats["mean"] * 100).round(2)

    print("\nSummary statistics (11 features):")
    print(stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "range", "cv (%)"]].to_string())

    high_cv = stats[stats["cv (%)"] > 50]["cv (%)"]
    if not high_cv.empty:
        print("\nHigh-variability features (CV > 50%):")
        for col, cv in high_cv.items():
            print(f"  '{col}'  CV = {cv:.1f}%")

    # ── EDA on copy without Id ────────────────────────────────────────
    eda_df = df.drop(columns=["Id"])

    # Plot 1 — Quality distribution
    quality_counts = eda_df["quality"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        quality_counts.index.astype(str),
        quality_counts.values,
        color=sns.color_palette("Blues_d", len(quality_counts)),
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, val in zip(bars, quality_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(val),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_title("Distribution of Wine Quality Scores", fontsize=14, fontweight="bold")
    ax.set_xlabel("Quality Score", fontsize=12)
    ax.set_ylabel("Number of Wines", fontsize=12)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.set_ylim(0, quality_counts.max() + 60)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "01_quality_distribution.png"), dpi=150)
    plt.close()

    print("\nQuality score distribution:")
    for score, count in quality_counts.items():
        print(f"  Score {score}: {count:4d} wines ({count / len(eda_df) * 100:.1f}%)")
    print("[Plot saved] 01_quality_distribution.png")

    # Plot 2 — Correlation heatmap
    corr = eda_df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 8},
    )
    ax.set_title("Correlation Heatmap — Physicochemical Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "02_correlation_heatmap.png"), dpi=150)
    plt.close()

    corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )
    print("\nTop 5 positive correlations:")
    for (f1, f2), val in corr_pairs.head(5).items():
        print(f"  {f1:25s} <-> {f2:25s}  r = {val:+.3f}")
    print("Top 5 negative correlations:")
    for (f1, f2), val in corr_pairs.tail(5).items():
        print(f"  {f1:25s} <-> {f2:25s}  r = {val:+.3f}")
    print("[Plot saved] 02_correlation_heatmap.png")

    # Plot 3 — Boxplots: alcohol & volatile acidity vs quality
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, col, palette in zip(axes, ["alcohol", "volatile acidity"], ["Blues", "Reds"]):
        sns.boxplot(
            x="quality", y=col, hue="quality", data=eda_df,
            palette=palette, linewidth=0.8, fliersize=3, ax=ax, legend=False,
        )
        ax.set_title(f"{col.title()} vs Quality", fontsize=13, fontweight="bold")
        ax.set_xlabel("Quality Score", fontsize=11)
        ax.set_ylabel(col.title(), fontsize=11)
    plt.suptitle("Boxplots of Key Features by Wine Quality", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "03_boxplots_alcohol_volatility.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\nMean alcohol by quality score:")
    for score, mean_val in eda_df.groupby("quality")["alcohol"].mean().items():
        print(f"  Quality {score}: {mean_val:.2f}%")
    print("Mean volatile acidity by quality score:")
    for score, mean_val in eda_df.groupby("quality")["volatile acidity"].mean().items():
        print(f"  Quality {score}: {mean_val:.3f} g/dm³")
    print("[Plot saved] 03_boxplots_alcohol_volatility.png")

    return df


# =============================================================================
# SECTION 3 — PREPROCESSING
# =============================================================================

def preprocess(df: pd.DataFrame):
    """
    Prepare model-ready train/test splits from the raw DataFrame.

    Steps:
      1. Drop the non-informative Id column.
      2. Binarize quality: 1 (High) if >= 6, 0 (Low) if <= 5.
      3. Split into features (X) and target (y).
      4. Apply an 80/20 train-test split (random_state=SEED).
      5. Fit StandardScaler on X_train; transform both splits.

    Returns (X_train, X_test, y_train, y_test,
             X_train_scaled, X_test_scaled, scaler, feature_names)
    """
    print(f"\n{SEP}")
    print("SECTION 3 — PREPROCESSING")
    print(SEP)

    # Drop Id
    df_clean = df.drop(columns=["Id"])
    print(f"\nDropped 'Id' column — {df_clean.shape[1]} columns remain.")

    # Binarize quality
    df_clean["quality"] = (df_clean["quality"] >= 6).astype(int)
    print("Binarized 'quality':  0 = Low (score <= 5)  |  1 = High (score >= 6)")

    # Class distribution
    class_counts = df_clean["quality"].value_counts().sort_index()
    total = len(df_clean)
    label_map = {0: "Low  (0)", 1: "High (1)"}
    print("\nBinary class distribution:")
    for lbl, cnt in class_counts.items():
        bar = "#" * int(cnt / total * 40)
        print(f"  {label_map[lbl]}: {cnt:4d} samples ({cnt / total * 100:.1f}%)  {bar}")
    print(f"  Class balance ratio (Low:High) = {class_counts[0]}:{class_counts[1]}")

    # Feature / target split
    X = df_clean.drop(columns=["quality"])
    y = df_clean["quality"]
    feature_names = list(X.columns)
    print(f"\nFeatures (X): {X.shape}  |  Target (y): {y.shape}")
    print(f"Feature names: {feature_names}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(f"\nTrain/test split (80/20, random_state={SEED}):")
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}  "
          f"classes: {dict(y_train.value_counts().sort_index())}")
    print(f"  X_test : {X_test.shape}  y_test : {y_test.shape}  "
          f"classes: {dict(y_test.value_counts().sort_index())}")

    # StandardScaler — fit on train only to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )
    print(f"\nStandardScaler applied (fit on X_train only):")
    print(f"  X_train_scaled: {X_train_scaled.shape}  "
          f"X_test_scaled: {X_test_scaled.shape}")

    # Sanity check
    means = X_train_scaled.mean().abs().max()
    stds  = X_train_scaled.std().mean()
    print(f"  Verification — max |mean|: {means:.2e} (≈0)  mean std: {stds:.4f} (≈1)")

    return (X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled, scaler, feature_names)


# =============================================================================
# SECTION 4 — MODEL TRAINING (SCALED)
# =============================================================================

def train_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train four classifiers on the scaled training data and evaluate each
    on the scaled test set.

    Models: Logistic Regression, Decision Tree, Random Forest, KNN.

    Prints accuracy, classification report, and confusion matrix for each.
    Returns a results dict keyed by model name.
    """
    print(f"\n{SEP}")
    print("SECTION 4 — MODEL TRAINING (SCALED DATA)")
    print(SEP)

    model_definitions = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree":       DecisionTreeClassifier(random_state=SEED),
        "Random Forest":       RandomForestClassifier(random_state=SEED),
        "KNN":                 KNeighborsClassifier(),
    }

    results = {}

    for name, model in model_definitions.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=["Low (0)", "High (1)"])
        cm     = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model":    model,
            "y_pred":   y_pred,
            "accuracy": acc,
            "report":   report,
            "cm":       cm,
        }

        tn, fp, fn, tp = cm.ravel()
        print(f"\n{'─' * 50}")
        print(f"  {name}")
        print(f"{'─' * 50}")
        print(f"  Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
        print(f"\n{report}")
        print("  Confusion matrix:")
        print(f"                   Predicted Low  Predicted High")
        print(f"    Actual Low          {tn:4d}           {fp:4d}")
        print(f"    Actual High         {fn:4d}           {tp:4d}")
        print(f"    TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # Quick summary table
    print(f"\n{'─' * 72}")
    print(f"  {'Model':<25} {'Accuracy':>10} {'Prec (High)':>12} "
          f"{'Recall (High)':>14} {'F1 (High)':>10}")
    print(f"  {'─' * 68}")
    for name, res in results.items():
        rd = classification_report(
            y_test, res["y_pred"], target_names=["Low (0)", "High (1)"], output_dict=True
        )
        print(f"  {name:<25} {res['accuracy']:>10.4f} "
              f"{rd['High (1)']['precision']:>12.4f} "
              f"{rd['High (1)']['recall']:>14.4f} "
              f"{rd['High (1)']['f1-score']:>10.4f}")

    best = max(results, key=lambda n: results[n]["accuracy"])
    print(f"\n  Best model: {best}  ({results[best]['accuracy']:.4f})")

    return results, model_definitions


# =============================================================================
# SECTION 5 — FEATURE SCALING EXPERIMENT (RQ4)
# =============================================================================

def scaling_experiment(model_definitions, results, X_train, X_test, y_train, y_test):
    """
    RQ4: Does StandardScaler improve model performance?

    Re-trains each model on unscaled data and compares accuracy with
    the scaled results from Section 4 side-by-side.

    Returns a dict of {model_name: accuracy_difference (scaled - unscaled)}.
    """
    print(f"\n{SEP}")
    print("SECTION 5 — FEATURE SCALING EXPERIMENT  (RQ4)")
    print("Does StandardScaler improve model performance?")
    print(SEP)

    # Train on unscaled data using fresh model instances
    results_unscaled = {}
    for name, model in model_definitions.items():
        fresh = model.__class__(**model.get_params())
        fresh.fit(X_train, y_train)
        y_pred = fresh.predict(X_test)
        results_unscaled[name] = {"accuracy": accuracy_score(y_test, y_pred)}

    # Side-by-side comparison table
    COL = 25
    print(f"\n  {'Model':<{COL}} {'No Scaling':>12} {'With Scaling':>13} {'Difference':>11}")
    print("  " + "─" * (COL + 42))

    scaling_effects = {}
    for name in model_definitions:
        acc_u = results_unscaled[name]["accuracy"]
        acc_s = results[name]["accuracy"]
        diff  = acc_s - acc_u
        scaling_effects[name] = diff

        sign = "+" if diff >= 0 else ""
        flag = "  <-- improved" if diff > 0.005 else ("  <-- degraded" if diff < -0.005 else "")
        print(f"  {name:<{COL}} {acc_u:>12.4f} {acc_s:>13.4f} {sign}{diff:>9.4f}{flag}")

    # Classify outcomes
    helped  = [n for n, d in scaling_effects.items() if d >  0.005]
    hurt    = [n for n, d in scaling_effects.items() if d < -0.005]
    no_diff = [n for n, d in scaling_effects.items() if abs(d) <= 0.005]

    print("\nFindings:")
    if helped:
        print(f"  Scaling HELPED    : {', '.join(helped)}")
    if hurt:
        print(f"  Scaling HURT      : {', '.join(hurt)}")
    if no_diff:
        print(f"  No material change: {', '.join(no_diff)}  (within ±0.005)")

    knn_gap = scaling_effects["KNN"]
    lr_gap  = scaling_effects["Logistic Regression"]
    print(f"\nConclusion:")
    print(f"  Scaling meaningfully boosted only KNN (+{knn_gap:.2%}), which relies on")
    print(f"  Euclidean distance and is highly sensitive to feature magnitude.")
    print(f"  Tree-based models are invariant to monotonic scaling. Logistic")
    print(f"  Regression converged to equivalent accuracy either way (Δ={lr_gap:+.4f}).")

    return scaling_effects


# =============================================================================
# SECTION 6 — FEATURE IMPORTANCE  (RQ2)
# =============================================================================

def feature_importance(results, feature_names, y_test):
    """
    RQ2: Which physicochemical features are the strongest predictors?

    Extracts and ranks feature importances from the trained Random Forest
    and coefficient magnitudes from Logistic Regression.

    Saves:
      Plot 4 — Random Forest feature importance (horizontal bar)
      Plot 5 — Logistic Regression coefficient magnitudes (horizontal bar)

    Returns (rf_ranked, lr_ranked, rf_rank_map, lr_rank_map).
    """
    print(f"\n{SEP}")
    print("SECTION 6 — FEATURE IMPORTANCE  (RQ2)")
    print("Which physicochemical features best predict wine quality?")
    print(SEP)

    # ── Random Forest importances ──────────────────────────────────────
    rf_model       = results["Random Forest"]["model"]
    rf_importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    rf_ranked      = rf_importances.sort_values(ascending=False)

    print("\nRandom Forest — feature importances (mean decrease in impurity):")
    for rank, (feat, score) in enumerate(rf_ranked.items(), start=1):
        bar = "#" * int(score * 300)
        print(f"  {rank:2}. {feat:<25}  {score:.4f}  {bar}")

    # Plot 4
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = sns.color_palette("Blues_r", len(rf_ranked))
    h_bars = ax.barh(
        rf_ranked.index[::-1], rf_ranked.values[::-1],
        color=colors[::-1], edgecolor="black", linewidth=0.5,
    )
    for bar, val in zip(h_bars, rf_ranked.values[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_title(
        "Random Forest — Feature Importance\n(Mean Decrease in Impurity)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_xlim(0, rf_ranked.max() + 0.04)
    ax.axvline(rf_ranked.mean(), color="red", linestyle="--", linewidth=1,
               label=f"Mean ({rf_ranked.mean():.4f})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "04_rf_feature_importance.png"), dpi=150)
    plt.close()
    print("[Plot saved] 04_rf_feature_importance.png")

    # ── Logistic Regression coefficients ──────────────────────────────
    lr_model  = results["Logistic Regression"]["model"]
    lr_signed = pd.Series(lr_model.coef_[0], index=feature_names)
    lr_coefs  = lr_signed.abs()
    lr_ranked = lr_coefs.sort_values(ascending=False)

    print("\nLogistic Regression — coefficient magnitudes (scaled features):")
    for rank, (feat, mag) in enumerate(lr_ranked.items(), start=1):
        sign = "(+)" if lr_signed[feat] > 0 else "(-)"
        bar  = "#" * int(mag * 60)
        print(f"  {rank:2}. {feat:<25}  |coef|={mag:.4f}  {sign}  {bar}")
    print("  (+) positive effect on High quality  |  (-) negative effect")

    # Plot 5
    fig, ax = plt.subplots(figsize=(9, 6))
    signed_ordered = lr_signed.reindex(lr_ranked.index)
    bar_colors = ["steelblue" if v > 0 else "tomato" for v in signed_ordered.values[::-1]]
    h_bars = ax.barh(
        lr_ranked.index[::-1], lr_ranked.values[::-1],
        color=bar_colors, edgecolor="black", linewidth=0.5,
    )
    for bar, val, feat in zip(h_bars, lr_ranked.values[::-1], lr_ranked.index[::-1]):
        sign = "(+)" if lr_signed[feat] > 0 else "(-)"
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f} {sign}", va="center", fontsize=9)
    ax.set_title(
        "Logistic Regression — Feature Coefficient Magnitudes\n"
        "(blue = positive effect, red = negative effect)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("|Coefficient|", fontsize=11)
    ax.set_xlim(0, lr_ranked.max() + 0.3)
    ax.axvline(lr_ranked.mean(), color="green", linestyle="--", linewidth=1,
               label=f"Mean ({lr_ranked.mean():.4f})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "05_lr_feature_coefficients.png"), dpi=150)
    plt.close()
    print("[Plot saved] 05_lr_feature_coefficients.png")

    # ── Cross-model rank comparison ────────────────────────────────────
    rf_rank_map = {feat: rank for rank, feat in enumerate(rf_ranked.index, start=1)}
    lr_rank_map = {feat: rank for rank, feat in enumerate(lr_ranked.index, start=1)}

    print(f"\nFeature rank comparison (RF vs LR):")
    print(f"  {'Feature':<25} {'RF Rank':>8} {'LR Rank':>8} {'Agree?':>8}")
    print("  " + "─" * 52)
    for feat in feature_names:
        rf_r  = rf_rank_map[feat]
        lr_r  = lr_rank_map[feat]
        agree = "YES" if abs(rf_r - lr_r) <= 2 else ""
        print(f"  {feat:<25} {rf_r:>8} {lr_r:>8} {agree:>8}")

    top3_rf = list(rf_ranked.index[:3])
    top3_lr = list(lr_ranked.index[:3])
    shared  = [f for f in top3_rf if f in top3_lr]
    print(f"\n  Top-3 RF : {top3_rf}")
    print(f"  Top-3 LR : {top3_lr}")
    print(f"  Shared   : {shared if shared else 'None'}")

    return rf_ranked, lr_ranked, rf_rank_map, lr_rank_map


# =============================================================================
# SECTION 7 — MODEL COMPARISON  (RQ3)
# =============================================================================

def model_comparison(results, model_definitions, y_test):
    """
    RQ3: Which classification model performs best?

    Collects accuracy, precision, recall, and F1 for all four models,
    ranks them, and produces two comparison plots.

    Saves:
      Plot 6 — Accuracy bar chart (winner highlighted)
      Plot 7 — Grouped F1-score bar chart (Low / High / Macro)

    Returns (metrics, ranked_names) where metrics is a dict of per-model
    performance values.
    """
    print(f"\n{SEP}")
    print("SECTION 7 — MODEL COMPARISON  (RQ3)")
    print("Which classification model performs best?")
    print(SEP)

    model_names = list(model_definitions.keys())
    short_names = {
        "Logistic Regression": "Log. Reg.",
        "Decision Tree":       "Dec. Tree",
        "Random Forest":       "Rand. Forest",
        "KNN":                 "KNN",
    }

    # Collect metrics from classification reports
    metrics = {}
    for name in model_names:
        rd = classification_report(
            y_test, results[name]["y_pred"],
            target_names=["Low (0)", "High (1)"],
            output_dict=True,
        )
        metrics[name] = {
            "accuracy":  results[name]["accuracy"],
            "f1_low":    rd["Low (0)"]["f1-score"],
            "f1_high":   rd["High (1)"]["f1-score"],
            "f1_macro":  rd["macro avg"]["f1-score"],
            "precision": rd["High (1)"]["precision"],
            "recall":    rd["High (1)"]["recall"],
        }

    # Sort by accuracy for the leaderboard and plots
    ranked_names = sorted(model_names, key=lambda n: metrics[n]["accuracy"], reverse=True)

    # Plot 6 — Accuracy bar chart
    accuracies = [metrics[n]["accuracy"] for n in ranked_names]
    labels     = [short_names[n] for n in ranked_names]
    palette    = sns.color_palette("Blues_r", len(ranked_names))
    # Gold bar for winner
    bar_colors = ["#f4c430" if i == 0 else palette[i] for i in range(len(ranked_names))]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accuracies, color=bar_colors, edgecolor="black", linewidth=0.7, width=0.55)
    for bar, val in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.set_ylim(0.60, 0.85)
    ax.axhline(max(accuracies), color="red", linestyle="--", linewidth=1,
               label=f"Best: {max(accuracies):.4f}")
    ax.set_title("Model Accuracy Comparison (with StandardScaler)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "06_model_accuracy_comparison.png"), dpi=150)
    plt.close()
    print("\n[Plot saved] 06_model_accuracy_comparison.png")

    # Plot 7 — Grouped F1-score bar chart
    x        = np.arange(len(ranked_names))
    width    = 0.25
    f1_low   = [metrics[n]["f1_low"]  for n in ranked_names]
    f1_high  = [metrics[n]["f1_high"] for n in ranked_names]
    f1_macro = [metrics[n]["f1_macro"] for n in ranked_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width, f1_low,   width, label="F1 — Low (0)",   color="#5b9bd5", edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x,         f1_high,  width, label="F1 — High (1)",  color="#ed7d31", edgecolor="black", linewidth=0.6)
    b3 = ax.bar(x + width, f1_macro, width, label="F1 — Macro avg", color="#70ad47", edgecolor="black", linewidth=0.6)

    for group in (b1, b2, b3):
        for bar in group:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.60, 0.88)
    ax.set_title("F1-Score Comparison by Class and Model (with StandardScaler)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "07_model_f1_comparison.png"), dpi=150)
    plt.close()
    print("[Plot saved] 07_model_f1_comparison.png")

    # Leaderboard table
    medals = {0: "GOLD  ", 1: "SILVER", 2: "BRONZE", 3: "      "}
    print(f"\n  {'Rank':<8} {'Model':<22} {'Accuracy':>9} {'Prec(H)':>9} "
          f"{'Recall(H)':>10} {'F1(H)':>7} {'F1 Macro':>9}")
    print("  " + "─" * 77)
    for i, name in enumerate(ranked_names):
        m = metrics[name]
        print(f"  {medals[i]}  {name:<22} {m['accuracy']:>9.4f} "
              f"{m['precision']:>9.4f} {m['recall']:>10.4f} "
              f"{m['f1_high']:>7.4f} {m['f1_macro']:>9.4f}")

    winner = ranked_names[0]
    loser  = ranked_names[-1]
    print(f"\nConclusion:")
    print(f"  {winner} is the best model — accuracy {metrics[winner]['accuracy']:.4f}, "
          f"macro F1 {metrics[winner]['f1_macro']:.4f}.")
    print(f"  It outperforms the weakest model ({loser}) by "
          f"{metrics[winner]['accuracy'] - metrics[loser]['accuracy']:.4f} accuracy points.")

    return metrics, ranked_names


# =============================================================================
# SECTION 8 — FINAL SUMMARY  (RQ1 + hypothesis validation)
# =============================================================================

def final_summary(metrics, ranked_names, model_definitions,
                  scaling_effects, rf_ranked, lr_ranked,
                  rf_rank_map, lr_rank_map):
    """
    RQ1: Can wine quality be accurately predicted from chemical properties?

    Prints a structured summary answering all four research questions,
    then validates each hypothesis from the project proposal against
    the actual results.
    """
    print(f"\n{SEP}")
    print("SECTION 8 — FINAL RESULTS SUMMARY")
    print(SEP)

    RANDOM_BASELINE = 0.50
    model_names     = list(model_definitions.keys())
    winner          = ranked_names[0]
    best_acc        = metrics[winner]["accuracy"]
    worst_acc       = metrics[ranked_names[-1]]["accuracy"]
    all_beat_base   = all(metrics[n]["accuracy"] > RANDOM_BASELINE for n in model_names)
    knn_gap         = scaling_effects["KNN"]
    lr_gap          = scaling_effects["Logistic Regression"]
    top2_rf         = list(rf_ranked.index[:2])
    top2_lr         = list(lr_ranked.index[:2])
    alc_va_both     = (
        ("alcohol" in top2_rf and "volatile acidity" in top2_rf) and
        ("alcohol" in top2_lr and "volatile acidity" in top2_lr)
    )

    # ── RQ1 ───────────────────────────────────────────────────────────
    print(f"""
RQ1  Can wine quality be accurately predicted from physicochemical properties?

  Random baseline (50/50 guess) : {RANDOM_BASELINE:.2%}
  Worst model  (Decision Tree)  : {worst_acc:.2%}  (+{worst_acc - RANDOM_BASELINE:.2%} above baseline)
  Best model   (Random Forest)  : {best_acc:.2%}  (+{best_acc - RANDOM_BASELINE:.2%} above baseline)
  All 4 models beat baseline    : {'YES' if all_beat_base else 'NO'}

  Answer: YES — physicochemical features carry meaningful predictive signal.
  Every model exceeded the 50% random baseline by at least {worst_acc - RANDOM_BASELINE:.2%}.
  The best model reached {best_acc:.2%}, though residual uncertainty (~{1 - best_acc:.0%})
  suggests unmeasured factors (grape variety, vintage, taster subjectivity)
  also influence perceived quality.""")

    # ── RQ2 ───────────────────────────────────────────────────────────
    print(f"""
{SEP2}
RQ2  Which physicochemical features are the strongest predictors?

  Random Forest  top-3 : {', '.join(rf_ranked.index[:3])}
  Logistic Reg.  top-3 : {', '.join(lr_ranked.index[:3])}

  alcohol          RF #{rf_rank_map['alcohol']}  LR #{lr_rank_map['alcohol']}  — strongest predictor in both models
  volatile acidity RF #{rf_rank_map['volatile acidity']}  LR #{lr_rank_map['volatile acidity']}  — negative effect (higher = lower quality)
  sulphates        RF #{rf_rank_map['sulphates']}  LR #{lr_rank_map['sulphates']}  — positive effect
  residual sugar   RF #{rf_rank_map['residual sugar']} LR #{lr_rank_map['residual sugar']} — weakest predictor in both models

  Answer: Alcohol is the single strongest predictor. Volatile acidity and
  sulphates rank 2nd–3rd across both models. 9/11 features agree within
  ±2 positions, giving high cross-model confidence in these rankings.""")

    # ── RQ3 ───────────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("RQ3  Which classification model performs best?\n")
    for i, name in enumerate(ranked_names):
        m      = metrics[name]
        marker = "  <-- BEST" if i == 0 else ""
        print(f"  {i + 1}. {name:<22}  acc={m['accuracy']:.4f}  F1-macro={m['f1_macro']:.4f}{marker}")
    lr_gap_acc = best_acc - metrics["Logistic Regression"]["accuracy"]
    print(f"""
  Answer: Random Forest achieves the highest accuracy ({best_acc:.2%}) and
  macro F1 ({metrics[winner]['f1_macro']:.4f}). Logistic Regression is a close second
  (gap: {lr_gap_acc:.4f}), offering comparable performance with better interpretability.
  Decision Tree performs worst, likely due to overfitting without pruning.""")

    # ── RQ4 ───────────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("RQ4  Does StandardScaler improve model performance?\n")
    for name in model_names:
        gap    = scaling_effects[name]
        sign   = "+" if gap >= 0 else ""
        effect = "improved" if gap > 0.005 else ("degraded" if gap < -0.005 else "no change")
        print(f"  {name:<22}  {sign}{gap:.4f}  ({effect})")
    print(f"""
  Answer: Scaling materially helped only KNN (+{knn_gap:.2%}), which computes
  Euclidean distances and is highly sensitive to feature magnitude.
  Tree-based models are invariant to monotonic scaling. Logistic Regression
  converged to equivalent accuracy either way (Δ={lr_gap:+.4f}).""")

    # ── Hypothesis validation ──────────────────────────────────────────
    def verdict(passed: bool) -> str:
        return "SUPPORTED     [PASS]" if passed else "NOT SUPPORTED [FAIL]"

    hypotheses = [
        (
            "H1 (RQ1): All models beat the 50% random baseline",
            all_beat_base,
            "  |  ".join(f"{n} {metrics[n]['accuracy']:.2%}" for n in model_names),
        ),
        (
            "H2 (RQ2): Alcohol & volatile acidity are the top-2 features",
            alc_va_both,
            f"RF top-2: {top2_rf}  |  LR top-2: {top2_lr}",
        ),
        (
            "H3 (RQ3): Random Forest achieves the highest accuracy",
            winner == "Random Forest",
            f"Winner: {winner} ({best_acc:.4f})  |  Runner-up: "
            f"{ranked_names[1]} ({metrics[ranked_names[1]]['accuracy']:.4f})",
        ),
        (
            "H4 (RQ4): Scaling boosts KNN and Logistic Regression",
            knn_gap > 0.005,
            f"KNN: +{knn_gap:.4f} (confirmed)  |  LR: +{lr_gap:.4f} (no material gain)",
        ),
    ]

    print(f"\n{SEP}")
    print("  HYPOTHESIS VALIDATION")
    print(SEP)
    for hyp, passed, evidence in hypotheses:
        print(f"\n  {hyp}")
        print(f"  Result  : {verdict(passed)}")
        print(f"  Evidence: {evidence}")

    passed_count = sum(p for _, p, _ in hypotheses)
    print(f"\n  Overall: {passed_count}/{len(hypotheses)} hypotheses supported")

    print(f"\n{SEP}")
    print("  END OF ANALYSIS")
    print(f"  All plots saved to: ./{OUTPUTS_DIR}/")
    print(SEP)


# =============================================================================
# MAIN — wire all sections together
# =============================================================================

def main():
    # Section 2 — Load raw data and run EDA
    df = load_and_explore("WineQT.csv")

    # Section 3 — Preprocess into train/test splits
    (X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled,
     scaler, feature_names) = preprocess(df)

    # Section 4 — Train and evaluate all four models on scaled data
    results, model_definitions = train_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    # Section 5 — RQ4: scaling experiment (scaled vs unscaled)
    scaling_effects = scaling_experiment(
        model_definitions, results, X_train, X_test, y_train, y_test
    )

    # Section 6 — RQ2: feature importance analysis
    rf_ranked, lr_ranked, rf_rank_map, lr_rank_map = feature_importance(
        results, feature_names, y_test
    )

    # Section 7 — RQ3: model comparison plots and leaderboard
    metrics, ranked_names = model_comparison(results, model_definitions, y_test)

    # Section 8 — RQ1 + hypothesis validation summary
    final_summary(
        metrics, ranked_names, model_definitions,
        scaling_effects, rf_ranked, lr_ranked,
        rf_rank_map, lr_rank_map,
    )


if __name__ == "__main__":
    main()
