"""Train and compare models for heart disease prediction.

Run from the project root:  python src/train.py
Outputs: models/best_model.joblib, models/metrics.csv, models/*.png
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "heart.csv"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
SEED = 42


def build_pipeline(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def main():
    df = pd.read_csv(DATA).drop_duplicates()
    X, y = df.drop(columns="target"), df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    candidates = {
        "Logistic Regression": (LogisticRegression(max_iter=2000),
                                {"model__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (RandomForestClassifier(random_state=SEED),
                          {"model__n_estimators": [200, 400], "model__max_depth": [3, 5, None],
                           "model__min_samples_leaf": [1, 3]}),
        "SVM": (SVC(probability=True, random_state=SEED),
                {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"]}),
        "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=SEED),
                    {"model__n_estimators": [100, 300], "model__max_depth": [2, 3, 4],
                     "model__learning_rate": [0.03, 0.1]}),
    }

    rows, fitted = [], {}
    for name, (model, grid) in candidates.items():
        # Tune for recall: in medical screening, missing a sick patient is the costly error.
        gs = GridSearchCV(build_pipeline(model), grid, cv=cv, scoring="recall", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        pred, proba = best.predict(X_te), best.predict_proba(X_te)[:, 1]
        rows.append({
            "model": name, "cv_recall": round(gs.best_score_, 3),
            "accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred),
            "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, proba), "best_params": gs.best_params_,
        })
        fitted[name] = best
        print(f"{name}: recall={rows[-1]['recall']:.3f} auc={rows[-1]['roc_auc']:.3f}")

    metrics = pd.DataFrame(rows).sort_values(["recall", "roc_auc"], ascending=False)
    metrics.round(3).to_csv(OUT / "metrics.csv", index=False)
    print("\n", metrics.drop(columns="best_params").round(3).to_string(index=False))

    best_name = metrics.iloc[0]["model"]
    best = fitted[best_name]
    joblib.dump({"pipeline": best, "name": best_name,
                 "features": NUMERIC + CATEGORICAL}, OUT / "best_model.joblib")
    print(f"\nSaved best model: {best_name}")

    # Plots: ROC curves for all models + confusion matrix of the best one
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, est in fitted.items():
        RocCurveDisplay.from_estimator(est, X_te, y_te, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curves (test set)")
    fig.tight_layout(); fig.savefig(OUT / "roc_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(best, X_te, y_te, ax=ax, cmap="Blues",
                                          display_labels=["No disease", "Disease"])
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=150); plt.close(fig)

    # Feature importance via permutation (works for every model type)
    from sklearn.inspection import permutation_importance
    r = permutation_importance(best, X_te, y_te, n_repeats=20, random_state=SEED, scoring="recall")
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.plot.barh(ax=ax, color="#c0392b")
    ax.set_title(f"Permutation importance - {best_name}")
    fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
