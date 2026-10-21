"""Навчання, підбір параметрів (GridSearchCV + крос-валідація), оцінка та збереження моделі.

Запуск з кореня репозиторію:  python -m src.train
"""
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.preprocessing import RAW_FEATURES, ROOT, build_pipeline, load_data, split_xy

SEED = 42
FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

CANDIDATES = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000),
        {"model__C": [0.01, 0.1, 1, 10]},
    ),
    "RandomForest": (
        RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1),
        {"model__max_depth": [8, 14], "model__min_samples_leaf": [5, 20]},
    ),
    "HistGradientBoosting": (
        HistGradientBoostingClassifier(random_state=SEED),
        {"model__learning_rate": [0.05, 0.1],
         "model__max_depth": [4, None],
         "model__max_iter": [200]},
    ),
}


def metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    X, y = split_xy(load_data())
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, fitted = [], {}
    for name, (model, grid) in CANDIDATES.items():
        t = time.time()
        gs = GridSearchCV(build_pipeline(model), grid, cv=cv, scoring="f1", n_jobs=1)
        gs.fit(X_tr, y_tr)
        proba = gs.predict_proba(X_te)[:, 1]
        m = metrics(y_te, (proba >= 0.5).astype(int), proba)
        rows.append({"model": name, "cv_f1": gs.best_score_, **m,
                     "best_params": json.dumps({k.replace("model__", ""): v for k, v in gs.best_params_.items()})})
        fitted[name] = gs.best_estimator_
        print(f"{name}: cv_f1={gs.best_score_:.4f} test={m} ({time.time() - t:.0f}s)", flush=True)

    table = pd.DataFrame(rows).sort_values("cv_f1", ascending=False)
    table.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
    best_name = table.iloc[0]["model"]
    best = fitted[best_name]
    print("\nНайкраща модель за CV F1:", best_name)

    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    final = metrics(y_te, pred, proba)
    (ROOT / "reports" / "metrics.json").write_text(
        json.dumps({"best_model": best_name, "test_metrics": final,
                    "best_params": json.loads(table.iloc[0]["best_params"])}, indent=2, ensure_ascii=False))

    # Графіки оцінки
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["залишився", "пішов"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion matrix — {best_name}")
    fig.tight_layout(); fig.savefig(FIG / "09_confusion_matrix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, est in fitted.items():
        RocCurveDisplay.from_estimator(est, X_te, y_te, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("ROC-криві (тестова вибірка)")
    fig.tight_layout(); fig.savefig(FIG / "10_roc_curves.png", dpi=130); plt.close(fig)

    # Важливість ознак: permutation importance на сирих ознаках (враховує весь пайплайн)
    sample = X_te.sample(8000, random_state=SEED)
    pi = permutation_importance(best, sample, y_te.loc[sample.index], scoring="f1",
                                n_repeats=5, random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=RAW_FEATURES).sort_values()
    imp.to_csv(ROOT / "reports" / "feature_importance.csv", header=["importance"])
    fig, ax = plt.subplots(figsize=(7, 4))
    imp.plot.barh(ax=ax)
    ax.set_xlabel("падіння F1 при перемішуванні ознаки")
    ax.set_title("Permutation importance")
    fig.tight_layout(); fig.savefig(FIG / "11_feature_importance.png", dpi=130); plt.close(fig)

    # Помилки за порогом: precision/recall при різних порогах
    thr = [0.3, 0.4, 0.5, 0.6, 0.7]
    pd.DataFrame([{"threshold": t, **{k: v for k, v in metrics(y_te, (proba >= t).astype(int), proba).items()
                                       if k in ("accuracy", "precision", "recall", "f1")}} for t in thr]
                 ).to_csv(ROOT / "reports" / "threshold_analysis.csv", index=False)

    joblib.dump(best, MODELS / "model.joblib", compress=3)
    (MODELS / "metadata.json").write_text(json.dumps(
        {"model": best_name, "features": RAW_FEATURES, "threshold": 0.5, "test_metrics": final}, indent=2))
    print("Збережено models/model.joblib")


if __name__ == "__main__":
    main()
