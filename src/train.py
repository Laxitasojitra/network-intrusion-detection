"""Train and evaluate the intrusion detection system.

Stage 1: a supervised classifier sorts each connection into Normal / DoS / Probe / R2L / U2R.
Stage 2: an Isolation Forest trained only on normal traffic flags anything that looks unusual,
         which catches new ("zero-day") attacks the classifier has never seen.
A connection is raised as an alert if either stage says it is an attack.
"""

import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from data_prep import (CATEGORICAL_FEATURES, CLASSES, NUMERIC_FEATURES, ROOT, SKEWED_FEATURES,
                       features, load_train_test)
from plot_style import AQUA, BLUE, INK_2, ORANGE, apply_style

RANDOM_STATE = 42
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"

# Target false-alarm rate for the anomaly detector: flag the top 2% most unusual normal connections
ANOMALY_QUANTILE = 0.98


def make_preprocessor():
    other_numeric = [c for c in NUMERIC_FEATURES if c not in SKEWED_FEATURES]
    return ColumnTransformer([
        ("skewed", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), SKEWED_FEATURES),
        ("numeric", StandardScaler(), other_numeric),
        # handle_unknown="ignore": the test set may contain services never seen in training
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def candidate_models():
    # Balanced class weights stop the model from ignoring rare classes like U2R (only 52 training rows)
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE),
    }


def detection_stats(is_attack, flagged, unseen):
    """Binary view used by security teams: did we raise an alert or not?"""
    is_attack, flagged, unseen = map(np.asarray, (is_attack, flagged, unseen))
    return {
        "detection_rate": float(flagged[is_attack == 1].mean()),      # attacks caught
        "false_alarm_rate": float(flagged[is_attack == 0].mean()),    # normal traffic wrongly flagged
        "unseen_attack_detection": float(flagged[unseen].mean()),     # zero-day attacks caught
        "known_attack_detection": float(flagged[(is_attack == 1) & ~unseen].mean()),
    }


def classification_stats(y_true, y_pred):
    per_class = recall_score(y_true, y_pred, labels=CLASSES, average=None)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro")),
        "recall_per_class": dict(zip(CLASSES, map(float, per_class))),
    }


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train, test, unseen_types = load_train_test()
    X_train, y_train = features(train), train["category"]
    X_test, y_test = features(test), test["category"]
    print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")
    print(f"{len(unseen_types)} attack types appear only in the test set ({test['unseen_attack'].sum():,} rows)\n")

    # 1. Compare classifiers with 3-fold cross-validation on the training set (macro F1 treats all classes equally)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_scores, fitted = {}, {}
    print("3-fold CV macro-F1 on training data:")
    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        start = time.time()
        cv_scores[name] = float(cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1).mean())
        fitted[name] = pipe.fit(X_train, y_train)
        print(f"  {name:<20} {cv_scores[name]:.4f}  ({time.time() - start:.0f}s)")

    # 2. Evaluate every classifier on the official test set
    results = {}
    for name, pipe in fitted.items():
        pred = pipe.predict(X_test)
        results[name] = {**classification_stats(y_test, pred),
                         **detection_stats(test["is_attack"], pred != "Normal", test["unseen_attack"])}
    best_name = max(cv_scores, key=cv_scores.get)
    best = fitted[best_name]
    print(f"\nSelected by CV: {best_name}")

    # 3. Anomaly detector trained on normal traffic only.
    #    The alert threshold is set on held-out normal data, never on the test set.
    normal = X_train[y_train == "Normal"]
    normal_fit, normal_val = train_test_split(normal, test_size=0.2, random_state=RANDOM_STATE)
    iso = Pipeline([
        ("prep", make_preprocessor()),
        ("model", IsolationForest(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
    ]).fit(normal_fit)
    threshold = float(np.quantile(-iso.score_samples(normal_val), ANOMALY_QUANTILE))
    anomaly_flag = -iso.score_samples(X_test) > threshold

    best_pred = best.predict(X_test)
    supervised_flag = best_pred != "Normal"
    hybrid_flag = supervised_flag | anomaly_flag

    detection = {
        "Supervised only": detection_stats(test["is_attack"], supervised_flag, test["unseen_attack"]),
        "Anomaly only": detection_stats(test["is_attack"], anomaly_flag, test["unseen_attack"]),
        "Hybrid": detection_stats(test["is_attack"], hybrid_flag, test["unseen_attack"]),
    }

    print("\nMulti-class results on test set:")
    table = pd.DataFrame({n: {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                              "detection": r["detection_rate"], "false_alarm": r["false_alarm_rate"],
                              "unseen_det": r["unseen_attack_detection"]} for n, r in results.items()}).T
    print(table.round(3).to_string())
    print(f"\nPer-class recall ({best_name}):")
    print(pd.Series(results[best_name]["recall_per_class"]).round(3).to_string())
    print(f"\nAttack detection on test set ({best_name} + Isolation Forest):")
    print(pd.DataFrame(detection).T.round(3).to_string())

    # 4. Save models and metrics
    joblib.dump({"classifier": best, "anomaly_detector": iso, "anomaly_threshold": threshold,
                 "classifier_name": best_name}, MODEL_DIR / "ids_model.joblib")
    metrics = {
        "train_rows": len(train), "test_rows": len(test),
        "unseen_attack_types": unseen_types, "unseen_attack_rows": int(test["unseen_attack"].sum()),
        "cv_macro_f1": {k: round(v, 4) for k, v in cv_scores.items()},
        "selected_classifier": best_name,
        "test_results": {n: {k: (round(v, 4) if isinstance(v, float) else {c: round(x, 4) for c, x in v.items()})
                             for k, v in r.items()} for n, r in results.items()},
        "anomaly_quantile": ANOMALY_QUANTILE,
        "detection_comparison": {n: {k: round(v, 4) for k, v in d.items()} for n, d in detection.items()},
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 5. Charts
    # Detection: supervised vs hybrid, for known and unseen attacks, plus false alarms
    labels = ["Known attacks", "Unseen (zero-day)\nattacks", "False alarms\non normal traffic"]
    sup = detection["Supervised only"]
    hyb = detection["Hybrid"]
    sup_vals = [sup["known_attack_detection"], sup["unseen_attack_detection"], sup["false_alarm_rate"]]
    hyb_vals = [hyb["known_attack_detection"], hyb["unseen_attack_detection"], hyb["false_alarm_rate"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - 0.19, np.array(sup_vals) * 100, width=0.36, color=BLUE, label="Supervised only")
    b2 = ax.bar(x + 0.19, np.array(hyb_vals) * 100, width=0.36, color=ORANGE, label="Hybrid (+ anomaly detection)")
    ax.bar_label(b1, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Share of connections flagged (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Hybrid detection catches far more zero-day attacks")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hybrid_detection.png", dpi=150)
    plt.close(fig)

    # Per-class recall for each classifier
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(CLASSES))
    for i, (name, color) in enumerate(zip(results, [BLUE, ORANGE, AQUA])):
        vals = [results[name]["recall_per_class"][c] * 100 for c in CLASSES]
        ax.bar(x + (i - 1) * 0.26, vals, width=0.24, color=color, label=name)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Recall on test set (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Rare attack classes (R2L, U2R) are the hardest to detect")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_class_recall.png", dpi=150)
    plt.close(fig)

    # Normalised confusion matrix for the selected classifier
    cm = confusion_matrix(y_test, best_pred, labels=CLASSES, normalize="true") * 100
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > 55 else INK_2)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix, % of each true class\n({best_name})", fontsize=11)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"\nModel, metrics and charts saved under {ROOT}")


if __name__ == "__main__":
    main()
