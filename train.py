"""
train.py — trains five classifiers on one dataset and exports everything the
Streamlit app needs.

Usage
-----
    python train.py                                  # uses the bundled dataset
    python train.py --csv data/other.csv --target y  # swap in your own

Outputs (all written next to this file)
---------------------------------------
    test_data.csv                    held-out 20% split, with labels
    model/<name>.joblib              five fitted Pipelines (preprocessing included)
    model/metadata.joblib            target column, class names, feature order
    model/metrics_comparison.csv     the comparison table
    model/confusion_matrices.png     one panel per model
"""

import argparse
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

FILE_KEYS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "kNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest (Ensemble)": "random_forest",
}


def build_preprocessor(numeric_cols, categorical_cols):
    """Impute + scale numerics; impute + one-hot encode categoricals.

    Wrapping this in a ColumnTransformer inside each Pipeline means the
    transformations are fitted on the training fold only and reapplied
    identically at inference time, so the Streamlit app can be handed raw CSV.
    """
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:                       # scikit-learn < 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_branch = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_branch = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", ohe),
    ])
    return ColumnTransformer([
        ("num", numeric_branch, numeric_cols),
        ("cat", categorical_branch, categorical_cols),
    ])


def score_model(y_true, y_pred, y_proba, is_binary):
    """The six metrics the assignment asks for."""
    average = "binary" if is_binary else "weighted"
    if is_binary:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the assignment models.")
    parser.add_argument("--csv", default=os.path.join(HERE, "data", "german_credit.csv"))
    parser.add_argument("--target", default="credit_risk")
    parser.add_argument("--drop", nargs="*", default=[],
                        help="identifier columns to discard")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---------------- load ----------------
    df = pd.read_csv(args.csv)
    df = df.drop(columns=[c for c in args.drop if c in df.columns])

    n_features = df.shape[1] - 1
    print(f"Loaded {args.csv}")
    print(f"  instances : {df.shape[0]}")
    print(f"  features  : {n_features}")
    assert n_features >= 12, "Assignment requires at least 12 features."
    assert df.shape[0] >= 500, "Assignment requires at least 500 instances."
    assert args.target in df.columns, f"target '{args.target}' not in {list(df.columns)}"

    X = df.drop(columns=[args.target])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[args.target])

    class_names = [str(c) for c in label_encoder.classes_]
    is_binary = len(class_names) == 2

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    print(f"  task      : {'binary' if is_binary else 'multi-class'} "
          f"({len(class_names)} classes -> {class_names})")
    print(f"  numeric   : {len(numeric_cols)}   categorical: {len(categorical_cols)}")
    print(f"  balance   : {dict(df[args.target].value_counts())}")

    # ---------------- split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  train/test: {X_train.shape[0]} / {X_test.shape[0]}\n")

    test_export = X_test.copy()
    test_export[args.target] = label_encoder.inverse_transform(y_test)
    test_export.to_csv(os.path.join(HERE, "test_data.csv"), index=False)
    print(f"wrote test_data.csv ({test_export.shape[0]} rows)\n")

    # ---------------- models ----------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000,
                                                  random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5,
                                                random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1),
    }

    rows, preds = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("classifier", estimator),
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)

        rows.append({"ML Model Name": name,
                     **score_model(y_test, y_pred, y_proba, is_binary)})
        preds[name] = y_pred

        joblib.dump(pipe, os.path.join(MODEL_DIR, f"{FILE_KEYS[name]}.joblib"))
        print(f"trained + saved  {name}")

    joblib.dump({
        "target_col": args.target,
        "class_names": class_names,
        "is_binary": is_binary,
        "feature_cols": list(X.columns),
        "label_classes": list(label_encoder.classes_),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }, os.path.join(MODEL_DIR, "metadata.joblib"))

    # ---------------- comparison table ----------------
    comparison = pd.DataFrame(rows).set_index("ML Model Name").round(4)
    comparison.to_csv(os.path.join(MODEL_DIR, "metrics_comparison.csv"))

    print("\n" + "=" * 78)
    print("COMPARISON TABLE")
    print("=" * 78)
    print(comparison.to_string())
    print("\n--- markdown for README.md ---\n")
    print(comparison.reset_index().to_markdown(index=False))

    print(f"\nBest Accuracy : {comparison['Accuracy'].idxmax()}")
    print(f"Best AUC      : {comparison['AUC'].idxmax()}")
    print(f"Best F1       : {comparison['F1'].idxmax()}")
    print(f"Best MCC      : {comparison['MCC'].idxmax()}")

    # ---------------- confusion matrices ----------------
    fig, axes = plt.subplots(1, len(preds), figsize=(4.4 * len(preds), 4.2))
    for ax, (name, y_pred) in zip(np.atleast_1d(axes), preds.items()):
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "confusion_matrices.png"),
                dpi=120, bbox_inches="tight")
    print("\nwrote model/confusion_matrices.png")

    # ---------------- per-model reports ----------------
    print("\n" + "=" * 78)
    print("PER-MODEL CLASSIFICATION REPORTS")
    print("=" * 78)
    for name, y_pred in preds.items():
        print(f"\n{name}\n{'-' * len(name)}")
        print(classification_report(y_test, y_pred, target_names=class_names,
                                    zero_division=0))

    # ---------------- pin the training versions ----------------
    # Version drift between the machine that pickles the models and the machine
    # that unpickles them is the most common Streamlit Cloud deploy failure, so
    # rewrite the pins to whatever actually produced these artifacts.
    import sklearn
    pins = {"scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__}
    req_path = os.path.join(HERE, "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path) as fh:
            lines = fh.read().splitlines()
        rewritten = []
        for line in lines:
            pkg = line.split("==")[0].split(">=")[0].strip()
            rewritten.append(f"{pkg}=={pins[pkg]}" if pkg in pins else line)
        with open(req_path, "w") as fh:
            fh.write("\n".join(rewritten) + "\n")
        print("\npinned in requirements.txt: " +
              ", ".join(f"{k}=={v}" for k, v in pins.items()))

    print("\nDone. Artifacts are in model/ and test_data.csv.")


if __name__ == "__main__":
    main()
