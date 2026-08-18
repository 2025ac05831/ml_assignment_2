"""
Streamlit front-end for ML Assignment 2.

Loads the classification pipelines trained in the Colab notebook, lets the user
upload a test CSV, and reports the six evaluation metrics along with a confusion
matrix and classification report for the selected model.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}

st.set_page_config(page_title="Classification Model Explorer",
                   page_icon="\U0001F4CA",
                   layout="wide")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_metadata():
    return joblib.load(os.path.join(MODEL_DIR, "metadata.joblib"))


@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


@st.cache_data
def load_metrics_table():
    path = os.path.join(MODEL_DIR, "metrics_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba, is_binary):
    average = "binary" if is_binary else "weighted"

    try:
        if is_binary:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    except ValueError:
        auc = float("nan")   # single class present in the uploaded slice

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(1.4 * len(class_names) + 3,
                                    1.1 * len(class_names) + 2.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("Controls")

try:
    meta = load_metadata()
except FileNotFoundError:
    st.error("`model/metadata.joblib` is missing. Run the training notebook and "
             "commit the `model/` folder before deploying.")
    st.stop()

TARGET_COL = meta["target_col"]
CLASS_NAMES = meta["class_names"]
IS_BINARY = meta["is_binary"]
FEATURE_COLS = meta["feature_cols"]
LABEL_CLASSES = [str(c) for c in meta["label_classes"]]

selected_model = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader(
    "Upload test data (CSV)",
    type=["csv"],
    help=f"The file must contain the feature columns and the target column "
         f"'{TARGET_COL}'.",
)
use_bundled = st.sidebar.checkbox("Use the bundled test_data.csv instead", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Task: {'Binary' if IS_BINARY else 'Multi-class'} classification  \n"
    f"Classes: {len(CLASS_NAMES)}  \n"
    f"Features: {len(FEATURE_COLS)}"
)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("Classification Model Explorer")
st.write(
    "Five classifiers trained on a single dataset, compared across six evaluation "
    "metrics. Upload a test CSV or use the bundled sample to see how each model "
    "behaves on unseen data."
)

overview_tab, evaluate_tab, predictions_tab, drivers_tab = st.tabs(
    ["Model comparison", "Evaluate selected model", "Predictions", "What drives the model"]
)

# ---- resolve the dataframe -----------------------------------------------
df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source_label = uploaded.name
elif use_bundled and os.path.exists("test_data.csv"):
    df = pd.read_csv("test_data.csv")
    source_label = "test_data.csv (bundled)"

with overview_tab:
    st.subheader("Metrics from the held-out test split")
    table = load_metrics_table()
    if table is not None:
        st.dataframe(table.style.format("{:.4f}")
                     .highlight_max(axis=0, color="#d7f0d7"), width="stretch")
        best = table["F1"].idxmax()
        st.success(f"Highest F1 on the held-out split: **{best}** "
                   f"({table.loc[best, 'F1']:.4f})")
    else:
        st.info("`model/metrics_comparison.csv` not found - run the notebook to "
                "generate it.")

with evaluate_tab:
    if df is None:
        st.info("Upload a CSV in the sidebar, or tick the bundled-data option.")
    else:
        st.caption(f"Source: {source_label}  |  {df.shape[0]} rows, "
                   f"{df.shape[1]} columns")

        if TARGET_COL not in df.columns:
            st.error(f"Column '{TARGET_COL}' not found in the uploaded file. "
                     f"Found: {list(df.columns)}")
        else:
            missing = [c for c in FEATURE_COLS if c not in df.columns]
            if missing:
                st.error(f"Missing feature columns: {missing}")
            else:
                X = df[FEATURE_COLS]
                y_true = pd.Series(df[TARGET_COL].astype(str)).map(
                    {name: idx for idx, name in enumerate(LABEL_CLASSES)}
                )

                if y_true.isna().any():
                    st.error("Some target values do not match the classes seen "
                             f"during training ({LABEL_CLASSES}).")
                else:
                    y_true = y_true.to_numpy()
                    pipe = load_model(MODEL_FILES[selected_model])
                    y_pred = pipe.predict(X)
                    y_proba = pipe.predict_proba(X)

                    scores = compute_metrics(y_true, y_pred, y_proba, IS_BINARY)

                    st.subheader(f"{selected_model} - evaluation metrics")
                    cols = st.columns(6)
                    for col, (metric_name, value) in zip(cols, scores.items()):
                        col.metric(metric_name,
                                   "n/a" if np.isnan(value) else f"{value:.4f}")

                    left, right = st.columns([1, 1])
                    with left:
                        st.subheader("Confusion matrix")
                        st.pyplot(plot_confusion(y_true, y_pred, CLASS_NAMES))
                    with right:
                        st.subheader("Classification report")
                        report = classification_report(
                            y_true, y_pred,
                            labels=list(range(len(CLASS_NAMES))),
                            target_names=CLASS_NAMES,
                            output_dict=True, zero_division=0,
                        )
                        st.dataframe(pd.DataFrame(report).transpose().round(4),
                                     width="stretch")

                    st.session_state["last_predictions"] = (df, y_pred, y_proba)

with predictions_tab:
    if "last_predictions" not in st.session_state:
        st.info("Run an evaluation first to see row-level predictions.")
    else:
        source_df, y_pred, y_proba = st.session_state["last_predictions"]
        out = source_df.copy()
        out["predicted_label"] = [CLASS_NAMES[i] for i in y_pred]
        out["confidence"] = y_proba.max(axis=1).round(4)

        st.subheader("Row-level predictions")
        st.dataframe(out.head(200), width="stretch")
        st.download_button(
            "Download predictions as CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

with drivers_tab:
    st.subheader(f"Most influential features — {selected_model}")
    pipe = load_model(MODEL_FILES[selected_model])
    classifier = pipe.named_steps["classifier"]

    try:
        feature_names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    except Exception:
        feature_names = None

    if feature_names is None:
        st.info("Feature names could not be recovered from the preprocessing step.")
    elif hasattr(classifier, "feature_importances_"):
        importances = pd.Series(classifier.feature_importances_,
                                index=feature_names).sort_values(ascending=False)
        st.caption("Impurity-based importance. Higher means the feature was used "
                   "for more informative splits across the ensemble.")
        top = importances.head(15).sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top.index, top.values, color="#4c72b0")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        st.pyplot(fig)
    elif hasattr(classifier, "coef_"):
        coefs = np.ravel(classifier.coef_[0]) if classifier.coef_.ndim > 1 \
            else np.ravel(classifier.coef_)
        series = pd.Series(coefs, index=feature_names)
        top = series.reindex(series.abs().sort_values(ascending=False).index).head(15)
        st.caption("Fitted coefficients on standardised features. Positive values "
                   f"push the prediction toward '{CLASS_NAMES[-1]}'.")
        top = top.iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top.index, top.values,
                color=["#c44e52" if v < 0 else "#55a868" for v in top.values])
        ax.axvline(0, color="#333", linewidth=0.8)
        ax.set_xlabel("Coefficient")
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info(f"{selected_model} does not expose per-feature importances. "
                "Try Random Forest, Decision Tree or Logistic Regression.")
