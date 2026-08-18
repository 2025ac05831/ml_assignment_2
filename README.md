# Credit Risk Classification — ML Assignment 2

**Name:** Govind Kumar Mishra
**BITS ID:** 2025ac05831
**Programme:** M.Tech (AIML) — Work Integrated Learning Programmes
**Course:** Machine Learning

Five classification models trained on the Statlog German Credit dataset, compared
across six evaluation metrics, with an interactive Streamlit front-end for
evaluating any test CSV against the trained pipelines.

---

## a. Problem Statement

Given twenty attributes describing a loan applicant — account history, credit
purpose and amount, employment record, existing obligations, age and housing
status — classify the applicant's credit risk as **good** or **bad**. This is a
binary classification problem.

The practical motivation is asymmetric cost. Approving a bad-risk applicant loses
the principal; declining a good-risk applicant loses only the margin on one loan.
A classifier that maximises headline accuracy on this dataset can do so simply by
approving nearly everyone, since 70% of applicants are good risks, so the useful
question is how well each model identifies the minority **bad** class.

---

## b. Dataset Description

| Property | Value |
|---|---|
| Source | Statlog (German Credit Data), UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data |
| Instances | 1,000 |
| Features | 20 (meets the ≥12 requirement) |
| Target variable | `credit_risk` |
| Classes | `good` (700), `bad` (300) |
| Class balance | 70 / 30 — moderately imbalanced |
| Missing values | None |
| Feature types | 7 numeric, 13 categorical |
| Train / test split | 800 / 200, stratified, `random_state=42` |

### Feature list

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | `checking_account_status` | categorical | Status of existing checking account |
| 2 | `duration_months` | numeric | Loan duration in months |
| 3 | `credit_history` | categorical | Repayment record on past credits |
| 4 | `purpose` | categorical | What the credit is for (car, education, business …) |
| 5 | `credit_amount` | numeric | Amount of credit requested (DM) |
| 6 | `savings_account` | categorical | Savings account / bonds balance band |
| 7 | `employment_since` | categorical | Length of current employment |
| 8 | `installment_rate_pct` | numeric | Instalment as % of disposable income |
| 9 | `personal_status_sex` | categorical | Personal status and sex |
| 10 | `other_debtors` | categorical | Co-applicants or guarantors |
| 11 | `present_residence_since` | numeric | Years at present residence |
| 12 | `property` | categorical | Type of property owned |
| 13 | `age_years` | numeric | Age in years |
| 14 | `other_installment_plans` | categorical | Other instalment plans held |
| 15 | `housing` | categorical | Rent / own / free |
| 16 | `existing_credits_count` | numeric | Number of existing credits at this bank |
| 17 | `job` | categorical | Job skill category |
| 18 | `dependents_count` | numeric | Number of dependents |
| 19 | `telephone` | categorical | Registered telephone |
| 20 | `foreign_worker` | categorical | Foreign worker flag |

The raw UCI file ships without a header row and encodes the target as 1/2. The
copy in `data/german_credit.csv` has the column names above applied and the target
mapped to the readable labels `good` / `bad`.

### Preprocessing

All preprocessing is defined inside a scikit-learn `Pipeline`, so every transform
is fitted on the training fold only and reapplied identically at inference time.
This matters for the Streamlit app: it can be handed a raw CSV and the loaded
pipeline handles scaling and encoding itself, with no risk of the app and the
training script disagreeing.

- Numeric features: median imputation, then `StandardScaler`. Scaling is
  load-bearing for Logistic Regression and kNN, which are distance- and
  magnitude-sensitive.
- Categorical features: most-frequent imputation, then `OneHotEncoder` with
  `handle_unknown="ignore"`, so an unseen category at prediction time degrades
  gracefully rather than raising.
- Target: label-encoded (`bad` → 0, `good` → 1).
- Split: stratified 80/20 to preserve the 70/30 class ratio in both folds.

---

## c. GitHub Repository Link

**Repository:** https://github.com/2025ac05831/ml_assignment_2

**Live Streamlit App:** https://mlassignment2-3vqzywrejjqutvpzv8ptll.streamlit.app/

```
ml-assignment-2/
├── app.py                          Streamlit application
├── train.py                        trains all five models, writes all artifacts
├── requirements.txt                dependency pins
├── README.md                       this file
├── test_data.csv                   held-out 200-row test split
├── data/
│   └── german_credit.csv           full dataset, headers applied
└── model/
    ├── ML_Assignment2_Colab.ipynb  notebook version (BITS Virtual Lab run)
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    ├── metadata.joblib             target, class names, feature order
    ├── metrics_comparison.csv      the table below, machine-readable
    └── confusion_matrices.png
```

---

## d. Models Used

All five models were trained on the identical 800-row training split with
identical preprocessing, and evaluated on the identical 200-row held-out split,
so the figures below are directly comparable.

Because the task is binary, Precision, Recall and F1 are reported for the positive
class (`good`), which is scikit-learn's default. The per-class breakdown for the
minority `bad` class is given separately underneath, since that is where the
models actually differ.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7050 | 0.7594 | 0.7755 | 0.8143 | 0.7944 | 0.2744 |
| Decision Tree | 0.7050 | 0.7270 | 0.7682 | 0.8286 | 0.7973 | 0.2613 |
| kNN | 0.7100 | 0.6942 | 0.7470 | 0.8857 | 0.8105 | 0.2266 |
| Naive Bayes | 0.6800 | 0.7109 | 0.8333 | 0.6786 | 0.7480 | 0.3350 |
| Random Forest (Ensemble) | 0.7600 | 0.7899 | 0.7805 | 0.9143 | 0.8421 | 0.3749 |

### Performance on the minority `bad` class

This table is not required by the assignment, but it is where the interesting
differences live — the headline table above can look flat while these numbers vary
by a factor of two.

| ML Model Name | Precision (bad) | Recall (bad) | F1 (bad) |
|:---|---:|---:|---:|
| Logistic Regression | 0.51 | 0.45 | 0.48 |
| Decision Tree | 0.51 | 0.42 | 0.46 |
| kNN | 0.53 | 0.30 | 0.38 |
| Naive Bayes | 0.48 | 0.68 | 0.56 |
| Random Forest (Ensemble) | 0.67 | 0.40 | 0.50 |

Confusion matrices for all five models: `model/confusion_matrices.png`.

### Observations

> **These are yours to write — see the note at the end of this file.**

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieves an Accuracy of 0.7050 and the second-highest AUC (0.7594) alongside an MCC of 0.2744. The substantial gap between its superior AUC and its mid-pack default accuracy illustrates the effect of the standard 0.5 decision threshold: while it reliably detects good applicants (Recall = 0.8143), it recovers only 45.00% (27/60) of bad credit risks. Strong AUC confirms that shifting the probability threshold can effectively control credit default costs. Key positive predictors for good credit include having no checking account (`checking_account_status_A14`, +0.930) and having a guarantor (`other_debtors_A103`, +0.906), whereas negative checking balances (`checking_account_status_A11`, -0.747) heavily increase risk. |
| Decision Tree | Yields an Accuracy of 0.7050, positive-class Recall of 0.8286, and F1 of 0.7973 (MCC: 0.2613), but records a lower AUC of 0.7270. A single decision tree relies on hard axis-aligned splits that produce piecewise-constant, uncalibrated leaf probability estimates. It identifies only 41.67% (25/60) of bad-risk applicants. Its splits are heavily concentrated on `checking_account_status_A14` (23.09% importance), `duration_months` (13.80%), and `credit_amount` (13.04%), exposing the high estimator variance that is subsequently tamed by ensembling. |
| kNN | Reaches 0.7100 Accuracy and the highest majority recall among individual models (0.8857, F1 = 0.8105), but suffers the lowest AUC (0.6942) and lowest MCC (0.2266) in the benchmark. It is the weakest at identifying credit defaults, catching only 30.00% (18/60) of bad risks while misclassifying 42 bad risks as good. This degradation is a direct consequence of the curse of dimensionality: one-hot encoding the 13 categorical features expands the feature space to over 60 dimensions, diluting Euclidean distance metrics and causing sparse minority-class instances to be overwhelmed by majority nearest neighbors. |
| Naive Bayes | Exhibits the lowest headline accuracy (0.6800) yet achieves the second-highest MCC (0.3350) and by far the highest default detection capability, capturing 68.33% (41/60) of bad risks (more than double kNN's recall). Although the conditional feature independence assumption is violated by interdependent loan terms like `credit_amount` and `duration_months`, the resulting skewed posterior probabilities heavily favor the minority class. In lending applications where false approvals carry severe financial penalties, Naive Bayes offers a uniquely cautious risk profile. |
| Random Forest (Ensemble) | Top performer across all six evaluation metrics: Accuracy (0.7600), AUC (0.7880), Precision (0.7805), Recall (0.9143), F1 (0.8421), and MCC (0.3749). Ensembling 300 randomized trees provides substantial variance reduction through bagging and feature subsampling, outperforming the single Decision Tree by +5.50% Accuracy, +0.061 AUC, and +0.1136 MCC. It also delivers the highest precision on bad risks (66.67%). Top impurity-based drivers are `credit_amount` (10.27%), `duration_months` (8.91%), and `age_years` (7.33%). |
| **Overall Winner** | **Random Forest (Ensemble)** is the clear overall winner. On a 70/30 class distribution, simple accuracy is deceptive—a naive majority classifier scores 0.7000 Accuracy but a useless 0.0 MCC. Random Forest's leading MCC of **0.3749**, AUC of **0.7880**, and balanced error distribution prove genuine discriminative power across both good and bad borrowers. |

---

## Streamlit App Features

| Requirement | Implementation |
|---|---|
| Dataset upload option (CSV) | Sidebar file uploader, with the bundled 200-row `test_data.csv` as a fallback so the app is never empty on first load |
| Model selection dropdown | Sidebar `selectbox` across all five trained pipelines |
| Display of evaluation metrics | "Evaluate selected model" tab — all six metrics as metric cards |
| Confusion matrix / classification report | Same tab, heatmap and per-class report side by side |
| Additional | Row-level predictions with confidence scores and CSV download; a "What drives the model" tab showing tree-based feature importances or fitted coefficients depending on the model selected |

Uploaded files are validated before scoring: the app checks that the target column
is present, that no feature columns are missing, and that the label values match
the classes seen during training, reporting a specific error rather than failing
on a stack trace.

---

## How to Run Locally

```bash
git clone https://github.com/2025ac05831/ml_assignment_2.git
cd ml_assignment_2
pip install -r requirements.txt

python train.py        # regenerates model/ and test_data.csv
streamlit run app.py   # opens on http://localhost:8501
```

To train on a different dataset instead:

```bash
python train.py --csv data/your_data.csv --target your_target_column --drop id
```

`train.py` asserts the ≥12 feature and ≥500 instance requirements, auto-detects
binary vs multi-class (switching the AUC strategy and averaging accordingly), and
rewrites the version pins in `requirements.txt` to match the environment that
produced the pickles.

---

## BITS Virtual Lab Execution

`model/ML_Assignment2_Colab.ipynb` is the notebook version of the training
pipeline. A screenshot of it executing on BITS Virtual Lab is included in the
submission PDF.

---

## Notes on Reproducibility

- `random_state=42` throughout — split and all stochastic estimators.
- `requirements.txt` pins `scikit-learn`, `pandas` and `numpy` to the exact
  versions that produced the committed `.joblib` files. Version drift between the
  machine that pickles a model and the machine that unpickles it is the most
  common cause of a Streamlit Cloud deployment failing on first load.
- Re-running `python train.py` reproduces every number in this README exactly.
