"""
save_model.py
Trains CatBoostClassifier and saves model.joblib + feature_columns.joblib
Run: python save_model.py
"""

import zipfile
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# ─── 1. Load data ─────────────────────────────────────────────────────────────
with zipfile.ZipFile("train.zip") as z:
    fname = [n for n in z.namelist() if n.endswith(".csv")][0]
    with z.open(fname) as f:
        train_raw = pd.read_csv(f)

with zipfile.ZipFile("test.zip") as z:
    fname = [n for n in z.namelist() if n.endswith(".csv")][0]
    with z.open(fname) as f:
        test_raw = pd.read_csv(f)

df = pd.concat([train_raw, test_raw], axis=0).reset_index(drop=True)

# ─── 2. Encoding ──────────────────────────────────────────────────────────────
employment_map = {
    "Employed": 0, "Self-employed": 1,
    "Unemployed": 2, "Retired": 3, "Student": 4,
}
df["employment_status"] = df["employment_status"].map(employment_map)

grade_map = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}
df["grade"]         = df["grade_subgrade"].str[0].map(grade_map)
df["subgrade"]      = df["grade_subgrade"].str[1:].astype(int)
df["grade_subgrade"] = df["grade"] * 10 + df["subgrade"]

# ─── 3. Feature engineering ───────────────────────────────────────────────────
df["high_dti"]           = (df["debt_to_income_ratio"] > 0.20).astype(int)
df["low_credit_score"]   = (df["credit_score"] < 650).astype(int)
df["high_interest_rate"] = (df["interest_rate"] > 13.0).astype(int)

df["credit_score_category"] = pd.cut(
    df["credit_score"],
    bins=[0, 580, 670, 740, 850],
    labels=["Poor", "Fair", "Good", "Excellent"],
)
df["dti_category"] = pd.cut(
    df["debt_to_income_ratio"],
    bins=[0, 0.1, 0.2, 0.3, 1],
    labels=["Low", "Medium", "High", "Very High"],
)
df["loan_amount_category"] = pd.cut(
    df["loan_amount"],
    bins=[0, 10000, 15000, 20000, 50000],
    labels=["Small", "Medium", "Large", "Very Large"],
)

# ─── 4. Split train / test ────────────────────────────────────────────────────
TOP_FEATURES = [
    "employment_status",
    "debt_to_income_ratio",
    "high_dti",
    "credit_score",
    "grade",
    "grade_subgrade",
    "low_credit_score",
    "interest_rate",
    "high_interest_rate",
    "loan_purpose",
    "credit_score_category",
    "dti_category",
    "loan_amount_category",
]

train_df = df[df["loan_paid_back"].notna()]
test_df  = df[df["loan_paid_back"].isna()]

x = train_df[TOP_FEATURES]
y = train_df["loan_paid_back"]
x_final_test = test_df[TOP_FEATURES]

x_all = pd.concat([x, x_final_test])
x_all = pd.get_dummies(x_all, drop_first=True)

x            = x_all.iloc[: len(x)]
x_final_test = x_all.iloc[len(x):]

feature_columns = x.columns.tolist()

# ─── 5. Train ─────────────────────────────────────────────────────────────────
print("Training CatBoostClassifier …")
model = CatBoostClassifier(verbose=0, random_state=42)
model.fit(x, y)
print("Training complete.")

# ─── 6. Save ──────────────────────────────────────────────────────────────────
joblib.dump(model,           "model.joblib")
joblib.dump(feature_columns, "feature_columns.joblib")
print("Saved → model.joblib")
print("Saved → feature_columns.joblib")
