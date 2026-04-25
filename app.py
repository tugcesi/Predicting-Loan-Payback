"""
💳 Loan Payback Prediction — Streamlit App
Model: CatBoostClassifier
Files: model.joblib, feature_columns.joblib
"""

from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="💳 Loan Payback Prediction",
    page_icon="💳",
    layout="wide",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH    = Path("model.joblib")
FEATURES_PATH = Path("feature_columns.joblib")

# ─── Constants ────────────────────────────────────────────────────────────────
EMPLOYMENT_MAP = {
    "Employed": 0, "Self-employed": 1,
    "Unemployed": 2, "Retired": 3, "Student": 4,
}
GRADE_MAP = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}

LOAN_PURPOSE_OPTIONS = [
    "Business", "Car", "Debt consolidation",
    "Education", "Home", "Medical", "Other", "Vacation",
]
GRADE_OPTIONS    = ["A", "B", "C", "D", "E", "F"]
SUBGRADE_OPTIONS = [1, 2, 3, 4, 5]


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        return None, None
    model           = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


def compute_features(p: dict) -> pd.DataFrame:
    grade_num     = GRADE_MAP[p["grade"]]
    subgrade_num  = p["subgrade"]
    grade_subgrade = grade_num * 10 + subgrade_num

    dti        = p["debt_to_income_ratio"]
    credit     = p["credit_score"]
    rate       = p["interest_rate"]
    loan_amt   = p["loan_amount"]

    high_dti           = int(dti > 0.20)
    low_credit_score   = int(credit < 650)
    high_interest_rate = int(rate > 13.0)

    # credit_score_category
    if credit <= 580:
        csc = "Poor"
    elif credit <= 670:
        csc = "Fair"
    elif credit <= 740:
        csc = "Good"
    else:
        csc = "Excellent"

    # dti_category
    if dti <= 0.1:
        dtc = "Low"
    elif dti <= 0.2:
        dtc = "Medium"
    elif dti <= 0.3:
        dtc = "High"
    else:
        dtc = "Very High"

    # loan_amount_category
    if loan_amt <= 10000:
        lac = "Small"
    elif loan_amt <= 15000:
        lac = "Medium"
    elif loan_amt <= 20000:
        lac = "Large"
    else:
        lac = "Very Large"

    row = pd.DataFrame([{
        "employment_status":      EMPLOYMENT_MAP[p["employment_status"]],
        "debt_to_income_ratio":   dti,
        "high_dti":               high_dti,
        "credit_score":           credit,
        "grade":                  grade_num,
        "grade_subgrade":         grade_subgrade,
        "low_credit_score":       low_credit_score,
        "interest_rate":          rate,
        "high_interest_rate":     high_interest_rate,
        "loan_purpose":           p["loan_purpose"],
        "credit_score_category":  csc,
        "dti_category":           dtc,
        "loan_amount_category":   lac,
    }])
    return row


def prepare_input(row: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    row = pd.get_dummies(row, drop_first=True)
    for col in feature_columns:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_columns]
    return row.astype(np.float32)


def create_gauge(prob: float) -> go.Figure:
    pct = prob * 100
    color = "#52c41a" if pct >= 60 else ("#faad14" if pct >= 40 else "#f5222d")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0,  40], "color": "#fff1f0"},
                {"range": [40, 60], "color": "#fffbe6"},
                {"range": [60, 100], "color": "#f6ffed"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75, "value": pct,
            },
        },
        title={"text": "Loan Payback Probability", "font": {"size": 18}},
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=0, l=30, r=30))
    return fig


def risk_banner(prob: float) -> None:
    pct = prob * 100
    if pct >= 60:
        st.success(f"✅ **Düşük Risk** — Krediyi geri ödeme ihtimali yüksek ({pct:.1f}%)")
    elif pct >= 40:
        st.warning(f"⚠️ **Orta Risk** — Krediyi geri ödeme ihtimali orta ({pct:.1f}%)")
    else:
        st.error(f"🚨 **Yüksek Risk** — Krediyi geri ödemeyebilir ({pct:.1f}%)")


def build_sidebar() -> dict:
    with st.sidebar:
        st.title("💳 Loan Payback Predictor")
        st.markdown("---")
        st.markdown("### 👤 Müşteri Bilgileri")
        employment_status = st.selectbox("Employment Status", list(EMPLOYMENT_MAP.keys()))

        st.markdown("### 📊 Kredi Bilgileri")
        loan_amount   = st.slider("Loan Amount ($)", 500, 50000, 10000, 500)
        interest_rate = st.slider("Interest Rate (%)", 3.2, 21.3, 12.4, 0.1)
        grade         = st.selectbox("Grade", GRADE_OPTIONS)
        subgrade      = st.selectbox("Subgrade", SUBGRADE_OPTIONS)
        loan_purpose  = st.selectbox("Loan Purpose", LOAN_PURPOSE_OPTIONS)

        st.markdown("### 💰 Finansal Bilgiler")
        credit_score         = st.slider("Credit Score", 395, 849, 680)
        debt_to_income_ratio = st.slider("Debt-to-Income Ratio", 0.01, 0.63, 0.12, 0.01)

        st.markdown("---")
        predict_clicked = st.button(
            "🔍 Predict", use_container_width=True, type="primary"
        )

    return {
        "payload": {
            "employment_status":      employment_status,
            "loan_amount":            loan_amount,
            "interest_rate":          interest_rate,
            "grade":                  grade,
            "subgrade":               subgrade,
            "loan_purpose":           loan_purpose,
            "credit_score":           credit_score,
            "debt_to_income_ratio":   debt_to_income_ratio,
        },
        "predict_clicked": predict_clicked,
    }


def main() -> None:
    model, feature_columns = load_artifacts()

    sidebar         = build_sidebar()
    payload         = sidebar["payload"]
    predict_clicked = sidebar["predict_clicked"]

    if model is None:
        st.error(
            "⚠️ **Model dosyaları bulunamadı!**\n\n"
            "Önce modeli eğit:\n```bash\npython save_model.py\n```"
        )
        st.stop()

    row      = compute_features(payload)
    x_input  = prepare_input(row, feature_columns)
    prob     = float(model.predict_proba(x_input)[0][1])
    pred     = int(model.predict(x_input)[0])

    st.title("💳 Loan Payback Prediction")
    st.markdown("Soldaki panelden müşteri bilgilerini gir, kredi geri ödeme tahminini gör.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Payback Prob.",    f"{prob * 100:.1f}%")
    c2.metric("📋 Prediction",       "✅ Ödeyecek" if pred == 1 else "❌ Ödemeyecek")
    c3.metric("📊 Credit Score",     payload["credit_score"]) 
    c4.metric("💹 Interest Rate",    f"{payload['interest_rate']}%" )

    if predict_clicked:
        st.markdown("---")
        risk_banner(prob)

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.plotly_chart(create_gauge(prob), use_container_width=True)

    with col_right:
        st.markdown("#### 📋 Girilen Bilgiler")
        dti = payload["debt_to_income_ratio"]
        cs  = payload["credit_score"]
        ir  = payload["interest_rate"]
        la  = payload["loan_amount"]

        # Computed values
        if dti <= 0.1:   dtc = "Low"
        elif dti <= 0.2: dtc = "Medium"
        elif dti <= 0.3: dtc = "High"
        else:            dtc = "Very High"

        if cs <= 580:    csc = "Poor"
        elif cs <= 670:  csc = "Fair"
        elif cs <= 740:  csc = "Good"
        else:            csc = "Excellent"

        if la <= 10000:  lac = "Small"
        elif la <= 15000: lac = "Medium"
        elif la <= 20000: lac = "Large"
        else:            lac = "Very Large"

        st.dataframe(
            pd.DataFrame({
                "Özellik": [
                    "Employment Status", "Loan Amount", "Interest Rate",
                    "Grade", "Subgrade", "Loan Purpose",
                    "Credit Score", "DTI Ratio",
                    "Credit Category", "DTI Category", "Loan Size",
                    "High DTI", "Low Credit", "High Rate",
                ],
                "Değer": [
                    payload["employment_status"],
                    f"${payload['loan_amount']:,}",
                    f"{payload['interest_rate']}%",
                    payload["grade"],
                    payload["subgrade"],
                    payload["loan_purpose"],
                    payload["credit_score"],
                    f"{payload['debt_to_income_ratio']:.3f}",
                    csc, dtc, lac,
                    "✅" if dti > 0.20 else "❌",
                    "✅" if cs < 650   else "❌",
                    "✅" if ir > 13.0  else "❌",
                ],
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.caption(
        "Powered by CatBoost & Streamlit | "
        "[GitHub](https://github.com/tugcesi/Predicting-Loan-Payback)"
    )


if __name__ == "__main__":
    main()