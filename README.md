# 💳 Predicting Loan Payback

A machine learning project that predicts whether a customer will pay back their loan using a **CatBoostClassifier** model and a Streamlit web application.

## ✨ Features

- **CatBoostClassifier** model (gradient boosting)
- 13 top features including engineered risk indicators
- Streamlit UI with:
  - Real-time loan payback probability gauge
  - Low / Medium / High risk banner
  - Input summary with computed categories

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python save_model.py   # trains and saves model.joblib
streamlit run app.py
```

## 🗂️ Project Structure

```text
.
├── app.py                         # Streamlit uygulaması
├── save_model.py                  # Model eğitip kaydeden script
├── predicting-loan-payback.ipynb  # EDA + Feature Engineering + Model
├── model.joblib                   # Eğitilmiş CatBoostClassifier (save_model.py ile üretilir)
├── feature_columns.joblib         # Feature listesi (save_model.py ile üretilir)
├── requirements.txt
└── README.md
```

## ⚙️ Technical Details

| Özellik | Detay |
|---------|-------|
| **Model** | `CatBoostClassifier(verbose=0, random_state=42)` |
| **Target** | `loan_paid_back` (0 / 1) |
| **Features** | 13 top feature (ham + engineered) |
| **Encoding** | `pd.get_dummies(drop_first=True)` |
| **Scaling** | Yok (CatBoost gerektirmez) |

## 📊 Engineered Features

| Feature | Kural |
|---------|-------|
| `high_dti` | `debt_to_income_ratio > 0.20` |
| `low_credit_score` | `credit_score < 650` |
| `high_interest_rate` | `interest_rate > 13.0` |
| `grade` | A=5, B=4, C=3, D=2, E=1, F=0 |
| `grade_subgrade` | `grade × 10 + subgrade` |
| `credit_score_category` | Poor / Fair / Good / Excellent |
| `dti_category` | Low / Medium / High / Very High |
| `loan_amount_category` | Small / Medium / Large / Very Large |

## 👤 Author

Tugce Basyigit

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE).