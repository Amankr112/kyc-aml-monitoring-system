# KYC Risk Scoring & AML Transaction Monitoring System

> **Targeted at:** JPMorgan Chase Data Analyst / Analyst Quality Assurance roles  
> **Domain:** Anti-Money Laundering · Know Your Customer · Operational Risk  
> **Stack:** Python · scikit-learn · SQLite · Streamlit · Plotly · Pytest

---

## Project Overview

A production-grade KYC/AML compliance system simulating core workflows at a global financial institution. The system ingests synthetic banking customer and transaction data, applies ML-driven risk scoring, flags suspicious activity, and surfaces insights through a real-time analyst dashboard.

**Directly aligned with JPMorgan Chase JD requirements:**

| JD Requirement | How This Project Covers It |
|---|---|
| AML / KYC / Compliance experience | Core domain of the entire project |
| Data analysis & insight generation | ML scoring + Plotly dashboard |
| Fraud / Operational Risk | Isolation Forest anomaly detection |
| Client onboarding (AML/KYC fulfillment) | New Customer Onboarding page with live KYC screening |
| MS Office / PC systems proficiency | SQL queries, structured data pipelines |
| Performance evaluation via data | Executive Overview KPI page |

---

## Architecture

```
kyc-aml-system/
│
├── data/
│   ├── generate_data.py       # Synthetic customer & transaction generator (Faker)
│   ├── customers.csv          # 500 customer profiles
│   └── transactions.csv       # 5,000 banking transactions
│
├── database/
│   ├── db_setup.py            # SQLite schema + data loader
│   └── kyc_aml.db             # SQLite database
│
├── models/
│   ├── kyc_model.py           # Random Forest KYC risk classifier
│   ├── aml_model.py           # Isolation Forest + GBM AML detector
│   └── *.pkl                  # Saved model artifacts
│
├── dashboard/
│   └── app.py                 # Streamlit 5-page analyst dashboard
│
├── tests/
│   └── test_kyc_aml.py        # Full pytest suite (30+ tests)
│
└── requirements.txt
```

---

## Models

### KYC Risk Scoring — Random Forest Classifier
- **Features:** Country risk tier, occupation risk score, PEP status, sanctions hit, adverse media, account balance, KYC score
- **Output:** Risk tier (LOW / MEDIUM / HIGH) with confidence probabilities
- **Performance:** CV F1 = 0.993 (5-fold)

### AML Transaction Monitoring — Dual Engine
1. **Isolation Forest** (unsupervised) — detects anomalous transaction patterns without labels
2. **Gradient Boosting Classifier** (supervised) — trained on labelled AML flags
   - Precision/Recall: 99%+ on test set
   - Key features: Structuring flag, destination country risk, cross-border indicator

---

## Dashboard Pages

| Page | Description |
|---|---|
| 📊 Executive Overview | KPI cards, risk distribution pie, alert type bar, volume timeline |
| 👤 KYC Customer Risk | Searchable customer register, risk histogram, customer deep-dive |
| 🚨 AML Alert Center | Alert heatmap, box plots by alert type, full alert table |
| 🔍 Transaction Deep-Dive | Type breakdown, hourly heatmap, cross-border risk matrix |
| ➕ New Customer Onboarding | Live KYC form with instant risk scoring & compliance decision |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data
python data/generate_data.py

# 3. Initialize database
python database/db_setup.py

# 4. Train ML models
python models/kyc_model.py
python models/aml_model.py

# 5. Launch dashboard
streamlit run dashboard/app.py

# 6. Run tests
pytest tests/ -v
```

---

## Key Results

- **500 customers** profiled with KYC risk scores (0–100)
- **5,000 transactions** monitored; **12.3% AML flag rate**
- **613 AML alerts** across 5 alert types (Structuring, High-Risk Jurisdiction, Sanctions Hit, Large Transaction, Unusual Activity)
- **GBM model:** 99%+ precision & recall on AML detection
- **KYC model:** 0.993 CV F1-score across LOW/MEDIUM/HIGH tiers
- **30+ pytest tests** covering data quality, DB integrity, model predictions, and scoring logic

---

## Resume Bullet Points

```
• Built a KYC Risk Scoring & AML Transaction Monitoring system using Python, 
  scikit-learn (Random Forest + GBM + Isolation Forest), and SQLite, flagging 
  12.3% of 5,000+ synthetic banking transactions with 99% precision

• Designed a 5-page Streamlit compliance dashboard with real-time KYC onboarding 
  screening, AML alert management, and cross-border transaction risk analysis 
  using Plotly visualizations

• Implemented dual AML detection engine combining supervised GBM (labelled flags) 
  and unsupervised Isolation Forest (anomaly detection) across 5 alert categories 
  aligned with FATF AML guidelines
```

---

## Tech Stack

`Python 3.10+` · `pandas` · `scikit-learn` · `Faker` · `SQLite` · `SQLAlchemy`  
`Streamlit` · `Plotly` · `joblib` · `pytest`

---

*Built to demonstrate AML/KYC domain knowledge and data analysis capability for financial services roles.*
