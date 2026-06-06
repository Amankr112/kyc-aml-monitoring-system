"""
KYC Risk Scoring Model
Trains a Random Forest classifier to predict customer risk tier (LOW/MEDIUM/HIGH).
Uses customer profile features: country risk, occupation risk, PEP status, etc.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

MODEL_DIR = os.path.dirname(__file__)
DATA_DIR  = os.path.join(MODEL_DIR, "..", "data")


COUNTRY_RISK_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
RISK_TIER_MAP    = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

OCCUPATION_RISK_MAP = {
    "Politician": 0.9, "Government Official": 0.85, "Arms Dealer": 0.95,
    "Cryptocurrency Trader": 0.7, "Lawyer": 0.5, "Doctor": 0.2,
    "Engineer": 0.15, "Teacher": 0.1, "Retail Worker": 0.1,
    "Banker": 0.3, "Business Owner": 0.45, "Journalist": 0.25,
}


def load_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame()
    X["country_risk_num"]    = df["country_risk"].map(COUNTRY_RISK_MAP).fillna(1)
    X["occupation_risk"]     = df["occupation"].map(OCCUPATION_RISK_MAP).fillna(0.3)
    X["is_pep"]              = df["is_pep"].astype(int)
    X["sanctions_hit"]       = df["sanctions_hit"].astype(int)
    X["adverse_media"]       = df["adverse_media"].astype(int)
    X["account_balance_log"] = np.log1p(df["account_balance"])
    X["kyc_score"]           = df["kyc_score"]
    return X


def train():
    print("Loading data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))

    X = load_features(df)
    y = df["risk_tier"].map(RISK_TIER_MAP)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Class distribution:\n{df['risk_tier'].value_counts()}\n")

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    print("=== Random Forest ===")
    print(classification_report(y_test, rf_preds, target_names=["LOW", "MEDIUM", "HIGH"]))

    # Cross-validation
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="f1_weighted")
    print(f"CV F1 (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Feature importance
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importances:")
    for feat, imp in feat_imp.items():
        print(f"  {feat:<25} {imp:.4f}")

    # Save model + feature list
    joblib.dump(rf, os.path.join(MODEL_DIR, "kyc_model.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "kyc_features.pkl"))
    print(f"\n✓ Model saved to {MODEL_DIR}/kyc_model.pkl")

    return rf, X_test, y_test


def predict_risk(customer_dict: dict) -> dict:
    """Predict risk for a single customer record."""
    model    = joblib.load(os.path.join(MODEL_DIR, "kyc_model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "kyc_features.pkl"))

    row = pd.DataFrame([customer_dict])
    X   = load_features(row)[features]

    proba    = model.predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    labels   = ["LOW", "MEDIUM", "HIGH"]

    return {
        "predicted_tier": labels[pred_idx],
        "confidence":     round(float(proba[pred_idx]), 4),
        "probabilities":  {labels[i]: round(float(p), 4) for i, p in enumerate(proba)},
    }


if __name__ == "__main__":
    train()
