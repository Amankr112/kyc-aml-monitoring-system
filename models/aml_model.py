"""
AML Transaction Monitoring Model
Uses Isolation Forest (unsupervised) + Rule Engine for AML detection.
Also trains a supervised GBM on labelled AML data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import joblib
import os

MODEL_DIR = os.path.dirname(__file__)
DATA_DIR  = os.path.join(MODEL_DIR, "..", "data")


def build_features(txn_df: pd.DataFrame, cust_df: pd.DataFrame) -> pd.DataFrame:
    df = txn_df.merge(
        cust_df[["customer_id", "kyc_score", "risk_tier", "is_pep", "sanctions_hit"]],
        on="customer_id", how="left"
    )

    RISK_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    DEST_RISK_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    df["txn_hour"]       = pd.to_datetime(df["txn_date"]).dt.hour
    df["amount_log"]     = np.log1p(df["amount"])
    df["risk_tier_num"]  = df["risk_tier"].map(RISK_MAP).fillna(1)
    df["dest_risk_num"]  = df["dest_risk"].map(DEST_RISK_MAP).fillna(0)

    features = [
        "amount_log", "amount", "txn_hour",
        "is_cross_border", "is_structuring", "is_round_number", "unusual_hours",
        "kyc_score", "risk_tier_num", "dest_risk_num",
        "is_pep", "sanctions_hit"
    ]
    return df[features + ["aml_flagged", "txn_id"]].copy()


def train():
    print("Loading data...")
    txn_df  = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    cust_df = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))

    feat_df = build_features(txn_df, cust_df)
    feature_cols = [c for c in feat_df.columns if c not in ["aml_flagged", "txn_id"]]

    X = feat_df[feature_cols].fillna(0)
    y = feat_df["aml_flagged"]

    print(f"Total transactions: {len(X)} | AML Flagged: {y.sum()} ({y.mean()*100:.1f}%)")

    # --- Isolation Forest (Unsupervised) ---
    print("\n=== Isolation Forest ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=200, contamination=0.12,
        random_state=42, n_jobs=-1
    )
    iso_preds = iso.fit_predict(X_scaled)
    iso_binary = (iso_preds == -1).astype(int)

    print(f"Isolation Forest flagged: {iso_binary.sum()} transactions")
    print(f"Precision: {precision_score(y, iso_binary):.3f}")
    print(f"Recall:    {recall_score(y, iso_binary):.3f}")
    print(f"F1:        {f1_score(y, iso_binary):.3f}")

    # --- Supervised GBM ---
    print("\n=== Gradient Boosting Classifier ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, random_state=42
    )
    gbm.fit(X_train, y_train)
    gbm_preds = gbm.predict(X_test)
    gbm_proba = gbm.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, gbm_preds, target_names=["CLEAN", "AML_FLAGGED"]))

    feat_imp = pd.Series(gbm.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Feature Importances:")
    for feat, imp in feat_imp.items():
        print(f"  {feat:<22} {imp:.4f}")

    # Save everything
    joblib.dump(iso,          os.path.join(MODEL_DIR, "aml_isolation.pkl"))
    joblib.dump(gbm,          os.path.join(MODEL_DIR, "aml_gbm.pkl"))
    joblib.dump(scaler,       os.path.join(MODEL_DIR, "aml_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "aml_features.pkl"))

    print(f"\n✓ AML models saved to {MODEL_DIR}")


def score_transaction(txn: dict, cust: dict) -> dict:
    """Score a single transaction for AML risk."""
    gbm      = joblib.load(os.path.join(MODEL_DIR, "aml_gbm.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "aml_features.pkl"))
    iso      = joblib.load(os.path.join(MODEL_DIR, "aml_isolation.pkl"))
    scaler   = joblib.load(os.path.join(MODEL_DIR, "aml_scaler.pkl"))

    RISK_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

    row = {
        "amount":          txn.get("amount", 0),
        "amount_log":      np.log1p(txn.get("amount", 0)),
        "txn_hour":        pd.to_datetime(txn.get("txn_date", "2024-01-01")).hour,
        "is_cross_border": int(txn.get("is_cross_border", 0)),
        "is_structuring":  int(txn.get("is_structuring", 0)),
        "is_round_number": int(txn.get("is_round_number", 0)),
        "unusual_hours":   int(txn.get("unusual_hours", 0)),
        "kyc_score":       cust.get("kyc_score", 50),
        "risk_tier_num":   RISK_MAP.get(cust.get("risk_tier", "MEDIUM"), 1),
        "dest_risk_num":   RISK_MAP.get(txn.get("dest_risk", "LOW"), 0),
        "is_pep":          int(cust.get("is_pep", 0)),
        "sanctions_hit":   int(cust.get("sanctions_hit", 0)),
    }

    X = pd.DataFrame([row])[features]
    proba      = gbm.predict_proba(X)[0][1]
    iso_score  = iso.decision_function(scaler.transform(X))[0]
    is_anomaly = iso.predict(scaler.transform(X))[0] == -1

    return {
        "aml_probability":  round(float(proba), 4),
        "is_anomaly":       bool(is_anomaly),
        "anomaly_score":    round(float(iso_score), 4),
        "aml_flag":         proba > 0.5 or is_anomaly,
        "risk_level":       "HIGH" if proba > 0.7 else ("MEDIUM" if proba > 0.4 else "LOW"),
    }


if __name__ == "__main__":
    train()
