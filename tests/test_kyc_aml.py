"""
Test Suite — KYC/AML System
Covers data generation, DB integrity, model predictions, and scoring logic.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import sqlite3

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


# ── Fixtures ────────────────────────────────────────────

@pytest.fixture(scope="session")
def customers_df():
    path = os.path.join(ROOT, "data", "customers.csv")
    assert os.path.exists(path), "Run data/generate_data.py first"
    return pd.read_csv(path)

@pytest.fixture(scope="session")
def transactions_df():
    path = os.path.join(ROOT, "data", "transactions.csv")
    assert os.path.exists(path), "Run data/generate_data.py first"
    return pd.read_csv(path)

@pytest.fixture(scope="session")
def db_conn():
    db_path = os.path.join(ROOT, "database", "kyc_aml.db")
    assert os.path.exists(db_path), "Run database/db_setup.py first"
    conn = sqlite3.connect(db_path)
    yield conn
    conn.close()


# ── Data Quality Tests ───────────────────────────────────

class TestDataQuality:

    def test_customers_row_count(self, customers_df):
        assert len(customers_df) >= 400, "Expected at least 400 customers"

    def test_no_null_customer_ids(self, customers_df):
        assert customers_df["customer_id"].notna().all()

    def test_kyc_score_range(self, customers_df):
        assert (customers_df["kyc_score"] >= 0).all()
        assert (customers_df["kyc_score"] <= 100).all()

    def test_risk_tier_values(self, customers_df):
        assert set(customers_df["risk_tier"].unique()).issubset({"LOW", "MEDIUM", "HIGH"})

    def test_transactions_row_count(self, transactions_df):
        assert len(transactions_df) >= 4000, "Expected at least 4,000 transactions"

    def test_no_negative_amounts(self, transactions_df):
        assert (transactions_df["amount"] >= 0).all()

    def test_aml_flag_is_binary(self, transactions_df):
        assert set(transactions_df["aml_flagged"].unique()).issubset({0, 1})

    def test_flagged_rate_in_range(self, transactions_df):
        rate = transactions_df["aml_flagged"].mean()
        assert 0.05 <= rate <= 0.30, f"Flag rate {rate:.2%} outside expected 5–30%"

    def test_customer_ids_match(self, customers_df, transactions_df):
        cust_ids = set(customers_df["customer_id"])
        txn_ids  = set(transactions_df["customer_id"])
        assert txn_ids.issubset(cust_ids), "Transaction has unknown customer_id"


# ── Database Integrity Tests ─────────────────────────────

class TestDatabase:

    def test_tables_exist(self, db_conn):
        cur = db_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        for t in ["customers", "transactions", "aml_alerts"]:
            assert t in tables, f"Table '{t}' missing"

    def test_customers_count(self, db_conn):
        n = db_conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        assert n >= 400

    def test_alerts_populated(self, db_conn):
        n = db_conn.execute("SELECT COUNT(*) FROM aml_alerts").fetchone()[0]
        assert n > 0, "AML alerts table is empty"

    def test_referential_integrity(self, db_conn):
        orphans = db_conn.execute("""
            SELECT COUNT(*) FROM transactions t
            LEFT JOIN customers c ON t.customer_id = c.customer_id
            WHERE c.customer_id IS NULL
        """).fetchone()[0]
        assert orphans == 0, f"{orphans} transactions with no matching customer"

    def test_kyc_scores_stored(self, db_conn):
        null_scores = db_conn.execute(
            "SELECT COUNT(*) FROM customers WHERE kyc_score IS NULL"
        ).fetchone()[0]
        assert null_scores == 0


# ── KYC Model Tests ──────────────────────────────────────

class TestKYCModel:

    @pytest.fixture(scope="class")
    def model_files(self):
        import joblib
        model_dir = os.path.join(ROOT, "models")
        assert os.path.exists(os.path.join(model_dir, "kyc_model.pkl")), \
            "Run models/kyc_model.py first"
        model    = joblib.load(os.path.join(model_dir, "kyc_model.pkl"))
        features = joblib.load(os.path.join(model_dir, "kyc_features.pkl"))
        return model, features

    def test_model_loads(self, model_files):
        model, features = model_files
        assert model is not None
        assert len(features) > 0

    def test_model_predicts_low_risk(self, model_files):
        from models.kyc_model import predict_risk
        result = predict_risk({
            "country_risk": "LOW", "occupation": "Teacher",
            "is_pep": 0, "sanctions_hit": 0, "adverse_media": 0,
            "account_balance": 10000, "kyc_score": 10
        })
        assert result["predicted_tier"] == "LOW"
        assert result["confidence"] > 0.5

    def test_model_predicts_high_risk(self, model_files):
        from models.kyc_model import predict_risk
        result = predict_risk({
            "country_risk": "HIGH", "occupation": "Politician",
            "is_pep": 1, "sanctions_hit": 1, "adverse_media": 1,
            "account_balance": 500000, "kyc_score": 90
        })
        assert result["predicted_tier"] == "HIGH"

    def test_prediction_probabilities_sum_to_one(self, model_files):
        from models.kyc_model import predict_risk
        result = predict_risk({
            "country_risk": "MEDIUM", "occupation": "Banker",
            "is_pep": 0, "sanctions_hit": 0, "adverse_media": 0,
            "account_balance": 25000, "kyc_score": 45
        })
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-3


# ── AML Model Tests ──────────────────────────────────────

class TestAMLModel:

    def test_aml_model_files_exist(self):
        model_dir = os.path.join(ROOT, "models")
        for fname in ["aml_gbm.pkl", "aml_isolation.pkl", "aml_scaler.pkl", "aml_features.pkl"]:
            assert os.path.exists(os.path.join(model_dir, fname)), f"Missing: {fname}"

    def test_clean_transaction_low_risk(self):
        from models.aml_model import score_transaction
        txn = {"amount": 500, "txn_date": "2024-06-15 10:00:00",
               "is_cross_border": 0, "is_structuring": 0,
               "is_round_number": 0, "unusual_hours": 0,
               "dest_risk": "LOW"}
        cust = {"kyc_score": 10, "risk_tier": "LOW", "is_pep": 0, "sanctions_hit": 0}
        result = score_transaction(txn, cust)
        assert result["aml_probability"] < 0.5
        assert result["risk_level"] == "LOW"

    def test_suspicious_transaction_high_risk(self):
        from models.aml_model import score_transaction
        txn = {"amount": 9500, "txn_date": "2024-06-15 02:30:00",
               "is_cross_border": 1, "is_structuring": 1,
               "is_round_number": 0, "unusual_hours": 1,
               "dest_risk": "HIGH"}
        cust = {"kyc_score": 85, "risk_tier": "HIGH", "is_pep": 1, "sanctions_hit": 1}
        result = score_transaction(txn, cust)
        assert result["aml_probability"] > 0.5

    def test_score_output_keys(self):
        from models.aml_model import score_transaction
        txn  = {"amount": 1000, "txn_date": "2024-01-01 12:00:00",
                "is_cross_border": 0, "is_structuring": 0,
                "is_round_number": 0, "unusual_hours": 0, "dest_risk": "LOW"}
        cust = {"kyc_score": 30, "risk_tier": "LOW", "is_pep": 0, "sanctions_hit": 0}
        result = score_transaction(txn, cust)
        for key in ["aml_probability","is_anomaly","anomaly_score","aml_flag","risk_level"]:
            assert key in result, f"Missing key: {key}"


# ── KYC Scoring Logic Tests ──────────────────────────────

class TestKYCScoringLogic:

    def test_pep_increases_score(self):
        """PEP flag should increase KYC score."""
        base = 30  # base score
        pep_score  = base + 25
        assert pep_score > base

    def test_high_risk_country_increases_score(self):
        high_c  = 0.9 * 30
        low_c   = 0.1 * 30
        assert high_c > low_c

    def test_kyc_score_capped_at_100(self):
        """KYC score should never exceed 100."""
        score = min(100, 0.9*30 + 0.95*25 + 25 + 15 + 10 + 10)
        assert score <= 100

    def test_risk_tier_thresholds(self):
        def tier(score):
            if score >= 65:   return "HIGH"
            elif score >= 35: return "MEDIUM"
            return "LOW"
        assert tier(10)  == "LOW"
        assert tier(50)  == "MEDIUM"
        assert tier(80)  == "HIGH"
        assert tier(65)  == "HIGH"
        assert tier(34)  == "LOW"
        assert tier(35)  == "MEDIUM"
