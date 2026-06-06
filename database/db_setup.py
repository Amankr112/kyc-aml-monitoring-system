"""
Database Setup — KYC/AML System
Creates SQLite schema and loads CSV data into tables.
"""

import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "kyc_aml.db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    email           TEXT,
    phone           TEXT,
    country         TEXT,
    country_risk    TEXT CHECK(country_risk IN ('LOW','MEDIUM','HIGH')),
    occupation      TEXT,
    is_pep          INTEGER DEFAULT 0,
    sanctions_hit   INTEGER DEFAULT 0,
    adverse_media   INTEGER DEFAULT 0,
    kyc_score       REAL,
    risk_tier       TEXT CHECK(risk_tier IN ('LOW','MEDIUM','HIGH')),
    onboarding_date TEXT,
    account_balance REAL
);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id          TEXT PRIMARY KEY,
    customer_id     TEXT REFERENCES customers(customer_id),
    txn_date        TEXT,
    amount          REAL,
    txn_type        TEXT,
    dest_country    TEXT,
    dest_risk       TEXT,
    is_cross_border INTEGER DEFAULT 0,
    is_structuring  INTEGER DEFAULT 0,
    is_round_number INTEGER DEFAULT 0,
    unusual_hours   INTEGER DEFAULT 0,
    aml_flagged     INTEGER DEFAULT 0,
    alert_type      TEXT,
    status          TEXT
);

CREATE TABLE IF NOT EXISTS aml_alerts (
    alert_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id          TEXT REFERENCES transactions(txn_id),
    customer_id     TEXT REFERENCES customers(customer_id),
    alert_type      TEXT,
    alert_date      TEXT,
    amount          REAL,
    risk_tier       TEXT,
    analyst_note    TEXT DEFAULT '',
    resolution      TEXT DEFAULT 'OPEN'
);

CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_txn_flagged  ON transactions(aml_flagged);
CREATE INDEX IF NOT EXISTS idx_alerts_cust  ON aml_alerts(customer_id);
"""


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.commit()

    # Load customers
    cust_df = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    cust_df.to_sql("customers", conn, if_exists="replace", index=False)

    # Load transactions
    txn_df = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    txn_df.to_sql("transactions", conn, if_exists="replace", index=False)

    # Populate alerts from flagged transactions
    flagged = txn_df[txn_df["aml_flagged"] == 1].copy()
    alerts = pd.DataFrame({
        "alert_id":     range(1, len(flagged) + 1),
        "txn_id":       flagged["txn_id"].values,
        "customer_id":  flagged["customer_id"].values,
        "alert_type":   flagged["alert_type"].values,
        "alert_date":   flagged["txn_date"].str[:10].values,
        "amount":       flagged["amount"].values,
        "risk_tier":    flagged["customer_id"].map(
                            cust_df.set_index("customer_id")["risk_tier"]
                        ).values,
        "analyst_note": "",
        "resolution":   "OPEN",
    })
    alerts.to_sql("aml_alerts", conn, if_exists="replace", index=False)

    conn.close()
    print(f"✓ DB initialized at {DB_PATH}")
    print(f"  Customers: {len(cust_df)} | Transactions: {len(txn_df)} | Alerts: {len(alerts)}")


def query(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


if __name__ == "__main__":
    init_db()
