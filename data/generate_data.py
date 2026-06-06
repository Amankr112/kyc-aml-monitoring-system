"""
Synthetic Banking Data Generator
Generates realistic customer profiles and transaction data for KYC/AML system.
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

fake = Faker()
random.seed(42)
np.random.seed(42)

# --- Country Risk Tiers ---
HIGH_RISK_COUNTRIES   = ["Iran", "North Korea", "Myanmar", "Syria", "Sudan", "Cuba", "Venezuela"]
MEDIUM_RISK_COUNTRIES = ["Pakistan", "Nigeria", "Kenya", "Bangladesh", "Egypt", "Ukraine", "Russia"]
LOW_RISK_COUNTRIES    = ["United States", "United Kingdom", "Germany", "France", "Japan",
                          "Canada", "Australia", "Singapore", "India", "Netherlands"]

OCCUPATION_RISK = {
    "Politician": 0.9, "Government Official": 0.85, "Arms Dealer": 0.95,
    "Cryptocurrency Trader": 0.7, "Lawyer": 0.5, "Doctor": 0.2,
    "Engineer": 0.15, "Teacher": 0.1, "Retail Worker": 0.1,
    "Banker": 0.3, "Business Owner": 0.45, "Journalist": 0.25,
}

def get_country_risk(country):
    if country in HIGH_RISK_COUNTRIES:   return "HIGH",   0.9
    if country in MEDIUM_RISK_COUNTRIES: return "MEDIUM", 0.5
    return "LOW", 0.1

def generate_customers(n=500):
    customers = []
    all_countries = HIGH_RISK_COUNTRIES + MEDIUM_RISK_COUNTRIES + LOW_RISK_COUNTRIES

    for i in range(n):
        country = random.choices(
            all_countries,
            weights=[10]*len(HIGH_RISK_COUNTRIES) + [20]*len(MEDIUM_RISK_COUNTRIES) + [70]*len(LOW_RISK_COUNTRIES),
            k=1
        )[0]
        country_risk_label, country_risk_score = get_country_risk(country)
        occupation = random.choice(list(OCCUPATION_RISK.keys()))
        occ_risk = OCCUPATION_RISK[occupation]

        # PEP = Politically Exposed Person
        is_pep = occupation in ["Politician", "Government Official"] or random.random() < 0.05
        sanctions_hit = (country in HIGH_RISK_COUNTRIES and random.random() < 0.15) or random.random() < 0.01
        adverse_media = random.random() < (0.2 if is_pep else 0.05)

        # KYC Score (0–100, higher = riskier)
        kyc_score = (
            country_risk_score * 30 +
            occ_risk * 25 +
            (25 if is_pep else 0) +
            (15 if sanctions_hit else 0) +
            (10 if adverse_media else 0) +
            random.uniform(-5, 5)
        )
        kyc_score = round(min(max(kyc_score, 0), 100), 2)

        if kyc_score >= 65:   risk_tier = "HIGH"
        elif kyc_score >= 35: risk_tier = "MEDIUM"
        else:                 risk_tier = "LOW"

        customers.append({
            "customer_id":      f"CUST{str(i+1).zfill(5)}",
            "name":             fake.name(),
            "email":            fake.email(),
            "phone":            fake.phone_number(),
            "country":          country,
            "country_risk":     country_risk_label,
            "occupation":       occupation,
            "is_pep":           int(is_pep),
            "sanctions_hit":    int(sanctions_hit),
            "adverse_media":    int(adverse_media),
            "kyc_score":        kyc_score,
            "risk_tier":        risk_tier,
            "onboarding_date":  fake.date_between(start_date="-3y", end_date="today").isoformat(),
            "account_balance":  round(random.uniform(500, 500000), 2),
        })

    return pd.DataFrame(customers)


def generate_transactions(customers_df, n=5000):
    transactions = []
    cust_ids = customers_df["customer_id"].tolist()
    high_risk_custs = customers_df[customers_df["risk_tier"] == "HIGH"]["customer_id"].tolist()

    for i in range(n):
        # High-risk customers generate more suspicious transactions
        if random.random() < 0.3 and high_risk_custs:
            cust_id = random.choice(high_risk_custs)
        else:
            cust_id = random.choice(cust_ids)

        cust = customers_df[customers_df["customer_id"] == cust_id].iloc[0]
        txn_date = fake.date_time_between(start_date="-1y", end_date="now")

        # Suspicious patterns
        is_structuring   = random.random() < (0.15 if cust["risk_tier"] == "HIGH" else 0.02)
        is_round_number  = random.random() < (0.2  if cust["risk_tier"] == "HIGH" else 0.05)
        is_unusual_hours = txn_date.hour in [1, 2, 3, 4] and random.random() < 0.3

        if is_structuring:
            # Just below $10,000 reporting threshold
            amount = round(random.uniform(9000, 9999), 2)
        elif is_round_number:
            amount = round(random.choice([5000, 10000, 25000, 50000, 100000]), 2)
        else:
            amount = round(random.expovariate(1/2000), 2)
            amount = min(amount, 200000)

        txn_type = random.choice(["WIRE_TRANSFER", "CASH_DEPOSIT", "CASH_WITHDRAWAL",
                                   "ONLINE_TRANSFER", "CHECK", "ACH"])

        # Cross-border flag
        dest_country = random.choice(LOW_RISK_COUNTRIES + MEDIUM_RISK_COUNTRIES + HIGH_RISK_COUNTRIES)
        is_cross_border = (dest_country != cust["country"]) and txn_type == "WIRE_TRANSFER"
        dest_risk = get_country_risk(dest_country)[0]

        # AML flag logic
        aml_flagged = (
            is_structuring or
            (amount > 50000 and cust["risk_tier"] == "HIGH") or
            (is_cross_border and dest_risk == "HIGH") or
            (cust["sanctions_hit"] and amount > 1000) or
            (is_unusual_hours and amount > 10000)
        )

        alert_type = None
        if aml_flagged:
            if is_structuring:                              alert_type = "STRUCTURING"
            elif is_cross_border and dest_risk == "HIGH":  alert_type = "HIGH_RISK_JURISDICTION"
            elif cust["sanctions_hit"]:                    alert_type = "SANCTIONS_HIT"
            elif amount > 50000:                           alert_type = "LARGE_TRANSACTION"
            else:                                          alert_type = "UNUSUAL_ACTIVITY"

        transactions.append({
            "txn_id":          f"TXN{str(i+1).zfill(6)}",
            "customer_id":     cust_id,
            "txn_date":        txn_date.isoformat(),
            "amount":          amount,
            "txn_type":        txn_type,
            "dest_country":    dest_country,
            "dest_risk":       dest_risk,
            "is_cross_border": int(is_cross_border),
            "is_structuring":  int(is_structuring),
            "is_round_number": int(is_round_number),
            "unusual_hours":   int(is_unusual_hours),
            "aml_flagged":     int(aml_flagged),
            "alert_type":      alert_type if aml_flagged else "NONE",
            "status":          random.choice(["PENDING", "CLEARED", "FLAGGED"]) if aml_flagged else "CLEARED",
        })

    return pd.DataFrame(transactions)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__))
    print("Generating 500 customers...")
    customers = generate_customers(500)
    customers.to_csv(f"{out}/customers.csv", index=False)
    print(f"  ✓ {len(customers)} customers | Risk: {customers['risk_tier'].value_counts().to_dict()}")

    print("Generating 5,000 transactions...")
    transactions = generate_transactions(customers, 5000)
    transactions.to_csv(f"{out}/transactions.csv", index=False)
    flagged = transactions["aml_flagged"].sum()
    print(f"  ✓ {len(transactions)} transactions | AML Flagged: {flagged} ({flagged/len(transactions)*100:.1f}%)")

    print("\nData generation complete.")
