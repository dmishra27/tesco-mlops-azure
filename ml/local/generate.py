"""
Synthetic transaction generator for Tesco propensity modelling.
Produces 5000 customers / 50000 transactions over 180 days with
3 personas that carry a known ground-truth propensity signal.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEED = 42
START_DATE = date(2024, 1, 1)
END_DATE   = date(2024, 6, 29)   # 180 days inclusive
N_CUSTOMERS = 5_000
N_TRANSACTIONS = 50_000

PERSONAS = {
    "A": {"n": 500,  "freq_range": (15, 30), "spend_range": (300, 800),  "online_range": (0.1, 0.3)},
    "B": {"n": 1000, "freq_range": (8,  15), "spend_range": (150, 400),  "online_range": (0.6, 0.9)},
    "C": {"n": 3500, "freq_range": (1,  5),  "spend_range": (20,  100),  "online_range": (0.2, 0.7)},
}

CATEGORIES = ["ready_meals", "bakery", "produce", "dairy", "beverages",
              "snacks", "frozen", "household", "personal_care", "alcohol"]

PROMOTED_CATEGORIES = ["ready_meals", "bakery", "beverages"]


def _power_law_basket(rng: np.random.Generator, n: int, low: float, high: float) -> np.ndarray:
    """Power-law distributed basket sizes between low and high."""
    raw = rng.power(0.5, n)                          # skewed toward lower end
    return low + raw * (high - low)


def _exponential_recency(rng: np.random.Generator, n: int, decay: float = 0.03) -> np.ndarray:
    """Days-ago drawn from exponential — recent purchases more likely."""
    raw = rng.exponential(scale=1.0 / decay, size=n)
    return np.clip(raw, 0, 179).astype(int)


def generate(out_dir: str = "data/synthetic") -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(out_dir, exist_ok=True)

    # ── Build customer table ──────────────────────────────────────────────────
    customer_rows = []
    cust_id = 1
    for persona, cfg in PERSONAS.items():
        for _ in range(cfg["n"]):
            freq = int(rng.integers(cfg["freq_range"][0], cfg["freq_range"][1] + 1))
            target_spend = float(rng.uniform(*cfg["spend_range"]))
            online_ratio = float(rng.uniform(*cfg["online_range"]))
            customer_rows.append({
                "customer_id":   f"CUST-{cust_id:05d}",
                "persona":       persona,
                "target_freq":   freq,
                "target_spend":  target_spend,
                "online_ratio":  online_ratio,
            })
            cust_id += 1

    customers_df = pd.DataFrame(customer_rows)

    # ── Generate transactions ────────────────────────────────────────────────
    txn_rows = []
    txn_id = 1

    for _, cust in customers_df.iterrows():
        freq    = cust["target_freq"]
        spend   = cust["target_spend"]
        o_ratio = cust["online_ratio"]
        persona = cust["persona"]

        # Recency-biased day offsets (exponential decay toward end of window)
        days_ago = _exponential_recency(rng, freq, decay=0.025)
        txn_dates = sorted([START_DATE + timedelta(days=int(179 - d)) for d in days_ago])

        # Target spend is total over all transactions; derive per-basket target
        avg_basket_target = spend / freq
        for txn_date in txn_dates:
            basket = float(_power_law_basket(rng, 1, avg_basket_target * 0.4, avg_basket_target * 1.8)[0])
            channel = "online" if rng.random() < o_ratio else "in-store"

            # Persona A skews heavily toward promoted categories
            if persona == "A":
                cat_weights = [0.25, 0.20, 0.05, 0.05, 0.20, 0.05, 0.05, 0.05, 0.05, 0.05]
            elif persona == "B":
                cat_weights = [0.15, 0.10, 0.10, 0.10, 0.25, 0.05, 0.05, 0.10, 0.05, 0.05]
            else:
                cat_weights = [0.10] * 10

            category = rng.choice(CATEGORIES, p=np.array(cat_weights) / sum(cat_weights))

            txn_rows.append({
                "transaction_id": f"TXN-{txn_id:07d}",
                "customer_id":    cust["customer_id"],
                "persona":        persona,
                "date":           txn_date,
                "category":       category,
                "channel":        channel,
                "basket_value":   round(basket, 2),
            })
            txn_id += 1

    txns_df = pd.DataFrame(txn_rows)

    # Trim/pad to target transaction count via sampling
    if len(txns_df) > N_TRANSACTIONS:
        txns_df = txns_df.sample(N_TRANSACTIONS, random_state=SEED).reset_index(drop=True)
    elif len(txns_df) < N_TRANSACTIONS:
        extra = txns_df.sample(N_TRANSACTIONS - len(txns_df), replace=True, random_state=SEED)
        txns_df = pd.concat([txns_df, extra], ignore_index=True)

    txns_df["date"] = pd.to_datetime(txns_df["date"])
    customers_df.to_csv(f"{out_dir}/customers.csv", index=False)
    txns_df.to_csv(f"{out_dir}/transactions.csv", index=False)

    # ── Summary stats ────────────────────────────────────────────────────────
    print("=" * 60)
    print("SYNTHETIC DATA GENERATION — SUMMARY")
    print("=" * 60)
    print(f"Customers  : {len(customers_df):,}")
    print(f"Transactions: {len(txns_df):,}")
    print(f"Date range : {txns_df['date'].min().date()} to {txns_df['date'].max().date()}")
    print()

    for persona in ["A", "B", "C"]:
        p = customers_df[customers_df["persona"] == persona]
        t = txns_df[txns_df["persona"] == persona]
        per_cust = t.groupby("customer_id")["basket_value"].agg(["count", "sum"])
        print(f"Persona {persona} ({len(p)} customers)")
        print(f"  Freq   : {per_cust['count'].mean():.1f} txns/cust  "
              f"(target {PERSONAS[persona]['freq_range']})")
        print(f"  Spend  : £{per_cust['sum'].mean():.0f}/cust  "
              f"(target £{PERSONAS[persona]['spend_range']})")
        print(f"  Online : {t[t['channel']=='online'].shape[0] / len(t):.1%}  "
              f"(target {PERSONAS[persona]['online_range']})")

    print(f"\nSaved to {out_dir}/transactions.csv")
    print(f"Saved to {out_dir}/customers.csv")


if __name__ == "__main__":
    generate()
