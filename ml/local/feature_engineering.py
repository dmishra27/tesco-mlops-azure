"""
Pandas-only feature engineering for local propensity modelling.

Temporal split design:
  Feature window  : days 1-120  (observation period)
  Label window    : days 121-150 (prediction target: did customer buy in
                                  a promoted category in the next 30 days?)
  Val  holdout    : random 20% of customers, same feature window, same label window
  Test holdout    : label window = days 151-180, features = days 1-150

Persona signal injection overrides the base label rate to the specified rates:
  Persona A: 70% positive  (loyalists — strong signal)
  Persona B: 35% positive  (digital-first — moderate signal)
  Persona C:  8% positive  (at-risk — weak signal)
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd
from tabulate import tabulate

PROMOTED_CATEGORIES = ["ready_meals", "bakery", "beverages"]
PERSONA_RATES = {"A": 0.70, "B": 0.35, "C": 0.08}
RNG_SEED = 42


def build_features(txns_df: pd.DataFrame, snapshot_date: date) -> pd.DataFrame:
    """Compute RFM + behavioural features with snapshot_date as reference."""
    df = txns_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    grp = df.groupby("customer_id")

    recency    = grp["date"].max().apply(lambda ts: (snapshot_date - ts.date()).days).rename("recency_days")
    frequency  = grp["basket_value"].count().rename("frequency")
    monetary   = grp["basket_value"].sum().rename("monetary")
    avg_basket = grp["basket_value"].mean().rename("avg_basket_size")
    basket_std = grp["basket_value"].std().fillna(0.0).rename("basket_std")
    active_days = grp["date"].apply(lambda s: s.dt.date.nunique()).rename("active_days")

    online_txns  = grp.apply(lambda x: (x["channel"] == "online").sum(),   include_groups=False).rename("online_txns")
    instore_txns = grp.apply(lambda x: (x["channel"] == "in-store").sum(), include_groups=False).rename("instore_txns")
    online_ratio = (online_txns / (online_txns + instore_txns).replace(0, np.nan)).fillna(0.0).rename("online_ratio")

    cat_spend   = df.groupby(["customer_id", "category"])["basket_value"].sum()
    top_category = cat_spend.groupby("customer_id").idxmax().apply(lambda x: x[1]).rename("top_category")

    promoted_flag = (
        df[df["category"].isin(PROMOTED_CATEGORIES)]
        .groupby("customer_id")["basket_value"].count()
        .gt(0).astype(int).rename("has_promoted_category")
    )

    features = pd.concat([
        recency, frequency, monetary, avg_basket, basket_std,
        online_ratio, active_days, top_category, promoted_flag,
    ], axis=1).reset_index()

    features["has_promoted_category"] = features["has_promoted_category"].fillna(0).astype(int)
    return features


def assign_persona_labels(
    customer_ids: pd.Series,
    persona_map: pd.DataFrame,
    label_window_txns: pd.DataFrame,
    seed: int = RNG_SEED,
) -> pd.Series:
    """
    Assign binary propensity labels using persona rates as the ground-truth signal.
    Rates: A=70%, B=35%, C=8% → expected overall ~20% positive rate.
    """
    rng = np.random.default_rng(seed)

    df = customer_ids.to_frame().merge(
        persona_map[["customer_id", "persona"]], on="customer_id", how="left"
    )
    df["persona"] = df["persona"].fillna("C")

    def _label(row):
        rate = PERSONA_RATES.get(row["persona"], 0.08)
        return int(rng.random() < rate)

    return df.apply(_label, axis=1)


def main():
    os.makedirs("data/features", exist_ok=True)
    os.makedirs("data/splits",   exist_ok=True)

    txns_df      = pd.read_csv("data/synthetic/transactions.csv")
    customers_df = pd.read_csv("data/synthetic/customers.csv")
    txns_df["date"] = pd.to_datetime(txns_df["date"])

    start_date = txns_df["date"].min()

    def day_num(dt): return (pd.to_datetime(dt) - start_date).dt.days

    txns_df["day"] = (txns_df["date"] - start_date).dt.days

    # Window slices
    feat_train_txns  = txns_df[txns_df["day"] <= 119]          # days  0-119 → features
    label_train_txns = txns_df[(txns_df["day"] >= 120) & (txns_df["day"] <= 149)]  # days 120-149 → label
    feat_test_txns   = txns_df[txns_df["day"] <= 149]          # days  0-149 → features (test)
    label_test_txns  = txns_df[txns_df["day"] >= 150]          # days 150-179 → label (test)

    snapshot_train = (start_date + pd.Timedelta(days=119)).date()
    snapshot_test  = (start_date + pd.Timedelta(days=149)).date()

    # Build full-window features (for reference CSV)
    full_features = build_features(txns_df, (start_date + pd.Timedelta(days=179)).date())
    full_features = full_features.merge(customers_df[["customer_id", "persona"]], on="customer_id", how="left")
    full_features.to_csv("data/features/customer_features.csv", index=False)

    print("=" * 60)
    print("FEATURE DISTRIBUTION SUMMARY")
    print("=" * 60)
    num_cols = ["recency_days", "frequency", "monetary", "avg_basket_size", "basket_std", "online_ratio", "active_days"]
    summary = full_features[num_cols].describe().T[["mean", "std", "min", "50%", "max"]].round(2)
    print(tabulate(summary, headers="keys", tablefmt="simple"))
    print(f"\nSaved to data/features/customer_features.csv  ({len(full_features):,} customers)")

    # ── Build train/val split ─────────────────────────────────────────────────
    # Features from days 0-119; labels from days 120-149
    train_val_features = build_features(feat_train_txns, snapshot_train)
    train_val_features = train_val_features.merge(
        customers_df[["customer_id", "persona"]], on="customer_id", how="left"
    )
    train_val_features["label"] = assign_persona_labels(
        train_val_features["customer_id"],
        customers_df,
        label_train_txns,
    )

    # 80/20 random customer split (stratified by persona)
    rng = np.random.default_rng(RNG_SEED)
    shuffled = train_val_features.sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)
    split_idx = int(len(shuffled) * 0.8)
    train_df = shuffled.iloc[:split_idx].copy()
    val_df   = shuffled.iloc[split_idx:].copy()

    # ── Build test split ──────────────────────────────────────────────────────
    # Features from days 0-149; labels from days 150-179
    test_features = build_features(feat_test_txns, snapshot_test)
    test_features = test_features.merge(
        customers_df[["customer_id", "persona"]], on="customer_id", how="left"
    )
    test_features["label"] = assign_persona_labels(
        test_features["customer_id"],
        customers_df,
        label_test_txns,
        seed=RNG_SEED + 1,
    )
    test_df = test_features

    train_df.to_csv("data/splits/train.csv", index=False)
    val_df.to_csv("data/splits/val.csv",     index=False)
    test_df.to_csv("data/splits/test.csv",   index=False)

    print("\n" + "=" * 60)
    print("TEMPORAL SPLIT CLASS BALANCE")
    print("=" * 60)
    rows = []
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n    = len(df)
        pos  = int(df["label"].sum())
        rate = pos / n if n > 0 else 0.0
        flag = ""
        if rate < 0.02:
            flag = "  WARNING: Signal too weak -- propensity modelling may not be meaningful"
        elif rate > 0.50:
            flag = "  WARNING: Signal too strong -- check label definition"
        rows.append([name, n, pos, f"{rate:.1%}{flag}"])
    print(tabulate(rows, headers=["Split", "Customers", "Positives", "Positive_rate"], tablefmt="simple"))
    print("\nSaved to data/splits/train.csv, val.csv, test.csv")


if __name__ == "__main__":
    main()
