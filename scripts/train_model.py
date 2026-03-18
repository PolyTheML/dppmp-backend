"""
Train a simple risk-multiplier model on synthetic data.
Saves a .pkl file that the API loads at startup.

Usage:
    python scripts/train_model.py
    → creates models/risk_model.pkl
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import joblib

SEED = 42
N_SAMPLES = 5000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "risk_model.pkl")

# ─── Region/gender/plan encoding maps (must match main.py) ──────────────────
REGION_MAP = {
    "Phnom Penh": 0, "Siem Reap": 1, "Battambang": 2,
    "Sihanoukville": 3, "Kampong Cham": 4, "Rural Areas": 5,
}
GENDER_MAP = {"Male": 0, "Female": 1, "Other": 2}
PLAN_MAP = {"Basic": 0, "Gold": 1, "Platinum": 2}

# ─── Ground-truth factor functions ───────────────────────────────────────────
REGION_FACTORS = [1.15, 1.05, 0.95, 1.08, 0.90, 0.85]   # indexed by region_enc
GENDER_FACTORS = [1.02, 0.98, 1.00]                        # indexed by gender_enc


def bmi_factor(bmi: float) -> float:
    if bmi < 18.5:
        return 1.08
    if bmi <= 24.9:
        return 1.00
    if bmi <= 29.9:
        return 1.12
    return 1.25


def generate_synthetic_data(n: int, seed: int) -> pd.DataFrame:
    """Generate synthetic applicant data with a known risk-multiplier formula + noise."""
    rng = np.random.RandomState(seed)

    ages = rng.randint(18, 76, size=n)
    genders = rng.choice([0, 1, 2], size=n, p=[0.48, 0.48, 0.04])
    regions = rng.choice(list(range(6)), size=n)
    plans = rng.choice([0, 1, 2], size=n, p=[0.35, 0.45, 0.20])
    smoking = rng.choice([0, 1], size=n, p=[0.75, 0.25])
    bmis = np.clip(rng.normal(25.0, 5.0, size=n), 14, 55).round(1)

    # Compute ground-truth multiplier
    multipliers = np.ones(n)
    for i in range(n):
        age_f = 1.0 + max(0, (ages[i] - 30)) * 0.004
        gender_f = GENDER_FACTORS[genders[i]]
        region_f = REGION_FACTORS[regions[i]]
        smoking_f = 1.35 if smoking[i] else 1.00
        bmi_f = bmi_factor(bmis[i])

        base = age_f * gender_f * region_f * smoking_f * bmi_f
        noise = rng.normal(0, 0.03)  # ±3% noise
        multipliers[i] = np.clip(base + noise, 0.70, 1.80)

    df = pd.DataFrame({
        "age": ages,
        "gender_enc": genders,
        "region_enc": regions,
        "plan_enc": plans,
        "smoking": smoking,
        "bmi": bmis,
        "risk_multiplier": multipliers.round(3),
    })
    return df


def train():
    print(f"Generating {N_SAMPLES} synthetic training samples...")
    df = generate_synthetic_data(N_SAMPLES, SEED)

    X = df[["age", "gender_enc", "region_enc", "plan_enc", "smoking", "bmi"]].values
    y = df["risk_multiplier"].values

    print("Training GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        random_state=SEED,
    )
    model.fit(X, y)

    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    rmse = -scores.mean()
    print(f"5-fold CV RMSE: {rmse:.4f}")

    # R² score
    r2 = model.score(X, y)
    print(f"Training R²:    {r2:.4f}")

    # Feature importance
    feature_names = ["age", "gender", "region", "plan", "smoking", "bmi"]
    importances = model.feature_importances_
    print("\nFeature Importance:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {name:10s} {imp:.3f} {bar}")

    # Stamp version on model object
    model._dppmp_version = "v1.0.0"

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, OUTPUT_PATH)
    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\nModel saved to {OUTPUT_PATH} ({size_kb:.0f} KB)")

    # Quick sanity check
    test_input = np.array([[35, 0, 0, 1, 0, 24.0]])  # 35yo male, Phnom Penh, Gold, non-smoker, BMI 24
    pred = model.predict(test_input)[0]
    print(f"Sanity check — 35M Phnom Penh Gold non-smoker BMI24: {pred:.3f}×")


if __name__ == "__main__":
    train()
