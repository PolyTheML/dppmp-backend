"""
DPPMP Backend — FastAPI
Single-service backend: rating engine + embedded ML model + API
Deploy to: Render / Railway
"""

import os
import time
import logging
import hashlib
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import asyncpg

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dppmp")

# ─── Config ──────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")
MODEL_PATH = os.getenv("MODEL_PATH", "models/risk_model.pkl")
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173"
).split(",")

# ─── Global state ────────────────────────────────────────────────────────────
db_pool: Optional[asyncpg.Pool] = None
model = None
model_version = "v1.0.0"


# ─── Startup / Shutdown ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + connect DB on startup, clean up on shutdown."""
    global db_pool, model, model_version

    # 1) Load ML model
    try:
        model = joblib.load(MODEL_PATH)
        model_version = getattr(model, "_dppmp_version", "v1.0.0")
        log.info(f"Model loaded: {MODEL_PATH} ({model_version})")
    except FileNotFoundError:
        log.warning(f"Model file not found at {MODEL_PATH} — using fallback rule-based model")
        model = None

    # 2) Connect to database
    if DATABASE_URL:
        try:
            db_pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=2,
                max_size=10,
                command_timeout=10,
            )
            log.info("Database connected")
        except Exception as e:
            log.warning(f"Database connection failed: {e} — using fallback rates")
            db_pool = None
    else:
        log.warning("DATABASE_URL not set — using fallback rates")

    yield  # ← app runs here

    # Cleanup
    if db_pool:
        await db_pool.close()
        log.info("Database pool closed")


app = FastAPI(
    title="DPPMP API",
    description="Dynamic Personal Pricing Model Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Schemas ──────────────────────────────────────────────
class PremiumRequest(BaseModel):
    age: int = Field(..., ge=18, le=80, description="Applicant age (18–80)")
    gender: str = Field(..., description="Male / Female / Other")
    region: str = Field(..., description="Rating region")
    plan: str = Field(..., description="Basic / Gold / Platinum")
    smoking: bool = Field(False, description="Smoker?")
    bmi: float = Field(24.0, ge=12, le=60, description="Body mass index")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        v = v.strip().title()
        if v not in ("Male", "Female", "Other"):
            raise ValueError("Gender must be Male, Female, or Other")
        return v

    @field_validator("plan")
    @classmethod
    def validate_plan(cls, v):
        v = v.strip().title()
        if v not in ("Basic", "Gold", "Platinum"):
            raise ValueError("Plan must be Basic, Gold, or Platinum")
        return v

    @field_validator("region")
    @classmethod
    def validate_region(cls, v):
        valid = {
            "Phnom Penh", "Siem Reap", "Battambang",
            "Sihanoukville", "Kampong Cham", "Rural Areas",
        }
        v = v.strip().title()
        if v not in valid:
            raise ValueError(f"Region must be one of: {', '.join(sorted(valid))}")
        return v


class PremiumResponse(BaseModel):
    quote_id: str
    base_rate: float
    risk_multiplier: float
    annual_premium: float
    monthly_premium: float
    model_version: str
    breakdown: dict
    calculated_at: str


# ─── Fallback Data (when DB is unavailable) ─────────────────────────────────
FALLBACK_BASE_RATES = {
    "Basic":    {18: 120, 25: 135, 30: 155, 35: 180, 40: 220, 45: 275, 50: 340, 55: 420, 60: 530, 65: 680},
    "Gold":     {18: 210, 25: 240, 30: 275, 35: 320, 40: 385, 45: 470, 50: 580, 55: 710, 60: 880, 65: 1100},
    "Platinum": {18: 350, 25: 395, 30: 450, 35: 520, 40: 625, 45: 760, 50: 940, 55: 1150, 60: 1420, 65: 1780},
}

REGION_FACTORS = {
    "Phnom Penh": 1.15, "Siem Reap": 1.05, "Battambang": 0.95,
    "Sihanoukville": 1.08, "Kampong Cham": 0.90, "Rural Areas": 0.85,
}

GENDER_FACTORS = {"Male": 1.02, "Female": 0.98, "Other": 1.00}
SMOKING_FACTORS = {True: 1.35, False: 1.00}


def interpolate_base_rate(plan: str, age: int) -> float:
    """Linear interpolation between age bands in the rate table."""
    rates = FALLBACK_BASE_RATES[plan]
    ages = sorted(rates.keys())
    if age <= ages[0]:
        return rates[ages[0]]
    if age >= ages[-1]:
        return rates[ages[-1]]
    for i in range(len(ages) - 1):
        if ages[i] <= age <= ages[i + 1]:
            lo, hi = ages[i], ages[i + 1]
            ratio = (age - lo) / (hi - lo)
            return round(rates[lo] + ratio * (rates[hi] - rates[lo]), 2)
    return rates[ages[-1]]


def bmi_factor(bmi: float) -> float:
    if bmi < 18.5:
        return 1.08
    if bmi <= 24.9:
        return 1.00
    if bmi <= 29.9:
        return 1.12
    return 1.25


# ─── Rating Engine ───────────────────────────────────────────────────────────
async def get_base_rate(plan: str, age: int) -> float:
    """Try database first, fall back to in-memory table."""
    if db_pool:
        try:
            row = await db_pool.fetchrow(
                """
                SELECT base_rate FROM rate_cards
                WHERE plan_code = $1
                  AND age_band_low <= $2
                  AND age_band_high >= $2
                  AND (expiry_date IS NULL OR expiry_date > CURRENT_DATE)
                ORDER BY effective_date DESC
                LIMIT 1
                """,
                plan,
                age,
            )
            if row:
                return float(row["base_rate"])
        except Exception as e:
            log.warning(f"DB lookup failed, using fallback: {e}")

    return interpolate_base_rate(plan, age)


def predict_risk_multiplier(age: int, gender: str, region: str, plan: str, smoking: bool, bmi_val: float) -> float:
    """
    Use the trained ML model if available, otherwise fall back to
    a deterministic factor-based calculation.
    """
    if model is not None:
        try:
            # Encode features for the sklearn model
            gender_enc = {"Male": 0, "Female": 1, "Other": 2}.get(gender, 2)
            region_enc = {
                "Phnom Penh": 0, "Siem Reap": 1, "Battambang": 2,
                "Sihanoukville": 3, "Kampong Cham": 4, "Rural Areas": 5,
            }.get(region, 0)
            plan_enc = {"Basic": 0, "Gold": 1, "Platinum": 2}.get(plan, 0)

            features = np.array([[age, gender_enc, region_enc, plan_enc, int(smoking), bmi_val]])
            prediction = model.predict(features)[0]
            return round(float(np.clip(prediction, 0.70, 1.80)), 3)
        except Exception as e:
            log.warning(f"Model prediction failed, using fallback: {e}")

    # Fallback: deterministic factors
    age_factor = 1.0 + max(0, (age - 30)) * 0.004
    multiplier = (
        age_factor
        * GENDER_FACTORS.get(gender, 1.0)
        * REGION_FACTORS.get(region, 1.0)
        * SMOKING_FACTORS.get(smoking, 1.0)
        * bmi_factor(bmi_val)
    )
    return round(multiplier, 3)


async def log_quote(quote_id: str, req: PremiumRequest, base: float, mult: float, annual: float, monthly: float):
    """Log quote to database (best-effort, non-blocking)."""
    if not db_pool:
        return
    try:
        await db_pool.execute(
            """
            INSERT INTO quote_log (quote_ref, input_json, base_rate, risk_multiplier,
                                   annual_premium, monthly_premium, model_version)
            VALUES ($1, $2::jsonb, $3, $4, $5, $6, $7)
            """,
            quote_id,
            req.model_dump_json(),
            base,
            mult,
            annual,
            monthly,
            model_version,
        )
    except Exception as e:
        log.warning(f"Failed to log quote: {e}")


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version,
        "database_connected": db_pool is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict-premium", response_model=PremiumResponse)
async def predict_premium(req: PremiumRequest, request: Request):
    """
    Main pricing endpoint.
    1. Fetch base rate from DB (or fallback table)
    2. Run ML model (or fallback factors) for risk multiplier
    3. Compute final premium
    4. Log the quote
    5. Return full breakdown
    """
    start = time.monotonic()

    # 1. Base rate
    base_rate = await get_base_rate(req.plan, req.age)

    # 2. Risk multiplier
    risk_mult = predict_risk_multiplier(
        req.age, req.gender, req.region, req.plan, req.smoking, req.bmi
    )

    # 3. Final premium
    annual = round(base_rate * risk_mult, 2)
    monthly = round(annual / 12, 2)

    # 4. Build breakdown
    breakdown = {
        "age_factor": round(1.0 + max(0, (req.age - 30)) * 0.004, 3),
        "gender_factor": GENDER_FACTORS.get(req.gender, 1.0),
        "region_factor": REGION_FACTORS.get(req.region, 1.0),
        "smoking_factor": SMOKING_FACTORS.get(req.smoking, 1.0),
        "bmi_factor": bmi_factor(req.bmi),
    }

    # 5. Quote ID
    ts = datetime.now(timezone.utc)
    raw = f"{req.age}{req.gender}{req.region}{req.plan}{ts.isoformat()}"
    quote_id = f"Q-{ts.strftime('%Y%m%d')}-{hashlib.sha256(raw.encode()).hexdigest()[:8].upper()}"

    # 6. Log (non-blocking best effort)
    await log_quote(quote_id, req, base_rate, risk_mult, annual, monthly)

    elapsed = round((time.monotonic() - start) * 1000, 1)
    log.info(f"Quote {quote_id} | {req.plan} age={req.age} | ${annual}/yr | {elapsed}ms")

    return PremiumResponse(
        quote_id=quote_id,
        base_rate=base_rate,
        risk_multiplier=risk_mult,
        annual_premium=annual,
        monthly_premium=monthly,
        model_version=model_version,
        breakdown=breakdown,
        calculated_at=ts.isoformat(),
    )


@app.get("/rate-card")
async def get_rate_card():
    """Return the full rate card for frontend display."""
    return {
        "plans": ["Basic", "Gold", "Platinum"],
        "regions": REGION_FACTORS,
        "gender_factors": GENDER_FACTORS,
        "smoking_factors": {"Yes": 1.35, "No": 1.00},
        "base_rates": FALLBACK_BASE_RATES,
    }


@app.get("/model-info")
async def model_info():
    """Return current model metadata."""
    return {
        "version": model_version,
        "type": type(model).__name__ if model else "FallbackRuleBased",
        "loaded": model is not None,
        "features": ["age", "gender", "region", "plan", "smoking", "bmi"],
        "output_range": [0.70, 1.80],
    }


# ─── Global Error Handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    log.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )
