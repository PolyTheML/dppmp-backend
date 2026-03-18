-- ─── DPPMP Database Setup (Supabase SQL Editor) ────────────────────────────
-- Run this once in: Supabase Dashboard → SQL Editor → New Query → Run
-- ─────────────────────────────────────────────────────────────────────────────

-- 1. Rate Cards
CREATE TABLE IF NOT EXISTS rate_cards (
    id             SERIAL PRIMARY KEY,
    plan_code      VARCHAR(20) NOT NULL,
    age_band_low   INT NOT NULL,
    age_band_high  INT NOT NULL,
    base_rate      DECIMAL(10,2) NOT NULL,
    effective_date DATE NOT NULL DEFAULT CURRENT_DATE,
    expiry_date    DATE
);

CREATE INDEX IF NOT EXISTS idx_rate_lookup
    ON rate_cards (plan_code, age_band_low, effective_date);

-- 2. Quote Log
CREATE TABLE IF NOT EXISTS quote_log (
    id              SERIAL PRIMARY KEY,
    quote_ref       VARCHAR(30) UNIQUE NOT NULL,
    created_at      TIMESTAMP DEFAULT NOW(),
    input_json      JSONB NOT NULL,
    base_rate       DECIMAL(10,2) NOT NULL,
    risk_multiplier DECIMAL(5,3) NOT NULL,
    annual_premium  DECIMAL(10,2) NOT NULL,
    monthly_premium DECIMAL(10,2) NOT NULL,
    model_version   VARCHAR(30) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_quote_created ON quote_log (created_at DESC);

-- ─── Seed Rate Card Data ────────────────────────────────────────────────────
INSERT INTO rate_cards (plan_code, age_band_low, age_band_high, base_rate) VALUES
    -- Basic Plan
    ('Basic', 18, 24, 120.00),
    ('Basic', 25, 29, 135.00),
    ('Basic', 30, 34, 155.00),
    ('Basic', 35, 39, 180.00),
    ('Basic', 40, 44, 220.00),
    ('Basic', 45, 49, 275.00),
    ('Basic', 50, 54, 340.00),
    ('Basic', 55, 59, 420.00),
    ('Basic', 60, 64, 530.00),
    ('Basic', 65, 80, 680.00),
    -- Gold Plan
    ('Gold', 18, 24, 210.00),
    ('Gold', 25, 29, 240.00),
    ('Gold', 30, 34, 275.00),
    ('Gold', 35, 39, 320.00),
    ('Gold', 40, 44, 385.00),
    ('Gold', 45, 49, 470.00),
    ('Gold', 50, 54, 580.00),
    ('Gold', 55, 59, 710.00),
    ('Gold', 60, 64, 880.00),
    ('Gold', 65, 80, 1100.00),
    -- Platinum Plan
    ('Platinum', 18, 24, 350.00),
    ('Platinum', 25, 29, 395.00),
    ('Platinum', 30, 34, 450.00),
    ('Platinum', 35, 39, 520.00),
    ('Platinum', 40, 44, 625.00),
    ('Platinum', 45, 49, 760.00),
    ('Platinum', 50, 54, 940.00),
    ('Platinum', 55, 59, 1150.00),
    ('Platinum', 60, 64, 1420.00),
    ('Platinum', 65, 80, 1780.00)
ON CONFLICT DO NOTHING;

-- ─── Verify ─────────────────────────────────────────────────────────────────
SELECT plan_code, COUNT(*) as bands, MIN(base_rate) as min_rate, MAX(base_rate) as max_rate
FROM rate_cards
GROUP BY plan_code
ORDER BY min_rate;
