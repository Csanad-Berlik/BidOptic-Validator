# BidOptic Dataset Validator

Validate your DSP auction log against the BidOptic schema before submitting for simulation. All checks run locally — no data leaves your infrastructure.

---

## Why this exists

BidOptic calibrates a digital twin of your RTB market from your historical auction logs. Before that process can run, your dataset needs to meet a set of statistical and structural requirements — sufficient rows for model training, required columns, healthy win and conversion rates, and clean timestamps.

This tool tells you exactly what is missing and what needs fixing before you share anything with us.

---

## Quickstart

Clone the repository and install the requirements:
```bash
git clone https://github.com/Csanad-Berlik/BidOptic-Validator.git
cd BidOptic-Validator
pip install pandas pyarrow
```

### 1. Run the test with our mock data:

See what a passing dataset looks like instantly:

```bash
python bidoptic_validator.py --data sample_data/sample_success.parquet
```

### 2. Run it against your own data:

Once you export your own 11-column CSV or Parquet file, validate it locally:

```bash
python bidoptic_validator.py --data /path/to/your/actual_logs.parquet
```
> [!NOTE]
> CSV files are also accepted, e.g., python bidoptic_validator.py --data your_logs.csv

If your dataset passes, a machine-readable receipt is written to the current directory. Send that file to your BidOptic contact to proceed — no raw data required.

---

## Required schema

Your file must contain these 11 columns. A 12th column improves latency model accuracy but is optional.

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | UTC datetime of the bid request. ISO 8601 or Unix seconds. |
| `user_id` | string | Anonymised user identifier. Any format accepted — cookie, GAID, hashed email, session ID. |
| `publisher_id` | string | Publisher or supply source identifier. |
| `ad_size` | string | Creative dimensions (`300x250`, `728x90`, etc.), `native`, `video_pre`, or `unknown`. Nulls are accepted and will be treated as `unknown`. |
| `bid_price` | float | Your submitted bid price. If your DSP logs in CPM, divide by 1,000 before export. |
| `clearing_price` | float | Price paid on wins. `0.0` on losses. |
| `is_won` | int | `1` if the auction was won, `0` otherwise. |
| `is_clicked` | int | `1` if the served impression was clicked. Requires a win. |
| `is_converted` | int | `1` if a conversion occurred downstream of a click. |
| `conversion_timestamp` | datetime | UTC datetime of conversion. Null if no conversion. |
| `conversion_value` | float | Revenue value of the conversion in your campaign currency. `0.0` if none. **Note:** If your data contains no revenue variance (e.g., all 0s or 1s), the system automatically enters **Binary Conversion Mode**. The LTV model is disabled, and every conversion is assigned a fixed value of 1.00 |
| `bid_latency_ms` | float | *(Optional)* Round-trip bid latency in milliseconds. Improves latency model accuracy. |

### Notes on `user_id`

BidOptic treats `user_id` as an opaque token and normalises all formats internally. Persistent identifiers (third-party cookies, authenticated IDs) yield richer user-level signal. Non-persistent identifiers (cookieless EU traffic, rotating session IDs) degrade gracefully to session-level signal — publisher and segment models remain accurate.

### Notes on `ad_size`

Native and video inventory commonly has no fixed dimensions. Pass `native`, `video_pre`, or `unknown` rather than leaving the field null. BidOptic's floor, CTR, and latency models learn population-average behaviour for uncategorised placements.

---

## What the validator checks

### Hard blockers — calibration cannot proceed

| Check | Threshold |
|---|---|
| Minimum row count | < 100,000 rows |
| Missing required column | Any of the 11 above |
| Critical column null rate | > 5% nulls in `bid_price`, `publisher_id`, `timestamp`, or `is_won` |
| Win rate too low | < 0.1% — usually a column encoding issue |
| Zero conversions | CVR, LTV, and delay models cannot train without positive labels |
| Insufficient conversions | < 50 total conversions — CVR and delay models will overfit |
| Timestamp unparseable | > 5% of values cannot be parsed |
| Date range too short | < 7 days — insufficient for temporal and pacing patterns |
| Negative values | `bid_latency_ms`, `conversion_value`, or `bid_price` contains negative values |
| Logical Impossibilities | Rows where `is_clicked=1` but `is_won=0`; rows where `is_converted=1` but `is_clicked=0`; or `clearing_price` > `bid_price` |

### Warnings — calibration runs but accuracy is reduced

| Check | Threshold |
|---|---|
| Row count below recommended | < 500,000 rows |
| ML CVR/LTV Threshold | 50–99 conversions — Simulation falls back to tabular segment-mean estimates. 100+ required for ML. |
| Non-critical column null rate | > 20% nulls in `clearing_price`, `is_clicked`, or `conversion_value` |
| `ad_size` nulls | Any null triggers an informational warning. `ad_size` is not subject to the 20% threshold — nulls are expected for native and video inventory and require no action if they reflect your inventory mix. |
| Win rate unusually high | > 60% — confirm the log covers all bid requests, not only wins |
| Missing conversion timestamps | > 20% of converted rows — AFT delay model accuracy reduced |
| Date range below recommended | < 14 days — temporal patterns will be less reliable |
| Date range moderately long | > 90 days — older data may reflect market regimes that no longer apply |
| Date range very long | > 180 days — strong recommendation to trim to the most recent 60–90 days before calibrating |
| Dataset end date stale | End date > 30 days before today — freshness warning; > 90 days — severe staleness warning, recalibration strongly recommended |
| Missing `bid_latency_ms` | Latency model falls back to market-average priors |

---

## Output

### Terminal report

```
──────────────────────────────────────────────────────────────
  BidOptic Dataset Validator
  File : your_logs.parquet
  Time : 2026-03-29 09:14 UTC
──────────────────────────────────────────────────────────────

  Rows                         2,473,810
  Win rate                     6.84%
  Conversion rate              0.0380%  (Est. 939 total conversions)
  Date range                   2026-02-01 → 2026-03-28 (55 days)
  Latency p50/p95/p99          44.2 / 71.8 / 118.3 ms

  No warnings.

──────────────────────────────────────────────────────────────
  RESULT: Dataset passed. Ready for BidOptic calibration.
──────────────────────────────────────────────────────────────

  NEXT STEPS:
  1. Receipt written → bidoptic_receipt_20260329T091432Z.json
     Send this JSON file to your BidOptic contact to proceed.
  2. On the scoping call, the evaluation accuracy threshold and licence terms
     are agreed in writing before calibration begins.
  3. To prepare your licensed container, provide your target Linux host specs:
       - Machine ID  (Run: cat /etc/machine-id on the Linux host that will run Docker)
       - CPU Cores   (Run: nproc on the same host)

──────────────────────────────────────────────────────────────
```

### Machine-readable output

Pass `--json` for CI-friendly output:

```bash
python bidoptic_validator.py --data your_logs.parquet --json
```

```json
{
  "errors": [],
  "warnings": [],
  "stats": {
    "row_count": 2473810,
    "days_of_data": 55,
    "date_range_start": "2026-02-01",
    "date_range_end": "2026-03-28"
  },
  "file": "your_logs.parquet",
  "validated": "2026-03-29T09:14:32.000000+00:00",
  "passed": true,
  "receipt_id": "A3F1B2C4D5E6F7A8"
}
```

> [!NOTE]
> Win rate, conversion rate, and latency percentiles are intentionally excluded from the JSON output. They are displayed in the terminal report only and are not written to disk or included in the receipt.

### The receipt
 
When validation passes in terminal mode (or silently when passing `--json`), a receipt JSON is written locally. It contains aggregate volume statistics and a deterministic SHA-256 fingerprint — no row-level data, no win rates, no conversion metrics. This file is what you send to BidOptic to initiate the next step.
 
> [!NOTE]
> The receipt file has **different field names** from the `--json` stdout output. Do not use `--json` stdout as your receipt — the file written to disk is what your BidOptic contact needs.
 
Receipt file schema:

```json
{
  "bidoptic_schema_version": "1.0",
  "validated_at": "2026-03-29T09:14:32.000000+00:00",
  "file": "your_logs.parquet",
  "passed": true,
  "blocker_count": 0,
  "warning_count": 0,
  "safe_volume_stats": {
    "row_count": 2473810,
    "days_of_data": 55,
    "date_range_start": "2026-02-01",
    "date_range_end": "2026-03-28"
  },
  "receipt_id": "A3F1B2C4D5E6F7A8"
}
```

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | All checks passed. Dataset is ready. |
| `1` | One or more hard blockers found. |
| `2` | File not found or unreadable. |

---

## Security

This script does not make network calls. Validation is entirely local. The receipt file contains only row counts, date ranges, and a SHA-256 fingerprint of those statistics. No row-level data, win rates, or conversion metrics are included or transmitted at any point.

BidOptic is delivered as an air-gapped SDK in an encrypted Docker image that runs inside your own infrastructure. Your data does not leave your environment at any stage of the calibration or simulation process.

---

## Minimum data requirements summary

- **100,000 rows** minimum (500,000+ recommended for best model accuracy)
- **7 days** of historical data minimum (14+ recommended)
- **At least 50 conversions** across the dataset
- All 11 required columns present
- `bid_price`, `publisher_id`, `timestamp`, `is_won` must have < 5% nulls

---

## Questions

> [!TIP]
> If your data has structural issues that are outside your control — no conversion signal in a specific log slice, non-standard identifier formats, missing latency logging — reach out before assuming your dataset won't work. Many common issues are handleable at the calibration layer.