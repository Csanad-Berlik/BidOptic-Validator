import sys
import os
import json
import argparse
from datetime import datetime, timezone

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required.  Run: pip install pandas")
    sys.exit(2)

REQUIRED_COLUMNS = [
    "timestamp",
    "user_id",
    "publisher_id",
    "ad_size",
    "bid_price",
    "clearing_price",
    "is_won",
    "is_clicked",
    "is_converted",
    "conversion_timestamp",
    "conversion_value",
]

CRITICAL_COLS     = ["bid_price", "publisher_id", "timestamp", "is_won"]
NULL_HARD_LIMIT   = 0.05

NON_CRITICAL_COLS = ["clearing_price", "is_clicked", "conversion_value", "ad_size"]
NULL_WARN_LIMIT   = 0.20

MIN_ROWS        = 100_000
WARN_ROWS       = 500_000
MIN_CONVERSIONS = 50

def validate(df: pd.DataFrame, actual_total_rows: int = None) -> dict:
    errors   = []
    warnings = []
    stats    = {}

    total = actual_total_rows if actual_total_rows is not None else len(df)
    stats["row_count"] = total

    if total == 0:
        errors.append("Dataset is empty.")
        return {"errors": errors, "warnings": warnings, "stats": stats}

    if total < MIN_ROWS:
        errors.append(
            f"Dataset has {total:,} rows. Minimum required is {MIN_ROWS:,}. "
            "Too few rows produce unreliable CTR/CVR/LTV model fits."
        )
    elif total < WARN_ROWS:
        warnings.append(
            f"Dataset has {total:,} rows. Calibration will run, but model accuracy "
            f"improves significantly above {WARN_ROWS:,} rows. "
            "Consider providing a larger historical export if available."
        )

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    for col in missing_cols:
        errors.append(f"Missing required column: '{col}'.")

    present_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]

    for col in CRITICAL_COLS:
        if col not in present_cols:
            continue
        null_rate = df[col].isnull().mean()
        stats[f"{col}_null_rate"] = round(null_rate, 4)
        if null_rate > NULL_HARD_LIMIT:
            errors.append(
                f"Column '{col}' has {null_rate:.1%} null values "
                f"(hard limit: {NULL_HARD_LIMIT:.0%}). "
                "Clean or impute before resubmitting."
            )

    for col in NON_CRITICAL_COLS:
        if col not in present_cols:
            continue
        null_rate = df[col].isnull().mean()
        stats[f"{col}_null_rate"] = round(null_rate, 4)

        if col == "ad_size":
            if null_rate > 0:
                warnings.append(
                    f"'ad_size' has {null_rate:.1%} null values. "
                    "These will be labeled 'unknown' — common for native and video inventory. "
                    "Floor, CTR, and latency models will learn population-average behavior "
                    "for uncategorized placements. No action needed if this reflects your inventory mix."
                )
            continue

        if null_rate > NULL_WARN_LIMIT:
            warnings.append(
                f"Column '{col}' has {null_rate:.1%} null values. "
                "Affected ML models will train on reduced signal."
            )

    if "is_won" in present_cols:
        try:
            win_rate = pd.to_numeric(df["is_won"], errors="coerce").mean()
            stats["win_rate"] = round(float(win_rate), 4)
            if win_rate < 0.001:
                errors.append(
                    f"Win rate is {win_rate:.3%} — suspiciously low. "
                    "Confirm 'is_won' is encoded as 1/0, not True/False strings."
                )
            elif win_rate > 0.60:
                warnings.append(
                    f"Win rate is {win_rate:.1%} — unusually high for RTB. "
                    "Confirm the log covers all bid requests, not only wins."
                )
        except Exception:
            errors.append("Column 'is_won' could not be parsed as numeric (expected 0/1).")

    if "is_converted" in present_cols:
        try:
            conv_series             = pd.to_numeric(df["is_converted"], errors="coerce")
            conv_rate               = conv_series.mean()
            chunk_conv_count        = int(conv_series.sum())
            extrapolated_conv_count = int(chunk_conv_count * (total / len(df)))

            stats["conversion_rate"]  = round(float(conv_rate), 6)
            stats["conversion_count"] = extrapolated_conv_count

            if conv_rate == 0.0:
                errors.append(
                    "Zero conversions found in the analyzed chunk. "
                    "CVR, LTV, and delay models require at least some positive conversion labels."
                )
            elif extrapolated_conv_count < MIN_CONVERSIONS:
                errors.append(
                    f"Est. {extrapolated_conv_count} conversions found. Minimum required is {MIN_CONVERSIONS}. "
                    "The CVR and conversion delay models will catastrophically overfit below this threshold. "
                    "Provide a longer date range or higher-volume traffic slice."
                )
            elif extrapolated_conv_count < 100:
                warnings.append(
                    f"Est. {extrapolated_conv_count} conversions found. "
                    "With 50-99 conversions, ML CVR/LTV models are disabled and simulation falls back to tabular segment-mean estimates. "
                    "100+ conversions are required for full ML calibration."
                )
        except Exception:
            errors.append("Column 'is_converted' could not be parsed as numeric (expected 0/1).")

    if "is_clicked" in present_cols and "is_won" in present_cols:
        try:
            won_mask = pd.to_numeric(df["is_won"], errors="coerce").fillna(0)
            click_mask = pd.to_numeric(df["is_clicked"], errors="coerce").fillna(0)
            invalid_clicks = ((won_mask == 0) & (click_mask == 1)).sum()
            if invalid_clicks > 0:
                errors.append(
                    f"Found {invalid_clicks:,} rows where is_clicked=1 but is_won=0. "
                    "You cannot register a click on an auction you did not win. Check your data joins."
                )
        except Exception:
            pass

    if "is_converted" in present_cols and "is_clicked" in present_cols:
        try:
            click_mask = pd.to_numeric(df["is_clicked"], errors="coerce").fillna(0)
            conv_mask  = pd.to_numeric(df["is_converted"], errors="coerce").fillna(0)
            invalid_convs = ((click_mask == 0) & (conv_mask == 1)).sum()
            if invalid_convs > 0:
                errors.append(
                    f"Found {invalid_convs:,} rows where is_converted=1 but is_clicked=0. "
                    "A conversion cannot occur without a preceding click. Check your data joins."
                )
        except Exception:
            pass

    if "clearing_price" in present_cols and "bid_price" in present_cols and "is_won" in present_cols:
        try:
            won_rows = df[pd.to_numeric(df["is_won"], errors="coerce") == 1]
            if len(won_rows) > 0:
                cp = pd.to_numeric(won_rows["clearing_price"], errors="coerce").fillna(0.0)
                bp = pd.to_numeric(won_rows["bid_price"], errors="coerce").fillna(0.0)
                invalid_clearing = (cp > bp).sum()
                if invalid_clearing > 0:
                    errors.append(
                        f"Found {invalid_clearing:,} won auctions where clearing_price exceeds bid_price. "
                        "This violates standard RTB auction mechanics."
                    )
        except Exception:
            pass

    if "conversion_value" in present_cols and "is_converted" in present_cols:
        try:
            converted_rows = df[pd.to_numeric(df["is_converted"], errors="coerce") == 1]
            if len(converted_rows) > 0:
                vals = pd.to_numeric(converted_rows["conversion_value"], errors="coerce").fillna(0.0)
                if vals.nunique() <= 1:
                    warnings.append(
                        "All conversions have a uniform or zero 'conversion_value'. "
                        "Simulation will run in Binary Conversion Mode (LTV disabled, each conversion assigned a fixed value of 1.00 in your campaign currency)."
                    )
        except Exception:
            pass

    if "conversion_timestamp" in present_cols and "is_converted" in present_cols:
        try:
            converted_mask = pd.to_numeric(df["is_converted"], errors="coerce") == 1
            converted_rows = df[converted_mask]
            if len(converted_rows) > 0:
                ts_null_rate = converted_rows["conversion_timestamp"].isnull().mean()
                stats["conv_ts_null_rate_on_converted"] = round(float(ts_null_rate), 4)
                if ts_null_rate > 0.20:
                    warnings.append(
                        f"{ts_null_rate:.1%} of converted rows are missing 'conversion_timestamp'. "
                        "The post-click delay model accuracy will be reduced. "
                        "Provide timestamps where available."
                    )
        except Exception:
            pass

    if "timestamp" in present_cols:
        try:
            parsed = pd.to_datetime(df["timestamp"], errors="coerce")
            ts_null_rate = parsed.isnull().mean()
            if ts_null_rate > NULL_HARD_LIMIT:
                errors.append(
                    f"{ts_null_rate:.1%} of 'timestamp' values could not be parsed as datetime. "
                    "Accepted formats: ISO 8601 (2024-03-15T14:32:00Z) or Unix timestamp (seconds)."
                )
            else:
                date_min = parsed.min()
                date_max = parsed.max()
                if pd.notna(date_min) and pd.notna(date_max):
                    stats["date_range_start"] = str(date_min.date())
                    stats["date_range_end"]   = str(date_max.date())
                    stats["days_of_data"]     = max((date_max - date_min).days, 1)

                    if stats["days_of_data"] < 7:
                        errors.append(
                            f"Dataset spans only {stats['days_of_data']} day(s). "
                            "A minimum of 7 days is required. "
                            "Too little data produces unreliable temporal and pacing patterns."
                        )
                    elif stats["days_of_data"] < 14:
                        warnings.append(
                            f"Dataset spans {stats['days_of_data']} day(s). "
                            "14 days or more is recommended for reliable temporal patterns."
                        )
                    elif stats["days_of_data"] < 63:
                        warnings.append(
                            f"Dataset spans {stats['days_of_data']} day(s). "
                            "Note: the BidOptic evaluation accuracy protocol calibrates on 8 weeks "
                            "of data and validates against a 9th week of live logs (~63 days total). "
                            "Your current export may be too short to run the full evaluation trigger — "
                            "consider pulling a longer historical window before beginning the evaluation."
                        )
                    if stats["days_of_data"] > 180:
                        warnings.append(
                            f"Dataset spans {stats['days_of_data']} days — well beyond the recommended "
                            "90-day window. Consider trimming to the most recent 60–90 days before "
                            "calibrating; older data may reflect floor prices, competitive dynamics, "
                            "and market conditions that no longer apply."
                        )
                    elif stats["days_of_data"] > 90:
                        warnings.append(
                            f"Dataset spans {stats['days_of_data']} days. "
                            "Data older than 90 days may reflect market conditions that no longer apply. "
                            "Calibration will run, but recent trends may be diluted."
                        )
                    try:
                        date_max_aware = (
                            date_max.tz_localize("UTC") if date_max.tzinfo is None else date_max
                        )
                        days_stale = (datetime.now(timezone.utc) - date_max_aware).days
                        if days_stale > 90:
                            warnings.append(
                                f"Your dataset ends {days_stale} days ago. "
                                "Data this stale may produce unreliable calibrations — market conditions, "
                                "floor prices, and competitive dynamics shift significantly over this window. "
                                "Recalibrating with more recent logs is strongly recommended."
                            )
                        elif days_stale > 30:
                            warnings.append(
                                f"Your dataset ends {days_stale} days ago. "
                                "For best accuracy, calibrate against data ending within the last 30 days."
                            )
                    except Exception:
                        pass

        except Exception:
            warnings.append("Could not parse 'timestamp' column — check format.")

    if "bid_latency_ms" not in df.columns:
        warnings.append(
            "'bid_latency_ms' column not found. "
            "The latency model will fall back to market-average priors. "
            "Include round-trip latency in milliseconds for higher accuracy."
        )
    elif "bid_latency_ms" in present_cols:
        try:
            lat = pd.to_numeric(df["bid_latency_ms"], errors="coerce").dropna()
            if len(lat) > 0:
                stats["latency_p50_ms"] = round(float(lat.quantile(0.50)), 1)
                stats["latency_p95_ms"] = round(float(lat.quantile(0.95)), 1)
                stats["latency_p99_ms"] = round(float(lat.quantile(0.99)), 1)
        except Exception:
            pass

    if "bid_latency_ms" in present_cols:
        if (pd.to_numeric(df["bid_latency_ms"], errors="coerce") < 0).any():
            errors.append("'bid_latency_ms' contains negative values. Network transit time must be >= 0.")

    if "conversion_value" in present_cols:
        if (pd.to_numeric(df["conversion_value"], errors="coerce") < 0).any():
            errors.append("'conversion_value' contains negative values. Revenue cannot be negative.")

    if "bid_price" in present_cols:
        if (pd.to_numeric(df["bid_price"], errors="coerce") < 0).any():
            errors.append("'bid_price' contains negative values. Bids must be >= 0.")
        if "is_won" in present_cols:
            try:
                won_mask      = pd.to_numeric(df["is_won"], errors="coerce") == 1
                zero_bid_wins = (pd.to_numeric(df.loc[won_mask, "bid_price"], errors="coerce") == 0).sum()
                if zero_bid_wins > 0:
                    errors.append(
                        f"Found {zero_bid_wins:,} won auctions where bid_price is 0.0. "
                        "A winning bid must be greater than zero. Check your data export."
                    )
            except Exception:
                pass

    return {"errors": errors, "warnings": warnings, "stats": stats}


def write_receipt(result: dict, file_path: str) -> str:
    import hashlib

    ts      = datetime.now(timezone.utc)
    ts_str  = ts.strftime("%Y%m%dT%H%M%SZ")

    safe_stats = {
        "row_count":        result["stats"].get("row_count"),
        "days_of_data":     result["stats"].get("days_of_data"),
        "date_range_start": result["stats"].get("date_range_start"),
        "date_range_end":   result["stats"].get("date_range_end"),
    }

    payload = {
        "bidoptic_schema_version": "1.0",
        "validated_at":            ts.isoformat(),
        "file":                    file_path,
        "passed":                  len(result["errors"]) == 0,
        "blocker_count":           len(result["errors"]),
        "warning_count":           len(result["warnings"]),
        "safe_volume_stats":       safe_stats,
    }

    fingerprint_src = json.dumps({"stats": safe_stats, "time": ts.isoformat()}, sort_keys=True).encode()
    payload["receipt_id"] = hashlib.sha256(fingerprint_src).hexdigest()[:16].upper()

    filename = f"bidoptic_receipt_{ts_str}.json"
    try:
        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)
    except PermissionError:
        print(f"Could not write receipt to '{filename}' — directory is read-only.")
        return None
    return filename

if os.name == 'nt':
    os.system("")

USE_COLOR = sys.stdout.isatty()
BOLD  = "\033[1m"  if USE_COLOR else ""
GREEN = "\033[92m" if USE_COLOR else ""
AMBER = "\033[93m" if USE_COLOR else ""
RED   = "\033[91m" if USE_COLOR else ""
CYAN  = "\033[96m" if USE_COLOR else ""
RESET = "\033[0m"  if USE_COLOR else ""


def print_report(result: dict, csv_path: str) -> None:
    errors   = result["errors"]
    warnings = result["warnings"]
    stats    = result["stats"]

    print()
    print(f"{BOLD}{'─' * 62}{RESET}")
    print(f"{BOLD}  BidOptic Dataset Validator{RESET}")
    print(f"  File : {csv_path}")
    print(f"  Time : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'─' * 62}")

    row_count = stats.get("row_count", 0)
    print(f"\n  {'Rows':<28} {row_count:,}")
    if "win_rate" in stats:
        print(f"  {'Win rate':<28} {stats['win_rate']:.2%}")
    if "conversion_rate" in stats:
        conv_count = stats.get("conversion_count", "n/a")
        print(f"  {'Conversion rate':<28} {stats['conversion_rate']:.4%}  (Est. {conv_count} total conversions)")
    if "date_range_start" in stats:
        print(
            f"  {'Date range':<28} "
            f"{stats['date_range_start']} → {stats['date_range_end']} "
            f"({stats['days_of_data']} days)"
        )
    if "latency_p50_ms" in stats:
        print(
            f"  {'Latency p50/p95/p99':<28} "
            f"{stats['latency_p50_ms']} / {stats['latency_p95_ms']} / {stats['latency_p99_ms']} ms"
        )

    print()
    if warnings:
        for w in warnings:
            print(f"  {AMBER}⚠  {w}{RESET}")
    else:
        print(f"  {GREEN}✓  No warnings.{RESET}")

    if errors:
        print()
        for e in errors:
            print(f"  {RED}✗  BLOCKER: {e}{RESET}")
        print()
        print(f"{'─' * 62}")
        print(
            f"  {BOLD}{RED}RESULT: {len(errors)} blocker(s) found. "
            f"Fix the issues above before submitting.{RESET}"
        )
    else:
        print()
        print(f"{'─' * 62}")
        print(f"  {BOLD}{GREEN}RESULT: Dataset passed. Ready for BidOptic calibration.{RESET}")

    print(f"{'─' * 62}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Validate a DSP auction log CSV or Parquet file against the BidOptic schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        metavar="PATH",
        help="Path to the CSV or Parquet file to validate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output a machine-readable JSON report instead of the formatted display.",
    )
    args = parser.parse_args()
    file_path = args.data

    print(f"Analyzing schema for: {file_path}", file=sys.stderr)

    try:
        if file_path.lower().endswith(".parquet"):
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(file_path)

            actual_total_rows = parquet_file.metadata.num_rows
            num_row_groups    = parquet_file.metadata.num_row_groups

            print(f"Parquet detected. Total rows: {actual_total_rows:,} across {num_row_groups} row group(s). Reading sample for validation...", file=sys.stderr)

            sample_indices = sorted({
                0,
                num_row_groups // 2,
                num_row_groups - 1,
            })

            chunks = [parquet_file.read_row_group(i).to_pandas() for i in sample_indices]
            df = pd.concat(chunks, ignore_index=True)

            if len(df) > 50_000:
                df = df.sample(n=50_000, random_state=42).reset_index(drop=True)

        elif file_path.lower().endswith(".csv"):
            print("CSV detected. Counting rows (this might take a moment)...", file=sys.stderr)

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                actual_total_rows = sum(1 for _ in f) - 1

            print(f"Total rows: {actual_total_rows:,}. Reading uniform sample across full time range...", file=sys.stderr)
            
            if actual_total_rows > 50_000:
                import random
                keep_prob = 50_000 / actual_total_rows

                skip_func = lambda i: i > 0 and random.random() > keep_prob
                df = pd.read_csv(file_path, skiprows=skip_func)

                if len(df) > 50_000:
                    df = df.sample(n=50_000, random_state=42).reset_index(drop=True)
            else:
                df = pd.read_csv(file_path)

        else:
            print("Error: File must be .parquet or .csv", file=sys.stderr)
            sys.exit(2)

    except ImportError:
        print("Error: pyarrow is required to read Parquet files. Run: pip install pyarrow", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Failed to read file: {e}", file=sys.stderr)
        sys.exit(2)

    result = validate(df, actual_total_rows=actual_total_rows)

    if args.json:
        import hashlib
        ts = datetime.now(timezone.utc)

        safe_stats = {
            "row_count":        result["stats"].get("row_count"),
            "days_of_data":     result["stats"].get("days_of_data"),
            "date_range_start": result["stats"].get("date_range_start"),
            "date_range_end":   result["stats"].get("date_range_end"),
        }
        result["stats"] = safe_stats

        fingerprint_src = json.dumps({"stats": safe_stats, "time": ts.isoformat()}, sort_keys=True).encode()

        result["file"]       = file_path
        result["validated"]  = ts.isoformat()
        result["passed"]     = len(result["errors"]) == 0
        result["receipt_id"] = hashlib.sha256(fingerprint_src).hexdigest()[:16].upper()

        if result["passed"]:
            write_receipt(result, file_path)

        print(json.dumps(result, indent=2))
    else:
        print_report(result, file_path)
        if len(result["errors"]) == 0:
            receipt_path = write_receipt(result, file_path)

            print(f"  {BOLD}{CYAN}NEXT STEPS:{RESET}")
            if receipt_path:
                print(f"  1. Receipt written → {receipt_path}")
                print(f"     Send this JSON file to your BidOptic contact to proceed.")
            else:
                print(f"  1. Send your validation receipt to your BidOptic contact to proceed.")
                
            print(f"  2. On the scoping call, the evaluation accuracy threshold and licence terms")
            print(f"     are agreed in writing before calibration begins.")
            print(f"  3. To prepare your licensed container, provide your target Linux host specs:")
            print(f"       - Machine ID  (Run: {BOLD}cat /etc/machine-id{RESET} on the Linux host that will run Docker)")
            print(f"       - CPU Cores   (Run: {BOLD}nproc{RESET} on the same host)\n")
            print(f"{'─' * 62}\n")

    sys.exit(0 if len(result["errors"]) == 0 else 1)


if __name__ == "__main__":
    main()