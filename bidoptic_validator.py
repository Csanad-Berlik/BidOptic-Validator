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

def check_user_volume_outliers(df):
    warnings = []
    stats = {}
    if "user_id" not in df.columns or len(df) == 0:
        return warnings, stats

    user_counts = df.groupby("user_id", observed=True).size()
    if len(user_counts) == 0:
        return warnings, stats

    median = user_counts.median()
    p99 = user_counts.quantile(0.99)
    extreme_threshold = max(median * 50, 200)
    extreme_users = user_counts[user_counts > extreme_threshold]

    stats["user_volume_p99"] = float(p99)
    stats["user_volume_median"] = float(median)
    stats["n_extreme_volume_users"] = int(len(extreme_users))
    traffic_share = float(extreme_users.sum() / len(df)) if len(df) > 0 else 0.0
    stats["extreme_volume_user_share_of_traffic"] = traffic_share

    if len(extreme_users) > 0:
        warnings.append(
            f"{len(extreme_users)} user_id(s) each generated more than {extreme_threshold:.0f} "
            f"impressions in the sampled chunk (median user: {median:.0f}), accounting for "
            f"{traffic_share:.1%} of sampled traffic. This volume pattern is inconsistent with "
            f"human browsing behavior and is a common signature of bot or scripted traffic. "
            f"Recommend reviewing these user_ids with your traffic-quality vendor before calibrating."
        )
    return warnings, stats

def check_publisher_ctr_cvr_divergence(df):
    warnings = []
    stats = {}
    required = {"is_won", "is_clicked", "is_converted", "publisher_id"}
    if not required.issubset(df.columns):
        return warnings, stats

    won = df[pd.to_numeric(df["is_won"], errors="coerce") == 1]
    if len(won) == 0:
        return warnings, stats

    pub_stats = won.groupby("publisher_id", observed=True).agg(
        ctr=("is_clicked", "mean"),
        impressions=("is_clicked", "size"),
    )
    clicked = won[pd.to_numeric(won["is_clicked"], errors="coerce") == 1]
    if len(clicked) == 0:
        return warnings, stats

    pub_cvr = clicked.groupby("publisher_id", observed=True)["is_converted"].mean()
    pub_stats["cvr"] = pub_stats.index.map(pub_cvr).fillna(0.0)
    pub_clicks = clicked.groupby("publisher_id", observed=True).size()
    pub_stats["n_clicks"] = pub_stats.index.map(pub_clicks).fillna(0).astype(int)

    MIN_CLICKS = 100
    eligible = pub_stats[pub_stats["n_clicks"] >= MIN_CLICKS]
    if len(eligible) == 0:
        return warnings, stats

    global_ctr = float(won["is_clicked"].mean())
    global_cvr = float(clicked["is_converted"].mean())

    suspicious = eligible[
        (eligible["ctr"] > global_ctr * 2.0) &
        (eligible["cvr"] < global_cvr * 0.25)
    ]

    stats["global_ctr"] = global_ctr
    stats["global_cvr"] = global_cvr
    stats["n_suspicious_publishers"] = int(len(suspicious))

    if len(suspicious) > 0:
        pub_ids = suspicious.index.tolist()[:10]
        warnings.append(
            f"{len(suspicious)} publisher(s) show CTR more than 2x the market average "
            f"combined with CVR less than 25% of the market average (min {MIN_CLICKS} clicks "
            f"each). This click-without-conversion pattern is a common click-fraud signature. "
            f"Sample publisher_ids: {pub_ids}. Consider excluding these from calibration or "
            f"flagging for your ad-fraud vendor."
        )
    return warnings, stats

# NOTE: on this tool's 50k-row sample, most users appear only 1-2 times,
# so this check has limited power here compared to the internal pipeline's
# larger validation chunk. Kept for parity and to catch egregious cases.
def check_timestamp_regularity(df, max_users_checked=2000):
    warnings = []
    stats = {}
    if "user_id" not in df.columns or "timestamp" not in df.columns:
        return warnings, stats

    user_counts = df.groupby("user_id", observed=True).size()
    candidates = user_counts[user_counts >= 20].nlargest(max_users_checked).index
    if len(candidates) == 0:
        return warnings, stats

    sub = df[df["user_id"].isin(candidates)].copy()
    sub["timestamp"] = pd.to_datetime(sub["timestamp"], errors="coerce")
    sub = sub.dropna(subset=["timestamp"]).sort_values(["user_id", "timestamp"])

    flagged_users = []
    for uid, group in sub.groupby("user_id", observed=True):
        deltas = group["timestamp"].diff().dt.total_seconds().dropna()
        if len(deltas) < 10:
            continue
        mean_delta = deltas.mean()
        std_delta = deltas.std()
        if mean_delta <= 0:
            continue
        cv = std_delta / mean_delta
        if cv < 0.15:
            flagged_users.append(uid)

    stats["n_users_with_metronomic_timing"] = len(flagged_users)

    if len(flagged_users) > 0:
        warnings.append(
            f"{len(flagged_users)} user_id(s) show near-perfectly-regular request timing "
            f"(coefficient of variation < 0.15 on inter-event intervals), inconsistent with "
            f"human browsing patterns. This is a common signature of scripted/automated traffic. "
            f"Sample: {flagged_users[:10]}."
        )
    return warnings, stats

def check_publisher_spend_concentration(df):
    warnings = []
    stats = {}
    required = {"is_won", "publisher_id", "clearing_price", "is_converted"}
    if not required.issubset(df.columns):
        return warnings, stats

    won = df[pd.to_numeric(df["is_won"], errors="coerce") == 1]
    if len(won) == 0:
        return warnings, stats

    pub_spend = won.groupby("publisher_id", observed=True)["clearing_price"].sum().sort_values(ascending=False)
    pub_convs = won.groupby("publisher_id", observed=True)["is_converted"].sum()

    total_spend = pub_spend.sum()
    if total_spend <= 0 or len(pub_spend) == 0:
        return warnings, stats

    top_n = max(1, int(len(pub_spend) * 0.05))
    top_publishers = pub_spend.head(top_n)
    top_spend_share = float(top_publishers.sum() / total_spend)
    total_convs = max(float(pub_convs.sum()), 1.0)
    top_conv_share = float(pub_convs.reindex(top_publishers.index).fillna(0).sum() / total_convs)

    stats["top5pct_publisher_spend_share"] = top_spend_share
    stats["top5pct_publisher_conversion_share"] = top_conv_share

    if top_spend_share > 0.40 and top_conv_share < (top_spend_share * 0.3):
        warnings.append(
            f"Top 5% of publishers by volume account for {top_spend_share:.1%} of total spend "
            f"but only {top_conv_share:.1%} of conversions. This spend/value mismatch warrants "
            f"review — it may reflect low-quality inventory concentration rather than fraud, "
            f"but the gap is large enough to check before calibrating the floor-price and "
            f"pub_quality models on this data."
        )
    return warnings, stats

def check_click_to_conversion_latency(df):
    warnings = []
    stats = {}
    required = {"click_timestamp", "is_converted", "conversion_timestamp"}
    if not required.issubset(df.columns):
        return warnings, stats

    conv_mask = pd.to_numeric(df["is_converted"], errors="coerce") == 1
    conv_rows = df[conv_mask].copy()
    if len(conv_rows) == 0:
        return warnings, stats

    click_ts = pd.to_datetime(conv_rows["click_timestamp"], errors="coerce")
    conv_ts = pd.to_datetime(conv_rows["conversion_timestamp"], errors="coerce")
    valid = click_ts.notna() & conv_ts.notna()
    if valid.sum() < 20:
        return warnings, stats

    delay_seconds = (conv_ts[valid] - click_ts[valid]).dt.total_seconds()
    near_zero = (delay_seconds < 1.0).sum()
    near_zero_rate = float(near_zero / valid.sum())

    stats["click_to_conversion_near_zero_rate"] = near_zero_rate

    if near_zero_rate > 0.10:
        warnings.append(
            f"{near_zero_rate:.1%} of conversions occur within 1 second of the click timestamp. "
            "Near-instant post-click conversion is inconsistent with genuine human purchase "
            "behavior and is a common signature of fabricated or bot-driven conversion events. "
            "Recommend reviewing affected traffic sources before calibrating."
        )
    return warnings, stats

def validate(df: pd.DataFrame, actual_total_rows: int = None) -> dict:
    errors   = []
    warnings = []
    stats    = {}

    total = actual_total_rows if actual_total_rows is not None else len(df)
    stats["row_count"] = total

    key_cols = [c for c in ["timestamp", "user_id", "publisher_id", "bid_price"] if c in df.columns]
    if len(key_cols) == 4:
        try:
            dup_count = df.duplicated(subset=key_cols).sum()
            stats["duplicate_rows"] = int(dup_count)
            if dup_count > 0:
                dup_rate = dup_count / len(df)
                if dup_rate > 0.05:
                    errors.append(
                        f"Found {dup_count:,} duplicate rows ({dup_rate:.2%}) with identical "
                        "timestamp, user_id, publisher_id, and bid_price. "
                        "At this rate, duplicates materially inflate CTR, CVR, and win rate. "
                        "Deduplicate before resubmitting. If timestamps are second-granularity "
                        "or bid prices are rounded, consider adding an auction_id column."
                    )
                else:
                    warnings.append(
                        f"Found {dup_count:,} duplicate rows ({dup_rate:.2%}) with identical "
                        "timestamp, user_id, publisher_id, and bid_price. "
                        "Low rates are expected when timestamps are second-granularity or bid prices "
                        "are rounded. Verify these are not logging artifacts."
                    )
        except Exception:
            pass

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
            elif win_rate > 0.30:
                errors.append(
                    f"Win rate is {win_rate:.1%} — far above the typical RTB range of 1–15%. "
                    "This strongly suggests the log contains only won impressions and is missing "
                    "lost bid requests. All ML models (CTR, CVR, win rate) will be trained on a "
                    "biased subset and will systematically overestimate performance. "
                    "Re-export the log to include all bid requests (wins AND losses)."
                )
            elif win_rate > 0.20:
                warnings.append(
                    f"Win rate is {win_rate:.1%} — above the typical RTB range of 1–15%. "
                    "Verify the log includes all bid requests, not only wins."
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

    if "is_converted" in present_cols and "is_won" in present_cols:
        try:
            won_mask2  = pd.to_numeric(df["is_won"],       errors="coerce").fillna(0)
            conv_mask2 = pd.to_numeric(df["is_converted"], errors="coerce").fillna(0)
            invalid_convs = ((won_mask2 == 0) & (conv_mask2 == 1)).sum()
            if invalid_convs > 0:
                errors.append(
                    f"Found {invalid_convs:,} rows where is_converted=1 but is_won=0. "
                    "A conversion cannot be attributed to an impression you did not win. "
                    "Check for view-through attribution or bad data joins."
                )
        except Exception:
            pass

    if "is_converted" in present_cols and "is_clicked" in present_cols and "is_won" in present_cols:
        try:
            _won_m   = pd.to_numeric(df["is_won"],       errors="coerce").fillna(0) == 1
            _click_m = pd.to_numeric(df["is_clicked"],   errors="coerce").fillna(0) == 0
            _conv_m  = pd.to_numeric(df["is_converted"], errors="coerce").fillna(0) == 1
            _vt      = (_won_m & _click_m & _conv_m).sum()
            _total_c = _conv_m.sum()
            if _total_c > 0 and _vt / _total_c > 0.05:
                warnings.append(
                    f"{_vt:,} conversions ({_vt / _total_c:.1%}) have is_clicked=0 on won rows. "
                    "These are view-through conversions. The CVR model trains on clicked rows only "
                    "and will not see these events, so calibrated CVR and CPA estimates will be "
                    "understated if view-through attribution is a meaningful part of your campaign."
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
            not_conv_mask = pd.to_numeric(df["is_converted"], errors="coerce").fillna(0) == 0
            bad_ltv = (pd.to_numeric(df.loc[not_conv_mask, "conversion_value"], errors="coerce") > 0).sum()
            if bad_ltv > 0:
                warnings.append(
                    f"{bad_ltv:,} rows have is_converted=0 but conversion_value > 0. "
                    "Non-converting rows must have conversion_value=0.0. "
                    "This will corrupt LTV model training."
                )
        except Exception:
            pass

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

    if "clearing_price" in present_cols and "is_won" in present_cols:
        try:
            _won_cp = df[pd.to_numeric(df["is_won"], errors="coerce") == 1]
            if len(_won_cp) > 0:
                _zero_rate = (
                    pd.to_numeric(_won_cp["clearing_price"], errors="coerce").fillna(0.0) == 0.0
                ).mean()
                if _zero_rate > 0.01:
                    errors.append(
                        f"{_zero_rate:.1%} of won auctions have clearing_price=0.0. "
                        "Won rows must carry the price paid — zero values indicate a data join "
                        "error or a null that was filled to 0 before export. "
                        "The floor price model will learn zero floors for affected publishers, "
                        "making the simulation unrealistically easy."
                    )
        except Exception:
            pass

    if "clearing_price" in present_cols and "bid_price" in present_cols and "is_won" in present_cols:
        try:
            _won_sp = df[pd.to_numeric(df["is_won"], errors="coerce") == 1].copy()
            if len(_won_sp) >= 100:
                _cp = pd.to_numeric(_won_sp["clearing_price"], errors="coerce")
                _bp = pd.to_numeric(_won_sp["bid_price"],      errors="coerce")
                _valid = _cp.notna() & _bp.notna() & (_bp > 0) & (_cp > 0)
                if _valid.sum() >= 50:
                    _ratio_median = float((_cp[_valid] / _bp[_valid]).median())
                    if _ratio_median < 0.70:
                        errors.append(
                            f"Median clearing_price / bid_price on won auctions is {_ratio_median:.2f}. "
                            "This is consistent with second-price (Vickrey) auction mechanics. "
                            "BidOptic does not support second-price auction data — the floor price, "
                            "win rate, and margin models require first-price clearing data to calibrate "
                            "correctly. Calibrating from second-price logs will produce systematically "
                            "wrong floor prices and unreliable simulation results. "
                            "If your exchange uses second-price, set clearing_price equal to bid_price "
                            "before export."
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
        try:
            median_bid = pd.to_numeric(df["bid_price"], errors="coerce").median()
            if median_bid > 10.0:
                warnings.append(
                    f"Median bid_price is ${median_bid:.2f}. All price columns must be per-impression USD, "
                    "not CPM. If your DSP logs in CPM, divide by 1,000 before export."
                )
        except Exception:
            pass

    if "clearing_price" in present_cols:
        if (pd.to_numeric(df["clearing_price"], errors="coerce") < 0).any():
            errors.append("'clearing_price' contains negative values. Clearing prices must be >= 0.")
        if "is_won" in present_cols:
            try:
                won_cp = pd.to_numeric(
                    df.loc[pd.to_numeric(df["is_won"], errors="coerce") == 1, "clearing_price"],
                    errors="coerce",
                ).dropna()
                if len(won_cp) > 0:
                    median_cp = float(won_cp.median())
                    if median_cp > 10.0:
                        errors.append(
                            f"Median clearing_price on won rows is ${median_cp:.2f}. "
                            "clearing_price must be per-impression USD, not CPM. "
                            "If your exchange reports clearing_price in CPM, divide by 1,000 before export. "
                            "Training the floor price model on CPM values will produce floors 1,000× too high "
                            "and make the simulation unwinnable."
                        )
            except Exception:
                pass

    if "publisher_id" in present_cols:
        try:
            pub_counts        = df["publisher_id"].value_counts(normalize=True)
            top_pub_share     = float(pub_counts.iloc[0])
            top_pub_id        = pub_counts.index[0]
            stats["publisher_count"]      = int(pub_counts.shape[0])
            stats["top_publisher_share"]  = round(top_pub_share, 4)
            if top_pub_share > 0.60:
                warnings.append(
                    f"Publisher '{top_pub_id}' accounts for {top_pub_share:.1%} of all rows "
                    f"({pub_counts.shape[0]} publishers total). Minority publishers have too few "
                    "impressions to fit reliable floor, CTR, and win-rate models independently — "
                    "they will fall back to population-average priors. If minority publisher accuracy "
                    "matters, consider pulling a longer window or filtering to your highest-volume supply sources."
                )
            elif top_pub_share > 0.40:
                warnings.append(
                    f"Publisher '{top_pub_id}' accounts for {top_pub_share:.1%} of all rows "
                    f"({pub_counts.shape[0]} publishers total). Models for minority publishers will "
                    "have less signal and higher variance. This is usually acceptable but worth noting "
                    "if minority publisher performance is a key evaluation metric."
                )
        except Exception:
            pass

    if "user_id" in present_cols:
        try:
            unique_users     = df["user_id"].nunique()
            uniqueness_ratio = unique_users / len(df)
            stats["unique_user_count"]    = int(unique_users)
            stats["user_id_unique_ratio"] = round(uniqueness_ratio, 4)
            if uniqueness_ratio > 0.80:
                warnings.append(
                    f"user_id uniqueness ratio is {uniqueness_ratio:.1%} ({unique_users:,} unique IDs "
                    f"across {len(df):,} rows). This suggests non-persistent identifiers — rotating "
                    "session IDs, cookieless traffic, or per-request tokens. User-level frequency, "
                    "recency, and LTV models will degrade to session-level signal. Publisher and "
                    "segment models are unaffected."
                )
            elif uniqueness_ratio > 0.50:
                warnings.append(
                    f"user_id uniqueness ratio is {uniqueness_ratio:.1%} ({unique_users:,} unique IDs "
                    f"across {len(df):,} rows). Moderate ID churn — possibly a mix of persistent and "
                    "non-persistent identifiers. User-level model accuracy may be reduced for the "
                    "non-persistent portion of your traffic."
                )
        except Exception:
            pass

    traffic_checks = [
        check_user_volume_outliers,
        check_publisher_ctr_cvr_divergence,
        check_timestamp_regularity,
        check_publisher_spend_concentration,
        check_click_to_conversion_latency,
    ]
    for check_fn in traffic_checks:
        try:
            check_warnings, check_stats = check_fn(df)
            warnings.extend(check_warnings)
            stats.update(check_stats)
        except Exception as e:
            warnings.append(f"Traffic anomaly check '{check_fn.__name__}' failed to run: {e}")

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

            TARGET_SAMPLE = 50_000

            if actual_total_rows <= TARGET_SAMPLE:
                df = parquet_file.read().to_pandas()
            else:
                MAX_GROUPS_TO_READ = 20

                rg_sizes = [
                    parquet_file.metadata.row_group(i).num_rows
                    for i in range(num_row_groups)
                ]

                if num_row_groups <= MAX_GROUPS_TO_READ:
                    selected = list(range(num_row_groups))
                else:
                    step = (num_row_groups - 1) / (MAX_GROUPS_TO_READ - 1)
                    selected = sorted({
                        round(i * step) for i in range(MAX_GROUPS_TO_READ)
                    })
                    selected = sorted(set(selected) | {0, num_row_groups - 1})

                selected_total = sum(rg_sizes[i] for i in selected)

                chunks = []
                for i in selected:
                    rg_df = parquet_file.read_row_group(i).to_pandas()
                    budget = max(1, round(TARGET_SAMPLE * rg_sizes[i] / selected_total))
                    if len(rg_df) > budget:
                        rg_df = rg_df.sample(n=budget, random_state=42)
                    chunks.append(rg_df)

                df = pd.concat(chunks, ignore_index=True)

                if len(df) > TARGET_SAMPLE:
                    df = df.sample(n=TARGET_SAMPLE, random_state=42).reset_index(drop=True)

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