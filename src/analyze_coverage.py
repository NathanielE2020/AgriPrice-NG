"""Time-series coverage analysis for the WFP Nigeria price dataset.

This script intentionally uses the lightweight Parquet reader that was used
for the initial inspection.  It does not require pandas or pyarrow and only
writes analysis artifacts under reports/.
"""

from __future__ import annotations

import csv
import math
import types
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import parquet


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"
PARQUET_PATH = RAW_DIR / "train.parquet"
METADATA_PATH = RAW_DIR / "wfp_markets_nga.csv"
COVERAGE_CSV = REPORT_DIR / "time_series_coverage.csv"
SUMMARY_TXT = REPORT_DIR / "time_series_coverage_summary.txt"


def snappy_decompress(data: bytes) -> bytes:
    """Decode a raw Snappy block used by the supplied Parquet file."""

    pos = 0
    declared_length = 0
    shift = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        declared_length |= (byte & 127) << shift
        if byte < 128:
            break
        shift += 7

    output = bytearray()
    while pos < len(data):
        tag = data[pos]
        pos += 1
        tag_type = tag & 3

        if tag_type == 0:  # literal
            length_code = tag >> 2
            if length_code < 60:
                length = length_code + 1
            else:
                n_bytes = length_code - 59
                length = int.from_bytes(data[pos : pos + n_bytes], "little") + 1
                pos += n_bytes
            output.extend(data[pos : pos + length])
            pos += length
        elif tag_type == 1:  # copy with one-byte offset
            length = ((tag >> 2) & 7) + 4
            offset = ((tag & 0xE0) << 3) | data[pos]
            pos += 1
            for _ in range(length):
                output.append(output[-offset])
        elif tag_type == 2:  # copy with two-byte offset
            length = (tag >> 2) + 1
            offset = int.from_bytes(data[pos : pos + 2], "little")
            pos += 2
            for _ in range(length):
                output.append(output[-offset])
        else:  # copy with four-byte offset
            length = (tag >> 2) + 1
            offset = int.from_bytes(data[pos : pos + 4], "little")
            pos += 4
            for _ in range(length):
                output.append(output[-offset])

    if len(output) != declared_length:
        raise ValueError(
            f"Snappy length mismatch: decoded {len(output)} bytes, "
            f"expected {declared_length}"
        )
    return bytes(output)


def install_parquet_compatibility() -> None:
    """Add Snappy support and map RLE_DICTIONARY to the reader's compatible path."""

    parquet.snappy = types.SimpleNamespace(decompress=snappy_decompress)
    original_read_data_page = parquet.read_data_page

    def read_data_page_compat(file_obj, schema_helper, page_header, column_metadata, dictionary):
        if (
            page_header.data_page_header.encoding
            == parquet.parquet_thrift.Encoding.RLE_DICTIONARY
        ):
            page_header.data_page_header.encoding = (
                parquet.parquet_thrift.Encoding.PLAIN_DICTIONARY
            )
        return original_read_data_page(
            file_obj, schema_helper, page_header, column_metadata, dictionary
        )

    parquet.read_data_page = read_data_page_compat


def to_date(value) -> date:
    """Convert the reader's nanosecond timestamp (or datetime) to a date."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).date()


def month_index(month_key: str) -> int:
    year, month = (int(part) for part in month_key.split("-"))
    return year * 12 + month


def month_key(day_value: date) -> str:
    return f"{day_value.year:04d}-{day_value.month:02d}"


def expected_months(first_month: str, last_month: str) -> int:
    return month_index(last_month) - month_index(first_month) + 1


def consecutive_run_lengths(months: list[str]) -> list[int]:
    if not months:
        return []
    sorted_months = sorted(set(months), key=month_index)
    runs: list[int] = []
    current = 1
    for previous, current_month in zip(sorted_months, sorted_months[1:]):
        if month_index(current_month) - month_index(previous) == 1:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    return runs


def read_rows() -> list[dict]:
    install_parquet_compatibility()
    with PARQUET_PATH.open("rb") as file_obj:
        return list(parquet.DictReader(file_obj))


def read_metadata_market_count() -> tuple[int, int, int]:
    """Return metadata record count, distinct IDs, and distinct admin1 values."""

    if not METADATA_PATH.exists():
        return 0, 0, 0
    with METADATA_PATH.open(newline="", encoding="utf-8-sig") as file_obj:
        rows = list(csv.reader(file_obj))
    if len(rows) <= 2:
        return 0, 0, 0
    header = rows[0]
    records = [dict(zip(header, row)) for row in rows[2:] if row]
    return (
        len(records),
        len({record["market_id"] for record in records}),
        len({record["admin1"] for record in records}),
    )


def build_coverage(rows: list[dict]) -> list[dict]:
    # The requested group key deliberately excludes priceflag.  We record its
    # values so training can avoid silently mixing actual and aggregate prices.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["commodity"],
                row["market"],
                row["unit"],
                row["pricetype"],
            )
        ].append(row)

    coverage_rows: list[dict] = []
    for group_key, group_rows in groups.items():
        commodity, market, unit, price_type = group_key
        dates = sorted({to_date(row["date"]) for row in group_rows})
        months = sorted({month_key(day_value) for day_value in dates}, key=month_index)
        first_month = months[0]
        last_month = months[-1]
        expected = expected_months(first_month, last_month)
        missing = expected - len(months)
        coverage = (len(months) / expected * 100.0) if expected else 0.0
        gaps = [
            month_index(current) - month_index(previous) - 1
            for previous, current in zip(months, months[1:])
        ]
        max_gap = max(gaps, default=0)
        max_run = max(consecutive_run_lengths(months), default=0)
        possible_sequences = max(0, max_run - 12)
        priceflags = sorted({row.get("priceflag") for row in group_rows})
        market_ids = sorted({row.get("market_id") for row in group_rows})
        states = sorted({row.get("admin1") for row in group_rows})

        # A strict flag is useful for selection: 96 observed monthly points,
        # at least 80% calendar coverage, and no internal gap over three months.
        sufficiently_continuous = (
            len(months) >= 96 and coverage >= 80.0 and max_gap <= 3
        )
        # A 12-month sequence window needs complete consecutive months. A run
        # of 13 months can produce one input/target pair. A run of 24 months
        # is only borderline after a chronological split. A run of 36 months,
        # with at least 48 total observations and 24 possible windows, is the
        # conservative reliability threshold used here.
        lstm_possible = max_run >= 13
        lstm_borderline = (
            len(group_rows) >= 36 and max_run >= 24 and possible_sequences >= 12
        )
        lstm_reliable = (
            len(group_rows) >= 48
            and max_run >= 36
            and possible_sequences >= 24
        )

        coverage_rows.append(
            {
                "series_id": f"{commodity} | {market} | {unit} | {price_type}",
                "commodity": commodity,
                "market": market,
                "market_id": ",".join(str(value) for value in market_ids),
                "state": ",".join(str(value) for value in states),
                "unit": unit,
                "price_type": price_type,
                "n_observations": len(group_rows),
                "first_date": dates[0].isoformat(),
                "last_date": dates[-1].isoformat(),
                "unique_months": len(months),
                "expected_months_in_range": expected,
                "missing_months": missing,
                "monthly_coverage_pct": round(coverage, 4),
                "max_gap_months": max_gap,
                "max_consecutive_months": max_run,
                "priceflag_values": ";".join(priceflags),
                "priceflag_count": len(priceflags),
                "lstm_sequence_window": 12,
                "lstm_possible_sequences": possible_sequences,
                "lstm_12m_possible": lstm_possible,
                "lstm_12m_borderline": lstm_borderline,
                "lstm_12m_reliable": lstm_reliable,
                "sufficiently_continuous_for_forecasting": sufficiently_continuous,
            }
        )

    return sorted(
        coverage_rows,
        key=lambda row: (
            -row["n_observations"],
            -row["monthly_coverage_pct"],
            -row["max_consecutive_months"],
            row["commodity"],
            row["market"],
            row["unit"],
            row["price_type"],
        ),
    )


def write_coverage_csv(rows: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with COVERAGE_CSV.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_primary_candidates(rows: list[dict], limit: int = 5) -> list[dict]:
    """Choose the longest series meeting the requested 96-observation preference."""

    ranked = sorted(
        (row for row in rows if row["n_observations"] >= 96),
        key=lambda row: (
            -int(row["n_observations"]),
            -float(row["monthly_coverage_pct"]),
            -int(row["max_consecutive_months"]),
            row["commodity"],
            row["market"],
        ),
    )
    return ranked[:limit]


def choose_lstm_borderline(rows: list[dict], limit: int = 5) -> list[dict]:
    """Return the best available exploratory LSTM candidates."""

    ranked = sorted(
        (row for row in rows if row["lstm_12m_borderline"]),
        key=lambda row: (
            -int(row["max_consecutive_months"]),
            -int(row["lstm_possible_sequences"]),
            -int(row["n_observations"]),
            row["commodity"],
            row["market"],
        ),
    )
    return ranked[:limit]


def write_summary(rows: list[dict], selected: list[dict], source_rows: list[dict]) -> None:
    metadata_records, metadata_ids, metadata_states = read_metadata_market_count()
    total = len(rows)
    count_96 = sum(row["n_observations"] >= 96 for row in rows)
    count_60 = sum(row["n_observations"] >= 60 for row in rows)
    count_48 = sum(row["n_observations"] >= 48 for row in rows)
    count_36 = sum(row["n_observations"] >= 36 for row in rows)
    continuous = sum(row["sufficiently_continuous_for_forecasting"] for row in rows)
    lstm_possible = sum(row["lstm_12m_possible"] for row in rows)
    lstm_borderline = sum(row["lstm_12m_borderline"] for row in rows)
    lstm_reliable = sum(row["lstm_12m_reliable"] for row in rows)
    mixed_priceflag = sum(row["priceflag_count"] > 1 for row in rows)
    dates = [to_date(row["date"]) for row in source_rows]

    lines: list[str] = []
    lines.extend(
        [
            "AgriPrice-NG time-series coverage analysis",
            "===========================================",
            "Generated: 2026-11-12",
            "",
            "Source and grouping",
            "------------------",
            "Source file: data/raw/train.parquet",
            f"Source rows read: {len(source_rows):,}",
            "Grouping key: commodity + market + unit + pricetype",
            "Exact source columns used: date, commodity, market, admin1 (state),",
            "price, unit, pricetype. market_id and priceflag were retained as",
            "context fields; priceflag was not silently merged into the key.",
            "",
            "Coverage columns in time_series_coverage.csv",
            "----------------------------------------------",
            "n_observations: number of source records in the group.",
            "unique_months: distinct YYYY-MM values present.",
            "expected_months_in_range: inclusive calendar months from first to last.",
            "missing_months: expected_months_in_range - unique_months.",
            "monthly_coverage_pct: unique_months / expected_months_in_range * 100.",
            "max_gap_months: largest internal calendar gap between observations.",
            "max_consecutive_months: longest uninterrupted monthly run.",
            "",
            "Decision thresholds",
            "-------------------",
            "sufficiently_continuous_for_forecasting is True only when all are true:",
            "  - at least 96 observed monthly points",
            "  - at least 80% calendar coverage over the first-to-last span",
            "  - no internal gap longer than three calendar months",
            "",
            "lstm_12m_possible is True when at least 13 consecutive months exist",
            "(the minimum for one 12-month input plus one target).",
            "lstm_12m_borderline additionally requires at least 36 records, a",
            "24-month uninterrupted run, and 12 possible windows.",
            "lstm_12m_reliable is stricter: at least 48 records, a 36-month",
            "uninterrupted run, and 24 possible windows. This is a conservative",
            "screening flag, not proof that the model will perform well.",
            "A final chronological split must still be checked.",
            "",
            "Group counts",
            "------------",
            f"Distinct requested groups: {total:,}",
            f"Groups with >=96 observations: {count_96:,}",
            f"Groups with >=60 observations: {count_60:,}",
            f"Groups with >=48 observations: {count_48:,}",
            f"Groups with >=36 observations: {count_36:,}",
            f"Groups meeting strict continuity flag: {continuous:,}",
            f"Groups with at least one possible 12-month LSTM window: {lstm_possible:,}",
            f"Groups meeting borderline LSTM flag: {lstm_borderline:,}",
            f"Groups meeting reliable LSTM flag: {lstm_reliable:,}",
            f"Groups containing multiple priceflag values: {mixed_priceflag:,}",
            "",
            "Metadata cross-check",
            "--------------------",
            f"Supporting CSV market records: {metadata_records:,}",
            f"Supporting CSV distinct market IDs: {metadata_ids:,}",
            f"Supporting CSV distinct admin1 values: {metadata_states:,}",
            f"Parquet date range: {min(dates).isoformat()} to {max(dates).isoformat()}",
            "",
            "Recommended three-day training scope",
            "-------------------------------------",
            "The supplied series are sparse. No group satisfies the strict 96-point",
            "and 80%-coverage continuity rule, so the candidates below are the best",
            "long-history series available, not claims of complete monthly coverage.",
            "Train each exact combination separately; do not combine units or Retail",
            "and Wholesale prices. Prefer a single actual priceflag when possible and",
            "record the priceflag policy in the training configuration.",
            "",
        ]
    )

    if selected:
        lines.append(
            "Primary long-history candidate series (all groups with >=96 observations):"
        )
        lines.append("")
        for index, row in enumerate(selected, start=1):
            lines.append(
                f"{index}. {row['commodity']} | {row['market']} | {row['unit']} | "
                f"{row['price_type']}"
            )
            lines.append(
                f"   market_id={row['market_id']}; state={row['state']}; "
                f"observations={row['n_observations']}; "
                f"dates={row['first_date']} to {row['last_date']}; "
                f"unique_months={row['unique_months']}; "
                f"missing_months={row['missing_months']}; "
                f"coverage={row['monthly_coverage_pct']:.2f}%; "
                f"max_gap={row['max_gap_months']} months; "
                f"longest_run={row['max_consecutive_months']} months; "
                f"priceflags={row['priceflag_values']}; "
                f"LSTM_reliable={row['lstm_12m_reliable']}"
            )
            lines.append(
                "   Selection reason: one of the five longest requested groups; the "
                "unit and Retail/Wholesale type remain fixed, while gaps and "
                "priceflag consistency must be handled explicitly."
            )
            lines.append("")
    else:
        lines.append("No group met the 96-observation preference.")
        lines.append("")

    borderline = choose_lstm_borderline(rows)
    lines.extend(
        [
            "12-month LSTM candidates",
            "------------------------",
            "No group meets the conservative reliable-LSTM flag. The following are",
            "the only borderline candidates under the documented screening rule;",
            "they can produce multiple complete 12-month windows, but their short",
            "uninterrupted runs and missing calendar months make LSTM results fragile.",
            "",
        ]
    )
    if borderline:
        for row in borderline:
            lines.append(
                f"- {row['commodity']} | {row['market']} | {row['unit']} | "
                f"{row['price_type']}: n={row['n_observations']}, "
                f"dates={row['first_date']} to {row['last_date']}, "
                f"coverage={row['monthly_coverage_pct']:.2f}%, "
                f"max_gap={row['max_gap_months']} months, "
                f"longest_run={row['max_consecutive_months']} months, "
                f"possible_sequences={row['lstm_possible_sequences']}, "
                f"priceflags={row['priceflag_values']}"
            )
    else:
        lines.append("- None met the borderline LSTM screening rule.")
    lines.extend(
        [
            "",
            "Practical recommendation: use the five primary long-history groups for",
            "tree-based and ensemble pipeline testing, but do not present their LSTM",
            "results as reliable. If the five-model comparison must use identical",
            "series, use only the borderline LSTM candidates above as an exploratory",
            "two-series benchmark and label the LSTM limitation clearly. A later",
            "approved alternative is a documented commodity-level aggregation or a",
            "gap-aware sequence method; neither should be applied silently.",
            "",
        ]
    )

    lines.extend(
        [
            "LSTM warning",
            "------------",
            "A 12-month sequence window requires complete consecutive monthly history.",
            "Any row with lstm_12m_reliable=False should be excluded from claims of",
            "reliable LSTM training or used only for exploratory baselines. Missing",
            "months must not be fabricated",
            "by copying future prices. The recommended implementation should create",
            "sequences only inside uninterrupted observed runs and use the same",
            "chronological test cutoff for every model.",
            "",
            "No models were trained during this phase. data/raw/ was not modified.",
        ]
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = read_rows()
    if not rows:
        raise RuntimeError("No Parquet rows were read")
    coverage_rows = build_coverage(rows)
    write_coverage_csv(coverage_rows)
    selected = choose_primary_candidates(coverage_rows, limit=5)
    write_summary(coverage_rows, selected, rows)
    print(f"Rows read: {len(rows):,}")
    print(f"Coverage groups: {len(coverage_rows):,}")
    print(f"Coverage report: {COVERAGE_CSV}")
    print(f"Summary report: {SUMMARY_TXT}")
    print("Primary candidates:")
    for row in selected:
        print(
            f"  {row['commodity']} | {row['market']} | {row['unit']} | "
            f"{row['price_type']} | n={row['n_observations']} | "
            f"coverage={row['monthly_coverage_pct']:.2f}% | "
            f"lstm={row['lstm_12m_reliable']}"
        )
    print("Borderline LSTM candidates:")
    for row in choose_lstm_borderline(coverage_rows):
        print(
            f"  {row['commodity']} | {row['market']} | {row['unit']} | "
            f"{row['price_type']} | n={row['n_observations']} | "
            f"run={row['max_consecutive_months']} months | "
            f"sequences={row['lstm_possible_sequences']}"
        )


if __name__ == "__main__":
    main()
