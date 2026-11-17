"""Supplementary commodity/month aggregate audit."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from .parquet_compat import month_key


def build_commodity_monthly_aggregates(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["commodity"], row["pricetype"], row["unit"], month_key(row["date"]))
        grouped[key].append(row)
    output: list[dict] = []
    for (commodity, price_type, unit, month), group in sorted(grouped.items()):
        prices = [float(row["price"]) for row in group]
        output.append(
            {
                "commodity": commodity,
                "price_type": price_type,
                "unit": unit,
                "month": month,
                "mean_price": statistics.fmean(prices),
                "median_price": statistics.median(prices),
                "n_markets": len({row["market"] for row in group}),
                "n_records": len(group),
                "currency_values": ";".join(sorted({str(row["currency"]) for row in group})),
            }
        )
    return output


def write_aggregates(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
