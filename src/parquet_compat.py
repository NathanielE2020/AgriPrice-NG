"""Read-only Parquet compatibility helpers for the supplied WFP file.

The normal path uses ``pyarrow.parquet``.  A lightweight ``parquet``/``thriftpy2``
reader remains available as a fallback for environments where pyarrow cannot be
installed.  Neither path writes to ``data/raw/``.
"""

from __future__ import annotations

import types
from datetime import date, datetime, timezone
from pathlib import Path


def _snappy_decompress(data: bytes) -> bytes:
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
        if tag_type == 0:
            length_code = tag >> 2
            if length_code < 60:
                length = length_code + 1
            else:
                n_bytes = length_code - 59
                length = int.from_bytes(data[pos : pos + n_bytes], "little") + 1
                pos += n_bytes
            output.extend(data[pos : pos + length])
            pos += length
        elif tag_type == 1:
            length = ((tag >> 2) & 7) + 4
            offset = ((tag & 0xE0) << 3) | data[pos]
            pos += 1
            for _ in range(length):
                output.append(output[-offset])
        elif tag_type == 2:
            length = (tag >> 2) + 1
            offset = int.from_bytes(data[pos : pos + 2], "little")
            pos += 2
            for _ in range(length):
                output.append(output[-offset])
        else:
            length = (tag >> 2) + 1
            offset = int.from_bytes(data[pos : pos + 4], "little")
            pos += 4
            for _ in range(length):
                output.append(output[-offset])

    if len(output) != declared_length:
        raise ValueError(
            f"Snappy decoded length {len(output)} does not match "
            f"declared length {declared_length}"
        )
    return bytes(output)


def _install_reader_compatibility():
    try:
        import parquet
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "The fallback Parquet reader is unavailable. Install parquet and "
            "thriftpy2, or install pandas and pyarrow."
        ) from exc

    parquet.snappy = types.SimpleNamespace(decompress=_snappy_decompress)
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
    return parquet


def read_parquet_rows(path: str | Path) -> list[dict]:
    parquet_path = Path(path)
    try:
        import pyarrow.parquet as pyarrow_parquet
    except ImportError:
        parquet = _install_reader_compatibility()
        with parquet_path.open("rb") as file_obj:
            return list(parquet.DictReader(file_obj))
    table = pyarrow_parquet.read_table(parquet_path)
    return table.to_pylist()


def to_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).date()


def month_key(value: date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def month_index(value: str) -> int:
    year, month = (int(part) for part in value.split("-"))
    return year * 12 + month


def month_from_index(index: int) -> str:
    year = (index - 1) // 12
    month = (index - 1) % 12 + 1
    return f"{year:04d}-{month:02d}"


def iter_months(first_month: str, last_month: str) -> list[str]:
    return [
        month_from_index(index)
        for index in range(month_index(first_month), month_index(last_month) + 1)
    ]
