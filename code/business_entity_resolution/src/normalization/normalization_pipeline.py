from pathlib import Path
import time

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .address_normalizer import normalize_address, normalize_addresses
from .country_rules import country_key
from .name_normalizer import normalize_name, normalize_names
from .unicode_utils import as_text_array


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
BLOCK_SIZE_BYTES = 64 << 20


def normalize_table(table: pa.Table) -> pa.Table:
    """Raw source records -> all views. Raw values and country are kept as given."""

    country = as_text_array(table["country"])

    columns = {
        "entity_id": table["entity_id"],
        "country_raw": country,
        "country_key": pa.array([country_key(c) for c in country.to_pylist()], type=pa.string()),
    }
    columns.update(normalize_names(table["business_name"], country))
    columns.update(normalize_addresses(table["business_address"]))

    return pa.table({k: _as_arrow(v) for k, v in columns.items()})


def _as_arrow(values):
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        return values
    return pa.array(values)


def normalize_record(business_name, business_address, country) -> dict:
    name = normalize_name(business_name, country)
    address = normalize_address(business_address)
    return {"country_raw": "" if country is None else str(country), **name, **address}


def read_source_batches(path: Path):
    """
    Stream a source TSV as record batches (all columns as strings).
    No null conversion: literal "NA"/"NULL" stay as written (train has
    businesses named "NA"); empty fields become "".
    """
    return pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=BLOCK_SIZE_BYTES),
        parse_options=pacsv.ParseOptions(delimiter="\t"),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in SOURCE_COLUMNS},
            null_values=[],
            strings_can_be_null=False,
        ),
    )


def normalize_file(in_path: Path, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rows = 0
    writer = None

    try:
        for batch in read_source_batches(in_path):
            normalized = normalize_table(pa.Table.from_batches([batch]))
            if writer is None:
                writer = pq.ParquetWriter(out_path, normalized.schema, compression="zstd")
            writer.write_table(normalized)
            rows += normalized.num_rows
    finally:
        if writer is not None:
            writer.close()

    return {"file": in_path.name, "rows": rows, "seconds": round(time.time() - start, 1)}
