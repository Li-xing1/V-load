# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "jsq"
DEFAULT_INPUT = DATA_DIR / "Weather.xlsx"
DEFAULT_MAPPING_OUTPUT = DATA_DIR / "Weather_category_mapping.json"
DEFAULT_COLUMNS = ("weatid", "winpid")


def sort_key(value):
    try:
        return 0, float(value), str(value)
    except (TypeError, ValueError):
        return 1, str(value)


def unique_values(series: pd.Series) -> list:
    values = series.dropna().drop_duplicates().tolist()
    return sorted(values, key=sort_key)


def build_encoding_map(series: pd.Series, start_index: int = 0) -> dict:
    return {
        value: index
        for index, value in enumerate(unique_values(series), start=start_index)
    }


def encode_column(frame: pd.DataFrame, column: str, start_index: int, replace: bool) -> tuple[str, dict]:
    mapping = build_encoding_map(frame[column], start_index=start_index)
    encoded = frame[column].map(mapping)
    encoded_column = column if replace else f"{column}_code"

    if encoded.isna().any():
        frame[encoded_column] = encoded.astype("Int64")
    else:
        frame[encoded_column] = encoded.astype(int)

    return encoded_column, mapping


def validate_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in weather excel: {missing_columns}")


def save_mapping(mapping_output: Path, metadata: dict) -> None:
    mapping_output.parent.mkdir(parents=True, exist_ok=True)
    with mapping_output.open("w", encoding="utf-8") as mapping_file:
        json.dump(metadata, mapping_file, ensure_ascii=False, indent=2)


def encode_weather_categories(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path | None = None,
    mapping_output: Path = DEFAULT_MAPPING_OUTPUT,
    columns: Iterable[str] = DEFAULT_COLUMNS,
    start_index: int = 0,
    replace: bool = True,
) -> dict:
    columns = tuple(columns)
    output_path = output_path or input_path
    frame = pd.read_excel(input_path)
    validate_columns(frame, columns)

    mapping_metadata = {
        "source": str(input_path),
        "output": str(output_path),
        "start_index": start_index,
        "columns": {},
    }

    for column in columns:
        encoded_column, mapping = encode_column(frame, column, start_index, replace)
        mapping_metadata["columns"][column] = {
            "encoded_column": encoded_column,
            "num_classes": len(mapping),
            "mapping": {str(value): int(code) for value, code in mapping.items()},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(output_path, index=False)
    save_mapping(mapping_output, mapping_metadata)
    return mapping_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode Weather.xlsx category columns into contiguous ids for one-hot and Embedding."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source weather xlsx path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Encoded weather xlsx path. Defaults to overwriting --input.",
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=DEFAULT_MAPPING_OUTPUT,
        help="JSON file used to save category-to-code mappings",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=list(DEFAULT_COLUMNS),
        help="Category columns to encode",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First category code. Use 0 for torch.nn.Embedding and one-hot.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        default=True,
        help="Replace source columns instead of adding *_code columns. This is the default.",
    )
    parser.add_argument(
        "--append-columns",
        action="store_false",
        dest="replace",
        help="Add *_code columns instead of replacing source columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.input
    metadata = encode_weather_categories(
        input_path=args.input,
        output_path=output_path,
        mapping_output=args.mapping_output,
        columns=args.columns,
        start_index=args.start_index,
        replace=args.replace,
    )

    print(f"Saved encoded weather excel: {output_path}")
    print(f"Saved category mapping: {args.mapping_output}")
    for column, column_metadata in metadata["columns"].items():
        print(
            f"{column} -> {column_metadata['encoded_column']}, "
            f"classes: {column_metadata['num_classes']}"
        )


if __name__ == "__main__":
    main()
