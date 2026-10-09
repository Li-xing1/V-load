# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent
PYTORCH_ENV = Path(r"D:\anaconda\envs\Pytorch")
SZ_ROOT = DATA_DIR / "SZ"
EXCEL_DIR = SZ_ROOT / "Excel"
OUTPUT_ROOT = SZ_ROOT / "data"
BRIDGE_INFO = SZ_ROOT / "data" / "bridge_information.xlsx"
WEATHER_FILE = SZ_ROOT / "Weather.xlsx"
WEATHER_MAPPING_FILE = SZ_ROOT / "Weather_category_mapping.json"
RESTDAY_FILE = SZ_ROOT / "restday.xlsx"
TIME_CONFIG = PROJECT_ROOT / "args" / "dataset_args.yaml"
DATASET_ARGS = PROJECT_ROOT / "args" / "dataset_args.yaml"
BUILD_GLOBAL = True

CANONICAL_LANES = ("上行右", "上行中", "上行左", "下行左", "下行中", "下行右")
LANES = list(range(6))
VEHICLE_TYPES = ["2C", "2F", "3", "4", "5", "6"]
VC_COLUMNS = VEHICLE_TYPES + ["flow"]
TIME_FREQ = "5min"
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
TWO_AXLE_PASSENGER_CODES = {"1", "2", "3", "4", "20", "21", "22", "23"}

REQUIRED_COLUMNS = ["time", "lane_id", "veh_type", "axle_num", "grossload", "speed"]
AXLE_COLUMNS = [f"axle{i}" for i in range(1, 9)]
OPTIONAL_COLUMNS = AXLE_COLUMNS + [f"axle_dis{i}" for i in range(1, 8)]
DEBUG_V_C_LIMIT = 1000
DEBUG_V_W_LIMIT = 3000
DEBUG_V_S_LIMIT = 3000
DEBUG_V_V_LIMIT = 1000
V_W_GROSSLOAD_MAX = 100000
V_V_SPEED_MAX = 50
MAX_V_S_TIME_GAP_SECONDS = 10 * 60


def weather_category_sort_key(value) -> tuple:
    try:
        return 0, float(value), str(value)
    except (TypeError, ValueError):
        return 1, str(value)


def preprocess_weather(
    weather_path: Path = WEATHER_FILE,
    mapping_path: Path = WEATHER_MAPPING_FILE,
) -> pd.DataFrame:
    weather = pd.read_excel(weather_path)
    category_columns = ("weatid", "winpid")
    missing = [column for column in category_columns if column not in weather.columns]
    if missing:
        raise ValueError(f"{weather_path.name} missing category columns: {missing}")

    metadata = {
        "source": str(weather_path),
        "start_index": 0,
        "columns": {},
    }
    for column in category_columns:
        if weather[column].isna().any():
            raise ValueError(f"{weather_path.name} contains missing values in {column}")
        categories = sorted(weather[column].drop_duplicates().tolist(), key=weather_category_sort_key)
        mapping = {value: index for index, value in enumerate(categories)}
        weather[column] = weather[column].map(mapping).astype(int)
        metadata["columns"][column] = {
            "encoded_column": column,
            "num_classes": len(mapping),
            "mapping": {str(value): index for value, index in mapping.items()},
        }

    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return weather


def parse_lane_ids(value) -> set[int]:
    if pd.isna(value) or str(value).strip() == "无数据":
        return set()
    return {int(item) for item in re.findall(r"\d+", str(value))}


def canonical_lane_mapping(bridge: pd.Series) -> dict[int, int]:
    directions = [parse_lane_ids(bridge["上行车道编码"]), parse_lane_ids(bridge["下行车道编码"])]
    lateral = {
        "right": parse_lane_ids(bridge["右车道编码"]),
        "middle": parse_lane_ids(bridge["中车道编码"]),
        "left": parse_lane_ids(bridge["左车道编码"]),
    }
    mapping = {}
    for lane in directions[0]:
        mapping[lane] = 0 if lane in lateral["right"] else 2 if lane in lateral["left"] else 1
    for lane in directions[1]:
        mapping[lane] = 5 if lane in lateral["right"] else 3 if lane in lateral["left"] else 4
    return mapping


def add_canonical_lane(data: pd.DataFrame, lane_mapping: dict[int, int]) -> pd.DataFrame:
    result = data.copy()
    result["standard_lane"] = result["lane_id"].map(lane_mapping)
    result = result[result["standard_lane"].notna()].copy()
    result["standard_lane"] = result["standard_lane"].astype(int)
    return result


def interval_lane_flow(data: pd.DataFrame) -> pd.DataFrame:
    frame = data[["time", "standard_lane"]].copy()
    frame["time_bin"] = frame["time"].dt.floor(TIME_FREQ)
    result = frame.groupby(["time_bin", "standard_lane"]).size().rename("vehicle_count").reset_index()
    result["minute_flow"] = result["vehicle_count"] / 5.0
    return result


def spacing_records(data: pd.DataFrame, vehicle_class_index: dict[str, int]) -> pd.DataFrame:
    frame = data.sort_values(["lane_id", "time"]).copy()
    tied = frame.groupby(["lane_id", "time"]).cumcount()
    tied_count = frame.groupby(["lane_id", "time"])["time"].transform("size")
    frame["precise_time"] = frame["time"] + pd.to_timedelta(tied / (tied_count + 1), unit="s")
    grouped = frame.groupby("lane_id", sort=False)
    frame["prev_time"] = grouped["precise_time"].shift(1)
    frame["prev_speed"] = grouped["speed_mps"].shift(1)
    frame["prev_vehicle_class"] = grouped["vehicle_class"].shift(1)
    frame = frame.dropna(subset=["prev_time"]).copy()
    frame["time_gap"] = (frame["precise_time"] - frame["prev_time"]).dt.total_seconds()
    frame = frame[(frame["time_gap"] > 0) & (frame["time_gap"] <= MAX_V_S_TIME_GAP_SECONDS)].copy()
    frame["distance"] = (frame["speed_mps"] + frame["prev_speed"]) * 0.5 * frame["time_gap"]
    frame["self_vehicle_class_idx"] = frame["vehicle_class"].map(vehicle_class_index)
    frame["prev_vehicle_class_idx"] = frame["prev_vehicle_class"].map(vehicle_class_index)
    return frame


def build_speed_grid(records: pd.DataFrame, times: pd.DatetimeIndex, lanes=LANES,
                     max_speed=V_V_SPEED_MAX):
    frame = records[["time", "standard_lane", "speed_mps"]].copy()
    frame["time_bin"] = frame["time"].dt.floor(TIME_FREQ)
    full_index = pd.MultiIndex.from_product([times, lanes], names=["time_bin", "standard_lane"])

    minute_flow = frame.groupby(["time_bin", "standard_lane"]).size().reindex(full_index, fill_value=0) / 5.0
    valid_speed = np.isfinite(frame["speed_mps"]) & frame["speed_mps"].between(
        0.0, max_speed, inclusive="right"
    )
    mean_speed = (
        frame.loc[valid_speed]
        .groupby(["time_bin", "standard_lane"])["speed_mps"]
        .mean()
        .reindex(full_index)
    )

    values = np.column_stack([
        mean_speed.fillna(0.0).to_numpy(dtype=np.float32),
        minute_flow.to_numpy(dtype=np.float32),
    ]).reshape(len(times), len(lanes), 2)
    mask = mean_speed.notna().to_numpy().reshape(len(times), len(lanes), 1)
    return values, mask


class V_V_profress:
    def __init__(self):
        self.start_time, self.end_time = self.load_time_range()
        self.weather, self.rest = self.load_context()
        self.bridge_metadata = pd.read_excel(BRIDGE_INFO)
        self.bridge_order, self.bridge_indices = self.load_bridge_metadata()
        code_column = "设施编码" if "设施编码" in self.bridge_metadata.columns else "编码"
        self.bridge_rows = {
            str(int(row[code_column])): row for _, row in self.bridge_metadata.iterrows()
        }
        self.spatial_lookup = self.build_spatial_lookup()

    def filter_v_w_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data

        mask = np.isfinite(data["grossload"]) & (data["grossload"] > 0) & (data["grossload"] <= V_W_GROSSLOAD_MAX)
        return data.loc[mask].copy()

    def filter_v_w_axles(self, data: pd.DataFrame, axle_columns: list[str]) -> pd.DataFrame:
        values = data[axle_columns]
        mask = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
        mask &= (values <= V_W_GROSSLOAD_MAX).all(axis=1)
        return data.loc[mask].copy()

    def filter_v_v_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data

        mask = np.isfinite(data["speed_mps"]) & (data["speed_mps"] > 0) & (data["speed_mps"] <= V_V_SPEED_MAX)
        return data.loc[mask].copy()

    def filter_v_s_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        columns = ["distance", "self_grossload"]
        mask = np.isfinite(data[columns]).all(axis=1) & (data[columns] > 0).all(axis=1)
        mask &= data["self_grossload"] <= V_W_GROSSLOAD_MAX
        return data.loc[mask].copy()

    def run(self) -> None:
        v_v_by_bridge: dict[str, np.ndarray] = {}
        masks_by_bridge: dict[str, np.ndarray] = {}
        shared_tf: np.ndarray | None = None
        ordered_items = self.ordered_excel_items()

        for bridge_code, excel_path in tqdm(ordered_items, desc="Processing V_V bridges", unit="bridge"):
            data = self.load_wim_excel(excel_path)
            data = add_canonical_lane(data, canonical_lane_mapping(self.bridge_rows[bridge_code]))
            v_v_data, v_v_tf, v_v_mask = self.build_v_v(data)
            v_v_by_bridge[bridge_code] = v_v_data
            masks_by_bridge[bridge_code] = v_v_mask
            shared_tf = v_v_tf if shared_tf is None else shared_tf

        if BUILD_GLOBAL:
            global_codes = [code for code in self.bridge_order if code in v_v_by_bridge]
            global_codes.extend(sorted(code for code in v_v_by_bridge if code not in global_codes))
            v_v_global = np.stack([v_v_by_bridge[code] for code in global_codes], axis=1)
            mask_global = np.stack([masks_by_bridge[code] for code in global_codes], axis=1)
            self.save_global_arrays("V_V", v_v_global, shared_tf, DEBUG_V_V_LIMIT)
            np.save(OUTPUT_ROOT / "npy" / "V_V" / "V_V_mask.npy", mask_global)
            np.save(OUTPUT_ROOT / "debug_npy" / "V_V" / "V_V_mask.npy", mask_global[:DEBUG_V_V_LIMIT])
            self.save_metadata(global_codes)

    def build_v_v(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = self.time_points()
        if data.empty:
            tf = self.build_tf(times)
            empty = np.zeros((len(times), 6, 2), dtype=np.float32)
            mask = np.zeros((len(times), 6, 1), dtype=bool)
            return empty, self.to_float32(tf, self.context_columns(), "V_V_tf"), mask

        frame = data[["time", "standard_lane", "speed_mps"]].copy()
        data_array, mask_array = build_speed_grid(frame, times)
        result = pd.DataFrame({"time_bin": times})
        result["time_index"] = result["time_bin"].map(self.time_index)
        result = self.add_context(result, "time_bin")
        tf = result[["time_bin"] + self.context_columns()].sort_values("time_bin")
        tf_array = self.to_float32(tf, self.context_columns(), "V_V_tf")
        return data_array, tf_array, mask_array

    def load_yaml(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def save_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def load_time_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        config = self.load_yaml(TIME_CONFIG)
        start_time = pd.to_datetime(config["start_time"])
        end_time = pd.to_datetime(config["end_time"])
        if end_time < start_time:
            raise ValueError(f"end_time earlier than start_time in {TIME_CONFIG}")
        return start_time, end_time

    def time_points(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start_time, self.end_time, freq=TIME_FREQ)

    def time_index(self, time_value: pd.Timestamp) -> int:
        time_value = pd.to_datetime(time_value)
        return min((time_value.hour * 60 + time_value.minute) // 5, 287)

    def timeline_index(self, time_value: pd.Timestamp) -> int:
        return int((pd.to_datetime(time_value) - self.start_time).total_seconds() // 300)

    def excel_column_index(self, reference: str) -> int:
        index = 0
        for char in reference:
            index = index * 26 + ord(char) - ord("A") + 1
        return index - 1

    def load_shared_strings(self, archive: ZipFile) -> list[str]:
        if "xl/sharedStrings.xml" not in archive.namelist():
            return []
        strings: list[str] = []
        with archive.open("xl/sharedStrings.xml") as file:
            for _, element in ET.iterparse(file, events=("end",)):
                if element.tag == XLSX_NS + "si":
                    strings.append("".join(text.text or "" for text in element.iter(XLSX_NS + "t")))
                    element.clear()
        return strings

    def first_sheet_path(self, archive: ZipFile) -> str:
        sheets = sorted(
            name
            for name in archive.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if not sheets:
            raise ValueError("xlsx file has no worksheet")
        return sheets[0]

    def read_cell(self, cell: ET.Element, shared_strings: list[str]):
        cell_type = cell.attrib.get("t")
        if cell_type == "inlineStr":
            return "".join(text.text or "" for text in cell.iter(XLSX_NS + "t"))
        value = cell.find(XLSX_NS + "v")
        if value is None:
            return None
        text = value.text
        if cell_type == "s":
            return shared_strings[int(text)]
        if cell_type == "b":
            return text == "1"
        return text

    def iter_xlsx_rows(self, path: Path):
        with ZipFile(path) as archive:
            shared_strings = self.load_shared_strings(archive)
            sheet_path = self.first_sheet_path(archive)
            with archive.open(sheet_path) as file:
                for _, row_element in ET.iterparse(file, events=("end",)):
                    if row_element.tag != XLSX_NS + "row":
                        continue
                    row = []
                    next_index = 0
                    for cell in row_element.findall(XLSX_NS + "c"):
                        match = re.match(r"([A-Z]+)", cell.attrib.get("r", ""))
                        cell_index = self.excel_column_index(match.group(1)) if match else next_index
                        while len(row) <= cell_index:
                            row.append(None)
                        row[cell_index] = self.read_cell(cell, shared_strings)
                        next_index = cell_index + 1
                    row_element.clear()
                    yield row

    def read_excel_fast(self, path: Path, columns: list[str]) -> pd.DataFrame:
        rows = self.iter_xlsx_rows(path)
        header = next(rows)
        header_index = {str(column).strip(): index for index, column in enumerate(header) if column is not None}
        missing = [column for column in columns if column not in header_index]
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")
        column_indices = [header_index[column] for column in columns]
        time_column_index = header_index["time"]

        def selected_rows():
            for row in rows:
                if time_column_index >= len(row) or row[time_column_index] is None:
                    continue
                row_time = pd.to_datetime(row[time_column_index], errors="coerce")
                if pd.isna(row_time) or row_time < self.start_time or row_time > self.end_time:
                    continue
                yield [row[index] if index < len(row) else None for index in column_indices]

        return pd.DataFrame(selected_rows(), columns=columns)

    def load_wim_excel(self, path: Path) -> pd.DataFrame:
        header = next(self.iter_xlsx_rows(path))
        header_set = {str(column).strip() for column in header if column is not None}
        columns = REQUIRED_COLUMNS + [column for column in OPTIONAL_COLUMNS if column in header_set]
        data = self.read_excel_fast(path, columns)
        if data.empty:
            data["vehicle_class"] = pd.Series(dtype="object")
            data["speed_mps"] = pd.Series(dtype="float64")
            return data

        data["time"] = pd.to_datetime(data["time"])
        for column in ["lane_id", "veh_type", "axle_num", "grossload", "speed"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        for column in [column for column in AXLE_COLUMNS if column in data.columns]:
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)

        data = data.dropna(subset=["time", "lane_id", "veh_type", "axle_num", "grossload", "speed"]).copy()
        data["lane_id"] = data["lane_id"].astype(int)
        data["vehicle_class"] = [self.classify_vehicle(row.veh_type, row.axle_num) for row in data.itertuples()]
        data = data[data["vehicle_class"].notna()].copy()
        data["speed_mps"] = data["speed"] / 3.6
        return data

    def classify_vehicle(self, veh_type, axle_num) -> str | None:
        axle = pd.to_numeric(axle_num, errors="coerce")
        if pd.isna(axle):
            return None
        axle = int(axle)
        if axle == 2:
            code = str(int(veh_type)) if pd.notna(veh_type) else ""
            return "2C" if code in TWO_AXLE_PASSENGER_CODES else "2F"
        if 3 <= axle <= 6:
            return str(axle)
        return None

    def load_context(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        weather = preprocess_weather()
        weather = weather.rename(
            columns={
                "uptime": "weather_time",
                "weatid": "weather_id",
                "temp": "temperature",
                "humidity": "humidity",
                "winpid": "wind_id",
                "aqi": "aqi",
            }
        )
        weather["weather_time"] = pd.to_datetime(weather["weather_time"])
        weather = weather.sort_values("weather_time")

        rest = pd.read_excel(RESTDAY_FILE)[["date", "restday"]]
        rest["date"] = pd.to_datetime(rest["date"]).dt.date
        return weather, rest

    def add_context(self, frame: pd.DataFrame, time_column: str) -> pd.DataFrame:
        result = pd.merge_asof(
            frame.sort_values(time_column),
            self.weather,
            left_on=time_column,
            right_on="weather_time",
            direction="nearest",
        )
        result["date"] = result[time_column].dt.date
        result = result.merge(self.rest, on="date", how="left")
        result["restday"] = result["restday"].fillna(1)
        return result

    def context_columns(self) -> list[str]:
        return ["time_index", "restday", "weather_id", "temperature", "humidity", "wind_id", "aqi"]

    def to_float32(self, frame: pd.DataFrame, columns: list[str], name: str) -> np.ndarray:
        values = frame[columns].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any():
            missing = values.columns[values.isna().any()].tolist()
            raise ValueError(f"{name} contains NaN in columns: {missing}")
        return values.to_numpy(dtype=np.float32, copy=True)

    def build_tf(self, times: pd.DatetimeIndex) -> pd.DataFrame:
        tf = pd.DataFrame({"time_bin": times})
        tf["time_index"] = tf["time_bin"].map(self.time_index)
        return self.add_context(tf, "time_bin")

    def save_global_arrays(self, family_name: str, data_array: np.ndarray, tf_array: np.ndarray | None, debug_limit: int) -> None:
        if tf_array is None:
            raise ValueError(f"Missing {family_name}_tf array")
        npy_dir = OUTPUT_ROOT / "npy" / family_name
        debug_dir = OUTPUT_ROOT / "debug_npy" / family_name
        npy_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_dir / f"{family_name}_data.npy", data_array)
        np.save(npy_dir / f"{family_name}_tf.npy", tf_array)
        np.save(debug_dir / f"{family_name}_data.npy", data_array[:debug_limit])
        np.save(debug_dir / f"{family_name}_tf.npy", tf_array[:debug_limit])

    def save_family_arrays(self, family_name: str, bridge_code: str, arrays: dict[str | int, np.ndarray], debug_limit: int) -> None:
        family_dir = OUTPUT_ROOT / "npy" / family_name / bridge_code
        debug_dir = OUTPUT_ROOT / "debug_npy" / family_name / bridge_code
        family_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
        for key, array in arrays.items():
            np.save(family_dir / f"{family_name}_{key}.npy", array)
            np.save(debug_dir / f"{family_name}_{key}.npy", array[:debug_limit])

    def save_archive(self, family_name: str, filename: str, arrays: dict[str, np.ndarray], debug_limit: int) -> None:
        output_dir = OUTPUT_ROOT / "npy" / family_name
        debug_dir = OUTPUT_ROOT / "debug_npy" / family_name
        output_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / filename, **arrays)
        row_count = len(arrays["target"])
        debug_index = np.linspace(0, row_count - 1, min(debug_limit, row_count), dtype=np.int64)
        np.savez_compressed(
            debug_dir / filename,
            **{
                key: value if np.asarray(value).ndim == 0 else value[debug_index]
                for key, value in arrays.items()
            },
        )

    def build_spatial_lookup(self) -> dict[str, np.ndarray]:
        locations = self.bridge_metadata["location"].str.split(",", expand=True).astype(float)
        longitude = np.deg2rad(locations[0].to_numpy())
        latitude = np.deg2rad(locations[1].to_numpy())
        longitude0 = longitude.mean()
        latitude0 = latitude.mean()
        x = 6_371_000.0 * np.cos(latitude0) * (longitude - longitude0)
        y = 6_371_000.0 * (latitude - latitude0)
        code_column = "设施编码" if "设施编码" in self.bridge_metadata.columns else "编码"
        result = {}
        for position, (_, row) in enumerate(self.bridge_metadata.iterrows()):
            code = str(int(row[code_column]))
            result[code] = np.asarray([
                row["总车道数"], row["建成年份编码"], row["道路等级编码"], x[position], y[position]
            ], dtype=np.float32)
        return result

    def add_bridge_context(self, frame: pd.DataFrame, bridge_code: str) -> pd.DataFrame:
        result = self.add_context(frame, "time")
        result["timeline_index"] = result["time"].map(self.timeline_index)
        result["bridge_index"] = int(self.bridge_indices[bridge_code])
        for column, value in zip(
            ["total_lanes", "construction_year", "road_class", "location_x", "location_y"],
            self.spatial_lookup[bridge_code],
        ):
            result[column] = value
        return result

    def save_metadata(self, global_codes: list[str]) -> None:
        self.save_json(OUTPUT_ROOT / "metadata.json", {
            "bridge_order": global_codes,
            "canonical_lanes": list(CANONICAL_LANES),
            "temporal_context": self.context_columns(),
            "spatial_context": [
                "total_lanes", "construction_year", "road_class", "location_x", "location_y"
            ],
        })

    def bridge_code_from_file(self, path: Path) -> str:
        match = re.search(r"(?:^|_)hsd_(\d+)(?:_|$)", path.stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        match = re.search(r"(\d{5,})", path.stem)
        return match.group(1) if match else path.stem

    def load_bridge_metadata(self) -> tuple[list[str], dict[str, str]]:
        frame = pd.read_excel(BRIDGE_INFO)
        code_column = "设施编码" if "设施编码" in frame.columns else "编码"
        index_column = "编码" if "编码" in frame.columns else code_column
        ordered_codes = frame[code_column].dropna().astype(int).astype(str).tolist()
        bridge_indices: dict[str, str] = {}
        for _, row in frame.iterrows():
            code_value = row.get(code_column)
            if pd.isna(code_value):
                continue
            code = str(int(code_value))
            index_value = row.get(index_column)
            bridge_indices[code] = str(int(index_value)) if pd.notna(index_value) else code
        return ordered_codes, bridge_indices

    def ordered_excel_items(self) -> list[tuple[str, Path]]:
        excel_files = sorted(EXCEL_DIR.glob("*.xlsx"))
        if not excel_files:
            raise FileNotFoundError(f"No xlsx files found in {EXCEL_DIR}")
        expected_codes = set(self.bridge_order)
        by_code = {}
        for path in excel_files:
            code = self.bridge_code_from_file(path)
            if code not in expected_codes:
                continue
            if code in by_code:
                raise ValueError(f"Multiple xlsx files found for bridge {code}: {by_code[code].name}, {path.name}")
            by_code[code] = path
        missing = [code for code in self.bridge_order if code not in by_code]
        if missing:
            raise FileNotFoundError(f"Missing xlsx files for bridges: {missing}")
        return [(code, by_code[code]) for code in self.bridge_order]


class V_W_profress(V_V_profress):
    def run(self) -> None:
        combined = {vehicle_type: [] for vehicle_type in VEHICLE_TYPES}
        for bridge_code, excel_path in tqdm(self.ordered_excel_items(), desc="Processing V_W bridges", unit="bridge"):
            data = self.load_wim_excel(excel_path)
            data = add_canonical_lane(data, canonical_lane_mapping(self.bridge_rows[bridge_code]))
            arrays = self.build_v_w(data, bridge_code)
            for vehicle_type in VEHICLE_TYPES:
                combined[vehicle_type].append(arrays[vehicle_type])
        for vehicle_type, parts in combined.items():
            merged = {
                key: np.concatenate([part[key] for part in parts], axis=0)
                for key in ("target", "context", "bridge_index", "timeline_index", "standard_lane")
            }
            order = np.lexsort((merged["bridge_index"], merged["timeline_index"]))
            merged = {key: value[order] for key, value in merged.items()}
            merged["num_time_steps"] = np.asarray(len(self.time_points()), dtype=np.int64)
            self.save_archive("V_W", f"V_W_{vehicle_type}.npz", merged, DEBUG_V_W_LIMIT)
        self.save_metadata(self.bridge_order)

    def build_v_w(self, data: pd.DataFrame, bridge_code: str) -> dict[str, dict[str, np.ndarray]]:
        data = self.filter_v_w_outliers(data)
        flow = interval_lane_flow(data)
        data = data.copy()
        data["time_bin"] = data["time"].dt.floor(TIME_FREQ)
        data = data.merge(flow[["time_bin", "standard_lane", "minute_flow"]], on=["time_bin", "standard_lane"])
        arrays = {}
        for vehicle_type in tqdm(VEHICLE_TYPES, desc="Building V_W", unit="class"):
            axle_count = 2 if vehicle_type in {"2C", "2F"} else int(vehicle_type)
            value_columns = AXLE_COLUMNS[:axle_count]
            frame = data.loc[
                data["vehicle_class"] == vehicle_type,
                ["time", "standard_lane", "speed_mps", "minute_flow", *value_columns],
            ].copy()
            frame = self.filter_v_w_axles(frame, value_columns)
            frame["time_index"] = frame["time"].map(self.time_index)
            frame = self.add_bridge_context(frame, bridge_code)
            context_columns = self.context_columns() + [
                "total_lanes", "construction_year", "road_class", "location_x", "location_y",
                "standard_lane", "speed_mps", "minute_flow",
            ]
            arrays[vehicle_type] = {
                "target": self.to_float32(frame, value_columns, f"V_W_{vehicle_type}_target"),
                "context": self.to_float32(frame, context_columns, f"V_W_{vehicle_type}_context"),
                "bridge_index": frame["bridge_index"].to_numpy(dtype=np.int64),
                "timeline_index": frame["timeline_index"].to_numpy(dtype=np.int64),
                "standard_lane": frame["standard_lane"].to_numpy(dtype=np.int64),
            }
        return arrays


class V_S_profress(V_V_profress):
    def __init__(self):
        super().__init__()
        self.v_c_index = {vehicle_type: index for index, vehicle_type in enumerate(VEHICLE_TYPES)}

    def run(self) -> None:
        parts = []
        for bridge_code, excel_path in tqdm(self.ordered_excel_items(), desc="Processing V_S bridges", unit="bridge"):
            data = self.load_wim_excel(excel_path)
            data = add_canonical_lane(data, canonical_lane_mapping(self.bridge_rows[bridge_code]))
            parts.append(self.build_v_s(data, bridge_code))
        merged = {
            key: np.concatenate([part[key] for part in parts], axis=0)
            for key in ("target", "context", "bridge_index", "timeline_index", "standard_lane")
        }
        order = np.lexsort((merged["bridge_index"], merged["timeline_index"]))
        merged = {key: value[order] for key, value in merged.items()}
        merged["num_time_steps"] = np.asarray(len(self.time_points()), dtype=np.int64)
        self.save_archive("V_S", "V_S.npz", merged, DEBUG_V_S_LIMIT)
        self.save_metadata(self.bridge_order)

    def build_v_s(self, data: pd.DataFrame, bridge_code: str) -> dict[str, np.ndarray]:
        data = self.filter_v_w_outliers(data)
        flow = interval_lane_flow(data)
        frame = spacing_records(data, self.v_c_index)
        frame["time_bin"] = frame["time"].dt.floor(TIME_FREQ)
        frame = frame.merge(flow[["time_bin", "standard_lane", "minute_flow"]], on=["time_bin", "standard_lane"])
        frame["time_index"] = frame["time"].map(self.time_index)
        frame["self_grossload"] = frame["grossload"]
        frame = self.add_bridge_context(frame, bridge_code)
        frame = self.filter_v_s_outliers(frame)
        context_columns = self.context_columns() + [
            "total_lanes", "construction_year", "road_class", "location_x", "location_y",
            "standard_lane", "speed_mps", "minute_flow", "prev_vehicle_class_idx",
            "self_vehicle_class_idx", "self_grossload",
        ]
        return {
            "target": self.to_float32(frame, ["distance"], "V_S_target"),
            "context": self.to_float32(frame, context_columns, "V_S_context"),
            "bridge_index": frame["bridge_index"].to_numpy(dtype=np.int64),
            "timeline_index": frame["timeline_index"].to_numpy(dtype=np.int64),
            "standard_lane": frame["standard_lane"].to_numpy(dtype=np.int64),
        }


class V_C_profress(V_V_profress):
    def run(self) -> None:
        v_c_by_bridge: dict[str, np.ndarray] = {}
        v_c_v_by_bridge: dict[str, np.ndarray] = {}
        shared_tf: np.ndarray | None = None
        ordered_items = self.ordered_excel_items()

        for bridge_code, excel_path in tqdm(ordered_items, desc="Processing V_C bridges", unit="bridge"):
            data = self.load_wim_excel(excel_path)
            data = add_canonical_lane(data, canonical_lane_mapping(self.bridge_rows[bridge_code]))
            v_c_data, v_c_tf, v_c_v = self.build_v_c(data)
            v_c_by_bridge[bridge_code] = v_c_data
            v_c_v_by_bridge[bridge_code] = v_c_v
            shared_tf = v_c_tf if shared_tf is None else shared_tf

        if BUILD_GLOBAL:
            global_codes = [code for code in self.bridge_order if code in v_c_by_bridge]
            global_codes.extend(sorted(code for code in v_c_by_bridge if code not in global_codes))
            v_c_global = np.stack([v_c_by_bridge[code] for code in global_codes], axis=1)
            v_c_v_global = np.stack([v_c_v_by_bridge[code] for code in global_codes], axis=1)
            self.save_global_arrays("V_C", v_c_global, shared_tf, DEBUG_V_C_LIMIT)
            mask_global = v_c_global[..., -1:] > 0
            np.save(OUTPUT_ROOT / "npy" / "V_C" / "V_C_mask.npy", mask_global)
            np.save(OUTPUT_ROOT / "debug_npy" / "V_C" / "V_C_mask.npy", mask_global[:DEBUG_V_C_LIMIT])
            np.save(OUTPUT_ROOT / "npy" / "V_C" / "V_C_v.npy", v_c_v_global)
            np.save(OUTPUT_ROOT / "debug_npy" / "V_C" / "V_C_v.npy", v_c_v_global[:DEBUG_V_C_LIMIT])
            self.save_metadata(global_codes)

    def build_v_c(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = self.time_points()
        if data.empty:
            tf = self.build_tf(times)
            return (
                np.zeros((len(times), 6, 7), dtype=np.float32),
                self.to_float32(tf, self.context_columns(), "V_C_tf"),
                np.zeros((len(times), 1), dtype=np.float32),
            )

        speed = self.filter_v_v_outliers(data)[["time", "speed_mps"]].copy()
        speed["time_bin"] = speed["time"].dt.floor(TIME_FREQ)
        speed = speed.groupby("time_bin")["speed_mps"].mean().reindex(times).ffill().bfill()
        speed_array = speed.to_numpy(dtype=np.float32).reshape(len(times), 1)

        frame = data[["time", "standard_lane", "vehicle_class"]].copy()
        frame["time_bin"] = frame["time"].dt.floor(TIME_FREQ)
        grouped = frame.groupby(["time_bin", "standard_lane", "vehicle_class"]).size().unstack(fill_value=0)
        for vehicle_type in VEHICLE_TYPES:
            if vehicle_type not in grouped.columns:
                grouped[vehicle_type] = 0
        grouped = grouped[VEHICLE_TYPES]
        grouped["flow"] = grouped.sum(axis=1)

        full_index = pd.MultiIndex.from_product([times, LANES], names=["time_bin", "standard_lane"])
        result = grouped.reindex(full_index, fill_value=0).reset_index()
        result["time_index"] = result["time_bin"].map(self.time_index)
        result = self.add_context(result, "time_bin")
        result = result.sort_values(["time_bin", "standard_lane"])

        data_array = self.to_float32(result, VC_COLUMNS, "V_C_data").reshape(len(times), 6, 7)
        tf = result[["time_bin"] + self.context_columns()].drop_duplicates("time_bin").sort_values("time_bin")
        tf_array = self.to_float32(tf, self.context_columns(), "V_C_tf")
        return data_array, tf_array, speed_array


def build_all_datasets() -> None:
    v_c_processor = V_C_profress()
    v_w_processor = V_W_profress()
    v_s_processor = V_S_profress()
    v_v_by_bridge = {}
    v_v_masks_by_bridge = {}
    v_c_by_bridge = {}
    v_c_v_by_bridge = {}
    v_w_by_class = {vehicle_type: [] for vehicle_type in VEHICLE_TYPES}
    v_s_parts = []
    shared_tf = None

    ordered_items = v_c_processor.ordered_excel_items()
    for bridge_code, excel_path in tqdm(ordered_items, desc="Processing all datasets", unit="bridge"):
        data = v_c_processor.load_wim_excel(excel_path)
        data = add_canonical_lane(data, canonical_lane_mapping(v_c_processor.bridge_rows[bridge_code]))

        v_c_data, v_c_tf, v_c_v = v_c_processor.build_v_c(data)
        v_v_data, _, v_v_mask = v_c_processor.build_v_v(data)
        v_w_arrays = v_w_processor.build_v_w(data, bridge_code)
        v_s_arrays = v_s_processor.build_v_s(data, bridge_code)
        v_c_by_bridge[bridge_code] = v_c_data
        v_c_v_by_bridge[bridge_code] = v_c_v
        v_v_by_bridge[bridge_code] = v_v_data
        v_v_masks_by_bridge[bridge_code] = v_v_mask
        v_s_parts.append(v_s_arrays)
        for vehicle_type in VEHICLE_TYPES:
            v_w_by_class[vehicle_type].append(v_w_arrays[vehicle_type])
        shared_tf = v_c_tf if shared_tf is None else shared_tf

    bridge_codes = [code for code in v_c_processor.bridge_order if code in v_c_by_bridge]
    v_c_global = np.stack([v_c_by_bridge[code] for code in bridge_codes], axis=1)
    v_c_v_global = np.stack([v_c_v_by_bridge[code] for code in bridge_codes], axis=1)
    v_v_global = np.stack([v_v_by_bridge[code] for code in bridge_codes], axis=1)
    v_v_mask = np.stack([v_v_masks_by_bridge[code] for code in bridge_codes], axis=1)
    v_c_mask = v_c_global[..., -1:] > 0
    v_c_processor.save_global_arrays("V_C", v_c_global, shared_tf, DEBUG_V_C_LIMIT)
    v_c_processor.save_global_arrays("V_V", v_v_global, shared_tf, DEBUG_V_V_LIMIT)
    np.save(OUTPUT_ROOT / "npy" / "V_C" / "V_C_mask.npy", v_c_mask)
    np.save(OUTPUT_ROOT / "debug_npy" / "V_C" / "V_C_mask.npy", v_c_mask[:DEBUG_V_C_LIMIT])
    np.save(OUTPUT_ROOT / "npy" / "V_C" / "V_C_v.npy", v_c_v_global)
    np.save(OUTPUT_ROOT / "debug_npy" / "V_C" / "V_C_v.npy", v_c_v_global[:DEBUG_V_C_LIMIT])
    np.save(OUTPUT_ROOT / "npy" / "V_V" / "V_V_mask.npy", v_v_mask)
    np.save(OUTPUT_ROOT / "debug_npy" / "V_V" / "V_V_mask.npy", v_v_mask[:DEBUG_V_V_LIMIT])

    for vehicle_type, parts in v_w_by_class.items():
        merged = {
            key: np.concatenate([part[key] for part in parts], axis=0)
            for key in ("target", "context", "bridge_index", "timeline_index", "standard_lane")
        }
        order = np.lexsort((merged["bridge_index"], merged["timeline_index"]))
        merged = {key: value[order] for key, value in merged.items()}
        merged["num_time_steps"] = np.asarray(len(v_c_processor.time_points()), dtype=np.int64)
        v_c_processor.save_archive("V_W", f"V_W_{vehicle_type}.npz", merged, DEBUG_V_W_LIMIT)

    v_s_merged = {
        key: np.concatenate([part[key] for part in v_s_parts], axis=0)
        for key in ("target", "context", "bridge_index", "timeline_index", "standard_lane")
    }
    order = np.lexsort((v_s_merged["bridge_index"], v_s_merged["timeline_index"]))
    v_s_merged = {key: value[order] for key, value in v_s_merged.items()}
    v_s_merged["num_time_steps"] = np.asarray(len(v_c_processor.time_points()), dtype=np.int64)
    v_c_processor.save_archive("V_S", "V_S.npz", v_s_merged, DEBUG_V_S_LIMIT)
    v_c_processor.save_metadata(bridge_codes)


if __name__ == "__main__":
    build_all_datasets()
