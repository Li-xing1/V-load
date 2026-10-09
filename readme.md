# V-load: Probabilistic Vehicle Load Modeling Across Multiple Bridges

This project provides data preprocessing, dataset loading, probabilistic model training, sampling, and distribution evaluation for vehicle observations across multiple bridges. The upload version contains source code and configuration files only; raw data, preprocessed datasets, model checkpoints, and experiment outputs are not included.

## Models and Project Structure

| Module | Purpose |
| --- | --- |
| `V_C` | Vehicle class and class-specific traffic flow modeling |
| `V_V` | Bivariate joint probabilistic modeling of vehicle speed and traffic flow |
| `V_W` | Conditional axle-weight modeling with separate models for each vehicle class |
| `V_S` | Conditional vehicle-spacing modeling |

```text
V-load/
├── args/                         # General, dataset, model, and plotting YAML configurations
├── dataset/                      # Data loading, normalization, sampling, and graph processing
├── models/                       # Four probabilistic models and shared components
├── utils/                        # Training, experiment logging, plotting, and evaluation
├── data/
│   ├── weather.py                # Historical weather retrieval and Excel export
│   ├── encode_weather_categories.py  # Weather category encoding and mapping export
│   └── preprocess.py             # Training and debug dataset generation
├── Train_main.py                 # Training and evaluation of enabled models
├── Train_V_C.py
├── Train_V_V.py
├── Train_V_W.py
├── Train_V_S.py
├── request.txt                   # Python dependencies
└── readme.md
```

## Environment and Installation

The syntax, module import, and command-line checks performed before upload used Python 3.9.18. `request.txt` lists the dependencies used by the code, pinned to the versions installed in the validation environment, including TensorBoard. It is not a full Conda environment export. Run the following commands from the project root after cloning.

```bash
git clone --branch 上传版本 --single-branch git@github.com:Li-xing1/V-load.git
cd V-load
python -m venv .venv
```

Activate the environment and install the dependencies:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r request.txt
```

```bash
# Linux / macOS
source .venv/bin/activate
python -m pip install -r request.txt
```

`args/args.yaml` defaults to `cuda:0`. If no GPU is available, change this setting to `cpu`, or pass `--device cpu` to a single-model training script. GPU training requires a CUDA-enabled PyTorch installation compatible with the local GPU driver. Upload validation used CPU-only PyTorch; full model training was not performed.

## Data Preparation

Datasets are not distributed with the code and must be supplied separately. The default layout is:

```text
data/SZ/
├── Excel/                       # Per-vehicle observations for each bridge (.xlsx)
├── Weather.xlsx                 # Weather observations
├── restday.xlsx                 # Must contain at least date and restday columns
└── data/
    ├── bridge_information.xlsx  # Bridge identifiers, lane mappings, and spatial attributes
    ├── G_distance.npy           # Bridge distance matrix, used by V_C / V_V
    ├── G_duration.npy           # Bridge travel-time matrix, used by V_C / V_V
    ├── npy/                    # Generated preprocessed datasets
    └── debug_npy/              # Generated debug subsets
```

Per-vehicle records must contain at least `time`, `lane_id`, `veh_type`, `axle_num`, `grossload`, and `speed`. Axle-weight models also require the corresponding axle-weight columns, such as `axle1`. Preprocessing attempts to read axle-spacing columns such as `axle_dis1`. The facility identifier in each filename must match the bridge metadata, for example `hsd_<facility_code>.xlsx`. Metadata must also include lane mappings, `location`, and the original column names `总车道数` (total lane count), `建成年份编码` (construction-year code), and `道路等级编码` (road-class code). These Chinese column names are retained because the code reads them literally. Refer to `data/preprocess.py` and `dataset/common.py` for the exact schema.

### Optional: Retrieve and Encode Weather Data

`data/weather.py` calls the K780 weather API and requires valid API credentials and a network connection. The upload version reads credentials from environment variables instead of hardcoding them. Do not write credentials into the source code or commit them to Git.

```powershell
$env:K780_APPKEY = 'your APPKEY'
$env:K780_SIGN = 'your SIGN'
python data/weather.py --start-date 20260714 --end-date 20260912 --wea-id 169
```

Weather retrieval writes its output to `data/jsq/`, while preprocessing reads `data/SZ/Weather.xlsx`. Create the destination parent directory first, then encode the retrieved data into the preprocessing directory:

```bash
python data/encode_weather_categories.py --input data/jsq/Weather.xlsx --output data/SZ/Weather.xlsx --mapping-output data/SZ/Weather_category_mapping.json
```

By default, encoding replaces `weatid` and `winpid` with category IDs starting at 0. Omitting `--output` overwrites the input file. Use `--append-columns` to retain the original columns and append encoded columns instead. Existing weather data can be supplied directly as `data/SZ/Weather.xlsx`; `preprocess.py` also encodes weather categories and saves the mappings.

### Generate Training Datasets

Check the time range, bridge nodes, lanes, and dataset paths in `args/dataset_args.yaml`, then run:

```bash
python data/preprocess.py
```

Preprocessing uses 5-minute intervals by default. It generates `.npy` files for `V_C` and `V_V`, class-specific `V_W_<vehicle_class>.npz` files, `V_S.npz`, and debug subsets. Bridge distance and travel-time matrices must be prepared separately; this script does not generate them.

## Training and Evaluation

### Unified Entry Point

```bash
python Train_main.py
```

The unified entry point reads configuration files directly and does not expose the command-line arguments of the single-model scripts. Select the models to train using `V_C_train`, `V_V_train`, `V_W_train`, and `V_S_train` in `args/args.yaml`. The current configuration enables only `V_W` and `V_S`; each model defaults to 100 training epochs. Sampling and evaluation use the corresponding best checkpoints after training.

### Single-Model Entry Points

```bash
python Train_V_C.py --device cpu --work-path vc_demo
python Train_V_V.py --device cpu --work-path vv_demo
python Train_V_W.py --vehicle-class 2C 2F --device cpu --work-path vw_demo
python Train_V_S.py --device cpu --work-path vs_demo
```

Single-model scripts support `--args-dir`, `--dataset-args`, `--work-path`, `--device`, and `--debug`. If `--vehicle-class` is omitted, `V_W` discovers available vehicle classes from the dataset files. Use commands such as `python Train_V_W.py --help` to inspect additional options.

```bash
python Train_V_S.py --debug --device cpu --work-path debug_vs
```

`--debug` uses `debug_npy` and reduces training to 2 epochs. Data preprocessing is still required; debug mode does not allow training without datasets.

### Configuration Files

| File | Purpose |
| --- | --- |
| `args/args.yaml` | Device, random seed, experiment name, enabled models, and debug settings |
| `args/dataset_args.yaml` | Dataset paths, time range, splits, bridge nodes, and lane definitions |
| `args/V_C.yaml` and corresponding model files | Model architecture and training hyperparameters |
| `args/figure.yaml` | Plot styles and output settings |

Dataset partitioning follows `split_order`; do not assume the three values represent training, validation, and test sets in that order. The current order is `[Test, Val, Train]`, with relative weights `[49, 5, 7]`. Update configurations and input data together when changing vehicle classes, bridge nodes, time ranges, or lane definitions.

## Outputs and Notes

Experiment outputs are stored in `exp/<experiment_name>/`, including `resolved_config.json`, `log/`, `checkpoint/`, `plot/figure/`, and `plot/data/`. When `save_name` is `time`, the experiment name is generated from the execution timestamp. The log directory also contains TensorBoard records:

```bash
tensorboard --logdir exp
```

The upload branch contains only the requested source files, the dependency list, this document, and `.gitignore`. Raw data, caches, IDE settings, the original development history, and experiment outputs are excluded. Full training and compatibility with actual input datasets must be verified after the required data has been prepared.
