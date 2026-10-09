# V-load：多桥车辆荷载概率建模

本项目包含多桥车辆数据预处理、数据集加载、概率模型训练、采样和分布评估代码。上传版本只包含源代码和配置，不包含原始数据、预处理结果、训练权重或实验输出。

## 模型与目录

| 模块 | 功能 |
| --- | --- |
| `V_C` | 车辆类别与类别流量建模 |
| `V_V` | 车速与流量的二元联合概率建模 |
| `V_W` | 按车型训练车辆轴重条件概率模型 |
| `V_S` | 车辆间距条件概率建模 |

```text
V-load/
├── args/                         # 通用、数据集、模型与绘图 YAML 配置
├── dataset/                      # 数据加载、标准化、采样及图结构处理
├── models/                       # 四类概率模型及公共组件
├── utils/                        # 训练、实验记录、绘图及分布评估工具
├── data/
│   ├── weather.py                # 历史天气采集及 Excel 导出
│   ├── encode_weather_categories.py  # 天气类别编码及映射保存
│   └── preprocess.py             # 生成训练和调试数据
├── Train_main.py                 # 按配置训练并评估启用的模型
├── Train_V_C.py
├── Train_V_V.py
├── Train_V_W.py
├── Train_V_S.py
├── request.txt                   # Python 依赖清单
└── readme.md
```

## 环境与安装

上传前的语法、模块导入与命令行检查使用 Python 3.9.18。`request.txt` 根据代码实际依赖及本机已安装版本编写，包含 TensorBoard；它不是整个 Conda 环境的导出文件。以下命令须在项目根目录运行。

```bash
git clone --branch 上传版本 --single-branch git@github.com:Li-xing1/V-load.git
cd V-load
python -m venv .venv
```

激活环境后安装依赖：

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

`args/args.yaml` 默认使用 `cuda:0`。无可用 GPU 时，请将该配置改为 `cpu`，或在单模型命令中传入 `--device cpu`。GPU 训练需要与本机驱动兼容的 CUDA 版 PyTorch。本次上传检查使用 CPU 版 PyTorch，没有执行完整训练。

## 数据准备

数据不随代码公开，需要自行提供。默认布局如下：

```text
data/SZ/
├── Excel/                       # 各桥逐车记录的 .xlsx 文件
├── Weather.xlsx                 # 天气数据
├── restday.xlsx                 # 至少含 date、restday 列
└── data/
    ├── bridge_information.xlsx  # 桥梁编码、车道与空间属性
    ├── G_distance.npy           # 桥梁距离矩阵，V_C / V_V 使用
    ├── G_duration.npy           # 桥梁通行时长矩阵，V_C / V_V 使用
    ├── npy/                    # 预处理后生成
    └── debug_npy/              # 预处理后生成的调试子集
```

逐车记录至少包含 `time`、`lane_id`、`veh_type`、`axle_num`、`grossload`、`speed`。轴重模型还需要相应的 `axle1` 等轴重列；预处理会尝试读取 `axle_dis1` 等轴距列。文件名中的设施编码应与桥梁元数据对应，例如 `hsd_设施编码.xlsx`。元数据还需要车道映射、`location`、`总车道数`、`建成年份编码`、`道路等级编码` 等字段，具体要求以 `data/preprocess.py` 和 `dataset/common.py` 为准。

### 可选：获取和编码天气

`data/weather.py` 调用 K780 天气接口，需要有效 API 凭证和网络连接。上传版本已移除硬编码凭证，改为环境变量读取；不要将凭证写回代码或提交到 Git。

```powershell
$env:K780_APPKEY = '你的 APPKEY'
$env:K780_SIGN = '你的 SIGN'
python data/weather.py --start-date 20260714 --end-date 20260912 --wea-id 169
```

采集结果写入 `data/jsq/`，而预处理读取 `data/SZ/Weather.xlsx`。提前创建目标父目录后，可将采集结果编码到预处理目录：

```bash
python data/encode_weather_categories.py --input data/jsq/Weather.xlsx --output data/SZ/Weather.xlsx --mapping-output data/SZ/Weather_category_mapping.json
```

编码默认替换 `weatid`、`winpid`，编号从 0 开始。不指定 `--output` 会覆盖输入文件；`--append-columns` 可保留原列并添加编码列。已有天气数据可直接提供到 `data/SZ/Weather.xlsx`，`preprocess.py` 也会进行类别编码并保存映射。

### 生成训练数据

先核对 `args/dataset_args.yaml` 中的时间范围、桥梁节点、车道及数据目录，再运行：

```bash
python data/preprocess.py
```

预处理默认以 5 分钟间隔组织数据，生成 `V_C`、`V_V` 的 `.npy` 文件，按车型生成 `V_W_车型.npz`，生成 `V_S.npz`，同时写出调试子集。桥梁距离和通行时长矩阵需另行准备，以上脚本不会生成这两个文件。

## 训练与评估

### 统一入口

```bash
python Train_main.py
```

统一入口直接读取配置，不提供单模型脚本的命令行参数。通过 `args/args.yaml` 中的 `V_C_train`、`V_V_train`、`V_W_train`、`V_S_train` 选择训练模块。当前配置仅开启 `V_W` 与 `V_S`，各模型默认训练 100 个 epoch。运行后使用相应最佳权重进行采样和评估。

### 单模型入口

```bash
python Train_V_C.py --device cpu --work-path vc_demo
python Train_V_V.py --device cpu --work-path vv_demo
python Train_V_W.py --vehicle-class 2C 2F --device cpu --work-path vw_demo
python Train_V_S.py --device cpu --work-path vs_demo
```

单模型脚本支持 `--args-dir`、`--dataset-args`、`--work-path`、`--device` 和 `--debug`。`V_W` 不指定 `--vehicle-class` 时会根据数据文件发现车型。额外参数可通过 `python Train_V_W.py --help` 等命令查看。

```bash
python Train_V_S.py --debug --device cpu --work-path debug_vs
```

`--debug` 使用 `debug_npy` 并将训练轮数设为 2，仍须先完成数据预处理；缺少数据时不能直接训练。

### 配置说明

| 文件 | 作用 |
| --- | --- |
| `args/args.yaml` | 设备、随机种子、实验名、启用模型与调试开关 |
| `args/dataset_args.yaml` | 数据路径、时间范围、切分、桥梁及车道定义 |
| `args/V_C.yaml` 等 | 各模型结构和训练超参数 |
| `args/figure.yaml` | 绘图样式和输出参数 |

数据切分遵循 `split_order`，不能直接假定三个数依次表示训练、验证、测试。当前顺序为 `[Test, Val, Train]`，比例权重为 `[49, 5, 7]`。车型、桥梁、时间范围或车道定义变化后，应同步调整配置和输入数据。

## 输出与注意事项

实验结果写入 `exp/<实验名>/`，包括 `resolved_config.json`、`log/`、`checkpoint/`、`plot/figure/` 和 `plot/data/`。`save_name` 为 `time` 时使用运行时间作为实验名。日志目录还包含 TensorBoard 记录：

```bash
tensorboard --logdir exp
```

本上传分支只发布用户指定的代码范围，额外包含依赖清单、本文档和 `.gitignore`。原始数据、缓存、IDE 配置、历史提交和实验输出均不包含在上传包中。完整训练及数据格式兼容性需在准备好实际数据后验证。
