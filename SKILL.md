---
name: binary-text-classification-training-project
description: Use this skill when Codex needs to create or update a complete Hugging Face Trainer project dedicated to binary text classification, with centralized config, preprocessing, train/valid/test splits, eval_loss best-checkpoint selection, evaluation, prediction, optional FastAPI service, and binary-only validation hooks.
metadata:
  short-description: 二分类文本模型训练工程规范
---

# 二分类文本模型训练工程

使用本 skill 生成或修改二分类文本模型训练代码时，应优先采用完整、可训练、可评估、可预测的 Hugging Face `datasets` + `transformers` 工程结构。该 skill 只面向 binary text classification（二分类文本分类），不面向单文件脚本，也不面向多分类、序列标注、文本匹配、问答、摘要、翻译或其他生成任务。

如果任务类型不是明确的二分类文本分类，应先检查数据和用户需求；无法确认时，不要直接套用本 skill，应切换到更合适的 skill 或向用户确认。

## 分块生成与校验规范

生成或修改项目代码前，必须先检查当前项目，确认 raw 数据格式、文本列、标签列、需要丢弃的列、标签取值和模型名称。

如果当前项目类似购物评论情感二分类示例，默认使用：

- `review` 作为文本输入列。
- `label` 作为二分类标签列。
- `cat` 作为可丢弃列。
- `差评` / `好评` 作为可读标签名称。

代码必须按模块分块生成：

- `architecture`：项目架构、目录和必要文件。
- `config`：`src/configuration/config.py`。
- `preprocess`：`src/process/preprocess.py`。
- `train`：`src/runner/train.py`。
- `evaluate`：`src/runner/evaluate.py`。
- `predict`：`src/runner/predict.py`。
- `web`：可选 Web 服务相关文件。

每完成一个 block 后，运行本 skill 自带校验脚本：

```bash
python3 scripts/check_binary_project.py --project-root project_root --block block_name
```

如果一次完成多个 block，可以在全部完成后运行：

```bash
python3 scripts/check_binary_project.py --project-root project_root --block all
```

校验脚本只做静态结构和关键规则检查，不替代真实运行测试。校验通过后，仍应执行 Python 语法检查；如果依赖和数据可用，应尽量使用少量样本跑通 `preprocess`、`train`、`evaluate` 和 `predict` 的最小流程。

## 项目架构规范

默认生成下面的完整工程化目录结构：

```text
project_root/
├── data/
│   ├── raw/
│   └── processed/
├── checkpoint/
│   ├── best/
│   ├── last/
│   └── labels.txt
├── logs/
└── src/
    ├── configuration/
    │   ├── __init__.py
    │   └── config.py
    ├── process/
    │   ├── __init__.py
    │   └── preprocess.py
    ├── runner/
    │   ├── __init__.py
    │   ├── train.py
    │   ├── evaluate.py
    │   └── predict.py
    ├── web/
    │   ├── __init__.py
    │   ├── app.py
    │   ├── schemas.py
    │   └── service.py
    └── main.py
```

如果用户明确要求简化版教学脚手架，可以使用扁平 `src/` 结构，只保留 `config.py`、`preprocess.py`、`train.py`、`evaluate.py` 和 `predict.py`。除此之外，优先使用上面的分层结构。

## config.py 规范

`src/configuration/config.py` 是二分类训练项目的统一配置中心。所有路径、文件名、模型名、字段名、标签名和核心训练超参数都应集中定义在这里。

推荐配置结构：

```python
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "checkpoint"
BEST_MODEL_DIR = MODEL_DIR / "best"
LAST_CHECKPOINT_DIR = MODEL_DIR / "last"
LOG_DIR = ROOT_DIR / "logs"
TENSORBOARD_LOGGING_DIR = LOG_DIR / "tensorboard"

BINARY_CLASSIFICATION_MODEL_NAME = "google-bert/bert-base-chinese"
TEXT_COLUMN = "review"
LABEL_COLUMN = "label"
UNUSED_COLUMNS = ["cat"]
LABEL_NAMES = ["差评", "好评"]
LABELS_FILE = "labels.txt"

SEQ_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 10
SAVE_STEPS = 50
SEED = 42
```

基本规则：

- 路径必须使用 `pathlib.Path`。
- 不要写本机绝对路径。
- `config.py` 只定义配置，不负责创建目录、预处理、训练、评估、预测或启动服务。
- 标签语义必须保持两个类别。如果 raw 标签是 `0/1`、`n/p` 或其他字符串，应在预处理阶段归一化为 id `0` 和 `1`。
- 最终的 id 到标签名称映射必须保存到 `checkpoint/labels.txt`，供训练、评估、预测和服务复用。
- 模型名称或本地模型路径只写在 `BINARY_CLASSIFICATION_MODEL_NAME` 中，其他模块不要重复硬编码。

## preprocess.py 规范

`src/process/preprocess.py` 只负责数据预处理，不负责训练、评估、预测或 Web 服务。

必须满足以下要求：

- 提供统一入口函数 `def preprocess():`。
- 从 `data/raw` 读取 CSV、TSV、JSON、JSONL、TXT 或 Parquet 等 raw 数据。
- 路径、文件名、字段名和模型名必须从 `configuration.config` 导入。
- 过滤缺失文本、空文本和缺失标签。
- 对文本执行 `strip()`，去除首尾空白。
- 按配置丢弃无用列，例如 `cat`。
- 将 raw 二分类标签转换为整数 id `0` 和 `1`。
- 保存可复用标签映射到 `MODEL_DIR / LABELS_FILE`。
- 如果只提供一个 raw 数据文件，必须切分为：
  - `train`: 80%
  - `valid`: 10%
  - `test`: 10%
- 如果标签列适合分层抽样，应优先使用 `stratify_by_column`，保证三个 split 中类别比例尽量一致。
- 保存包含 `train`、`valid` 和 `test` 的 Hugging Face `DatasetDict` 到 `PROCESSED_DATA_DIR`。
- 使用 `AutoTokenizer.from_pretrained(BINARY_CLASSIFICATION_MODEL_NAME)` 加载 tokenizer。
- 分词时使用 `padding="max_length"`、`max_length=SEQ_LEN` 和 `truncation=True`。
- 为 `Trainer` 生成字段名为 `labels` 的标签列。

`preprocess()` 内部必须保留中文编号步骤注释，说明每一步做什么。

## train.py 规范

`src/runner/train.py` 只负责二分类模型训练，不负责 raw 数据预处理、最终测试集评估、预测或 Web 服务。

必须满足以下要求：

- 提供统一入口函数 `def train():`。
- 从 `PROCESSED_DATA_DIR` 加载预处理后的 `DatasetDict`。
- 使用 `train` split 训练，使用 `valid` split 选择最佳 checkpoint。
- `test` split 只允许在 `evaluate.py` 中做最终评估，不能参与训练阶段的最佳 checkpoint 选择。
- 从 `checkpoint/labels.txt` 读取标签映射。
- 构造 `id2label` 和 `label2id`。
- 使用 `AutoTokenizer` 和 `DataCollatorWithPadding`。
- 使用 `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2, id2label=id2label, label2id=label2id)`。
- 定义 `compute_metrics()`，至少计算 `accuracy` 和 `weighted f1`。
- 通过 `build_training_args()` 创建 `TrainingArguments`。
- `build_training_args()` 必须使用 `inspect.signature()` 兼容不同 `transformers` 版本中的 `eval_strategy` / `evaluation_strategy` 参数差异。
- 默认每个 epoch 评估一次、保存一次 checkpoint。
- 必须设置 `load_best_model_at_end=True`、`metric_for_best_model="eval_loss"` 和 `greater_is_better=False`。
- 使用 `get_last_checkpoint()` 检查断点，并传入 `trainer.train(resume_from_checkpoint=last_checkpoint)`。
- 训练结束后保存最佳模型和 tokenizer 到 `checkpoint/best`。
- 创建 `Trainer` 时必须通过 `build_trainer()` 兼容新版 `processing_class` 和旧版 `tokenizer` 参数。

禁止手写完整 PyTorch epoch 循环、batch 循环、`loss.backward()`、`optimizer.step()`、验证循环、日志记录、checkpoint 保存或最佳模型加载逻辑。

## evaluate.py 规范

`src/runner/evaluate.py` 只负责加载最佳模型做评估，不负责 raw 数据预处理、重新训练、预测或启动服务。

必须满足以下要求：

- 提供统一入口函数 `def evaluate(split: str = "test"):`。
- 从 `checkpoint/best` 加载 tokenizer 和最佳模型。
- 从 `data/processed` 加载预处理后的数据集。
- 默认评估 `test` split。
- 使用 `Trainer.evaluate()` 完成评估。
- 创建评估用 `Trainer` 时必须复用 `build_trainer()`。
- 指标逻辑应和训练阶段保持一致，至少包含 `accuracy` 和 `weighted f1`。
- 打印并返回评估指标。

评估阶段不能重新训练、重新划分数据或读取 raw 数据。

## predict.py 规范

`src/runner/predict.py` 只负责加载训练好的最佳模型并执行推理，不负责预处理、训练、评估或启动 Web 服务。

必须满足以下要求：

- 提供可复用的 `Predictor` 类。
- 优先从 `checkpoint/best` 加载 tokenizer 和模型。
- 设备选择使用 CUDA 优先，否则使用 CPU。
- 加载模型后必须调用 `model.eval()`。
- 推理时必须使用 `torch.no_grad()`。
- 同时支持单条字符串输入和字符串列表输入。
- 对 `logits` 做 `softmax`，选择概率最高的类别，并返回：

```python
{"label": "好评", "confidence": 0.98}
```

- 提供 `--text` 参数，支持单条命令行预测。
- 如果没有传入 `--text`，默认进入交互模式。
- 交互模式启动时只加载一次模型。
- 用户输入 `q`、`quit` 或 `exit` 时退出。
- 用户输入空文本时提示重新输入，不执行预测。

本 skill 是二分类预测逻辑，禁止使用 `model.generate()` 或 Seq2Seq 解码。

## 统一入口规范

`src/main.py` 应通过 action 分发不同任务：

- `preprocess`
- `train`
- `evaluate`
- `predict`
- `serve`：仅当生成 Web 服务模块时提供

推荐入口命令：

```bash
python3 src/main.py preprocess
python3 src/main.py train
python3 src/main.py evaluate
python3 src/main.py predict --text "这条评论很好"
```

## Web 服务规范

只有当用户明确要求服务化，或完整项目要求包含服务入口时，才生成 `src/web/`。

基本规则：

- `web/service.py` 必须复用 `runner.predict.Predictor`。
- 模型应在服务启动时或缓存服务对象中加载一次，不要每次请求都重新加载。
- `web/app.py` 提供简单的 FastAPI 预测接口。
- `web/schemas.py` 定义 Pydantic 请求和响应结构。
- Web 代码不能重新实现分词、模型加载或推理逻辑。

## 注释与兼容性规范

- 每个重要函数或类前必须写 `# 功能：...` 中文功能注释。
- `preprocess()`、`train()`、`evaluate()` 和预测流程中应保留中文编号步骤注释。
- 多行关键参数调用中，重要参数旁边应写简短中文行内注释。
- 避免不必要的 dataclass、factory、registry、callback 系统和自定义 Trainer 类。
- 不要直接写 `Trainer(..., tokenizer=...)` 或 `Trainer(..., processing_class=...)`，必须通过 `build_trainer()` 创建。
- 二分类代码中禁止出现 `AutoModelForSeq2SeqLM`、`Seq2SeqTrainer`、`Seq2SeqTrainingArguments`、`DataCollatorForSeq2Seq` 或 `model.generate()`。

## 最终验证规范

生成或修改完成后，执行：

```bash
python3 scripts/check_binary_project.py --project-root project_root --block all
python3 -m py_compile src/main.py src/process/preprocess.py src/runner/train.py src/runner/evaluate.py src/runner/predict.py
```

如果依赖和数据可用，应使用小样本或临时降低训练轮数，尽量跑通预处理、训练、评估和预测的 smoke test。最终回复中应说明静态校验、语法检查和 smoke test 的结果。
