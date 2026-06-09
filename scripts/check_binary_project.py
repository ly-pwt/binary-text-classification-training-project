#!/usr/bin/env python3
"""Static validator for binary-text-classification-training-project outputs."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


BLOCKS = ("architecture", "config", "preprocess", "train", "evaluate", "predict", "web")
ABSOLUTE_PATH_RE = re.compile(r"(/Users/|/home/|[A-Za-z]:\\\\)")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
FUNCTION_COMMENT_RE = re.compile(r"#\s*功能[：:].+")
FORBIDDEN_GENERATION_TOKENS = (
    "AutoModelForSeq2SeqLM",
    "Seq2SeqTrainer",
    "Seq2SeqTrainingArguments",
    "DataCollatorForSeq2Seq",
    ".generate(",
)


def add_issue(issues: list[dict[str, str]], block: str, file_name: str, message: str) -> None:
    issues.append({"block": block, "file": file_name, "message": message})


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require_path(root: Path, block: str, relative_path: str, issues: list[dict[str, str]]) -> Path:
    path = root / relative_path
    if not path.exists():
        add_issue(issues, block, relative_path, "缺少必须路径。")
    return path


def parse_python(root: Path, block: str, relative_path: str, issues: list[dict[str, str]]) -> ast.Module | None:
    path = require_path(root, block, relative_path, issues)
    if not path.exists() or not path.is_file():
        return None
    try:
        return ast.parse(read_text(path), filename=str(path))
    except SyntaxError as exc:
        add_issue(issues, block, relative_path, f"Python 语法错误：{exc}")
        return None


def has_function(tree: ast.Module | None, name: str) -> bool:
    if tree is None:
        return False
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name for node in ast.walk(tree))


def has_class(tree: ast.Module | None, name: str) -> bool:
    if tree is None:
        return False
    return any(isinstance(node, ast.ClassDef) and node.name == name for node in ast.walk(tree))


def check_no_absolute_paths(root: Path, block: str, relative_path: str, issues: list[dict[str, str]]) -> str:
    path = root / relative_path
    if not path.exists() or not path.is_file():
        return ""
    text = read_text(path)
    if ABSOLUTE_PATH_RE.search(text):
        add_issue(issues, block, relative_path, "不要写本机绝对路径。")
    return text


def check_no_generation_tokens(root: Path, block: str, relative_path: str, issues: list[dict[str, str]]) -> None:
    text = check_no_absolute_paths(root, block, relative_path, issues)
    for token in FORBIDDEN_GENERATION_TOKENS:
        if token in text:
            add_issue(issues, block, relative_path, f"二分类项目中不应出现生成任务组件：{token}")


def check_function_comments(root: Path, block: str, relative_path: str, issues: list[dict[str, str]]) -> None:
    path = root / relative_path
    if not path.exists() or not path.is_file():
        return
    text = read_text(path)
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return
    lines = text.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        index = node.lineno - 2
        while index >= 0 and not lines[index].strip():
            index -= 1
        if index < 0 or not FUNCTION_COMMENT_RE.search(lines[index]):
            add_issue(issues, block, relative_path, f"{node.name} 前缺少 '# 功能：...' 中文功能注释。")


def check_chinese_comments(root: Path, block: str, relative_path: str, issues: list[dict[str, str]]) -> None:
    path = root / relative_path
    if not path.exists() or not path.is_file():
        return
    comments = [line for line in read_text(path).splitlines() if line.strip().startswith("#")]
    if sum(1 for line in comments if CHINESE_RE.search(line)) < 3:
        add_issue(issues, block, relative_path, "中文流程注释较少。")


def check_common(root: Path, block: str, relative_path: str, issues: list[dict[str, str]]) -> ast.Module | None:
    check_no_generation_tokens(root, block, relative_path, issues)
    tree = parse_python(root, block, relative_path, issues)
    check_function_comments(root, block, relative_path, issues)
    check_chinese_comments(root, block, relative_path, issues)
    return tree


def check_architecture(root: Path, issues: list[dict[str, str]]) -> None:
    required = [
        "data/raw",
        "data/processed",
        "checkpoint/best",
        "checkpoint/last",
        "logs",
        "src/configuration/__init__.py",
        "src/configuration/config.py",
        "src/process/__init__.py",
        "src/process/preprocess.py",
        "src/runner/__init__.py",
        "src/runner/train.py",
        "src/runner/evaluate.py",
        "src/runner/predict.py",
        "src/main.py",
    ]
    for relative_path in required:
        require_path(root, "architecture", relative_path, issues)


def check_config(root: Path, issues: list[dict[str, str]]) -> None:
    block = "config"
    relative_path = "src/configuration/config.py"
    tree = check_common(root, block, relative_path, issues)
    if tree is None:
        return
    text = read_text(root / relative_path)
    required = {
        "from pathlib import Path": "必须使用 pathlib.Path。",
        "ROOT_DIR = Path(__file__).parent.parent.parent": "ROOT_DIR 必须按分层 src 结构计算。",
        'RAW_DATA_DIR = ROOT_DIR / "data" / "raw"': "缺少 RAW_DATA_DIR。",
        'PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"': "缺少 PROCESSED_DATA_DIR。",
        'MODEL_DIR = ROOT_DIR / "checkpoint"': "模型目录必须是 checkpoint。",
        'BEST_MODEL_DIR = MODEL_DIR / "best"': "缺少 BEST_MODEL_DIR。",
        'LAST_CHECKPOINT_DIR = MODEL_DIR / "last"': "缺少 LAST_CHECKPOINT_DIR。",
        'TENSORBOARD_LOGGING_DIR = LOG_DIR / "tensorboard"': "缺少 TensorBoard 日志目录。",
        "BINARY_CLASSIFICATION_MODEL_NAME": "缺少二分类模型名称配置。",
        "TEXT_COLUMN": "缺少 TEXT_COLUMN。",
        "LABEL_COLUMN": "缺少 LABEL_COLUMN。",
        "LABEL_NAMES": "缺少 LABEL_NAMES。",
        "LABELS_FILE": "缺少 LABELS_FILE。",
        "SEQ_LEN": "缺少 SEQ_LEN。",
        "BATCH_SIZE": "缺少 BATCH_SIZE。",
        "LEARNING_RATE": "缺少 LEARNING_RATE。",
        "EPOCHS": "缺少 EPOCHS。",
        "SEED": "缺少 SEED。",
    }
    for snippet, message in required.items():
        if snippet not in text:
            add_issue(issues, block, relative_path, message)
    if ".mkdir" in text:
        add_issue(issues, block, relative_path, "config.py 不应主动创建目录。")


def check_preprocess(root: Path, issues: list[dict[str, str]]) -> None:
    block = "preprocess"
    relative_path = "src/process/preprocess.py"
    tree = check_common(root, block, relative_path, issues)
    if tree is None:
        return
    text = read_text(root / relative_path)
    if not has_function(tree, "preprocess"):
        add_issue(issues, block, relative_path, "缺少 preprocess() 入口函数。")
    required_tokens = {
        "AutoTokenizer": "必须加载 AutoTokenizer。",
        "BINARY_CLASSIFICATION_MODEL_NAME": "tokenizer 模型名必须来自 config。",
        "TEXT_COLUMN": "必须从 config 使用文本列。",
        "LABEL_COLUMN": "必须从 config 使用标签列。",
        "LABELS_FILE": "必须保存标签映射文件。",
        "train_test_split": "单文件数据必须支持切分。",
        "DatasetDict": "必须保存 train/valid/test DatasetDict。",
        '"train"': "DatasetDict 必须包含 train。",
        '"valid"': "DatasetDict 必须包含 valid。",
        '"test"': "DatasetDict 必须包含 test。",
        "stratify_by_column": "二分类应优先分层抽样。",
        "labels": "必须生成 Trainer 使用的 labels 字段。",
        "save_to_disk": "必须保存处理后的数据集。",
    }
    for token, message in required_tokens.items():
        if token not in text:
            add_issue(issues, block, relative_path, message)


def check_trainer_compat(root: Path, block: str, relative_path: str, issues: list[dict[str, str]]) -> None:
    text = read_text(root / relative_path)
    if "Trainer(" in text and re.search(r"\bTrainer\s*\([\s\S]*?\b(?:tokenizer|processing_class)\s*=", text):
        add_issue(issues, block, relative_path, "不要直接写 Trainer tokenizer/processing_class 参数，必须通过 build_trainer()。")
    for token in ("build_trainer", "signature", "processing_class", "tokenizer"):
        if token not in text:
            add_issue(issues, block, relative_path, f"缺少 Trainer API 兼容要素：{token}")


def check_train(root: Path, issues: list[dict[str, str]]) -> None:
    block = "train"
    relative_path = "src/runner/train.py"
    tree = check_common(root, block, relative_path, issues)
    if tree is None:
        return
    text = read_text(root / relative_path)
    if not has_function(tree, "train"):
        add_issue(issues, block, relative_path, "缺少 train() 入口函数。")
    required_tokens = {
        "AutoModelForSequenceClassification": "必须使用序列分类模型。",
        "num_labels=2": "必须显式设置 num_labels=2。",
        "id2label": "必须传入 id2label。",
        "label2id": "必须传入 label2id。",
        "DataCollatorWithPadding": "必须使用 DataCollatorWithPadding。",
        "compute_metrics": "必须定义二分类指标。",
        "accuracy": "必须计算 accuracy。",
        "f1": "必须计算 F1。",
        "load_best_model_at_end": "必须自动加载最佳模型。",
        "metric_for_best_model": "必须配置最佳模型指标。",
        "eval_loss": "默认必须用 eval_loss 选最佳模型。",
        "greater_is_better": "必须指定 eval_loss 越小越好。",
        "get_last_checkpoint": "必须支持断点续训。",
        "BEST_MODEL_DIR": "必须保存到 checkpoint/best。",
        "dataset_dict[\"valid\"]": "训练中必须使用 valid 作为验证集。",
    }
    for token, message in required_tokens.items():
        if token not in text:
            add_issue(issues, block, relative_path, message)
    check_trainer_compat(root, block, relative_path, issues)


def check_evaluate(root: Path, issues: list[dict[str, str]]) -> None:
    block = "evaluate"
    relative_path = "src/runner/evaluate.py"
    tree = check_common(root, block, relative_path, issues)
    if tree is None:
        return
    text = read_text(root / relative_path)
    if not has_function(tree, "evaluate"):
        add_issue(issues, block, relative_path, "缺少 evaluate() 入口函数。")
    required_tokens = {
        "BEST_MODEL_DIR": "评估必须加载最佳模型目录。",
        "AutoModelForSequenceClassification": "必须使用序列分类模型。",
        "dataset_dict[split]": "evaluate(split=...) 应按 split 评估。",
        "compute_metrics": "必须计算指标。",
        "accuracy": "必须计算 accuracy。",
        "f1": "必须计算 F1。",
        "Trainer": "必须使用 Trainer.evaluate()。",
        ".evaluate(": "必须调用 evaluate。",
    }
    for token, message in required_tokens.items():
        if token not in text:
            add_issue(issues, block, relative_path, message)
    check_trainer_compat(root, block, relative_path, issues)


def check_predict(root: Path, issues: list[dict[str, str]]) -> None:
    block = "predict"
    relative_path = "src/runner/predict.py"
    tree = check_common(root, block, relative_path, issues)
    if tree is None:
        return
    text = read_text(root / relative_path)
    if not has_class(tree, "Predictor"):
        add_issue(issues, block, relative_path, "必须提供 Predictor 类。")
    required_tokens = {
        "BEST_MODEL_DIR": "预测必须优先加载 checkpoint/best。",
        "AutoTokenizer": "必须加载 tokenizer。",
        "AutoModelForSequenceClassification": "必须加载序列分类模型。",
        "torch.no_grad": "推理必须使用 torch.no_grad()。",
        "softmax": "二分类预测必须对 logits 做 softmax。",
        "confidence": "返回结果必须包含置信度。",
        "argparse": "必须支持 --text 命令行预测。",
        "--text": "必须提供 --text 参数。",
        "quit": "交互模式必须支持退出命令。",
    }
    for token, message in required_tokens.items():
        if token not in text:
            add_issue(issues, block, relative_path, message)


def check_web(root: Path, issues: list[dict[str, str]]) -> None:
    web_dir = root / "src" / "web"
    if not web_dir.exists():
        return
    for relative_path in ("src/web/__init__.py", "src/web/app.py", "src/web/schemas.py", "src/web/service.py"):
        tree = check_common(root, "web", relative_path, issues)
        if tree is None and relative_path.endswith(".py"):
            continue
    service = root / "src/web/service.py"
    if service.exists() and "Predictor" not in read_text(service):
        add_issue(issues, "web", "src/web/service.py", "Web 服务必须复用 runner.predict.Predictor。")


def check_main(root: Path, issues: list[dict[str, str]]) -> None:
    block = "architecture"
    relative_path = "src/main.py"
    tree = check_common(root, block, relative_path, issues)
    if tree is None:
        return
    text = read_text(root / relative_path)
    for token in ("preprocess", "train", "evaluate", "predict"):
        if token not in text:
            add_issue(issues, block, relative_path, f"main.py 缺少 action：{token}")


def run_checks(root: Path, blocks: list[str]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    selected = BLOCKS if "all" in blocks else tuple(blocks)
    if "architecture" in selected:
        check_architecture(root, issues)
        check_main(root, issues)
    if "config" in selected:
        check_config(root, issues)
    if "preprocess" in selected:
        check_preprocess(root, issues)
    if "train" in selected:
        check_train(root, issues)
    if "evaluate" in selected:
        check_evaluate(root, issues)
    if "predict" in selected:
        check_predict(root, issues)
    if "web" in selected:
        check_web(root, issues)
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--block", action="append", choices=(*BLOCKS, "all"), required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    issues = run_checks(root, args.block)
    if issues:
        print("Binary project validation failed:")
        for issue in issues:
            print(f"- [{issue['block']}] {issue['file']}: {issue['message']}")
        return 1
    print("Binary project validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
