import json
import os
import re
import csv
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple

BASE_DIR = Path(__file__).resolve().parent

task_dict: Dict[str, str] = {
    "PIQA": "../CLASSLA-PIQA-Benchmark/results/piqa_results.json",
    "COPA": "../DIALECT-COPA-Benchmark/results/copa_results.json",
    "Genre": "../Genre-Automatic-Identification-Benchmark/evaluation-for-the-paper/results_combined.json",
    "News Topic": "../IPTC-NewsTopic-Benchmark/evaluation-for-the-paper/results.json",
    "Parliamentary Speech Topic": "../ParlaCAP-Topic-Benchmark/evaluation-for-the-paper/results.json",
    "Sentiment": "../ParlaSent-Benchmark/evaluation-for-the-paper/results.json"
}
task_dict = {task: str((BASE_DIR / path).resolve()) for task, path in task_dict.items()}

MATRIX_PATHS = {task: Path(task_dict[task]) for task in ("PIQA", "COPA")}

LANGUAGE_NAMES = {
    "bg": "Bulgarian", "bs": "Bosnian", "en": "English", "hr": "Croatian",
    "hr-ckm": "Chakavian Dialect", "mk": "Macedonian", "sl": "Slovenian",
    "sl-cer": "Cerkno Dialect", "sl-prl": "Prlekija Dialect",
    "sr": "Serbian", "sr_latn": "Serbian", "sr_cyrl": "Serbian (cyrillic)",
    "sr-tor": "Torlak Dialect", "sr-tor_latn": "Torlak Dialect",
    "sr-tor_cyrl": "Torlak Dialect (cyrillic)",
    "sr-satro": "Šatrovački Dialect", "sr-satra_latn": "Šatrovački Dialect",
    "sr-satra_cyrl": "Šatrovački Dialect (cyrillic)",
}


# Output CSV file for the dashboard
OUTPUT_CSV = BASE_DIR / "results.csv"


def load_jsonl(path: str) -> List[dict]:
    """
    Load a JSONL file (one JSON object per line).
    Returns a list of dicts.
    """
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path} at line {i}: {e}")
            if not isinstance(obj, dict):
                raise ValueError(f"Non-object JSON at line {i} in {path}")
            records.append(obj)
    return records


def parse_key_to_language_metric(key: str) -> Tuple[str, str]:
    """
    Parse keys like 'Macedonian (accuracy)' into ('Macedonian', 'accuracy')
    or 'Macedonian (micro-F1)' into ('Macedonian', 'micro_f1').

    If there is no '(metric)' part, return ('key', 'score').
    """
    pattern = r"^(.*)\s*\(([^()]+)\)\s*$"
    m = re.match(pattern, key)
    if m:
        language = m.group(1).strip()
        metric = m.group(2).strip()
    else:
        language = key.strip()
        metric = "score"

    # Normalize metric: lowercase, spaces and hyphens → underscores
    metric_norm = (
        metric.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    return language, metric_norm

def rename_model(model: str) -> str:
    """
    Replace model name using model_name_map if present.
    Otherwise return original name.
    """
    model_name_map = {
    "gpt-4o-2024-08-06": "GPT-4o",
    "gpt-3.5-turbo-0125": "GPT-3.5-Turbo",
    "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
    "gemma3:27b": "Gemma 3",
    "gemma2:27b": "Gemma 2",
    "llama3.3:latest": "LLaMA 3.3",
    "Llama 3.3": "LLaMA 3.3",
    "deepseek-r1:14b": "DeepSeek-R1-Distill-Qwen-14B",
    "DeekSeek-R1": "DeepSeek-R1-Distill-Qwen-14B",
    "deepseek/deepseek-r1-0528": "DeepSeek-R1-0528",
    "dummy-most_frequent": "Dummy (Most Frequent)",
    "dummy-stratified": "Dummy (Stratified)",
    "SVC": "Support Vector Machine",
    "COMPLEMENTNB": "Naive Bayes Classifier",
    "gpt-5-mini-2025-08-07": "GPT-5-mini",
    "gpt-5": "GPT-5",
    "gpt-5-2025-08-07": "GPT-5",
    "gpt-5-nano-2025-08-07": "GPT-5-Nano",
    "GPT-5-nano": "GPT-5-Nano",
    "llama4:scout": "LLaMA 4 Scout",
    "qwen3:32b": "Qwen 3",
    "Qwen3": "Qwen 3",
    'google/gemini-2.5-flash-lite': "Gemini 2.5 Flash Lite",
    'google/gemini-2.5-flash': "Gemini 2.5 Flash",
    'google/gemini-2.5-pro': "Gemini 2.5 Pro",
    "google/gemini-3-flash-preview": "Gemini 3 Flash Preview",
    'mistralai/mistral-medium-3.1': "Mistral Medium 3.1",
    'mistralai/mistral-small-3.2-24b-instruct': "Mistral Small 3.2",
    'cohere/command-a': "Command A",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "anthropic/claude-opus-4.6": "Claude Opus 4.6",
    "anthropic/claude-sonnet-4.6": "Claude Sonnet 4.6",
    "openai/gpt-5.4": "GPT-5.4",
    "openai/gpt-5.4-pro": "GPT-5.4 Pro",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemma-4-31b-it": "Gemma 4 31B",
    "google/gemma-4-26b-a4b-it": "Gemma 4 26B",
    "google/gemma-3-27b-it": "Gemma 3",
    "google/gemma3:27b": "Gemma 3",
    "meta-llama/llama-4-maverick": "LLaMA 4 Maverick",
    "meta-llama/llama-3.3-70b-instruct": "LLaMA 3.3",
    "mistralai/mistral-large-2512": "Mistral Large",
    "mistralai/mistral-small-2603": "Mistral Small",
    "qwen/qwen3-32b": "Qwen 3",
    "openai/gpt-4o": "GPT-4o",
    "openai/gpt-5": "GPT-5",
    "openai/gpt-3.5-turbo": "GPT-3.5-Turbo",
    "GaMS-27B-quantized": "GaMS-27B-Instruct (quantized)",
    "GaMS-27B": "GaMS-27B-Instruct",
    "GaMS-Instruct 27B": "GaMS-27B-Instruct",
    "cjvt/GaMS-27B-Instruct-i1": "GaMS-27B-Instruct-i1",
    "hf.co/mradermacher/GaMS-27B-Instruct-i1-GGUF:latest": "GaMS-27B-Instruct-i1 (GGUF)",
    }

    return model_name_map.get(model, model)

def extract_rows_for_task(task_name: str, jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Given a task name and path to its JSONL file,
    return a list of rows with columns:
    task, model, language, metric, value, source_model
    """
    data = load_jsonl(jsonl_path)
    rows: List[Dict[str, Any]] = []

    for record in data:
        # Try both 'Model' and 'model'
        model = record.get("Model") or record.get("model")
        if model is None:
            # Skip records without model info
            continue

        # Apply renaming dictionary
        model_clean = rename_model(model)

        for key, value in record.items():
            if key in ("Model", "model"):
                continue

            # Parse language and metric from the key
            language, metric = parse_key_to_language_metric(key)

            # Convert value to float if possible
            try:
                value_num = float(value)
            except (TypeError, ValueError):
                # Skip non-numeric values (e.g., if something is text)
                continue

            # Correct Serbian (latin) to Serbian
            if language == "Serbian (latin)":
                language = "Serbian"

            # PIQA/COPA include every model and script. Other tasks retain their filters.
            if task_name in MATRIX_PATHS or language not in ("Serbian (cyrillic)", "Torlak Dialect (cyrillic)"):
                # Skip certain models:
                if task_name in MATRIX_PATHS or model_clean not in ["Command A", "Dummy (Most Frequent)", "Dummy (Stratified)",
                       "Gemini 2.5 Flash Lite", "Llama 4 Scout", "Logistic Regression", "NLI zero-shot model",
                       "Naive Bayes Classifier", "Support Vector Machine", "fastText", "GaMS-27B-Instruct (quantized)", "GaMS-27B-Instruct (quantized)"]:
                    rows.append(
                        {
                            "task": task_name,
                            "model": model_clean,
                            "language": language,
                            "metric": metric,
                            "value": value_num,
                            "source_model": model,
                        }
                )
            
    return rows


def extract_matrix_rows(task_name: str, path: Path) -> List[Dict[str, Any]]:
    """Read the reviewed model-by-language JSON matrix without filtering results."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a nonempty list of result records in {path}")
    cells = {}
    for record in data:
        if not isinstance(record, dict) or not isinstance(record.get("Model"), str) or not record["Model"].strip():
            raise ValueError(f"Missing model identifier in {path}: {record!r}")
        model = record["Model"]
        for code, value in record.items():
            if code == "Model":
                continue
            if code not in LANGUAGE_NAMES:
                raise ValueError(f"Unknown language {code!r} in {path}")
            if value is None or value == "":
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid accuracy for {model} / {code} in {path}: {value!r}")
            language = LANGUAGE_NAMES[code]
            key = (rename_model(model), language, "accuracy")
            if key in cells and cells[key]["value"] != value:
                raise ValueError(f"Conflicting scores for {task_name} / {key} in {path}")
            cells.setdefault(key, {
                "task": task_name, "model": key[0], "language": language,
                "metric": "accuracy", "value": float(value), "source_model": model,
            })
    return list(cells.values())


def main():
    all_rows: List[Dict[str, Any]] = []

    for task_name, path in task_dict.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing results for task {task_name}: {path}")

        print(f"Processing task {task_name} from {path} ...")
        if task_name in MATRIX_PATHS:
            # These matrices define the complete current benchmark results.
            rows = extract_matrix_rows(task_name, path)
        else:
            rows = extract_rows_for_task(task_name, path)
        print(f"  -> extracted {len(rows)} rows")
        all_rows.extend(rows)

    if not all_rows:
        print("No rows extracted. Check your task_dict and JSONL formats.")
        return

    # Ensure output directory exists
    out_dir = os.path.dirname(OUTPUT_CSV)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = ["task", "model", "language", "metric", "value", "source_model"]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {OUTPUT_CSV}")

    # print out all different models
    print("List of evaluated models:")
    print(sorted({row["model"] for row in all_rows}))

    print("List of evaluated languages:")
    print(sorted({row["language"] for row in all_rows}))


if __name__ == "__main__":
    main()
