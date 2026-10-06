"""
AgentShield: An Adaptive Security Runtime for Autonomous AI Agents
Web Dataset Preprocessing Pipeline

This script loads, validates, standardizes text representation, groups
duplicates by SHA-256 hash to prevent cross-split data leakage, and generates
stratified Train/Validation/Test splits for the AgentShield Web Dataset.
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

# Force UTF-8 stdout for Windows console compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("AgentShield.WebPreprocessing")


def preprocess_web_dataset(
    raw_web_dir: Path,
    output_dir: Path,
    seed: int = 42
):
    """
    Reads benign and malicious web JSONL files, standardizes format,
    applies SHA-256 group stratification, and saves train/val/test splits.
    """
    raw_web_dir = Path(raw_web_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benign_dir = raw_web_dir / "benign"
    malicious_dir = raw_web_dir / "Malicious"

    if not benign_dir.exists() or not malicious_dir.exists():
        raise FileNotFoundError(f"Raw web subdirectories missing in {raw_web_dir}")

    records = []
    hashes = defaultdict(list)
    corrupted_count = 0

    def process_dir(dir_path: Path, label: int):
        nonlocal corrupted_count
        for file_path in sorted(dir_path.glob("*.jsonl")):
            category = file_path.stem
            file_count = 0
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        data = json.loads(line_str)
                    except json.JSONDecodeError:
                        corrupted_count += 1
                        continue

                    raw_text = str(data.get("text", "")).strip()
                    if not raw_text:
                        corrupted_count += 1
                        continue

                    standardized_text = f"Web content: {raw_text}".strip()
                    text_hash = hashlib.sha256(standardized_text.encode("utf-8")).hexdigest()

                    record = {
                        "domain": "web",
                        "text": standardized_text,
                        "label": label,
                        "category": category,
                        "raw_id": data.get("id"),
                        "sha256": text_hash
                    }

                    records.append(record)
                    hashes[text_hash].append(record)
                    file_count += 1

            logger.info(f"Loaded {file_count} samples from {file_path.name} (category='{category}', label={label})")

    logger.info("--- Processing Benign Web Data ---")
    process_dir(benign_dir, label=0)

    logger.info("--- Processing Malicious Web Data ---")
    process_dir(malicious_dir, label=1)

    logger.info(f"Total Web samples loaded: {len(records)} (Corrupted/Empty skipped: {corrupted_count})")

    # Duplicate detection & grouping
    dup_groups = [g for g in hashes.values() if len(g) > 1]
    duplicate_count = sum(len(g) - 1 for g in dup_groups)
    unique_hashes = list(hashes.keys())

    logger.info(
        f"Deduplication: {duplicate_count} duplicate copies detected across {len(dup_groups)} groups. "
        f"Unique content groups: {len(unique_hashes)}"
    )

    # Stratified Split by unique hash groups (80% Train, 10% Val, 10% Test)
    hash_labels = [hashes[h][0]["label"] for h in unique_hashes]

    h_train, h_temp, _, y_temp = train_test_split(
        unique_hashes, hash_labels, test_size=0.20, random_state=seed, stratify=hash_labels
    )
    h_val, h_test, _, _ = train_test_split(
        h_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    set_train, set_val, set_test = set(h_train), set(h_val), set(h_test)

    train_records, val_records, test_records = [], [], []
    for r in records:
        h = r["sha256"]
        if h in set_train:
            train_records.append(r)
        elif h in set_val:
            val_records.append(r)
        elif h in set_test:
            test_records.append(r)

    def write_jsonl(file_path: Path, recs: list):
        with open(file_path, "w", encoding="utf-8") as out_f:
            for rec in recs:
                clean_rec = {k: v for k, v in rec.items() if k != "sha256"}
                out_f.write(json.dumps(clean_rec, ensure_ascii=False) + "\n")

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "validation.jsonl", val_records)
    write_jsonl(output_dir / "test.jsonl", test_records)

    logger.info(
        f"Splits saved -> Train: {len(train_records)} | Validation: {len(val_records)} | Test: {len(test_records)}"
    )

    # Compute detailed metrics report
    report = {
        "dataset_summary": {
            "total_samples": len(records),
            "benign_samples": sum(1 for r in records if r["label"] == 0),
            "malicious_samples": sum(1 for r in records if r["label"] == 1),
            "corrupted_lines": corrupted_count,
            "duplicate_samples_count": duplicate_count,
            "unique_samples_count": len(unique_hashes)
        },
        "category_distribution": dict(sorted(Counter(r["category"] for r in records).items())),
        "split_statistics": {
            "train": {
                "total": len(train_records),
                "benign": sum(1 for r in train_records if r["label"] == 0),
                "malicious": sum(1 for r in train_records if r["label"] == 1)
            },
            "validation": {
                "total": len(val_records),
                "benign": sum(1 for r in val_records if r["label"] == 0),
                "malicious": sum(1 for r in val_records if r["label"] == 1)
            },
            "test": {
                "total": len(test_records),
                "benign": sum(1 for r in test_records if r["label"] == 0),
                "malicious": sum(1 for r in test_records if r["label"] == 1)
            }
        }
    }

    report_path = output_dir / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=4, ensure_ascii=False)

    logger.info(f"Preprocessing report written to: {report_path}")

    # Display sample records
    print("\n" + "=" * 70)
    print("SAMPLE PREPROCESSED WEB RECORDS")
    print("=" * 70)
    for i, s in enumerate(train_records[:3], 1):
        clean = {k: v for k, v in s.items() if k != "sha256"}
        if len(clean["text"]) > 180:
            clean["text"] = clean["text"][:180] + " ...[TRUNCATED]"
        print(f"\nSample {i} (category={clean['category']}, label={clean['label']}):")
        print(json.dumps(clean, indent=2, ensure_ascii=False))
    print("=" * 70 + "\n")

    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_web_dir = project_root / "data" / "raw" / "web"
    output_dir = project_root / "data" / "processed" / "web"
    preprocess_web_dataset(raw_web_dir, output_dir)
