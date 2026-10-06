"""
AgentShield: An Adaptive Security Runtime for Autonomous AI Agents
PDF Dataset Preprocessing Pipeline

This script validates, preprocesses, standardizes text representation,
groups duplicates by SHA-256 hash to prevent cross-split data leakage,
and generates stratified Train/Validation/Test splits for the AgentShield PDF Dataset.
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
logger = logging.getLogger("AgentShield.PDFPreprocessing")


def preprocess_pdf_dataset(
    raw_pdf_path: Path,
    output_dir: Path,
    seed: int = 42
):
    """
    Standardizes raw PDF dataset (dataset_pdf.jsonl), performs SHA-256 deduplication,
    and produces leakage-free train (80%), validation (10%), and test (10%) splits.
    """
    raw_pdf_path = Path(raw_pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_pdf_path.exists():
        raise FileNotFoundError(f"Raw PDF dataset not found at: {raw_pdf_path}")

    logger.info(f"Loading raw PDF dataset from: {raw_pdf_path}")

    records = []
    hashes = defaultdict(list)
    corrupted_lines = 0
    total_lines = 0

    with open(raw_pdf_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            total_lines += 1

            try:
                data = json.loads(line_str)
            except json.JSONDecodeError:
                corrupted_lines += 1
                continue

            if "label" not in data or data["label"] not in (0, 1):
                corrupted_lines += 1
                continue

            label = int(data["label"])
            context = str(data.get("context", "")).strip()
            user_intent = str(data.get("user_intent", "")).strip()
            source = str(data.get("source", "unknown")).strip()

            # Standardized text field
            if user_intent:
                standardized_text = f"User intent: {user_intent} Document content: {context}".strip()
            else:
                standardized_text = f"Document content: {context}".strip()

            # Compute SHA-256 hash of standardized text for deduplication
            text_hash = hashlib.sha256(standardized_text.encode("utf-8")).hexdigest()

            record = {
                "domain": "pdf",
                "text": standardized_text,
                "label": label,
                "category": source,
                "user_intent": user_intent,
                "source": source,
                "sha256": text_hash
            }

            records.append(record)
            hashes[text_hash].append(record)

            if total_lines % 20000 == 0:
                logger.info(f"Processed {total_lines} PDF samples...")

    logger.info(f"Finished reading: {len(records)} valid samples (corrupted/skipped: {corrupted_lines})")

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
            "corrupted_lines": corrupted_lines,
            "duplicate_samples_count": duplicate_count,
            "unique_samples_count": len(unique_hashes)
        },
        "source_distribution": dict(Counter(r["source"] for r in records)),
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
    print("SAMPLE PREPROCESSED PDF RECORDS")
    print("=" * 70)
    for i, s in enumerate(train_records[:3], 1):
        clean = {k: v for k, v in s.items() if k != "sha256"}
        if len(clean["text"]) > 180:
            clean["text"] = clean["text"][:180] + " ...[TRUNCATED]"
        print(f"\nSample {i} (label={clean['label']}, source={clean['source']}):")
        print(json.dumps(clean, indent=2, ensure_ascii=False))
    print("=" * 70 + "\n")

    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_pdf_path = project_root / "data" / "raw" / "pdf" / "dataset_pdf.jsonl"
    output_dir = project_root / "data" / "processed" / "pdf"
    preprocess_pdf_dataset(raw_pdf_path, output_dir)
