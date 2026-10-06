"""
AgentShield: An Adaptive Security Runtime for Autonomous AI Agents
Combined Dataset Preparation Pipeline

This script merges the standardized Image OCR, PDF, and Web datasets
into a unified multi-domain dataset under data/processed/combined/.

Key Features:
1. Empty OCR Text Filtering: Excludes textless image records ("Image content:")
   that create conflicting labels in the pure text classifier.
2. Cross-Label Conflict Resolution: Detects and removes ambiguous texts that appear
   with contradictory labels (label 0 AND label 1) due to adversarial visual perturbations.
3. Intra-Domain Class Balancing: Eliminates confounding domain bias (Simpson's Paradox)
   by balancing benign vs. malicious classes within each domain's training set.
4. Domain Balancing: Downsamples large PDF corpus to maintain equal domain representation
   across Image, Web, and PDF modalities.
"""

import os
import sys
import json
import logging
import argparse
import random
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
logger = logging.getLogger("AgentShield.CombineDatasets")


def load_domain_jsonl(file_path: Path, domain: str, filter_empty_ocr: bool = True) -> tuple:
    """
    Reads a domain JSONL file and returns (valid_records, dropped_records_info).
    Filters out empty OCR or boilerplate-only text records if filter_empty_ocr is True.
    """
    records = []
    dropped = []
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return records, dropped

    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                text = str(data.get("text", "")).strip()
                label = int(data.get("label"))
                category = str(data.get("category", domain))

                # Image domain specific validation:
                if domain == "image":
                    extracted = str(data.get("extracted_text", "")).strip()
                    ocr_status = str(data.get("ocr_status", "")).strip()

                    # Detect textless images or boilerplate prefix
                    is_empty = (
                        ocr_status == "no_text"
                        or not extracted
                        or text in ("Image content:", "Image content")
                    )

                    if filter_empty_ocr and is_empty:
                        dropped.append({
                            "line": line_idx,
                            "label": label,
                            "category": category,
                            "reason": "empty_ocr_text"
                        })
                        continue

                # General empty text check across all domains
                if not text:
                    dropped.append({
                        "line": line_idx,
                        "label": label,
                        "category": category,
                        "reason": "empty_text"
                    })
                    continue

                records.append({
                    "text": text,
                    "label": label,
                    "domain": domain,
                    "category": category
                })
            except Exception:
                continue

    return records, dropped


def filter_conflicting_label_texts(records: list) -> tuple:
    """
    Finds texts that exist with conflicting labels (e.g. label 0 and label 1)
    and removes them to ensure zero contradictory gradient signals in the text model.
    """
    text_to_labels = defaultdict(set)
    for r in records:
        text_to_labels[r["text"]].add(r["label"])

    conflicting_texts = {t for t, labels in text_to_labels.items() if len(labels) > 1}
    if not conflicting_texts:
        return records, 0

    clean_records = [r for r in records if r["text"] not in conflicting_texts]
    dropped_count = len(records) - len(clean_records)
    logger.info(
        f"Filtered {len(conflicting_texts)} ambiguous texts with contradictory labels "
        f"({dropped_count} total records dropped)"
    )
    return clean_records, len(conflicting_texts)


def stratified_subset(records: list, target_size: int, seed: int = 42) -> list:
    """Selects a stratified subset of records balanced by label."""
    if len(records) <= target_size:
        return records
    labels = [r["label"] for r in records]
    sampled, _ = train_test_split(
        records, train_size=target_size, random_state=seed, stratify=labels
    )
    return sampled


def balance_classes_in_domain(
    records: list,
    max_samples_per_class: int = None,
    seed: int = 42
) -> tuple:
    """
    Balances benign (label 0) and malicious (label 1) samples 1:1 within a domain split.
    If max_samples_per_class is None, balances to the minority class count.
    """
    benign = [r for r in records if r["label"] == 0]
    malicious = [r for r in records if r["label"] == 1]

    minority_count = min(len(benign), len(malicious))
    target = minority_count if max_samples_per_class is None else min(minority_count, max_samples_per_class)

    rng = random.Random(seed)
    sampled_benign = rng.sample(benign, target) if len(benign) > target else benign
    sampled_malicious = rng.sample(malicious, target) if len(malicious) > target else malicious

    balanced = sampled_benign + sampled_malicious
    rng.shuffle(balanced)

    audit = {
        "original_benign": len(benign),
        "original_malicious": len(malicious),
        "balanced_per_class": target,
        "total_balanced": len(balanced)
    }
    return balanced, audit


def combine_all_datasets(
    project_root: Path,
    output_dir: Path,
    pdf_train_samples: int = 1600,
    pdf_val_samples: int = 400,
    pdf_test_samples: int = 400,
    filter_empty_ocr: bool = True,
    filter_conflicts: bool = True,
    balance_domain_classes: bool = True,
    full_pdf: bool = False,
    seed: int = 42
):
    """
    Merges Image OCR, Web, and PDF splits into combined dataset with
    empty-OCR filtering, cross-label conflict resolution, and intra-domain class balancing.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = project_root / "data" / "processed" / "image"
    pdf_dir = project_root / "data" / "processed" / "pdf"
    web_dir = project_root / "data" / "processed" / "web"

    # --------------------------------------------------------------
    # 1. Load Datasets with Filtering
    # --------------------------------------------------------------
    img_train, img_drop_train = load_domain_jsonl(image_dir / "train.jsonl", "image", filter_empty_ocr)
    img_val,   img_drop_val   = load_domain_jsonl(image_dir / "validation.jsonl", "image", filter_empty_ocr)
    img_test,  img_drop_test  = load_domain_jsonl(image_dir / "test.jsonl", "image", filter_empty_ocr)

    total_img_dropped = len(img_drop_train) + len(img_drop_val) + len(img_drop_test)
    logger.info(
        f"Loaded Image splits (Filter Empty OCR: {filter_empty_ocr}) -> "
        f"Train: {len(img_train)} (dropped {len(img_drop_train)}), "
        f"Val: {len(img_val)} (dropped {len(img_drop_val)}), "
        f"Test: {len(img_test)} (dropped {len(img_drop_test)}) | Total dropped: {total_img_dropped}"
    )

    web_train, _ = load_domain_jsonl(web_dir / "train.jsonl", "web", filter_empty_ocr)
    web_val,   _ = load_domain_jsonl(web_dir / "validation.jsonl", "web", filter_empty_ocr)
    web_test,  _ = load_domain_jsonl(web_dir / "test.jsonl", "web", filter_empty_ocr)
    logger.info(f"Loaded Web splits -> Train: {len(web_train)}, Val: {len(web_val)}, Test: {len(web_test)}")

    pdf_train, _ = load_domain_jsonl(pdf_dir / "train.jsonl", "pdf", filter_empty_ocr)
    pdf_val,   _ = load_domain_jsonl(pdf_dir / "validation.jsonl", "pdf", filter_empty_ocr)
    pdf_test,  _ = load_domain_jsonl(pdf_dir / "test.jsonl", "pdf", filter_empty_ocr)
    logger.info(f"Loaded PDF splits (Raw) -> Train: {len(pdf_train)}, Val: {len(pdf_val)}, Test: {len(pdf_test)}")

    # --------------------------------------------------------------
    # 2. PDF Domain Subsampling
    # --------------------------------------------------------------
    if not full_pdf:
        pdf_train = stratified_subset(pdf_train, pdf_train_samples, seed)
        pdf_val   = stratified_subset(pdf_val, pdf_val_samples, seed)
        pdf_test  = stratified_subset(pdf_test, pdf_test_samples, seed)
        logger.info(
            f"Domain-Balanced PDF splits -> Train: {len(pdf_train)}, Val: {len(pdf_val)}, Test: {len(pdf_test)}"
        )

    # --------------------------------------------------------------
    # 3. Filter Contradictory Label Collisions
    # --------------------------------------------------------------
    conflicts_audit = {}
    if filter_conflicts:
        logger.info("Checking and resolving contradictory label text collisions...")
        # Resolve within each domain before combining
        img_train, conflicts_audit["image_train"] = filter_conflicting_label_texts(img_train)
        img_val,   conflicts_audit["image_val"]   = filter_conflicting_label_texts(img_val)
        img_test,  conflicts_audit["image_test"]  = filter_conflicting_label_texts(img_test)

    # --------------------------------------------------------------
    # 4. Intra-Domain Class Balancing (Training Set)
    # --------------------------------------------------------------
    balancing_audit = {}
    if balance_domain_classes:
        logger.info("Applying Intra-Domain Class Balancing on Training Set (1:1 Benign:Malicious)...")
        img_train, balancing_audit["image"] = balance_classes_in_domain(img_train, seed=seed)
        web_train, balancing_audit["web"]   = balance_classes_in_domain(web_train, seed=seed)
        pdf_train, balancing_audit["pdf"]   = balance_classes_in_domain(pdf_train, seed=seed)

        for dom, aud in balancing_audit.items():
            logger.info(
                f"  [{dom.upper()}] Original: {aud['original_benign']} B / {aud['original_malicious']} M -> "
                f"Balanced: {aud['balanced_per_class']} B / {aud['balanced_per_class']} M (Total: {aud['total_balanced']})"
            )

    # --------------------------------------------------------------
    # 5. Merge & Shuffle Splits
    # --------------------------------------------------------------
    rng = random.Random(seed)

    combined_train = img_train + web_train + pdf_train
    if filter_conflicts:
        combined_train, conflicts_audit["cross_domain_train"] = filter_conflicting_label_texts(combined_train)
    rng.shuffle(combined_train)

    combined_val = img_val + web_val + pdf_val
    if filter_conflicts:
        combined_val, conflicts_audit["cross_domain_val"] = filter_conflicting_label_texts(combined_val)
    rng.shuffle(combined_val)

    combined_test = img_test + web_test + pdf_test
    if filter_conflicts:
        combined_test, conflicts_audit["cross_domain_test"] = filter_conflicting_label_texts(combined_test)
    rng.shuffle(combined_test)

    def write_jsonl(file_path: Path, recs: list):
        with open(file_path, "w", encoding="utf-8") as out_f:
            for rec in recs:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_jsonl(output_dir / "train.jsonl", combined_train)
    write_jsonl(output_dir / "validation.jsonl", combined_val)
    write_jsonl(output_dir / "test.jsonl", combined_test)

    # --------------------------------------------------------------
    # 6. Compute Detailed Audit Report
    # --------------------------------------------------------------
    def get_stats(recs):
        by_domain_class = defaultdict(lambda: {"benign": 0, "malicious": 0, "total": 0})
        for r in recs:
            dom = r["domain"]
            by_domain_class[dom]["total"] += 1
            if r["label"] == 1:
                by_domain_class[dom]["malicious"] += 1
            else:
                by_domain_class[dom]["benign"] += 1

        return {
            "total": len(recs),
            "benign": sum(1 for r in recs if r["label"] == 0),
            "malicious": sum(1 for r in recs if r["label"] == 1),
            "by_domain": dict(Counter(r["domain"] for r in recs)),
            "by_domain_and_label": dict(by_domain_class)
        }

    report = {
        "dataset_summary": {
            "total_combined_samples": len(combined_train) + len(combined_val) + len(combined_test),
            "domain_balancing_enabled": not full_pdf,
            "filter_empty_ocr_enabled": filter_empty_ocr,
            "filter_conflicts_enabled": filter_conflicts,
            "intra_domain_class_balancing_enabled": balance_domain_classes,
            "total_empty_ocr_dropped": total_img_dropped,
            "empty_ocr_dropped_details": {
                "train": len(img_drop_train),
                "validation": len(img_drop_val),
                "test": len(img_drop_test)
            }
        },
        "conflicts_audit": conflicts_audit,
        "intra_domain_balancing_audit": balancing_audit,
        "split_statistics": {
            "train": get_stats(combined_train),
            "validation": get_stats(combined_val),
            "test": get_stats(combined_test)
        }
    }

    report_path = output_dir / "combined_dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=4, ensure_ascii=False)

    logger.info(f"Combined splits successfully written to: {output_dir}")
    logger.info(
        f"Train: {len(combined_train)} | Validation: {len(combined_val)} | Test: {len(combined_test)}"
    )
    logger.info(f"Report saved to: {report_path}")

    # Console Summary
    print("\n" + "=" * 78)
    print("AGENTSHIELD COMBINED DATASET SUMMARY & AUDIT")
    print("=" * 78)
    for split_name in ["train", "validation", "test"]:
        s = report["split_statistics"][split_name]
        pct_b = round((s["benign"] / s["total"]) * 100, 1) if s["total"] > 0 else 0
        pct_m = round((s["malicious"] / s["total"]) * 100, 1) if s["total"] > 0 else 0
        print(
            f"{split_name.upper():<12}: Total = {s['total']:<6} | "
            f"Benign = {s['benign']:<5} ({pct_b}%) | Malicious = {s['malicious']:<5} ({pct_m}%)"
        )
        for dom, dstat in s["by_domain_and_label"].items():
            print(f"   -> {dom:<6}: Total={dstat['total']:<5} (Benign={dstat['benign']}, Malicious={dstat['malicious']})")
    print("=" * 78 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine Image OCR, PDF, and Web datasets with quality filtering")
    parser.add_argument("--pdf_train_samples", type=int, default=1600, help="Number of PDF samples in train set")
    parser.add_argument("--pdf_val_samples", type=int, default=400, help="Number of PDF samples in validation set")
    parser.add_argument("--pdf_test_samples", type=int, default=400, help="Number of PDF samples in test set")
    parser.add_argument("--filter_empty_ocr", action="store_true", default=True, help="Filter out empty OCR / boilerplate text")
    parser.add_argument("--no_filter_empty_ocr", action="store_false", dest="filter_empty_ocr", help="Do not filter empty OCR")
    parser.add_argument("--filter_conflicts", action="store_true", default=True, help="Filter out cross-label conflicting identical texts")
    parser.add_argument("--no_filter_conflicts", action="store_false", dest="filter_conflicts", help="Do not filter conflicts")
    parser.add_argument("--balance_domain_classes", action="store_true", default=True, help="Balance benign/malicious 1:1 within each domain")
    parser.add_argument("--no_balance_classes", action="store_false", dest="balance_domain_classes", help="Do not balance domain classes")
    parser.add_argument("--full_pdf", action="store_true", help="Include all PDF samples without sub-sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "data" / "processed" / "combined"

    combine_all_datasets(
        project_root=project_root,
        output_dir=output_dir,
        pdf_train_samples=args.pdf_train_samples,
        pdf_val_samples=args.pdf_val_samples,
        pdf_test_samples=args.pdf_test_samples,
        filter_empty_ocr=args.filter_empty_ocr,
        filter_conflicts=args.filter_conflicts,
        balance_domain_classes=args.balance_domain_classes,
        full_pdf=args.full_pdf,
        seed=args.seed
    )
