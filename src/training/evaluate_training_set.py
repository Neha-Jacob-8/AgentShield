"""
AgentShield: Training Set Evaluation Script
Calculates complete classification metrics (Accuracy, Precision, Recall, F1, and Confusion Matrix)
for both the overall training set and broken down by domain (Web, PDF, Image OCR).
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TORCH"] = "1"

import sys
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Force UTF-8 stdout for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def evaluate_training_set(
    model_dir: Path,
    train_path: Path,
    batch_size: int = 16,
    max_length: int = 256
):
    print("=" * 70)
    print("EVALUATING COMMON DEBERTA ON TRAINING DATASET")
    print(f"Model Path  : {model_dir}")
    print(f"Dataset Path: {train_path}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. Load Dataset
    records = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))

    total_samples = len(records)
    print(f"[INFO] Loaded {total_samples} training samples.")

    # 2. Load Model & Tokenizer
    print("[INFO] Loading Tokenizer and Model...")
    load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded in {round(time.time() - load_start, 2)}s.")

    # 3. Batch Evaluation
    texts = [r["text"] for r in records]
    true_labels = [r["label"] for r in records]
    domains = [r.get("domain", "unknown") for r in records]

    all_preds = []
    start_eval = time.time()

    print(f"[INFO] Running evaluation across {total_samples} samples with batch_size={batch_size}...")
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(batch_preds)

            if (i // batch_size + 1) % 25 == 0 or (i + batch_size) >= total_samples:
                processed = min(i + batch_size, total_samples)
                elapsed = time.time() - start_eval
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_samples - processed) / rate if rate > 0 else 0
                print(f"  Processed {processed}/{total_samples} ({round(processed / total_samples * 100, 1)}%) "
                      f"- Speed: {round(rate, 1)} samples/s - ETA: {round(eta)}s", flush=True)

    all_preds = np.array(all_preds)
    true_labels = np.array(true_labels)

    # 4. Compute Overall Metrics
    acc = float(accuracy_score(true_labels, all_preds))
    prec = float(precision_score(true_labels, all_preds, zero_division=0))
    rec = float(recall_score(true_labels, all_preds, zero_division=0))
    f1 = float(f1_score(true_labels, all_preds, zero_division=0))
    cm = confusion_matrix(true_labels, all_preds, labels=[0, 1])

    # 5. Compute Per-Domain Metrics
    domain_breakdown = {}
    for dom in sorted(set(domains)):
        idx = [j for j, d in enumerate(domains) if d == dom]
        sub_true = true_labels[idx]
        sub_preds = all_preds[idx]
        sub_cm = confusion_matrix(sub_true, sub_preds, labels=[0, 1])

        domain_breakdown[dom] = {
            "samples": len(idx),
            "accuracy": float(accuracy_score(sub_true, sub_preds)),
            "precision": float(precision_score(sub_true, sub_preds, zero_division=0)),
            "recall": float(recall_score(sub_true, sub_preds, zero_division=0)),
            "f1": float(f1_score(sub_true, sub_preds, zero_division=0)),
            "confusion_matrix": {
                "TN": int(sub_cm[0, 0]),
                "FP": int(sub_cm[0, 1]),
                "FN": int(sub_cm[1, 0]),
                "TP": int(sub_cm[1, 1])
            }
        }

    results = {
        "dataset": "train",
        "total_samples": total_samples,
        "overall": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": {
                "TN": int(cm[0, 0]),
                "FP": int(cm[0, 1]),
                "FN": int(cm[1, 0]),
                "TP": int(cm[1, 1])
            }
        },
        "by_domain": domain_breakdown
    }

    # 6. Save results
    save_path = model_dir / "training_set_evaluation_results.json"
    with open(save_path, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("TRAINING SET EVALUATION METRICS REPORT")
    print("=" * 70)
    print(f"Overall Training Accuracy : {round(acc * 100, 2)}%")
    print(f"Overall Training Precision: {round(prec * 100, 2)}%")
    print(f"Overall Training Recall   : {round(rec * 100, 2)}%")
    print(f"Overall Training F1 Score : {round(f1 * 100, 2)}%")
    print("-" * 70)
    print("CONFUSION MATRIX (Training Set):")
    print(f"  TN (Benign Correct)  : {cm[0, 0]:<5} | FP (False Alarms)   : {cm[0, 1]:<5}")
    print(f"  FN (Missed Attacks)  : {cm[1, 0]:<5} | TP (Attacks Blocked): {cm[1, 1]:<5}")
    print("-" * 70)
    print("PER-DOMAIN TRAINING BREAKDOWN:")
    for dom, m in domain_breakdown.items():
        print(f"  {dom.upper():<8}: Samples={m['samples']:<5} | Acc={round(m['accuracy'] * 100, 2)}% | "
              f"Prec={round(m['precision'] * 100, 2)}% | Rec={round(m['recall'] * 100, 2)}% | "
              f"F1={round(m['f1'] * 100, 2)}%")
    print("=" * 70 + "\n")
    print(f"[INFO] Results saved to: {save_path}")

    return results


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    model_dir = project_root / "models" / "deberta-common"
    train_path = project_root / "data" / "processed" / "combined" / "train.jsonl"

    evaluate_training_set(model_dir=model_dir, train_path=train_path, batch_size=16)
