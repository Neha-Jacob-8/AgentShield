"""
AgentShield: Vision Model Training Set Evaluation Script
Calculates complete classification metrics (Accuracy, Precision, Recall, F1, and Confusion Matrix)
for the Direct Vision Transformer (ViT) on the full image training set (2,380 samples),
including per-category performance breakdown.
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
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Force UTF-8 stdout for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add src/vision to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_vision import load_image_dataset_jsonl, AgentShieldVisionDataset, custom_image_collator


def evaluate_vision_training_set(
    model_dir: Path,
    train_path: Path,
    batch_size: int = 32
):
    print("=" * 70)
    print("EVALUATING VISION TRANSFORMER (ViT) ON TRAINING DATASET")
    print(f"Model Path  : {model_dir}")
    print(f"Dataset Path: {train_path}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cpu":
        torch.set_num_threads(8)
        print(f"[INFO] PyTorch CPU threads set to: {torch.get_num_threads()}")

    # 1. Load Dataset
    records, stats = load_image_dataset_jsonl(train_path)
    total_samples = len(records)
    print(f"[INFO] Loaded {total_samples} image training records.")

    # 2. Load Model & Image Processor
    print("[INFO] Loading ViT Model and Image Processor...")
    load_start = time.time()
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = AutoModelForImageClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    print(f"[INFO] ViT model loaded in {round(time.time() - load_start, 2)}s.")

    # 3. Create DataLoader
    dataset = AgentShieldVisionDataset(records, processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_image_collator
    )

    # 4. Batch Evaluation
    all_preds = []
    all_labels = []
    categories = [r["category"] for r in records]

    start_eval = time.time()
    print(f"[INFO] Running evaluation across {total_samples} images with batch_size={batch_size}...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, 1):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]

            outputs = model(pixel_values=pixel_values)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

            if batch_idx % 10 == 0 or (batch_idx * batch_size) >= total_samples:
                processed = min(batch_idx * batch_size, total_samples)
                elapsed = time.time() - start_eval
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_samples - processed) / rate if rate > 0 else 0
                print(f"  Processed {processed}/{total_samples} ({round(processed / total_samples * 100, 1)}%) "
                      f"- Speed: {round(rate, 1)} images/s - ETA: {round(eta)}s", flush=True)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 5. Compute Overall Metrics
    acc = float(accuracy_score(all_labels, all_preds))
    prec = float(precision_score(all_labels, all_preds, pos_label=1, zero_division=0))
    rec = float(recall_score(all_labels, all_preds, pos_label=1, zero_division=0))
    f1 = float(f1_score(all_labels, all_preds, pos_label=1, zero_division=0))
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    # 6. Compute Category Breakdown
    cat_metrics = {}
    unique_cats = sorted(set(categories))
    for cat in unique_cats:
        idx = [j for j, c in enumerate(categories) if c == cat]
        c_true = all_labels[idx]
        c_preds = all_preds[idx]
        is_malicious = c_true[0] == 1 if len(c_true) > 0 else False

        if is_malicious:
            c_rec = float(recall_score(c_true, c_preds, pos_label=1, zero_division=0))
            detected = int(np.sum(c_preds == 1))
            cat_metrics[cat] = {
                "type": "malicious",
                "total_samples": len(idx),
                "detected_malicious": detected,
                "missed_malicious": len(idx) - detected,
                "recall": c_rec
            }
        else:
            c_acc = float(accuracy_score(c_true, c_preds))
            correct = int(np.sum(c_preds == 0))
            cat_metrics[cat] = {
                "type": "benign",
                "total_samples": len(idx),
                "correct_benign": correct,
                "false_alarm": len(idx) - correct,
                "accuracy": c_acc
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
        "by_category": cat_metrics
    }

    # 7. Save Results
    save_path = model_dir / "training_set_evaluation_results.json"
    with open(save_path, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, indent=4, ensure_ascii=False)

    consolidated_path = model_dir / "training_and_test_results.json"
    if consolidated_path.exists():
        try:
            with open(consolidated_path, "r", encoding="utf-8") as f:
                consolidated_data = json.load(f)
            consolidated_data["training_metrics"] = results["overall"]
            consolidated_data["training_by_category"] = results["by_category"]
            with open(consolidated_path, "w", encoding="utf-8") as f:
                json.dump(consolidated_data, f, indent=4, ensure_ascii=False)
            print(f"[INFO] Updated consolidated metrics in: {consolidated_path}")
        except Exception as e:
            print(f"[WARNING] Could not update {consolidated_path}: {e}")

    print("\n" + "=" * 70)
    print("VISION MODEL (ViT) TRAINING SET EVALUATION REPORT")
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
    print("CATEGORY BREAKDOWN (Training Set):")
    for cat, m in cat_metrics.items():
        if m["type"] == "malicious":
            print(f"  [MALICIOUS] {cat:<22}: Samples={m['total_samples']:<4} | "
                  f"Detected={m['detected_malicious']:<4} | Recall={round(m['recall'] * 100, 2)}%")
        else:
            print(f"  [BENIGN]    {cat:<22}: Samples={m['total_samples']:<4} | "
                  f"Accepted={m['correct_benign']:<4} | Accuracy={round(m['accuracy'] * 100, 2)}%")
    print("=" * 70 + "\n")
    print(f"[INFO] Results saved to: {save_path}")

    return results


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    model_dir = project_root / "models" / "vision-image"
    train_path = project_root / "data" / "processed" / "image" / "train.jsonl"

    evaluate_vision_training_set(model_dir=model_dir, train_path=train_path, batch_size=32)
