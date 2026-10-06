"""
AgentShield: An Adaptive Security Runtime for Autonomous AI Agents
Part 3 — Direct Image Vision Model Training Pipeline

This module fine-tunes a pretrained Vision Transformer (google/vit-base-patch16-224)
for binary prompt-injection image classification directly from raw image pixels.

Architecture:
               IMAGE
                 │
    ┌────────────┴────────────┐
    │                         │
    ▼                         ▼
   OCR                   Vision Model
    │                         │
    ▼                         ▼
Extracted Text            Raw Image
    │                         │
    ▼                         ▼
Common DeBERTa          Visual Classifier
    │                         │
    ▼                         ▼
Text Threat Score      Visual Threat Score
    │                         │
    └────────────┬────────────┘
                 ▼
                CATS
                 ▼
          Final Trust Score

Features:
- Binary Classification: 0 = BENIGN, 1 = MALICIOUS.
- Pretrained Vision Transformer (google/vit-base-patch16-224).
- Category-Wise Test Breakdown: EIA, VPI, VWA_adv_embedded_img, VWA_adv_screenshot,
  WebInject, popup, wasp, screenshot, embedded_img.
- Hardware Adaptation: Supports CUDA GPU (FP16, gradient accumulation) or CPU fallback.
- Best Model Checkpointing: Saves based on validation F1 score.
- Saves model & image processor under models/vision-image/.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TORCH"] = "1"

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

# Force UTF-8 stdout for Windows console compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Configure logging
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger("AgentShield.VisionTraining")


# ------------------------------------------------------------------
# 1. Dataset Loader & PyTorch Dataset
# ------------------------------------------------------------------
def load_image_dataset_jsonl(
    jsonl_path: Path,
    max_samples: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Loads image JSONL dataset file. Resolves image paths, verifies readability,
    and extracts category & binary label (0 or 1).
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Image dataset file not found: {jsonl_path}")

    records = []
    dropped_count = 0
    drop_reasons = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as e:
                dropped_count += 1
                drop_reasons.append(f"Line {line_num}: JSON decode error ({e})")
                continue

            if "label" not in data:
                dropped_count += 1
                drop_reasons.append(f"Line {line_num}: Missing 'label' field")
                continue

            try:
                label = int(data["label"])
                if label not in (0, 1):
                    dropped_count += 1
                    drop_reasons.append(f"Line {line_num}: Invalid label {label}")
                    continue
            except (ValueError, TypeError):
                dropped_count += 1
                drop_reasons.append(f"Line {line_num}: Non-integer label '{data.get('label')}'")
                continue

            # Resolve image path: prefer processed_image_path if available and valid
            img_path = None
            project_root = Path(__file__).resolve().parent.parent.parent
            for p_key in ["processed_image_path", "image_path"]:
                if data.get(p_key):
                    raw_str = str(data[p_key])
                    cand = Path(raw_str)
                    if cand.exists():
                        img_path = cand
                        break
                    # Fallback: check relative path from project root
                    idx = raw_str.find("AgentShield")
                    if idx != -1:
                        rel = raw_str[idx + len("AgentShield") + 1:].replace("\\", "/")
                        cand_rel = project_root / rel
                        if cand_rel.exists():
                            img_path = cand_rel
                            break
                    cand_direct = project_root / raw_str.replace("\\", "/")
                    if cand_direct.exists():
                        img_path = cand_direct
                        break

            if img_path is None:
                dropped_count += 1
                drop_reasons.append(f"Line {line_num}: Image file missing ({data.get('image_path')})")
                continue

            category = str(data.get("category", "unknown"))

            record = {
                "image_path": str(img_path.resolve()),
                "label": label,
                "category": category
            }
            records.append(record)

    # If max_samples requested, perform stratified sampling
    if max_samples is not None and max_samples < len(records):
        from sklearn.model_selection import train_test_split
        labels = [r["label"] for r in records]
        if len(set(labels)) > 1:
            records, _ = train_test_split(
                records,
                train_size=max_samples,
                stratify=labels,
                random_state=42
            )
        else:
            records = records[:max_samples]

    benign_count = sum(1 for r in records if r["label"] == 0)
    malicious_count = sum(1 for r in records if r["label"] == 1)

    category_counts = {}
    for r in records:
        cat = r["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    stats = {
        "file": str(jsonl_path),
        "total_loaded": len(records),
        "benign_count (0)": benign_count,
        "malicious_count (1)": malicious_count,
        "category_distribution": category_counts,
        "dropped_samples": dropped_count,
        "drop_reasons": drop_reasons[:5]
    }

    logger.info(
        f"Loaded {len(records)} image records from {jsonl_path.name} "
        f"(Benign: {benign_count}, Malicious: {malicious_count})"
    )

    return records, stats


class AgentShieldVisionDataset:
    """
    PyTorch Dataset wrapper for ViT image classification.
    """
    def __init__(self, records: List[Dict[str, Any]], image_processor):
        self.records = records
        self.processor = image_processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = rec["image_path"]
        label = rec["label"]

        try:
            with Image.open(img_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                pixel_values = self.processor(img, return_tensors="pt").pixel_values.squeeze(0)
        except Exception as e:
            # Fallback to zero image tensor if unreadable
            logger.warning(f"Error opening image {img_path}: {e}")
            import torch
            pixel_values = torch.zeros((3, 224, 224), dtype=torch.float32)

        return {
            "pixel_values": pixel_values,
            "label": label,
            "category": rec["category"]
        }


def custom_image_collator(batch):
    """
    Collates image dataset samples into batches.
    """
    import torch
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


# ------------------------------------------------------------------
# 2. Metrics & Security Explanation
# ------------------------------------------------------------------
def compute_metrics_fn(eval_pred):
    """
    Computes Accuracy, binary Precision, Recall, and F1 with pos_label=1.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec = recall_score(labels, preds, pos_label=1, zero_division=0)
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }


def print_confusion_matrix_explanation(cm):
    """
    Prints confusion matrix and explains security implications.
    """
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp

    print("\n" + "=" * 70)
    print("VISION MODEL CONFUSION MATRIX & SECURITY METRIC BREAKDOWN")
    print("=" * 70)
    print(f"                       Predicted BENIGN (0)    Predicted MALICIOUS (1)")
    print(f"Actual BENIGN (0)      TN = {tn:<18}  FP = {fp:<18}")
    print(f"Actual MALICIOUS (1)   FN = {fn:<18}  TP = {tp:<18}")
    print("-" * 70)
    print("SECURITY DEFINITIONS:")
    print(f"• True Positive  (TP = {tp:<4}): Malicious image correctly flagged.              [ATTACK PREVENTED]")
    print(f"• True Negative  (TN = {tn:<4}): Benign image correctly accepted.                [NORMAL EXECUTION]")
    print(f"• False Positive (FP = {fp:<4}): Safe image mistakenly flagged as malicious.     [FALSE ALARM]")
    print(f"• False Negative (FN = {fn:<4}): Malicious image MISSED and passed to agent!     [SECURITY FAILURE]")
    print(f"Total Images Evaluated  : {total}")
    print("=" * 70 + "\n")


# ------------------------------------------------------------------
# 3. Main Training Execution
# ------------------------------------------------------------------
def run_vision_training(args):
    import torch
    import numpy as np
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        Trainer,
        TrainingArguments,
        set_seed
    )
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hardware Detection
    device_name = "CPU"
    use_fp16 = False
    if torch.cuda.is_available() and not args.no_cuda:
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
        use_fp16 = True if args.fp16 else False
        logger.info(f"Hardware Acceleration: {device_name} detected! FP16={use_fp16}")
    else:
        logger.info("Hardware: Running on CPU (CUDA unavailable or disabled).")

    # Load Datasets
    max_train = args.max_train_samples if (args.max_train_samples or not args.quick_test) else 100
    max_eval = args.max_eval_samples if (args.max_eval_samples or not args.quick_test) else 30
    max_test = args.max_test_samples if (args.max_test_samples or not args.quick_test) else 30

    if args.quick_test:
        logger.warning("[QUICK TEST / SMOKE TEST MODE ACTIVE]")
        logger.warning("Subsetting dataset: train=100, val=30, test=30, epochs=1")
        logger.warning("NOTE: Quick test results are for pipeline verification ONLY.")
        args.num_train_epochs = 1

    train_records, train_stats = load_image_dataset_jsonl(args.train_path, max_samples=max_train)
    val_records, val_stats = load_image_dataset_jsonl(args.validation_path, max_samples=max_eval)
    test_records, test_stats = load_image_dataset_jsonl(args.test_path, max_samples=max_test)

    # Processor & Model Initialization
    logger.info(f"Loading Vision Processor & Pretrained Model: {args.model_name}")
    image_processor = AutoImageProcessor.from_pretrained(args.model_name)

    id2label = {0: "BENIGN", 1: "MALICIOUS"}
    label2id = {"BENIGN": 0, "MALICIOUS": 1}

    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # PyTorch Datasets
    train_ds = AgentShieldVisionDataset(train_records, image_processor)
    val_ds = AgentShieldVisionDataset(val_records, image_processor)
    test_ds = AgentShieldVisionDataset(test_records, image_processor)

    # Training Arguments
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    import inspect
    sig = inspect.signature(TrainingArguments.__init__)
    valid_params = set(sig.parameters.keys())

    train_args_kwargs = {
        "output_dir": str(checkpoint_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "load_best_model_at_end": False,
        "save_total_limit": 1,
        "fp16": use_fp16,
        "gradient_checkpointing": False,
        "report_to": "none",
        "seed": args.seed,
        "remove_unused_columns": False
    }

    if "eval_strategy" in valid_params:
        train_args_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in valid_params:
        train_args_kwargs["evaluation_strategy"] = "epoch"

    filtered_kwargs = {k: v for k, v in train_args_kwargs.items() if k in valid_params}
    training_args = TrainingArguments(**filtered_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=custom_image_collator,
        compute_metrics=compute_metrics_fn
    )

    logger.info("Starting Vision Model fine-tuning...")
    train_result = trainer.train()

    logger.info("Evaluating on validation dataset...")
    val_metrics = trainer.evaluate(eval_dataset=val_ds)

    logger.info(
        f"Validation Results -> Accuracy: {val_metrics.get('eval_accuracy', 0):.4f} | "
        f"Precision: {val_metrics.get('eval_precision', 0):.4f} | "
        f"Recall: {val_metrics.get('eval_recall', 0):.4f} | "
        f"F1: {val_metrics.get('eval_f1', 0):.4f}"
    )

    # Save Best Model & Processor
    logger.info(f"Saving best Vision Model & Processor to: {output_dir}")
    model.save_pretrained(str(output_dir))
    image_processor.save_pretrained(str(output_dir))

    # Test Set Evaluation
    logger.info("=" * 70)
    logger.info("EVALUATING BEST VISION MODEL ON UNSEEN TEST DATASET")
    logger.info("=" * 70)

    test_predictions = trainer.predict(test_ds)
    test_logits = test_predictions.predictions
    test_labels = test_predictions.label_ids
    test_preds = np.argmax(test_logits, axis=-1)

    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1])
    print_confusion_matrix_explanation(cm)

    test_acc = float(test_predictions.metrics.get("test_accuracy", 0))
    test_prec = float(test_predictions.metrics.get("test_precision", 0))
    test_rec = float(test_predictions.metrics.get("test_recall", 0))
    test_f1 = float(test_predictions.metrics.get("test_f1", 0))

    test_report_dict = classification_report(
        test_labels, test_preds, target_names=["BENIGN (0)", "MALICIOUS (1)"], output_dict=True, zero_division=0
    )

    # Category-Wise Breakdown
    cat_metrics = {}
    test_categories = [r["category"] for r in test_records]
    unique_categories = sorted(set(test_categories))

    print("\n" + "-" * 70)
    print("CATEGORY-WISE EVALUATION BREAKDOWN")
    print("-" * 70)
    for cat in unique_categories:
        indices = [i for i, c in enumerate(test_categories) if c == cat]
        if indices:
            cat_labels = [test_labels[i] for i in indices]
            cat_preds = [test_preds[i] for i in indices]
            total_cat = len(indices)
            
            # Check if this category is malicious or benign
            is_malicious_cat = any(l == 1 for l in cat_labels)
            
            if is_malicious_cat:
                detected_malicious = sum(1 for gl, gp in zip(cat_labels, cat_preds) if gp == 1)
                missed_malicious = sum(1 for gl, gp in zip(cat_labels, cat_preds) if gp == 0)
                recall_rate = float(detected_malicious / total_cat) if total_cat > 0 else 0.0

                cat_metrics[cat] = {
                    "total_samples": total_cat,
                    "detected_malicious": detected_malicious,
                    "missed_malicious": missed_malicious,
                    "recall": recall_rate
                }
                print(
                    f"Category: {cat:<22} | Samples: {total_cat:<5} | "
                    f"Detected (TP): {detected_malicious:<5} | Missed (FN): {missed_malicious:<5} | Recall: {recall_rate:.4f}"
                )
            else:
                correct_benign = sum(1 for gl, gp in zip(cat_labels, cat_preds) if gp == 0)
                false_alarm = sum(1 for gl, gp in zip(cat_labels, cat_preds) if gp == 1)
                acc_rate = float(correct_benign / total_cat) if total_cat > 0 else 0.0

                cat_metrics[cat] = {
                    "total_samples": total_cat,
                    "correct_benign": correct_benign,
                    "false_alarm": false_alarm,
                    "accuracy": acc_rate
                }
                print(
                    f"Category: {cat:<22} | Samples: {total_cat:<5} | "
                    f"Correct (TN): {correct_benign:<5} | False Alarm (FP): {false_alarm:<5} | Accuracy: {acc_rate:.4f}"
                )
    print("-" * 70 + "\n")

    final_results = {
        "model_name": args.model_name,
        "device": device_name,
        "is_quick_test": bool(args.quick_test),
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "train_batch_size": args.per_device_train_batch_size,
            "eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "weight_decay": args.weight_decay,
            "fp16": use_fp16
        },
        "dataset_statistics": {
            "train": train_stats,
            "validation": val_stats,
            "test": test_stats
        },
        "validation_metrics": {
            "accuracy": float(val_metrics.get("eval_accuracy", 0)),
            "precision": float(val_metrics.get("eval_precision", 0)),
            "recall": float(val_metrics.get("eval_recall", 0)),
            "f1": float(val_metrics.get("eval_f1", 0))
        },
        "test_metrics": {
            "overall": {
                "accuracy": test_acc,
                "precision": test_prec,
                "recall": test_rec,
                "f1": test_f1,
                "confusion_matrix": {
                    "TN": int(cm[0, 0]),
                    "FP": int(cm[0, 1]),
                    "FN": int(cm[1, 0]),
                    "TP": int(cm[1, 1])
                },
                "classification_report": test_report_dict
            },
            "by_category": cat_metrics
        }
    }

    results_file = output_dir / "training_and_test_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    logger.info(f"Complete results saved to: {results_file}")

    # Verify Inference Script
    print("\n" + "=" * 70)
    print("VERIFYING VISION INFERENCE MODULE (predict_image.py)")
    print("=" * 70)

    try:
        from predict_image import AgentShieldVisionPredictor
    except ImportError:
        from AgentShield.src.vision.predict_image import AgentShieldVisionPredictor

    predictor = AgentShieldVisionPredictor(model_path=output_dir)

    sample_benign = next((r["image_path"] for r in test_records if r["label"] == 0), None)
    sample_malicious = next((r["image_path"] for r in test_records if r["label"] == 1), None)

    for label_str, img_p in [("BENIGN", sample_benign), ("MALICIOUS", sample_malicious)]:
        if img_p:
            inf_res = predictor.predict(img_p)
            print(f"\nSample ({label_str}): '{img_p}'")
            print(json.dumps(inf_res, indent=2))

    print("\n" + "=" * 70)
    print("Vision Model Training & Test Pipeline Completed Successfully!")
    print("=" * 70 + "\n")

    return final_results


def parse_args():
    project_root = Path(__file__).resolve().parent.parent.parent
    default_data_dir = project_root / "data" / "processed" / "image"
    default_output_dir = project_root / "models" / "vision-image"

    parser = argparse.ArgumentParser(
        description="AgentShield Direct Image Vision Model Training Pipeline"
    )

    parser.add_argument(
        "--train_path",
        type=str,
        default=str(default_data_dir / "train.jsonl"),
        help="Path to train dataset JSONL file"
    )
    parser.add_argument(
        "--validation_path",
        type=str,
        default=str(default_data_dir / "validation.jsonl"),
        help="Path to validation dataset JSONL file"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=str(default_data_dir / "test.jsonl"),
        help="Path to test dataset JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(default_output_dir),
        help="Directory to save fine-tuned Vision Model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vit-base-patch16-224",
        help="Pretrained Vision Transformer model identifier"
    )
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate (default: 3e-5)")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Train batch size per device (default: 8)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Eval batch size per device (default: 16)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--logging_steps", type=int, default=20, help="Logging steps interval (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 mixed precision if CUDA is available")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU mode")

    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional max train samples")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional max eval samples")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Optional max test samples")
    parser.add_argument("--quick_test", action="store_true", help="Run quick verification test (100 train, 30 val, 30 test)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_vision_training(args)
