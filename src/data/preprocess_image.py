"""
AgentShield: An Adaptive Security Runtime for Autonomous AI Agents
Part 1 — Image Dataset Preprocessing Pipeline

This script validates, preprocesses, extracts OCR text (with timeout safety),
groups duplicates by SHA-256 hash, and generates stratified Train/Validation/Test
splits for the AgentShield Image Dataset.

OCR Note:
- Images are downscaled to max 640px before OCR for speed.
- Per-image OCR timeout (30s) prevents infinite hangs on large screenshots.
- Images that time out get ocr_status='timeout' but remain in the dataset.
"""

import os
import sys
import json
import hashlib
import numpy as np
import ctypes
import threading

# Force UTF-8 stdout for Windows console compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# OCR Engine Init (det_limit_side_len controls internal resize)
# ------------------------------------------------------------------
OCR_ENGINE = None
OCR_ENGINE_TYPE = None

try:
    from rapidocr_onnxruntime import RapidOCR
    # det_limit_side_len tells RapidOCR to NOT internally upscale large images
    OCR_ENGINE = RapidOCR(det_limit_side_len=736)
    OCR_ENGINE_TYPE = "rapidocr"
    print("[INFO] OCR Engine: RapidOCR (ONNX) det_limit_side_len=736.", flush=True)
except Exception as e1:
    try:
        import easyocr
        OCR_ENGINE = easyocr.Reader(['en'], gpu=False)
        OCR_ENGINE_TYPE = "easyocr"
        print("[INFO] OCR Engine: EasyOCR initialized.", flush=True)
    except Exception as e2:
        try:
            import pytesseract
            OCR_ENGINE = pytesseract
            OCR_ENGINE_TYPE = "pytesseract"
            print("[INFO] OCR Engine: PyTesseract initialized.", flush=True)
        except Exception as e3:
            print("[WARNING] No OCR library available.", flush=True)
            OCR_ENGINE_TYPE = "none"


# ------------------------------------------------------------------
# Thread-based OCR with timeout
# ------------------------------------------------------------------
class OCRResult:
    def __init__(self):
        self.text = ""
        self.status = "no_text"
        self.done = False


def _ocr_worker(img_np, result: OCRResult):
    """Thread worker: run RapidOCR on a numpy array."""
    try:
        res, _ = OCR_ENGINE(img_np)
        if res:
            lines = [line[1].strip() for line in res if line and len(line) > 1 and line[1]]
            full_text = " ".join(lines).strip()
            result.text = full_text if full_text else ""
            result.status = "text_found" if full_text else "no_text"
        else:
            result.text = ""
            result.status = "no_text"
    except Exception as e:
        result.text = ""
        result.status = f"error: {str(e)}"
    finally:
        result.done = True


def _kill_thread(t: threading.Thread):
    """Force-kill a Python thread via ctypes (Windows-compatible)."""
    if not t.is_alive():
        return
    tid = t.ident
    if tid is None:
        return
    try:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid),
            ctypes.py_object(SystemExit)
        )
    except Exception:
        pass


def extract_ocr_text(pil_img, img_path, timeout_sec=30):
    """
    Extracts OCR text with a per-image timeout.
    Returns (extracted_text, status)
    status values: 'text_found', 'no_text', 'timeout', 'error: ...'
    """
    if OCR_ENGINE_TYPE == "rapidocr" and OCR_ENGINE is not None:
        try:
            # Scale image to max 640px on longest side before OCR
            w, h = pil_img.size
            max_side = max(w, h)
            if max_side > 640:
                scale = 640 / max_side
                ocr_img = pil_img.resize(
                    (int(w * scale), int(h * scale)),
                    Image.Resampling.BILINEAR
                )
            else:
                ocr_img = pil_img

            if ocr_img.mode != 'RGB':
                ocr_img = ocr_img.convert('RGB')

            img_np = np.array(ocr_img)

            result = OCRResult()
            t = threading.Thread(target=_ocr_worker, args=(img_np, result), daemon=True)
            t.start()
            t.join(timeout=timeout_sec)

            if not result.done:
                _kill_thread(t)
                return "", "timeout"

            return result.text, result.status

        except Exception as e:
            return "", f"error: {str(e)}"

    elif OCR_ENGINE_TYPE == "easyocr" and OCR_ENGINE is not None:
        try:
            results = OCR_ENGINE.readtext(str(img_path), detail=0)
            full_text = " ".join([r.strip() for r in results if r.strip()]).strip()
            return (full_text, "text_found") if full_text else ("", "no_text")
        except Exception as e:
            return "", f"error: {str(e)}"

    elif OCR_ENGINE_TYPE == "pytesseract" and OCR_ENGINE is not None:
        try:
            text = OCR_ENGINE.image_to_string(pil_img).strip()
            return (text, "text_found") if text else ("", "no_text")
        except Exception as e:
            return "", f"error: {str(e)}"

    return "", "no_text"


# ------------------------------------------------------------------
# Dataset Discovery
# ------------------------------------------------------------------
def find_raw_image_dir(project_root):
    candidates = [
        project_root / "data" / "raw" / "image",
        project_root.parent / "Dataset_image",
        project_root / "Dataset_image"
    ]
    for cand in candidates:
        if cand.exists() and any(cand.iterdir()):
            return cand
    raise FileNotFoundError("Could not locate raw image dataset folder!")


# ------------------------------------------------------------------
# Main Preprocessing
# ------------------------------------------------------------------
def process_dataset(project_root):
    raw_dir = find_raw_image_dir(project_root)
    processed_dir = project_root / "data" / "processed" / "image"
    processed_img_copies_dir = processed_dir / "processed_images"

    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_img_copies_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Raw image directory  : {raw_dir}", flush=True)
    print(f"[INFO] Output directory     : {processed_dir}", flush=True)

    # Collect all image paths
    all_image_paths = []
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']:
                all_image_paths.append(Path(root) / f)

    all_image_paths = sorted(all_image_paths)
    total_found = len(all_image_paths)
    print(f"[INFO] Found {total_found} image files.", flush=True)

    records = []
    hashes = defaultdict(list)
    corrupted_files = []
    ocr_failures = []

    for idx, img_path in enumerate(all_image_paths, 1):
        if idx % 50 == 0 or idx == total_found:
            print(f"[PROGRESS] {idx}/{total_found} images processed...", flush=True)

        # PART B — Validate
        if not img_path.exists():
            corrupted_files.append({"path": str(img_path), "error": "File missing"})
            continue

        try:
            with Image.open(img_path) as img:
                img.verify()

            with Image.open(img_path) as img:
                width, height = img.size
                image_format = img.format or img_path.suffix[1:].upper()

                # PART C — Save processed RGB copy
                rel_subpath = img_path.relative_to(raw_dir)
                processed_img_path = processed_img_copies_dir / rel_subpath
                processed_img_path.parent.mkdir(parents=True, exist_ok=True)

                if img.mode != 'RGB':
                    rgb_img = img.convert('RGB')
                    rgb_img.save(processed_img_path)
                    working_img = rgb_img
                else:
                    img.save(processed_img_path)
                    working_img = img.copy()

                # PART D — OCR (with timeout)
                extracted_text, ocr_status = extract_ocr_text(working_img, processed_img_path)

        except Exception as e:
            corrupted_files.append({"path": str(img_path), "error": str(e)})
            continue

        # SHA-256 for duplicate detection
        with open(img_path, 'rb') as fp:
            file_hash = hashlib.sha256(fp.read()).hexdigest()

        # Label and category from directory structure
        parts = rel_subpath.parts
        if parts[0] == 'benign':
            label = 0
            category = parts[1] if len(parts) > 1 else 'benign'
        elif parts[0] == 'malicious':
            label = 1
            category = parts[1] if len(parts) > 1 else 'malicious'
        else:
            label = -1
            category = 'unknown'

        # Track OCR errors
        if ocr_status.startswith("error"):
            ocr_failures.append({"path": str(img_path), "error": ocr_status})
            ocr_status = "error"
            extracted_text = ""

        # PART F — Standardized text field
        text_field = f"Image content: {extracted_text}".strip() if extracted_text else "Image content:"

        # PART G — Record
        record = {
            "domain": "image",
            "image_path": str(img_path.resolve()),
            "processed_image_path": str(processed_img_path.resolve()),
            "category": category,
            "label": label,
            "width": width,
            "height": height,
            "image_format": image_format,
            "extracted_text": extracted_text,
            "ocr_status": ocr_status,
            "text": text_field,
            "sha256": file_hash
        }
        records.append(record)
        hashes[file_hash].append(record)

    # PART H — Duplicate Detection
    dup_groups = [g for g in hashes.values() if len(g) > 1]
    duplicate_count = sum(len(g) - 1 for g in dup_groups)
    unique_hashes = list(hashes.keys())

    print(f"\n[INFO] Duplicates: {duplicate_count} copies across {len(dup_groups)} groups.", flush=True)

    # PART I — Stratified Split by unique hash groups
    hash_labels = [hashes[h][0]["label"] for h in unique_hashes]

    h_train, h_temp, _, y_temp = train_test_split(
        unique_hashes, hash_labels, test_size=0.20, random_state=42, stratify=hash_labels
    )
    h_val, h_test, _, _ = train_test_split(
        h_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
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

    def save_jsonl(filepath, rec_list):
        with open(filepath, 'w', encoding='utf-8') as f:
            for r in rec_list:
                clean_r = {k: v for k, v in r.items() if k != "sha256"}
                f.write(json.dumps(clean_r, ensure_ascii=False) + '\n')

    save_jsonl(processed_dir / "train.jsonl", train_records)
    save_jsonl(processed_dir / "validation.jsonl", val_records)
    save_jsonl(processed_dir / "test.jsonl", test_records)

    print(f"[INFO] Train : {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}", flush=True)

    # PART J — Statistics Report
    total_images = len(records)
    benign_records = [r for r in records if r["label"] == 0]
    malicious_records = [r for r in records if r["label"] == 1]

    benign_ocr = [r for r in benign_records if r["ocr_status"] == "text_found"]
    malicious_ocr = [r for r in malicious_records if r["ocr_status"] == "text_found"]
    benign_no_ocr = [r for r in benign_records if r["ocr_status"] in ("no_text", "timeout")]
    malicious_no_ocr = [r for r in malicious_records if r["ocr_status"] in ("no_text", "timeout")]
    ocr_timeouts = [r for r in records if r["ocr_status"] == "timeout"]

    report = {
        "dataset_summary": {
            "total_images": total_images,
            "benign_images": len(benign_records),
            "malicious_images": len(malicious_records),
            "corrupted_images": len(corrupted_files),
            "duplicate_images_count": duplicate_count,
            "duplicate_groups_count": len(dup_groups),
            "ocr_engine_used": OCR_ENGINE_TYPE
        },
        "category_breakdown": dict(sorted(Counter(r["category"] for r in records).items())),
        "ocr_statistics": {
            "images_with_ocr_text": len([r for r in records if r["ocr_status"] == "text_found"]),
            "images_with_no_ocr_text": len([r for r in records if r["ocr_status"] == "no_text"]),
            "ocr_timeouts": len(ocr_timeouts),
            "ocr_failures": len(ocr_failures),
            "malicious_images_with_ocr_text": len(malicious_ocr),
            "malicious_images_without_ocr_text": len(malicious_no_ocr),
            "pct_malicious_with_ocr": round(len(malicious_ocr) / len(malicious_records) * 100, 2) if malicious_records else 0,
            "benign_images_with_ocr_text": len(benign_ocr),
            "benign_images_without_ocr_text": len(benign_no_ocr),
            "pct_benign_with_ocr": round(len(benign_ocr) / len(benign_records) * 100, 2) if benign_records else 0
        },
        "image_properties": {
            "format_distribution": dict(Counter(r["image_format"] for r in records)),
            "dimension_stats": {
                "min_width": int(min(r["width"] for r in records)) if records else 0,
                "max_width": int(max(r["width"] for r in records)) if records else 0,
                "avg_width": float(round(np.mean([r["width"] for r in records]), 2)) if records else 0,
                "min_height": int(min(r["height"] for r in records)) if records else 0,
                "max_height": int(max(r["height"] for r in records)) if records else 0,
                "avg_height": float(round(np.mean([r["height"] for r in records]), 2)) if records else 0
            }
        },
        "split_statistics": {
            "train": {"total": len(train_records),
                      "benign": len([r for r in train_records if r["label"] == 0]),
                      "malicious": len([r for r in train_records if r["label"] == 1])},
            "validation": {"total": len(val_records),
                           "benign": len([r for r in val_records if r["label"] == 0]),
                           "malicious": len([r for r in val_records if r["label"] == 1])},
            "test": {"total": len(test_records),
                     "benign": len([r for r in test_records if r["label"] == 0]),
                     "malicious": len([r for r in test_records if r["label"] == 1])}
        }
    }

    report_path = processed_dir / "preprocessing_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Report saved: {report_path}", flush=True)

    # PART K — 5 Sample Records
    print("\n" + "="*70, flush=True)
    print("PART K — SAMPLE PROCESSED RECORDS", flush=True)
    print("="*70, flush=True)

    benign_no_txt = next((r for r in benign_records if r["ocr_status"] in ("no_text", "timeout")), None)
    benign_txt    = next((r for r in benign_records if r["ocr_status"] == "text_found"), None)
    screenshot_ex = next((r for r in records if "screenshot" in r["category"]), None)
    mal_txt       = next((r for r in malicious_records if r["ocr_status"] == "text_found"), None)
    mal_no_txt    = next((r for r in malicious_records if r["ocr_status"] in ("no_text", "timeout")), None)

    samples = [s for s in [benign_no_txt, benign_txt, screenshot_ex, mal_txt, mal_no_txt] if s]
    if len(samples) < 5:
        samples = records[:5]

    for i, s in enumerate(samples[:5], 1):
        clean = {k: v for k, v in s.items() if k != "sha256"}
        if len(clean["extracted_text"]) > 120:
            clean["extracted_text"] = clean["extracted_text"][:120] + " ...[TRUNCATED]"
        if len(clean["text"]) > 140:
            clean["text"] = clean["text"][:140] + " ...[TRUNCATED]"
        print(f"\n--- Sample {i} ({s['category']} | label={s['label']} | ocr={s['ocr_status']}) ---", flush=True)
        print(json.dumps(clean, indent=2, ensure_ascii=False), flush=True)

    return report


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    process_dataset(project_root)
