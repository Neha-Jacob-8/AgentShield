# AgentShield: An Adaptive Security Runtime for Autonomous AI Agents

AgentShield is a security middleware designed to analyze external tool responses (Image, PDF, Web) and detect indirect prompt injection attacks before they reach autonomous AI agents.

---

## 1. Core Architecture — Dual-Branch Multi-Modal Defense

AgentShield uses a dual-branch architecture combining a single common text classifier with a direct vision model:

```
                            TOOL RESPONSE INPUTS
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
           ▼                         ▼                         ▼
      Image Input                PDF Input                 Web Input
           │                         │                         │
      Image Validation       Text Extraction / Parsing   HTML Parsing / Clean
           │                         │                         │
        RapidOCR                     │                         │
           │                         │                         │
           ▼                         ▼                         ▼
   {"text": "...",           {"text": "...",           {"text": "...",
    "domain": "image"}        "domain": "pdf"}          "domain": "web"}
           │                         │                         │
           └─────────────────────────┼─────────────────────────┘
                                     │
                                     ▼
                      Standardized Combined Dataset
                    (data/processed/combined/*.jsonl)
                                     │
                                     ▼
                       ONE COMMON DeBERTa MODEL
                   (models/deberta-common/deberta-v3-base)
                                     │
                                     ▼
                      Text Threat Score (DeBERTa)
```

### Dual-Branch Parallel Pipeline for Image Tool Outputs:
```
                                  Image Input
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            │                                                     │
            ▼                                                     ▼
 OCR → Common DeBERTa → Text Threat Score           Vision Model → Visual Threat Score
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       │
                                       ▼
                                     CATS
                                       │
                                       ▼
                            Trust Score / Decision
```

---

## 2. Image Preprocessing Pipeline (Completed)

The image preprocessing pipeline standardizes the raw image dataset (2,962 images), validates file integrity, extracts embedded text via RapidOCR (ONNX engine with timeout safety), preserves visual image assets, prevents duplicate data leakage across splits using SHA-256 group stratification, and generates stratified train/validation/test datasets.

### Image Dataset Statistics:

| Metric | Value |
| :--- | :--- |
| **Total Image Samples** | **2,962** |
| **Benign Samples (`label = 0`)** | **948** (32.01%) |
| **Malicious Samples (`label = 1`)** | **2,014** (67.99%) |
| **Corrupted / Unreadable Images** | **0** |
| **Exact Duplicate Images (SHA-256)** | **197 duplicate copies across 146 groups** |
| **Images with OCR Text Found** | **2,585** (87.27%) |
| **OCR Timeouts / Failures** | **1 timeout (retained safely with `ocr_status='timeout'`), 0 failures** |

### Category Breakdown:
- **Benign (948)**: `screenshot` (722), `embedded_img` (226)
- **Malicious (2,014)**: `WebInject` (500), `EIA` (496), `VWA_adv_embedded_img` (298), `VWA_adv_screenshot` (283), `popup` (216), `VPI` (145), `wasp` (76)

### Split Distribution (Stratified & Leakage-Free via SHA-256):
- **Training Set (80%)**: `2,380` images → `data/processed/image/train.jsonl`
- **Validation Set (10%)**: `286` images → `data/processed/image/validation.jsonl`
- **Testing Set (10%)**: `296` images → `data/processed/image/test.jsonl`

---

## 3. Common DeBERTa Training & Inference Pipeline

The DeBERTa training pipeline is **dataset-independent and reusable across all domains**. It requires only two standardized fields:
- `"text"`: String representation of content
- `"label"`: `0` (Benign) or `1` (Malicious)
- `"domain"` *(optional)*: `"image"`, `"pdf"`, or `"web"` for per-domain metric reporting (`Image F1`, `PDF F1`, `Web F1`).

### Running a Pipeline Smoke Test (Quick Verification):
```bash
python src/training/train_deberta.py --quick_test \
    --train_path data/processed/image/train.jsonl \
    --validation_path data/processed/image/validation.jsonl \
    --test_path data/processed/image/test.jsonl \
    --output_dir models/deberta-common
```

### Eventual Final Common Model Training Command:
*(To be executed after PDF and Web preprocessing are complete and combined dataset is created)*:
```bash
python src/training/train_deberta.py \
    --train_path data/processed/combined/train.jsonl \
    --validation_path data/processed/combined/validation.jsonl \
    --test_path data/processed/combined/test.jsonl \
    --output_dir models/deberta-common
```

---

## 4. Direct Image Vision Model Pipeline (Part 3)

### Why a Vision Model is Required:
The OCR + DeBERTa branch extracts textual prompt injections. However, image-based tool outputs may contain:
- Visual prompt injections (VPI) where text layout or adversarial patterns carry threats.
- Images with low OCR readability, unreadable text, or OCR timeouts.
- Visual structural threats (such as fake popup overlays or adversarial web injected artifacts).

The **Direct Image Vision Model** evaluates raw image pixels using a fine-tuned **Vision Transformer (`google/vit-base-patch16-224`)** for binary classification (`0 = Benign`, `1 = Malicious`).

### Model & Preprocessing:
- **Base Architecture**: Pretrained `google/vit-base-patch16-224` (`AutoImageProcessor`, `AutoModelForImageClassification`).
- **Binary Labels**: `0 = BENIGN`, `1 = MALICIOUS`.
- **Saved Model Location**: `models/vision-image/`

### Running Quick Verification Test (Smoke Test):
```bash
python src/vision/train_vision.py --quick_test
```

### Running Full Vision Model Fine-Tuning:
```bash
python src/vision/train_vision.py \
    --train_path data/processed/image/train.jsonl \
    --validation_path data/processed/image/validation.jsonl \
    --test_path data/processed/image/test.jsonl \
    --output_dir models/vision-image
```

### Running Image Inference (`predict_image.py`):
```bash
python src/vision/predict_image.py --image data/processed/image/processed_images/malicious/EIA/51.png
```

Example Inference Output:
```json
{
  "image_path": "data/processed/image/processed_images/malicious/EIA/51.png",
  "label": 1,
  "prediction": "malicious",
  "benign_probability": 0.4324,
  "malicious_probability": 0.5676
}
```

### Future CATS Connection:
The returned `malicious_probability` serves as the **Visual Threat Score** ($V_{threat}$). Downstream Context-Aware Trust Scoring (CATS) will fuse:
$$\text{Trust Score} = f(T_{threat}, V_{threat}, S_{semantic})$$
to make final Accept / Sanitize / Reject decisions.

---

## 5. Current Project Status

| Module / Component | Status | Location / Output |
| :--- | :--- | :--- |
| **Image Preprocessing & OCR** | **COMPLETED** | `src/data/preprocess_image.py` → `data/processed/image/` |
| **PDF Preprocessing** | **COMPLETED** | `src/data/preprocess_pdf.py` → `data/processed/pdf/` (70,000 samples) |
| **Web Preprocessing** | **COMPLETED** | `src/data/preprocess_web.py` → `data/processed/web/` (3,698 samples) |
| **Combined Dataset (`combined/`)** | **COMPLETED & OPTIMIZED** | `src/data/combine_datasets.py` → `data/processed/combined/` (4,494 train / 1,018 val / 988 test; 1:1 intra-domain class balanced, 0 label conflicts, 377 empty-OCR textless samples filtered) |
| **Direct Image Vision Model (ViT)** | **TRAINED & EVALUATED** | `src/vision/train_vision.py` → `models/vision-image/` (Test F1: 97.13%, Recall: 100%, 0 FN) |
| **Common DeBERTa Model (v3-base)** | **PIPELINE READY / TRAINED** | `src/training/train_deberta.py` → `models/deberta-common/` |
| **CATS (Trust Scoring Engine)** | **IMPLEMENTED (baseline formula & thresholds, NOT yet validated)** | `src/cats/` — see `docs/CATS_DOCUMENTATION.md` |
| **AgentShield Security Runtime** | **NOT YET IMPLEMENTED** | Future component |

---

## 6. Combined Dataset Optimization & Balance

The unified multi-modal text dataset (`data/processed/combined/`) features three key quality safeguards:
1. **Empty OCR Gating**: Textless images (377 samples across splits) are excluded from the pure text dataset to prevent identical `"Image content:"` boilerplate strings from receiving contradictory labels. (All images remain preserved for the Direct Vision ViT branch).
2. **Cross-Label Conflict Resolution**: Ambiguous text strings that appeared in both benign and malicious image splits due to imperceptible visual-only attacks are detected and filtered, ensuring zero contradictory gradient signals.
3. **Intra-Domain Class Balancing**: In the training split, each domain contributes an exact 1:1 benign-to-malicious balance (Web: 803/803, PDF: 800/800, Image: 644/644), eliminating confounding format shortcuts (Simpson's Paradox).

---

## 7. Evaluation Summary

### Direct Image Vision Model (`models/vision-image/`)
- **Architecture**: `google/vit-base-patch16-224`
- **Test Accuracy**: **95.95%**
- **Test Precision**: **94.42%**
- **Test Recall**: **100.00%** (203/203 malicious samples detected)
- **Test F1 Score**: **97.13%**
- **Confusion Matrix**: `TP=203`, `TN=81`, `FP=12`, `FN=0` (Zero missed attacks)

### Common DeBERTa-v3-base Model (`models/deberta-common/`)
- **Architecture**: `microsoft/deberta-v3-base` (Fine-tuned on cleaned, balanced multi-domain dataset)
- **Overall Test Accuracy**: **91.50%** (up from 84.00%, +7.50%)
- **Overall Test Precision**: **94.35%** (up from 79.22%, +15.13%)
- **Overall Test Recall**: **86.29%** (384/445 attacks detected)
- **Overall Test F1 Score**: **90.14%** (up from 83.56%, +6.58%)
- **Confusion Matrix**: `TP=384`, `TN=520`, `FP=23`, `FN=61` (False alarms dropped by 64% from 64 to 23)
- **Per-Domain Breakdown**:
  - **Web Text**: **97.13% Accuracy**, **98.57% Precision**, **88.46% Recall**, **F1: 93.24%** (up from 90.11%)
  - **PDF Text**: **90.75% Accuracy**, **94.51% Precision**, **86.43% Recall**, **F1: 90.29%** (up from 86.67%)
  - **Image OCR**: **84.52% Accuracy** (up from 65.50%), **92.26% Precision** (up from 67.47%), **85.12% Recall**, **F1: 88.54%** (up from 79.15%)

---

## 8. Data and Model Weights

The datasets (`data/`, ~3.9 GB) and the trained model weights are not stored in this repository because of GitHub's file-size limits. The model configs, tokenizer files and evaluation results are included.

To run training, inference or evaluation, place the files at:

```
data/raw/{image,pdf,web}/                   # raw datasets
data/processed/{image,pdf,web,combined}/    # produced by src/data/*.py
models/deberta-common/model.safetensors     # fine-tuned DeBERTa-v3-base (~738 MB)
models/vision-image/model.safetensors       # fine-tuned ViT (~343 MB)
```

The CATS unit tests run without data or weights:

```bash
python -m unittest tests.test_cats
```

