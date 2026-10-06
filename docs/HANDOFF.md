# CATS handoff (for the next team member)

## What you receive
A working Trust & Analysis layer. Input: the DeBERTa/ViT threat probabilities for a tool response (plus the user's task).
Output: risk, trust and an **ACCEPT / SANITIZE / REJECT** decision with an explanation. See `CATS_DOCUMENTATION.md` for the maths.

## How to call it
```python
from src.cats import CATSEngine, load_config
from src.cats.integration import CATSAgentShield

shield = CATSAgentShield(engine=CATSEngine(load_config("configs/cats_default.json")))
out = shield.assess_pdf(pdf_text, user_intent="Extract the invoice total.")   # also assess_web / assess_image / assess_text
cats = out["cats"]
if cats["decision"] == "REJECT":   ...   # block the tool response
elif cats["decision"] == "SANITIZE": ... # NOT implemented yet: your layer decides what to do (strip, quarantine, ask user)
else: ...                                # pass to the agent
```
CLI: `python src/predict_pipeline.py --pdf "<text>" --cats --user_intent "<task>" --json`

Output keys: `decision, risk_score, trust_score, text_threat, visual_threat, semantic_alignment, reason, notes, details`.
Real example (our run): text_threat 0.9851, semantic_alignment 0.5315, risk 0.9885, trust 0.0115, REJECT.

## What you must supply / know
* **`user_intent` must come from the agent runtime.** Without it alignment is unavailable and CATS reduces to threat fusion (still works).
* Models: `models/deberta-common/` and `models/vision-image/` (weights not in the source zip). The embedding model `all-MiniLM-L6-v2` downloads on first use (needs internet once, or pre-cache it).
* Tunables live in `configs/cats_default.json` (weights, lambda, thresholds, fusion mode, per-modality `profiles`). Defaults are BASELINE values, not validated.
* `SANITIZE` is only a label. No sanitiser exists.

## Measured behaviour (validation unless stated; see docs section 13-14)
* Image val: baseline CATS F1 0.9592 vs ViT-only 0.9720. DeBERTa-only 0.8750.
* PDF val (n=7005): CATS REJECT F1 0.8770 (precision 0.9605), SANITIZE|REJECT F1 0.8956 (recall 0.9546), DeBERTa-only 0.9076.
  REJECT is the high-precision setting; SANITIZE|REJECT is the high-recall setting.
* Fusion and semantic alignment did **not** measurably improve ranking (ROC-AUC) over the best single signal.
* Image test with the validation-tuned config: CATS F1 0.9829, but its AUC equals ViT-only (0.9940): gain is threshold calibration.

## Final findings (see docs section 15)
* Tuned CATS on images is exactly equivalent to ViT-only with threshold 0.85 (validation F1 0.9766; test F1 0.9805 on 296 images). Calibrating thresholds helped; fusion did not add anything.
* Alignment alone has AUC 0.59 on PDF data; adds +0.0002 AUC on top of DeBERTa.
* Recommendation for the runtime: treat thresholds as policy knobs. For images, a calibrated V threshold of about 0.85 is the evidence-supported REJECT cut-off. DeBERTa's 0.5 cut-off is not optimal either (0.15 scored better on image-OCR validation).

## Open items
1. Implement the SANITIZE action and decide the runtime policy for each decision.
2. Calibrate DeBERTa thresholds per domain (web, PDF) on validation before final use; test once.
3. Optional: evaluate on web data; try a richer alignment signal (e.g. per-sentence instruction detection).

## Verify the install
`python -m unittest tests.test_cats` -> expect `OK` at the end of the output.
