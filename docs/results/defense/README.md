# Defense & Integration evaluation reports

Produced by `python -m src.defense.evaluate` with the real DeBERTa / ViT weights (Apple-silicon GPU).
Every setting was chosen on the **validation** files; the **test** files were produced once, afterwards.

| File | Split | What it contains |
|---|---|---|
| `test_report_realdocs_model.json` | test | **Final results** with the text model fine-tuned on real documents (`models/deberta-common-realdocs`, the runtime default): detection, text-tool decisions, images, adaptive defense, sanitizer and whole real papers (`docs/DEFENSE_INTEGRATION.md` §14) |
| `test_realdocs_original_model.json` | test | Whole real papers with the original `deberta-common` (for comparison) |
| `model_choice/` | validation | The comparison that chose the text model and left the instruction gate off: original / fine-tuned model, each with and without the gate (detection, text tools, real papers), and the other parts for both models |
| `test_report.json` | test | Results with the original `deberta-common`: detection, text-tool decisions, images, adaptive defense, sanitizer (`docs/DEFENSE_INTEGRATION.md` §10). The parts that use the sanitizer were re-run after each segmentation fix with unchanged settings (see its `note`). |
| `test_images_cats_default.json` | test | Image baseline with the team's original `cats_default.json` ("before" row for images) |
| `validation_report_run1.json` | validation | First run: detection and adaptive parts are valid. Its runtime part is **superseded**: image-OCR records were run through the text path and the lenient variant was mis-derived (both fixed). |
| `validation_report_run2.json` | validation | Runtime, images and sanitizer with the fixes (spec post-check default at the time) |
| `validation_post_check_modes.json` | validation | Text-tool decisions for spec / lenient / evidence-aware post-check: the basis for choosing `evidence_aware` |
| `validation_images_cats_runtime.json` | validation | Images with the image profile of `configs/cats_runtime.json`: the basis for adopting it |
