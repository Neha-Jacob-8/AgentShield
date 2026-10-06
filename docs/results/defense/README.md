# Defense & Integration evaluation reports

Produced by `python -m src.defense.evaluate` with the real DeBERTa / ViT weights (Apple-silicon GPU).
Every setting was chosen on the **validation** files; the **test** files were produced once, afterwards.

| File | Split | What it contains |
|---|---|---|
| `test_report.json` | test | **Final results**: detection, text-tool decisions, images, adaptive defense, sanitizer (`docs/DEFENSE_INTEGRATION.md` §10). The default-row key was re-labelled after the run (see its `note`); numbers unchanged. |
| `test_images_cats_default.json` | test | Image baseline with the team's original `cats_default.json` ("before" row for images) |
| `validation_report_run1.json` | validation | First run: detection and adaptive parts are valid. Its runtime part is **superseded**: image-OCR records were run through the text path and the lenient variant was mis-derived (both fixed). |
| `validation_report_run2.json` | validation | Runtime, images and sanitizer with the fixes (spec post-check default at the time) |
| `validation_post_check_modes.json` | validation | Text-tool decisions for spec / lenient / evidence-aware post-check: the basis for choosing `evidence_aware` |
| `validation_images_cats_runtime.json` | validation | Images with the image profile of `configs/cats_runtime.json`: the basis for adopting it |
