"""
Evaluate CATS against ground-truth labels and compare it with DeBERTa-only / ViT-only decisions.

Data: any AgentShield *.jsonl (data/processed/{image,pdf,web,combined}/{validation,test}.jsonl)
with fields: text, label, domain, and optionally user_intent, image_path /
processed_image_path, extracted_text, ocr_status.

Two ways to get model scores
  1. --scores_in FILE   : reuse pre-computed scores (fields t_threat / v_threat). No model weights needed.
  2. default            : run DeBERTa (and ViT for image records) from models/. Use --scores_out FILE
                          to save the scores so CATS settings can be re-evaluated without re-running models.

Examples
  python -m src.cats.evaluate --data data/processed/image/test.jsonl --scores_out outputs/cats/image_test_scores.jsonl
  python -m src.cats.evaluate --data data/processed/pdf/test.jsonl  --scores_out outputs/cats/pdf_test_scores.jsonl
  python -m src.cats.evaluate --scores_in outputs/cats/image_test_scores.jsonl --sweep_reject 0.5,0.6,0.7,0.8,0.9

Methodology notes
  * Tune thresholds/weights on the VALIDATION split only, then report once on TEST.
  * Baselines are the existing models' own decisions: DeBERTa-only: T >= 0.5, ViT-only: V >= 0.5.
  * CATS is reported two ways: REJECT-only counts as 'malicious', and SANITIZE+REJECT counts as 'malicious'.
  * Image/web datasets carry no user intent, so semantic alignment is only active on PDF records
    (their text contains 'User intent: ... Document content: ...').
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import load_config
from .engine import CATSEngine
from .integration import split_pdf_text, strip_domain_prefix

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- data
def load_jsonl(paths: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def record_context(rec: Dict[str, Any]):
    """-> (user_intent, content_for_embedding, has_text). Uses only fields that exist in the data."""
    domain, text = rec.get("domain", "text"), rec.get("text", "") or ""
    if domain == "pdf":
        intent, content = split_pdf_text(text)
        intent = rec.get("user_intent") or intent
    elif domain == "image":
        intent = None
        content = (rec.get("extracted_text") or strip_domain_prefix(text)).strip()
    else:
        intent, content = rec.get("user_intent"), strip_domain_prefix(text)
    has_text = bool(content)
    if "ocr_status" in rec:
        has_text = has_text and rec["ocr_status"] == "text_found"
    return intent, content, has_text


# ----------------------------------------------------------------------------- model scoring
def score_records(rows, deberta_dir=None, vision_dir=None, max_length=512, log_every=50):
    """Attach t_threat / v_threat using the EXISTING predictors (no model code is changed)."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.predict_pipeline import AgentShieldPredictor
    pred = AgentShieldPredictor(deberta_model_path=deberta_dir, vision_model_path=vision_dir)
    for i, rec in enumerate(rows, 1):
        _, _, has_text = record_context(rec)
        rec["t_threat"] = rec["v_threat"] = None
        if has_text:
            rec["t_threat"] = pred.text_predictor.predict(rec["text"], max_length=max_length)["malicious_probability"]
        if rec.get("domain") == "image":
            for key in ("processed_image_path", "image_path"):
                if rec.get(key):
                    try:
                        rec["v_threat"] = pred.vision_predictor.predict(rec[key])["malicious_probability"]
                        break
                    except FileNotFoundError:
                        continue
        if i % log_every == 0:
            print(f"[INFO] scored {i}/{len(rows)}", flush=True)
    return rows


# ----------------------------------------------------------------------------- metrics
def binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n = tp + tn + fp + fn
    div = lambda a, b: a / b if b else 0.0
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {"n": n, "accuracy": div(tp + tn, n), "precision": prec, "recall": rec,
            "f1": div(2 * prec * rec, prec + rec),
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
            "false_positives": fp, "false_negatives": fn}


def run_cats(engine: CATSEngine, rows) -> List[Dict[str, Any]]:
    out = []
    for rec in rows:
        intent, content, has_text = record_context(rec)
        res = engine.assess(domain=rec.get("domain", "text"), text_threat=rec.get("t_threat"),
                            visual_threat=rec.get("v_threat"), content=content,
                            user_intent=intent, has_text=has_text)
        out.append({"label": int(rec["label"]), "domain": rec.get("domain", "text"),
                    "t": res.text_threat, "v": res.visual_threat, "decision": res.decision,
                    "risk": res.risk_score, "alignment": res.semantic_alignment,
                    "text_preview": (content or "")[:100]})
    return out


def make_methods(t_th: float = 0.5, v_th: float = 0.5):
    """Decision rules compared. Baseline thresholds default to 0.5 (the classifiers' native cut-off); pass
    calibrated values (see tune.py) for a FAIR comparison against CATS, whose thresholds are tuned."""
    return {
        f"DeBERTa-only (T>={t_th:g})": lambda r: None if r["t"] is None else int(r["t"] >= t_th),
        f"ViT-only (V>={v_th:g})": lambda r: None if r["v"] is None else int(r["v"] >= v_th),
        "CATS (REJECT = malicious)": lambda r: int(r["decision"] == "REJECT"),
        "CATS (SANITIZE|REJECT = malicious)": lambda r: int(r["decision"] != "ACCEPT"),
    }


METHODS = make_methods()
SUBSETS = {
    "all records": lambda r: True,
    "DeBERTa signal available": lambda r: r["t"] is not None,
    "ViT signal available": lambda r: r["v"] is not None,
    "both signals available": lambda r: r["t"] is not None and r["v"] is not None,
    "image, no readable OCR text": lambda r: r["domain"] == "image" and r["t"] is None,
}


def roc_auc(y: List[int], score: List[float]) -> Optional[float]:
    """Rank-based ROC-AUC (Mann-Whitney U, ties get average rank). None if one class is missing."""
    pos = sum(y); neg = len(y) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(y)), key=lambda i: score[i])
    ranks, i = [0.0] * len(y), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and score[order[j + 1]] == score[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return (sum(r for r, t in zip(ranks, y) if t == 1) - pos * (pos + 1) / 2) / (pos * neg)


def average_precision(y: List[int], score: List[float]) -> Optional[float]:
    """Average precision (area under the precision-recall curve, step-wise; ties grouped)."""
    pos = sum(y)
    if pos == 0:
        return None
    pairs = sorted(zip(score, y), key=lambda x: -x[0])
    tp = fp = 0
    ap, prev_recall, i = 0.0, 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            tp += pairs[j][1]; fp += 1 - pairs[j][1]; j += 1
        recall = tp / pos
        ap += (recall - prev_recall) * (tp / (tp + fp))
        prev_recall, i = recall, j
    return ap


THRESHOLD_FREE = {          # score used per source; independent of any decision threshold
    "DeBERTa T_threat": lambda r: r["t"],
    "ViT V_threat": lambda r: r["v"],
    "CATS risk": lambda r: r["risk"],
    "Misalignment (1-A) alone": lambda r: None if r["alignment"] is None else 1.0 - r["alignment"],
}


def build_report(results: List[Dict[str, Any]], t_threshold: float = 0.5, v_threshold: float = 0.5) -> Dict[str, Any]:
    METHODS = make_methods(t_threshold, v_threshold)
    report: Dict[str, Any] = {"n_records": len(results), "subsets": {}, "threshold_free": {},
                              "by_domain": {}, "decision_distribution": {}}
    seen: Dict[frozenset, str] = {}
    for sname, sfn in SUBSETS.items():
        sub = [r for r in results if sfn(r)]
        if not sub:
            continue
        key = frozenset(id(r) for r in sub)
        if key in seen:                       # identical record set as an earlier subset -> skip duplicate table
            report.setdefault("duplicate_subsets", {})[sname] = seen[key]
            continue
        seen[key] = sname
        report["threshold_free"][sname] = {}
        for tname, tfn in THRESHOLD_FREE.items():
            pairs = [(r["label"], tfn(r)) for r in sub if tfn(r) is not None]
            if pairs:
                ys, sc = map(list, zip(*pairs))
                report["threshold_free"][sname][tname] = {
                    "n": len(ys), "roc_auc": roc_auc(ys, sc), "average_precision": average_precision(ys, sc)}
        report["subsets"][sname] = {}
        for mname, mfn in METHODS.items():
            pairs = [(r["label"], mfn(r)) for r in sub if mfn(r) is not None]
            if pairs:
                report["subsets"][sname][mname] = binary_metrics(*map(list, zip(*pairs)))
    for dom in sorted({r["domain"] for r in results}):
        sub = [r for r in results if r["domain"] == dom]
        report["by_domain"][dom] = {m: binary_metrics([r["label"] for r in sub], [f(r) for r in sub])
                                    for m, f in METHODS.items() if m.startswith("CATS")}
    for lab in (0, 1):
        report["decision_distribution"][f"true_label_{lab}"] = dict(
            Counter(r["decision"] for r in results if r["label"] == lab))
    return report


def print_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print(f"CATS EVALUATION  (n={report['n_records']})  -- baseline thresholds, NOT validated")
    print("=" * 100)
    for sname, methods in report["subsets"].items():
        print(f"\n### Subset: {sname}")
        print(f"{'Method':<38}{'n':>6}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'TP':>6}{'TN':>6}{'FP':>6}{'FN':>6}")
        for m, x in methods.items():
            c = x["confusion_matrix"]
            print(f"{m:<38}{x['n']:>6}{x['accuracy']:>8.4f}{x['precision']:>8.4f}{x['recall']:>8.4f}"
                  f"{x['f1']:>8.4f}{c['TP']:>6}{c['TN']:>6}{c['FP']:>6}{c['FN']:>6}")
    print("\n### Threshold-free ranking quality (does the score separate the classes, regardless of threshold?)")
    print(f"{'Subset':<32}{'Score':<20}{'n':>6}{'ROC-AUC':>10}{'AvgPrec':>10}")
    for sname, d in report["threshold_free"].items():
        for tname, x in d.items():
            auc = "n/a" if x["roc_auc"] is None else f"{x['roc_auc']:.4f}"
            ap = "n/a" if x["average_precision"] is None else f"{x['average_precision']:.4f}"
            print(f"{sname:<32}{tname:<20}{x['n']:>6}{auc:>10}{ap:>10}")
    for dup, orig in report.get("duplicate_subsets", {}).items():
        print(f"(subset '{dup}' has exactly the same records as '{orig}'; table not repeated)")
    print("\n### Decision distribution by true label (0=benign, 1=malicious)")
    for k, v in report["decision_distribution"].items():
        print(f"  {k}: {v}")
    print("\nNOTE: a difference between rows is only evidence of improvement if it is measured on real "
          "model scores from a held-out split with thresholds fixed beforehand.\n")


def sweep(engine: CATSEngine, rows, thresholds: List[float]) -> None:
    print("\n### Reject-threshold sweep (use VALIDATION data to choose; then freeze and report on TEST)")
    print(f"{'reject_at':>10}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'FP':>6}{'FN':>6}")
    for th in thresholds:
        cfg = dataclasses.replace(engine.config, reject_at_or_above=th,
                                  accept_below=min(engine.config.accept_below, th - 1e-6), profiles={})
        res = run_cats(engine.with_config(cfg), rows)
        m = binary_metrics([r["label"] for r in res], [int(r["decision"] == "REJECT") for r in res])
        print(f"{th:>10.2f}{m['accuracy']:>8.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['f1']:>8.4f}"
              f"{m['false_positives']:>6}{m['false_negatives']:>6}")


def apply_overrides(cfg, pairs: Optional[List[str]]):
    """--set key=value  (global CATSConfig fields only), e.g. --set align_lambda=0 --set w_text=0.6"""
    valid = {f.name for f in dataclasses.fields(cfg)} - {"profiles"}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got '{item}'")
        k, v = item.split("=", 1)
        if k not in valid:
            raise SystemExit(f"Unknown config key '{k}'. Valid: {sorted(valid)}")
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass                                  # keep as plain string (e.g. fusion_mode=max)
        setattr(cfg, k, v)
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate CATS vs DeBERTa-only / ViT-only", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--data", nargs="+", help="jsonl file(s) with ground-truth labels")
    ap.add_argument("--scores_in", help="jsonl with precomputed t_threat/v_threat (skips model inference)")
    ap.add_argument("--scores_out", help="save per-record model scores here")
    ap.add_argument("--config", help="CATS config JSON")
    ap.add_argument("--embedder", choices=["sentence-transformers", "lexical", "auto"])
    ap.add_argument("--fusion", choices=["noisy_or", "max", "weighted_mean"])
    ap.add_argument("--t_threshold", type=float, default=0.5, help="DeBERTa-only baseline cut-off (default 0.5)")
    ap.add_argument("--v_threshold", type=float, default=0.5, help="ViT-only baseline cut-off (default 0.5)")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="override any global config value, repeatable (e.g. --set align_lambda=0 --set w_text=0.6)")
    ap.add_argument("--deberta_model"); ap.add_argument("--vision_model")
    ap.add_argument("--max_length", type=int, default=512,
                    help="DeBERTa max tokens (predictor default 512; training evaluation used 256)")
    ap.add_argument("--limit", type=int, help="only the first N records (quick check)")
    ap.add_argument("--sweep_reject", help="comma-separated reject thresholds, e.g. 0.5,0.7,0.9")
    ap.add_argument("--report_out", help="write full report JSON here")
    ap.add_argument("--dump_errors", type=int, default=0, help="print N CATS false positives / negatives")
    a = ap.parse_args(argv)

    if not (a.data or a.scores_in):
        ap.error("provide --data or --scores_in")
    rows = load_jsonl([a.scores_in] if a.scores_in else a.data)
    if a.limit:
        rows = rows[: a.limit]
    if not a.scores_in:
        rows = score_records(rows, a.deberta_model, a.vision_model, a.max_length)
    elif any("t_threat" not in r for r in rows):
        ap.error("--scores_in file must contain t_threat/v_threat fields")
    if a.scores_out:
        Path(a.scores_out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.scores_out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cfg = load_config(a.config)
    if a.embedder: cfg.embedding_backend = a.embedder
    if a.fusion: cfg.fusion_mode = a.fusion
    apply_overrides(cfg, a.set)
    engine = CATSEngine(cfg.validate())
    results = run_cats(engine, rows)
    report = build_report(results, a.t_threshold, a.v_threshold)
    report["config"] = cfg.to_dict()
    print_report(report)

    if a.dump_errors:
        for name, bad in (("FALSE POSITIVES", lambda r: r["label"] == 0 and r["decision"] != "ACCEPT"),
                          ("FALSE NEGATIVES", lambda r: r["label"] == 1 and r["decision"] == "ACCEPT")):
            print(f"--- CATS {name} (first {a.dump_errors}) ---")
            for r in [r for r in results if bad(r)][: a.dump_errors]:
                print(f"  [{r['domain']}] T={r['t']} V={r['v']} A={r['alignment']} risk={r['risk']} "
                      f"-> {r['decision']} | {r['text_preview']!r}")
    if a.sweep_reject:
        sweep(engine, rows, [float(x) for x in a.sweep_reject.split(",")])
    if a.report_out:
        Path(a.report_out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.report_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] report written to {a.report_out}")


if __name__ == "__main__":
    main()
