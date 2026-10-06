"""
Grid-search CATS parameters on a VALIDATION scores file, then freeze the winner for TEST.

Workflow (avoids tuning on the test set):
  1. python -m src.cats.evaluate --data data/processed/image/validation.jsonl --scores_out outputs/cats/img_val.jsonl
  2. python -m src.cats.evaluate --data data/processed/image/test.jsonl       --scores_out outputs/cats/img_test.jsonl
  3. python -m src.cats.tune --scores_in outputs/cats/img_val.jsonl --save_best outputs/cats/best_config.json
  4. python -m src.cats.evaluate --scores_in outputs/cats/img_test.jsonl --config outputs/cats/best_config.json   # ONCE

Caution: a grid of hundreds of settings on a few hundred samples over-fits (selection bias). Differences of a
couple of samples are noise. Prefer simple settings, report the grid size, and always confirm on the test split.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from typing import Any, Dict, List

from . import scoring
from .config import CATSConfig, load_config
from .decision import decide
from .engine import CATSEngine
from .evaluate import binary_metrics, load_jsonl, record_context


def prepare(engine: CATSEngine, rows) -> List[Dict[str, Any]]:
    """Compute everything that does NOT depend on the grid once (alignment embeddings are the slow part)."""
    prep = []
    for rec in rows:
        intent, content, has_text = record_context(rec)
        t = rec.get("t_threat") if has_text else None
        align, _, _ = engine.combined_alignment(content if has_text else None, intent, None)
        prep.append({"label": int(rec["label"]), "domain": rec.get("domain", "text"),
                     "t": t, "v": rec.get("v_threat"), "a": align})
    return prep


def decisions(prep, cfg: CATSConfig) -> List[str]:
    out = []
    for r in prep:
        params = cfg.resolve(r["domain"])
        sb = scoring.score(r["t"], r["v"], r["a"], params, cfg.fusion_mode)
        out.append(decide(sb.risk, params, cfg.no_evidence_decision))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores_in", required=True, help="VALIDATION scores jsonl (t_threat / v_threat present)")
    ap.add_argument("--config"); ap.add_argument("--embedder", choices=["sentence-transformers", "lexical", "auto"])
    ap.add_argument("--objective", choices=["reject", "flag"], default="reject",
                    help="reject: positive = REJECT (tunes reject threshold); flag: positive = SANITIZE|REJECT (tunes accept threshold)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--save_best")
    a = ap.parse_args(argv)

    base = load_config(a.config)
    if a.embedder:
        base.embedding_backend = a.embedder
    rows = load_jsonl([a.scores_in])
    engine = CATSEngine(base)
    prep = prepare(engine, rows)
    y = [r["label"] for r in prep]
    has_align = any(r["a"] is not None for r in prep)

    grid = {
        "fusion_mode": ["noisy_or", "max", "weighted_mean"],
        "w_text": [0.4, 0.6, 0.8, 1.0],
        "w_visual": [0.6, 0.8, 1.0],
        "align_lambda": [0.0, 0.5, 1.0] if has_align else [base.align_lambda],  # no alignment data -> lambda irrelevant
        "threshold": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if a.objective == "reject" else [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    keys = list(grid)
    results = []
    for combo in itertools.product(*grid.values()):
        c = dict(zip(keys, combo))
        th = c.pop("threshold")
        kw = dict(c)
        if a.objective == "reject":
            kw["reject_at_or_above"] = th; kw["accept_below"] = min(base.accept_below, th - 1e-6)
        else:
            kw["accept_below"] = th; kw["reject_at_or_above"] = max(base.reject_at_or_above, th + 1e-6)
        cfg = dataclasses.replace(base, profiles={}, **kw)
        dec = decisions(prep, cfg)
        pred = [int(d == "REJECT") if a.objective == "reject" else int(d != "ACCEPT") for d in dec]
        m = binary_metrics(y, pred)
        results.append((m["f1"], m, cfg, kw))

    results.sort(key=lambda x: (-x[0], x[1]["false_negatives"]))
    default_pred = decisions(prep, base)
    dm = binary_metrics(y, [int(d == "REJECT") if a.objective == "reject" else int(d != "ACCEPT") for d in default_pred])
    n_cfg = len(results)
    print(f"Grid size: {n_cfg} configurations on n={len(prep)} validation records (objective: {a.objective}).")
    print("WARNING: best-of-grid F1 is optimistically biased. Confirm on TEST once.\n")
    print(f"{'F1':>8}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'FP':>5}{'FN':>5}  config")
    show = lambda m, kw: print(f"{m['f1']:>8.4f}{m['accuracy']:>8.4f}{m['precision']:>8.4f}{m['recall']:>8.4f}"
                               f"{m['false_positives']:>5}{m['false_negatives']:>5}  {kw}")
    print("-- baseline default config:")
    show(dm, {k: getattr(base, k) for k in ("fusion_mode", "w_text", "w_visual", "align_lambda", "accept_below", "reject_at_or_above")})
    print(f"-- top {a.top}:")
    for f1, m, cfg, kw in results[: a.top]:
        show(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in kw.items()})
    # ---- fair reference: the SAME tuning budget given to single-signal baselines (threshold only) ----
    ths = [round(x / 100, 2) for x in range(5, 100, 5)]
    print("\n-- calibrated single-signal baselines (threshold tuned on the same validation data):")
    for name, key in (("ViT-only", "v"), ("DeBERTa-only", "t")):
        sub = [r for r in prep if r[key] is not None]
        if not sub:
            continue
        best = max(((binary_metrics([r["label"] for r in sub], [int(r[key] >= th) for r in sub]), th) for th in ths),
                   key=lambda x: (x[0]["f1"], -x[0]["false_negatives"]))
        show(best[0], {"rule": f"{name}: score >= {best[1]}", "n": len(sub)})
    print("   If the best CATS row is not clearly above these, the gain comes from threshold calibration, not fusion.")
    if a.save_best:
        results[0][2].to_json(a.save_best)
        print(f"\n[INFO] best config saved to {a.save_best}")


if __name__ == "__main__":
    main()
