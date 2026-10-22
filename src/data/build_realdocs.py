"""
AgentShield: An Adaptive Security Runtime for Autonomous AI Agents
Real-Document Dataset (hard negatives for the text detector)

The benign PDF training documents are all GPT-4o-mini prose, so DeBERTa never saw a real document:
on lecture slides and research papers it scores ordinary sentences, figure captions and headings as
attacks (12 of 13 real benign documents blocked). This script builds training and evaluation data
from real documents:

  benign    real document text: 200-token windows (what the runtime scores) and single sentences
            (what the sanitizer scores), with the "Document content: " prefix the runtime uses
  malicious the same kind of real text with one BIPIA attack instruction inserted at a sentence
            boundary (start, middle or end), and attack sentences on their own
  benign    real text with a harmless sentence from another paper inserted the same way (a heading or a
            caption, half of them in capitals), so that an out-of-place sentence alone is not an attack:
            trained without these, the model flagged a slide heading ("A SIMPLE ROBOTIC VISION TASK")
            inside lecture notes, because it did not fit the text around it

Inputs:
  data/raw/arxiv/*.pdf                         open-access arXiv papers (any field), one file per paper
  data/raw/bipia/{text,code}_attack_{train,test}.json   BIPIA attack lists (github.com/microsoft/BIPIA)

Leakage rules: papers are split 70 / 15 / 15 into train / validation / test (stratified by arXiv
category, seed 42), so no paper appears in two splits; training and validation use BIPIA's TRAIN
attacks, the test split uses BIPIA's TEST attacks only. Papers about prompt injection or jailbreaks
are left out, because they quote attacks.

Outputs (data/processed/realdocs/):
  {train,validation,test}.jsonl        windows and sentences, domain "realdoc"
  docs_{validation,test}.jsonl         whole documents, each once clean and once with an inserted attack
  train_mixed.jsonl / validation_mixed.jsonl   the combined dataset plus the realdoc records

  python src/data/build_realdocs.py
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.defense.documents import read_pdf            # noqa: E402  the runtime's own PDF adapter
from src.defense.sanitizer import segment_text        # noqa: E402

PREFIX = "Document content: "
OFF_TOPIC = re.compile(r"prompt[\s-]+injection|jailbreak|adversarial\s+prompt|injected\s+instruction", re.I)


def load_attacks(raw: Path, split: str):
    out = []
    for kind in ("text", "code"):
        for items in json.loads((raw / f"{kind}_attack_{split}.json").read_text()).values():
            out += [a.strip() for a in items if a.strip()]
    return out


def split_papers(pdfs, seed):
    by_cat = defaultdict(list)
    for p in sorted(pdfs):
        by_cat[p.name.split("__")[0]].append(p)
    rnd, out = random.Random(seed), {"train": [], "validation": [], "test": []}
    for cat in sorted(by_cat):
        papers = by_cat[cat]
        rnd.shuffle(papers)
        n_val = n_test = max(1, round(len(papers) * 0.15))
        out["test"] += papers[:n_test]
        out["validation"] += papers[n_test:n_test + n_val]
        out["train"] += papers[n_test + n_val:]
    return out


def windows(ids, size):
    return [ids[i:i + size] for i in range(0, len(ids) - size + 1, size)]


def insert(text, attack, rnd, where=None):
    """Insert the attack at a sentence boundary: start, middle or end."""
    segs = [s.text for s in segment_text(text)]
    where = where or rnd.choice(("start", "middle", "end"))
    pos = {"start": 0, "middle": len(segs) // 2, "end": len(segs)}[where]
    return " ".join(segs[:pos] + [attack] + segs[pos:]), where


def build(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rnd = random.Random(a.seed)
    attacks = {"train": load_attacks(a.bipia, "train"), "test": load_attacks(a.bipia, "test")}
    texts, skipped = {}, []
    for p in sorted(a.pdfs.glob("*.pdf")):
        try:
            t = read_pdf(p.read_bytes(), max_preview_pages=0).text
        except Exception:
            skipped.append(p.name); continue
        if len(t) < 3000 or OFF_TOPIC.search(t):
            skipped.append(p.name); continue
        texts[p] = t
    splits = split_papers(list(texts), a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, papers in splits.items():
        pool = attacks["test" if split == "test" else "train"]
        recs, docs = [], []
        for p in papers:
            text, paper = texts[p], p.stem
            ids = tok(" ".join(text.split()), add_special_tokens=False)["input_ids"]
            wins = windows(ids, a.window_tokens)
            rnd.shuffle(wins)
            benign_w, attack_w = wins[:a.benign_windows], wins[a.benign_windows:a.benign_windows + a.attack_windows]
            for w in benign_w:
                recs.append({"text": PREFIX + tok.decode(w), "label": 0, "domain": "realdoc", "paper": paper, "kind": "window"})
            for w in attack_w:
                att = rnd.choice(pool)
                room = a.window_tokens - len(tok(att, add_special_tokens=False)["input_ids"])
                if room < 60:
                    continue
                body, where = insert(tok.decode(w[:room]), att, rnd)
                recs.append({"text": PREFIX + body, "label": 1, "domain": "realdoc", "paper": paper,
                             "kind": f"window+attack@{where}"})
            sents = [s.text for s in segment_text(text) if len(s.text) > 2]
            for w in wins[a.benign_windows + a.attack_windows:][:a.benign_inserts]:
                other = rnd.choice([q for q in papers if q != p] or [p])
                pool_b = [s.text for s in segment_text(texts[other]) if 2 <= len(s.text.split()) <= 14]
                if not pool_b:
                    continue
                extra = rnd.choice(pool_b)
                extra = extra.upper() if rnd.random() < 0.5 else extra
                body, where = insert(tok.decode(w[:a.window_tokens - 30]), extra, rnd)
                recs.append({"text": PREFIX + body, "label": 0, "domain": "realdoc", "paper": paper,
                             "kind": f"window+benign@{where}"})
            for s in rnd.sample(sents, min(a.benign_sentences, len(sents))):
                recs.append({"text": PREFIX + s, "label": 0, "domain": "realdoc", "paper": paper, "kind": "sentence"})
            for att in rnd.sample(pool, a.attack_sentences):
                recs.append({"text": PREFIX + att, "label": 1, "domain": "realdoc", "paper": paper, "kind": "attack sentence"})
            if split != "train":
                att = rnd.choice(pool)
                body, where = insert(text, att, rnd, ("start", "middle", "end")[len(docs) // 2 % 3])
                docs += [{"id": paper, "text": text, "label": 0},
                         {"id": paper, "text": body, "label": 1, "position": where, "attack": att}]
        rnd.shuffle(recs)
        with open(a.out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
        if docs:
            with open(a.out / f"docs_{split}.jsonl", "w", encoding="utf-8") as f:
                f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in docs)
        stats[split] = {"papers": len(papers), "records": len(recs),
                        "benign": sum(r["label"] == 0 for r in recs), "malicious": sum(r["label"] == 1 for r in recs),
                        "documents": len(docs)}
    for split in ("train", "validation"):          # the combined dataset plus the real-document records
        base = (a.combined / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        extra = (a.out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        lines = [l for l in base + extra if l.strip()]
        rnd.shuffle(lines)
        (a.out / f"{split}_mixed.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats[f"{split}_mixed"] = len(lines)
    stats["skipped_papers"] = len(skipped)
    stats["attacks"] = {k: len(v) for k, v in attacks.items()}
    (a.out / "build_report.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdfs", type=Path, default=PROJECT_ROOT / "data" / "raw" / "arxiv")
    ap.add_argument("--bipia", type=Path, default=PROJECT_ROOT / "data" / "raw" / "bipia")
    ap.add_argument("--combined", type=Path, default=PROJECT_ROOT / "data" / "processed" / "combined")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "processed" / "realdocs")
    ap.add_argument("--tokenizer", default=str(PROJECT_ROOT / "models" / "deberta-common"))
    ap.add_argument("--window_tokens", type=int, default=200)
    ap.add_argument("--benign_windows", type=int, default=10)
    ap.add_argument("--attack_windows", type=int, default=10)
    ap.add_argument("--benign_inserts", type=int, default=6)
    ap.add_argument("--benign_sentences", type=int, default=8)
    ap.add_argument("--attack_sentences", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    build(ap.parse_args())
