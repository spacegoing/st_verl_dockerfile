"""
BSPO hard-eval ETL: build a verl-compatible parquet from 3 sources.

Sources:
  1. BeyondAIME (100 rows)                             → data_source=beyondaime
  2. AIME 2025 I+II jsonl (15+15=30 rows)              → data_source=aime2025
  3. OlympiadBench / OE_TO_maths_en_COMP, first 100
     numerical single-answer rows                       → data_source=olympiadbench

Output: /mnt/public/lichang93/downloads/datasets/eval_bspo_hard/eval.parquet  (230 rows)

Schema matches 0320_split/eval_nemogym_math.parquet:
  data_source, prompt, reward_model, ability, extra_info

extra_info is a JSON-dumped string containing at minimum:
  index, agent_ref, question, expected_answer, source_dataset

This keeps the math-judge server route stable (it pulls expected_answer out of extra_info).

Usage (inside vrl container or anywhere with pandas+pyarrow):
  python3 etl_eval_hard.py
"""

from __future__ import annotations
import json
import os
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
# k8s pods and the vrl container both mount PFS at /root/myCodeLab/host/downloads.
# On the bare host the same data lives at /mnt/public/lichang93/downloads.
_CANDIDATES = [
    Path("/root/myCodeLab/host/downloads/datasets"),
    Path("/mnt/public/lichang93/downloads/datasets"),
]
DL_BASE = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])
OUT_DIR = DL_BASE / "eval_bspo_hard"
OUT_FILE = OUT_DIR / "eval.parquet"

# ── Prompt scaffolding (matches current nemogym_math eval exactly) ───────────
SYSTEM_PROMPT = "You are JoyAI, a large language model trained by JD (京东). Answer as concisely as possible."
USER_SUFFIX = "\n\nRemember to put your final answer inside \\boxed{}."
AGENT_REF = {"type": "responses_api_agents", "name": "math_with_judge_simple_agent"}


def make_row(idx: int, problem: str, answer: str, data_source: str, source_dataset: str) -> dict:
    """Build one parquet row."""
    question = f"\n\n{problem}{USER_SUFFIX}"
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    extra_info = {
        "index": idx,
        "agent_ref": AGENT_REF,
        "question": question,
        "expected_answer": str(answer),
        "source_dataset": source_dataset,
    }
    return {
        "data_source": data_source,
        "prompt": prompt,
        "reward_model": {"ground_truth": "", "style": "rule"},
        "ability": "",
        "extra_info": json.dumps(extra_info, ensure_ascii=False),
    }


# ── Source 1: BeyondAIME ────────────────────────────────────────────────────
def load_beyondaime() -> list[dict]:
    src = DL_BASE / "BeyondAIME/data/test.parquet"
    df = pd.read_parquet(src)
    rows = []
    for i, r in df.iterrows():
        rows.append(make_row(
            idx=int(i),
            problem=r["problem"],
            answer=int(r["answer"]),
            data_source="beyondaime",
            source_dataset="BeyondAIME",
        ))
    print(f"[beyondaime]   loaded {len(rows)} rows from {src}")
    return rows


# ── Source 2: AIME 2025 (I + II) ────────────────────────────────────────────
def load_aime2025() -> list[dict]:
    rows = []
    for split in ("aime2025-I.jsonl", "aime2025-II.jsonl"):
        src = DL_BASE / f"AIME2025/{split}"
        with src.open() as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                rows.append(make_row(
                    idx=len(rows),
                    problem=d["question"],
                    answer=d["answer"],
                    data_source="aime2025",
                    source_dataset=f"AIME2025/{split[:-6]}",  # strip .jsonl
                ))
    print(f"[aime2025]     loaded {len(rows)} rows")
    return rows


# ── Source 3: OlympiadBench / OE_TO_maths_en_COMP (first 100 numerical) ─────
def load_olympiadbench(n: int = 100) -> list[dict]:
    src = DL_BASE / "OlympiadBench/OlympiadBench/OE_TO_maths_en_COMP/OE_TO_maths_en_COMP.parquet"
    df = pd.read_parquet(src)
    # Keep numerical, single-answer, no image (OE_TO = Text-Only already)
    sel = df[
        (df["answer_type"] == "Numerical")
        & (~df["is_multiple_answer"])
    ].reset_index(drop=True)
    sel = sel.head(n)
    rows = []
    for i, r in sel.iterrows():
        # final_answer is numpy array of str; take first
        raw = r["final_answer"]
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if isinstance(raw, list) and raw:
            ans = str(raw[0])
        else:
            ans = str(raw)
        # Strip LaTeX $...$ wrapping if present (judge handles it either way; cleaner for log inspection)
        ans = ans.strip()
        if ans.startswith("$") and ans.endswith("$") and len(ans) > 2:
            ans = ans[1:-1]
        rows.append(make_row(
            idx=int(r["id"]),
            problem=r["question"],
            answer=ans,
            data_source="olympiadbench",
            source_dataset="OlympiadBench/OE_TO_maths_en_COMP",
        ))
    print(f"[olympiadbench] loaded {len(rows)} rows (filtered numerical/single-answer)")
    return rows


# ── Assemble and write ──────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_rows.extend(load_beyondaime())
    all_rows.extend(load_aime2025())
    all_rows.extend(load_olympiadbench(100))

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_FILE, index=False)
    total = len(df)
    per_source = df["data_source"].value_counts().to_dict()
    print(f"\nwrote {OUT_FILE}")
    print(f"  total rows: {total}")
    print(f"  per data_source: {per_source}")
    # Quick sanity check: first row of each source
    for ds in df["data_source"].unique():
        r = df[df["data_source"] == ds].iloc[0]
        ei = json.loads(r["extra_info"])
        print(f"  sample [{ds}] expected_answer={ei['expected_answer']!r}  question[:100]={ei['question'][:100]!r}")


if __name__ == "__main__":
    main()
