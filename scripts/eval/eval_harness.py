"""Lightweight evaluation harness for Synapse retrieval and answer quality.

Usage:
  python scripts/eval/eval_harness.py --input scripts/eval/sample_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from src.rag.grounding import compute_grounding_score

WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def normalize_text(text: str) -> str:
    return " ".join(WORD_PATTERN.findall(text.lower()))


def token_f1(reference: str, prediction: str) -> float:
    ref_tokens = normalize_text(reference).split()
    pred_tokens = normalize_text(prediction).split()
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    ref_counter = Counter(ref_tokens)
    pred_counter = Counter(pred_tokens)
    overlap = sum((ref_counter & pred_counter).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(reference: str, prediction: str) -> float:
    return 1.0 if normalize_text(reference) == normalize_text(prediction) else 0.0


def source_recall(gold_sources: list[str], retrieved_sources: list[str]) -> float:
    if not gold_sources:
        return 0.0
    gold = {item.strip().lower() for item in gold_sources if item.strip()}
    retrieved = {item.strip().lower() for item in retrieved_sources if item.strip()}
    if not gold:
        return 0.0
    return len(gold & retrieved) / len(gold)


def evaluate(records: list[dict]) -> dict[str, float | int]:
    em_scores = []
    f1_scores = []
    grounding_scores = []
    source_recalls = []

    for record in records:
        reference_answer = record.get("reference_answer", "")
        predicted_answer = record.get("predicted_answer", "")
        retrieved_chunks = record.get("retrieved_chunks", [])
        gold_sources = record.get("gold_sources", [])
        retrieved_sources = record.get("retrieved_sources", [])

        em_scores.append(exact_match(reference_answer, predicted_answer))
        f1_scores.append(token_f1(reference_answer, predicted_answer))
        grounding_scores.append(compute_grounding_score(predicted_answer, retrieved_chunks))

        if gold_sources:
            source_recalls.append(source_recall(gold_sources, retrieved_sources))

    return {
        "samples": len(records),
        "exact_match": round(mean(em_scores), 4) if em_scores else 0.0,
        "token_f1": round(mean(f1_scores), 4) if f1_scores else 0.0,
        "grounding_score": round(mean(grounding_scores), 4) if grounding_scores else 0.0,
        "source_recall": round(mean(source_recalls), 4) if source_recalls else 0.0,
    }


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG predictions from JSONL")
    parser.add_argument("--input", required=True, help="Path to predictions JSONL")
    args = parser.parse_args()

    records = load_jsonl(Path(args.input))
    if not records:
        raise SystemExit("No records found in evaluation file")

    result = evaluate(records)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
