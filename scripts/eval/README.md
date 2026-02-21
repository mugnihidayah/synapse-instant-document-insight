# Eval Harness

Run:

```bash
python scripts/eval/eval_harness.py --input scripts/eval/sample_predictions.jsonl
```

JSONL fields per row:

- `question`: string
- `reference_answer`: ground truth answer
- `predicted_answer`: model answer
- `retrieved_chunks`: list of chunk texts used as citations
- `gold_sources`: optional list of expected source ids
- `retrieved_sources`: optional list of retrieved source ids

Output metrics:

- `exact_match`
- `token_f1`
- `grounding_score`
- `source_recall`
