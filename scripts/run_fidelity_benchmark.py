#!/usr/bin/env python3
"""One-click fidelity benchmark runner.

Runs IOI fidelity benchmark for one or more models and writes both
JSON and CSV outputs under benchmarks/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformer_lens import HookedTransformer

from tests.benchmark_fidelity import DEFAULT_SCORING_FORMULA, run_fidelity_benchmark

IOI_CASES = [
    ("IOI_ABB", "When Mary and John went to the store, John gave a drink to", "When Mary and John went to the store, Mary gave a drink to", " Mary"),
    ("IOI_BAB", "When Alice and Bob went to the park, Bob gave a ball to", "When Alice and Bob went to the park, Alice gave a ball to", " Alice"),
    ("IOI_name_move", "When Emma and James went to the office, James gave a report to", "When Emma and James went to the office, Emma gave a report to", " Emma"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt2"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scoring-formula", default=DEFAULT_SCORING_FORMULA)
    parser.add_argument("--out-dir", default="benchmarks")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    rows = []
    for model_name in args.models:
        print(f"[load] {model_name} on {args.device}")
        model = HookedTransformer.from_pretrained(model_name, device=args.device)
        for idx, (task, clean, corrupted, target) in enumerate(IOI_CASES, start=42):
            result = run_fidelity_benchmark(
                model=model,
                clean_prompt=clean,
                corrupted_prompt=corrupted,
                target_token=target,
                scoring_formula=args.scoring_formula,
                model_name=model_name,
                task_name=task,
            )
            row = {
                "model": model_name,
                "task": task,
                "seed": idx,
                "device": args.device,
                "torch_version": torch.__version__,
                "transformer_lens_version": "unknown",
                "scoring_formula": args.scoring_formula,
                "pearson": float(result.pearson_pcc),
                "spearman": float(result.spearman_rho),
                "top_5_overlap": float(result.top_k_overlaps.get(5, 0.0)),
                "top_10_overlap": float(result.top_k_overlaps.get(10, 0.0)),
                "top_20_overlap": float(result.top_k_overlaps.get(20, 0.0)),
                "reip_runtime_s": float(result.reip_runtime_s),
                "ap_runtime_s": float(result.ap_runtime_s),
                "speedup": float(result.ap_runtime_s / max(result.reip_runtime_s, 1e-6)),
                "n_components": int(result.n_components),
                "gpu_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu",
                "timestamp": timestamp,
            }
            rows.append(row)
            print(result.summary())

    json_path = out_dir / f"fidelity_results_{timestamp}.json"
    csv_path = out_dir / f"fidelity_results_{timestamp}.csv"
    latest_json = out_dir / "fidelity_results_latest.json"
    latest_csv = out_dir / "fidelity_results_latest.csv"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with latest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] {json_path}")
    print(f"[done] {csv_path}")


if __name__ == "__main__":
    main()
