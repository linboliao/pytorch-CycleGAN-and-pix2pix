#!/usr/bin/env python3
"""Run formal registration_v1_component for a deterministic dual_v1 shard.

This runner intentionally does not call registration_v1_component.main(), because
that CLI rewrites the global registration_summary.csv on every invocation.
Instead it calls process_pair() directly and writes shard-local status files.
Existing per-pair formal transforms are preserved and skipped.
"""

import argparse
import csv
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import registration_utils as utils
import registration_v1_component as reg


STATUS_FIELDS = [
    "pair_id",
    "case_id",
    "shard_index",
    "status",
    "component_count",
    "medium_high_components",
    "low_components",
    "error_type",
    "error_message",
]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_status(path, rows):
    utils.write_csv_atomic(path, rows, STATUS_FIELDS)


def existing_component_files(root, pair_id):
    return sorted((Path(root) / "transforms" / pair_id).glob("component_*.json"))


def summarize_existing(files):
    confidence = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        confidence.append(payload.get("confidence", ""))
    return {
        "component_count": len(files),
        "medium_high_components": sum(x in {"medium", "high"} for x in confidence),
        "low_components": sum(x not in {"medium", "high"} for x in confidence),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "configs/vsseg/dual_v1.json"),
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_json(args.config)
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    public = {
        r["pair_id"]: r
        for r in utils.read_csv(cfg["public_pair_inventory"])
        if r.get("pair_type") == cfg["pair_type"] and r.get("status") == "paired"
    }
    private = {
        r["pair_id"]: r
        for r in utils.read_csv(cfg["private_pair_inventory"])
        if r.get("pair_type") == cfg["pair_type"]
    }
    pair_ids = sorted(public)
    if len(pair_ids) != 36:
        raise RuntimeError("Expected 36 paired dual-stain rows, got %d" % len(pair_ids))
    missing_private = [p for p in pair_ids if p not in private]
    if missing_private:
        raise RuntimeError("Missing private inventory rows: %s" % missing_private)

    shard_pair_ids = [
        p for i, p in enumerate(pair_ids) if i % args.num_shards == args.shard_index
    ]

    reg_cfg = reg.load_config(cfg["registration_config"])
    reg_cfg["output_root"] = cfg["registration_output_root"]
    reg_cfg["report_root"] = cfg["registration_report_root"]
    reg_args = SimpleNamespace(**reg_cfg)
    reg_args.run_dhr = False

    Aslide = utils.import_aslide(reg_args.aslide_root)
    status_root = Path(cfg["cohort_report_root"]) / "registration_shards"
    status_root.mkdir(parents=True, exist_ok=True)
    status_path = status_root / ("shard_%02d.csv" % args.shard_index)
    rows = []

    for pair_id in shard_pair_ids:
        pub = public[pair_id]
        files = existing_component_files(reg_args.output_root, pair_id)
        if files and not args.force:
            summary = summarize_existing(files)
            row = {
                "pair_id": pair_id,
                "case_id": pub["case_id"],
                "shard_index": args.shard_index,
                "status": "existing",
                **summary,
                "error_type": "",
                "error_message": "",
            }
            rows.append(row)
            write_status(status_path, rows)
            print("EXISTING", pair_id, summary, flush=True)
            continue

        try:
            _, component_rows = reg.process_pair(private[pair_id], reg_args, Aslide)
            files = existing_component_files(reg_args.output_root, pair_id)
            summary = summarize_existing(files)
            row = {
                "pair_id": pair_id,
                "case_id": pub["case_id"],
                "shard_index": args.shard_index,
                "status": "completed",
                **summary,
                "error_type": "",
                "error_message": "",
            }
            print("COMPLETED", pair_id, summary, flush=True)
        except Exception as exc:
            row = {
                "pair_id": pair_id,
                "case_id": pub["case_id"],
                "shard_index": args.shard_index,
                "status": "failed",
                "component_count": 0,
                "medium_high_components": 0,
                "low_components": 0,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
            }
            print("FAILED", pair_id, type(exc).__name__, str(exc), flush=True)
            traceback.print_exc()

        rows.append(row)
        write_status(status_path, rows)

    print(
        "DONE shard=%d pairs=%d completed=%d failed=%d existing=%d"
        % (
            args.shard_index,
            len(rows),
            sum(r["status"] == "completed" for r in rows),
            sum(r["status"] == "failed" for r in rows),
            sum(r["status"] == "existing" for r in rows),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
