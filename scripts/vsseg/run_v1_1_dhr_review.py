#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import registration_v1_auto_component as base
import registration_v1_1_structure_component as reg


def read_inventory(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return {row["pair_id"]: row for row in csv.DictReader(handle)}


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    default_config = Path(__file__).resolve().parents[2] / "configs" / "vsseg" / "registration_v1_1_structure_component.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--transform-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--include-low", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    inventory = read_inventory(config["inventory"])
    if args.pair_id not in inventory:
        raise KeyError("Unknown pair_id: %s" % args.pair_id)
    row = inventory[args.pair_id]

    runtime = argparse.Namespace(**config)
    if args.device:
        runtime.dhr_device = args.device

    pair_transform_root = Path(args.transform_root) / args.pair_id
    report_dir = Path(args.report_root) / args.pair_id
    report_dir.mkdir(parents=True, exist_ok=True)
    transform_files = sorted(pair_transform_root.glob("component_*.json"))
    if not transform_files:
        raise RuntimeError("No component transforms found: %s" % pair_transform_root)

    Aslide = base.import_aslide(config["aslide_root"])
    he_slide = Aslide(row["he_wsi_path"])
    ihc_slide = Aslide(row["ihc_wsi_path"])
    review_rows = []
    try:
        for transform_file in transform_files:
            payload = json.loads(transform_file.read_text(encoding="utf-8"))
            component_id = int(payload["component_id"])
            confidence = payload["confidence"]
            diagnostic_only = confidence == "low"
            if diagnostic_only and not args.include_low:
                review_rows.append({
                    "pair_id": args.pair_id,
                    "component_id": component_id,
                    "confidence": confidence,
                    "status": "skipped_low_confidence",
                    "diagnostic_only": True,
                    "output_path": "",
                })
                continue
            matrix = payload.get("matrix_level0_HE_to_IHC_2x3")
            if matrix is None:
                raise RuntimeError("Missing level-0 affine in %s" % transform_file)
            he_bbox0 = payload["he_component_bbox_level0"]
            he_crop, he_origin, he_sx, he_sy = reg.read_component_crop(
                he_slide, he_bbox0, runtime.component_crop_max_side
            )
            he_component = reg.component_from_local_tissue(he_crop)
            anchor_x, anchor_y = base.component_anchor(he_component)
            center0 = reg.local_point_to_level0(
                anchor_x, anchor_y, he_origin, he_sx, he_sy
            )
            montage, dhr_info = reg.dhr_smoke_at_center(
                he_slide, ihc_slide, center0, matrix, runtime
            )
            if diagnostic_only:
                montage = base.add_header(montage, [
                    "DIAGNOSTIC ONLY: coarse registration confidence=low; do not use for dataset materialization"
                ])
                output_name = "04_component_%02d_dhr_smoke_DIAGNOSTIC_LOW.png" % component_id
            else:
                output_name = "04_component_%02d_dhr_smoke.png" % component_id
            output_path = report_dir / output_name
            Image.fromarray(montage).save(output_path)
            review_rows.append({
                "pair_id": args.pair_id,
                "component_id": component_id,
                "confidence": confidence,
                "status": "generated",
                "diagnostic_only": diagnostic_only,
                "output_path": str(output_path),
                "valid_fraction": dhr_info["final_patch_valid_fraction"],
                "folding_fraction": dhr_info["dhr_qc"]["folding_fraction"],
                "displacement_p95_px": dhr_info["dhr_qc"]["displacement_p95_px"],
            })
            print(
                args.pair_id,
                "component", component_id,
                confidence,
                "diagnostic_only=%s" % diagnostic_only,
                "valid=%.4f" % dhr_info["final_patch_valid_fraction"],
                "disp_p95=%.2f" % dhr_info["dhr_qc"]["displacement_p95_px"],
                flush=True,
            )
    finally:
        base.close_slide(he_slide)
        base.close_slide(ihc_slide)

    fields = [
        "pair_id", "component_id", "confidence", "status", "diagnostic_only",
        "output_path", "valid_fraction", "folding_fraction", "displacement_p95_px",
    ]
    base.write_csv_atomic(report_dir / "dhr_review_summary.csv", review_rows, fields)


if __name__ == "__main__":
    main()
