#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


REGISTRATION_VERSION = "registration_v0_manual"
PIPELINE_SCHEMA_VERSION = "2"

PAIR_SPECS = {
    "dual": {
        "pair_type": "HE-p63+CKpan_dual",
        "cohort": "dual_stain_g5",
        "stain": "p63+CKpan",
    },
    "ckpan": {
        "pair_type": "HE-CKpan",
        "cohort": "ckpan_single",
        "stain": "CKpan",
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8-sig",
        newline="",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temp_path), str(path))


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(str(temp_path), str(path))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def git_dirty(path):
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except Exception:
        return True


def canonical(text):
    return re.sub(r"[^A-Za-z0-9]", "", text or "").upper()


def load_json(path):
    error = None
    for encoding in ("utf-8-sig", "gbk"):
        try:
            with Path(path).open(encoding=encoding) as handle:
                return json.load(handle)
        except Exception as exc:
            error = exc
    raise RuntimeError("Could not decode JSON %s: %s" % (path, error))


def parse_positions(path):
    obj = load_json(path)
    if not isinstance(obj, list):
        raise ValueError("Landmark file must be a JSON list")
    points = []
    for index, item in enumerate(obj):
        if not isinstance(item, dict) or item.get("type") != "Position":
            continue
        region = item.get("region") or {}
        if "x" not in region or "y" not in region:
            continue
        points.append(
            {
                "index": index,
                "name": str(item.get("name", "")),
                "imageindex": str(item.get("imageindex", "")),
                "x": float(region["x"]),
                "y": float(region["y"]),
            }
        )
    return points


def normalized_names(points):
    return [re.sub(r"\s+", "", point["name"]) for point in points]


def pair_landmarks_by_name(he_points, ihc_points, min_landmarks):
    he_names = normalized_names(he_points)
    ihc_names = normalized_names(ihc_points)

    if he_names == ihc_names:
        return he_points, ihc_points, "exact_sequence", [], []

    if len(set(he_names)) != len(he_names) or len(set(ihc_names)) != len(ihc_names):
        return [], [], "ambiguous_duplicate_names", he_names, ihc_names

    ihc_by_name = {name: point for name, point in zip(ihc_names, ihc_points)}
    matched_he = []
    matched_ihc = []
    matched_names = []
    for name, point in zip(he_names, he_points):
        if name and name in ihc_by_name:
            matched_he.append(point)
            matched_ihc.append(ihc_by_name[name])
            matched_names.append(name)

    unmatched_he = [name for name in he_names if name not in set(matched_names)]
    unmatched_ihc = [name for name in ihc_names if name not in set(matched_names)]
    if len(matched_he) < min_landmarks:
        return [], [], "insufficient_common_names", unmatched_he, unmatched_ihc
    return matched_he, matched_ihc, "name_intersection", unmatched_he, unmatched_ihc


def fit_affine_lstsq(source, target):
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Expected matching Nx2 landmark arrays")
    design = np.column_stack(
        [source.astype(np.float64), np.ones(len(source), dtype=np.float64)]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(
        design, target.astype(np.float64), rcond=None
    )
    if rank < 3:
        raise ValueError("Degenerate landmark configuration")
    return coefficients.T


def affine_errors(matrix, source, target):
    predicted = np.column_stack([source, np.ones(len(source))]) @ matrix.T
    return np.linalg.norm(predicted - target, axis=1)


def ransac_qc(source, target, threshold):
    matrix, inliers = cv2.estimateAffine2D(
        source.astype(np.float32),
        target.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(threshold),
        maxIters=5000,
        confidence=0.999,
        refineIters=50,
    )
    if matrix is None:
        return {"matrix": None, "inliers": 0, "rmse": None}
    mask = (
        inliers.ravel().astype(bool)
        if inliers is not None
        else np.ones(len(source), dtype=bool)
    )
    errors = affine_errors(matrix, source, target)
    rmse = float(np.sqrt(np.mean(errors[mask] ** 2))) if mask.any() else None
    return {"matrix": matrix, "inliers": int(mask.sum()), "rmse": rmse}


def patient_key(case_key):
    if "." in case_key:
        return case_key.split(".", 1)[0]
    match = re.match(r"^(\d{6,9})", case_key or "")
    return match.group(1) if match else (case_key or "")


def patient_group_id(case_key):
    return "PAT_" + hashlib.sha256(patient_key(case_key).encode()).hexdigest()[:12].upper()


def deterministic_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fallback_split(group_id):
    value = int(deterministic_hash(group_id)[:8], 16) % 100
    return "train" if value < 70 else ("val" if value < 85 else "test")


def stratified_patient_splits(rows):
    eligible = [row for row in rows if row["landmark_status"] == "eligible"]
    cohorts_by_group = defaultdict(set)
    for row in eligible:
        cohorts_by_group[row["patient_group_id"]].add(row["cohort"])

    mapping = {}
    shared_groups = {
        group_id for group_id, cohorts in cohorts_by_group.items() if len(cohorts) > 1
    }
    for group_id in shared_groups:
        mapping[group_id] = fallback_split(group_id)

    for cohort in sorted({row["cohort"] for row in eligible}):
        groups = sorted(
            {
                row["patient_group_id"]
                for row in eligible
                if row["cohort"] == cohort and row["patient_group_id"] not in mapping
            },
            key=lambda group_id: deterministic_hash(cohort + "|" + group_id),
        )
        count = len(groups)
        if not count:
            continue
        n_test = max(1, int(round(count * 0.15))) if count >= 3 else 0
        n_val = max(1, int(round(count * 0.15))) if count >= 3 else 0
        while n_test + n_val >= count:
            if n_val:
                n_val -= 1
            elif n_test:
                n_test -= 1
            else:
                break
        for index, group_id in enumerate(groups):
            if index < n_test:
                mapping[group_id] = "test"
            elif index < n_test + n_val:
                mapping[group_id] = "val"
            else:
                mapping[group_id] = "train"

    for row in rows:
        mapping.setdefault(row["patient_group_id"], fallback_split(row["patient_group_id"]))
    return mapping


def sample_id(pair_id, x, y, patch_size, context_size):
    raw = "%s|%d|%d|%d|%d" % (pair_id, x, y, patch_size, context_size)
    return "SMP_" + hashlib.sha256(raw.encode()).hexdigest()[:16].upper()


def discover_landmark_jsons(project_root):
    raw = Path(project_root) / "raw_links"
    return {
        "dual": list((raw / "dual_stain_landmarks").rglob("*.json"))
        + list((raw / "dual_stain_labels").rglob("*.json")),
        "ckpan": list((raw / "ckpan_single").rglob("*.json")),
    }


def landmark_role(family, path):
    text = str(path)
    if family == "dual":
        return "ihc" if "ihc" in text.lower() else "he"
    if "CK pan图像及坐标" in text:
        return "ihc"
    if "HE图像及坐标" in text:
        return "he"
    return "unknown"


def find_landmark_jsons(family, case_key, jsons):
    key = canonical(case_key)
    candidates = [
        path
        for path in jsons
        if key and key in canonical(path.parent.parent.name)
    ]
    if not candidates:
        candidates = [path for path in jsons if key and key in canonical(str(path))]
    return (
        [path for path in candidates if landmark_role(family, path) == "he"],
        [path for path in candidates if landmark_role(family, path) == "ihc"],
    )


def audit(args):
    project_root = Path(args.project_root)
    output_root = Path(args.output_root)
    private_root = output_root / "private"
    transform_root = output_root / "transforms"
    private_root.mkdir(parents=True, exist_ok=True)
    transform_root.mkdir(parents=True, exist_ok=True)

    pairs = read_csv(project_root / "manifests" / "pair_inventory.csv")
    images = read_csv(
        project_root / "manifests" / "private" / "image_inventory_internal.csv"
    )
    image_by_id = {row["image_id"]: row for row in images}
    landmark_sets = discover_landmark_jsons(project_root)

    public_rows = []
    private_rows = []
    transform_rows = []

    for family in args.families:
        spec = PAIR_SPECS[family]
        family_pairs = [row for row in pairs if row["pair_type"] == spec["pair_type"]]
        for pair in family_pairs:
            he_info = image_by_id[pair["he_image_id"]]
            ihc_info = image_by_id[pair["ihc_image_id"]]
            case_key = he_info.get("internal_case_key") or ihc_info.get("internal_case_key")
            he_jsons, ihc_jsons = find_landmark_jsons(
                family, case_key, landmark_sets[family]
            )
            he_points = []
            ihc_points = []
            matched_he_points = []
            matched_ihc_points = []
            matching_method = ""
            unmatched_he_names = []
            unmatched_ihc_names = []
            status = "eligible"
            reason = ""

            if len(he_jsons) != 1 or len(ihc_jsons) != 1:
                status = "needs_review"
                reason = "landmark_json_mapping_not_1to1"
            else:
                he_points = parse_positions(he_jsons[0])
                ihc_points = parse_positions(ihc_jsons[0])
                if len(he_points) < args.min_landmarks or len(ihc_points) < args.min_landmarks:
                    status = "insufficient_landmarks"
                    reason = "fewer_than_%d" % args.min_landmarks
                elif args.require_matching_names:
                    (
                        matched_he_points,
                        matched_ihc_points,
                        matching_method,
                        unmatched_he_names,
                        unmatched_ihc_names,
                    ) = pair_landmarks_by_name(
                        he_points, ihc_points, args.min_landmarks
                    )
                    if not matched_he_points:
                        status = "needs_review"
                        reason = matching_method
                else:
                    if len(he_points) != len(ihc_points):
                        status = "needs_review"
                        reason = "landmark_count_mismatch"
                    else:
                        matched_he_points = he_points
                        matched_ihc_points = ihc_points
                        matching_method = "legacy_sequence"

            transform_path = ""
            rmse = ""
            median_error = ""
            max_error = ""
            ransac_inliers = ""
            ransac_rmse = ""
            determinant = ""

            if status == "eligible":
                he_xy = np.asarray(
                    [[point["x"], point["y"]] for point in matched_he_points],
                    dtype=np.float64,
                )
                ihc_xy = np.asarray(
                    [[point["x"], point["y"]] for point in matched_ihc_points],
                    dtype=np.float64,
                )
                matrix = fit_affine_lstsq(he_xy, ihc_xy)
                errors = affine_errors(matrix, he_xy, ihc_xy)
                ransac = ransac_qc(he_xy, ihc_xy, args.ransac_threshold_px)

                rmse = float(np.sqrt(np.mean(errors ** 2)))
                median_error = float(np.median(errors))
                max_error = float(np.max(errors))
                determinant = float(np.linalg.det(matrix[:, :2]))
                ransac_inliers = ransac["inliers"]
                ransac_rmse = "" if ransac["rmse"] is None else ransac["rmse"]

                if rmse > args.max_affine_rmse_px:
                    status = "needs_review"
                    reason = "affine_rmse_exceeds_limit"
                    transform_path = ""
                else:
                    payload = {
                        "registration_version": args.registration_version,
                        "pair_id": pair["pair_id"],
                        "case_id": pair["case_id"],
                        "cohort": spec["cohort"],
                        "pair_type": spec["pair_type"],
                        "stain": spec["stain"],
                        "direction": "HE_to_IHC",
                        "coordinate_system": "WSI_level0_pixels",
                        "fit_method": "all_matched_landmarks_least_squares_affine",
                        "landmark_matching_method": matching_method,
                        "matrix_2x3": matrix.tolist(),
                        "landmark_count": len(matched_he_points),
                        "landmark_errors_px": errors.tolist(),
                        "rmse_px": rmse,
                        "median_error_px": median_error,
                        "max_error_px": max_error,
                        "affine_determinant": determinant,
                        "ransac_qc": {
                            "threshold_px": args.ransac_threshold_px,
                            "inlier_count": ransac_inliers,
                            "inlier_rmse_px": ransac_rmse,
                            "matrix_2x3": (
                                None
                                if ransac["matrix"] is None
                                else ransac["matrix"].tolist()
                            ),
                        },
                    }
                    transform_file = transform_root / (pair["pair_id"] + ".json")
                    write_json_atomic(transform_file, payload)
                    transform_path = str(transform_file)

                    transform_rows.append(
                        {
                            "pair_id": pair["pair_id"],
                            "case_id": pair["case_id"],
                            "cohort": spec["cohort"],
                            "pair_type": spec["pair_type"],
                            "stain": spec["stain"],
                            "landmark_count": len(matched_he_points),
                            "matching_method": matching_method,
                            "affine_rmse_px": rmse,
                            "affine_median_error_px": median_error,
                            "affine_max_error_px": max_error,
                            "affine_determinant": determinant,
                            "ransac_inliers": ransac_inliers,
                            "ransac_inlier_rmse_px": ransac_rmse,
                            "transform_path": transform_path,
                        }
                    )

            group_id = patient_group_id(case_key)
            public_row = {
                "pair_id": pair["pair_id"],
                "case_id": pair["case_id"],
                "patient_group_id": group_id,
                "split": "",
                "cohort": spec["cohort"],
                "pair_type": spec["pair_type"],
                "stain": spec["stain"],
                "landmark_status": status,
                "reason": reason,
                "he_landmark_json_count": len(he_jsons),
                "ihc_landmark_json_count": len(ihc_jsons),
                "he_point_count": len(he_points),
                "ihc_point_count": len(ihc_points),
                "matched_point_count": len(matched_he_points),
                "matching_method": matching_method,
                "unmatched_he_name_count": len(unmatched_he_names),
                "unmatched_ihc_name_count": len(unmatched_ihc_names),
                "point_names_equal": (
                    normalized_names(he_points) == normalized_names(ihc_points)
                    if he_points and ihc_points
                    else False
                ),
                "affine_rmse_px": rmse,
                "affine_median_error_px": median_error,
                "affine_max_error_px": max_error,
                "ransac_inliers": ransac_inliers,
                "transform_path": transform_path,
            }
            private_row = dict(public_row)
            private_row.update(
                {
                    "internal_case_key": case_key,
                    "he_wsi_path": he_info["project_link_path"],
                    "ihc_wsi_path": ihc_info["project_link_path"],
                    "he_landmark_json": (
                        str(he_jsons[0])
                        if len(he_jsons) == 1
                        else "|".join(map(str, he_jsons))
                    ),
                    "ihc_landmark_json": (
                        str(ihc_jsons[0])
                        if len(ihc_jsons) == 1
                        else "|".join(map(str, ihc_jsons))
                    ),
                }
            )
            public_rows.append(public_row)
            private_rows.append(private_row)

    split_map = stratified_patient_splits(public_rows)
    for row in public_rows:
        row["split"] = split_map[row["patient_group_id"]]
    for row in private_rows:
        row["split"] = split_map[row["patient_group_id"]]

    write_csv_atomic(
        output_root / "manual_pair_inventory.csv",
        public_rows,
        list(public_rows[0].keys()) if public_rows else [],
    )
    write_csv_atomic(
        private_root / "manual_pair_inventory_internal.csv",
        private_rows,
        list(private_rows[0].keys()) if private_rows else [],
    )
    write_csv_atomic(
        output_root / "affine_transform_summary.csv",
        transform_rows,
        list(transform_rows[0].keys()) if transform_rows else [],
    )

    print(
        "PAIRS",
        len(public_rows),
        "STATUS",
        dict(Counter(row["landmark_status"] for row in public_rows)),
    )
    for family in args.families:
        cohort = PAIR_SPECS[family]["cohort"]
        rows = [row for row in public_rows if row["cohort"] == cohort]
        print(
            "COHORT",
            cohort,
            "total",
            len(rows),
            "eligible",
            sum(row["landmark_status"] == "eligible" for row in rows),
        )


def import_aslide(args):
    root = Path(args.aslide_root)
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from aslide.aslide import Aslide
    return Aslide


def close_slide(slide):
    for candidate in (
        getattr(slide, "close", None),
        getattr(getattr(slide, "_osr", None), "close", None),
    ):
        if callable(candidate):
            try:
                candidate()
                return
            except Exception:
                pass


def rgb_array(image):
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        array = array[..., :3]
    return np.array(array, dtype=np.uint8, copy=True, order="C")


def tissue_mask(rgb):
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] > 20) | (hsv[..., 2] < 235)).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


def planning_thumbnail(slide, max_downsample):
    downsamples = [float(value) for value in slide.level_downsamples]
    eligible = [
        index for index, downsample in enumerate(downsamples)
        if downsample <= max_downsample
    ]
    level = (
        max(eligible, key=lambda index: downsamples[index])
        if eligible
        else int(np.argmin(downsamples))
    )
    width, height = slide.level_dimensions[level]
    downsample = downsamples[level]
    thumbnail = rgb_array(
        slide.read_region((0, 0), level, (int(width), int(height)))
    )
    return thumbnail, downsample


def mask_fraction(mask, center_x, center_y, size, downsample):
    half = size / 2.0
    x0 = max(0, int(math.floor((center_x - half) / downsample)))
    y0 = max(0, int(math.floor((center_y - half) / downsample)))
    x1 = min(mask.shape[1], int(math.ceil((center_x + half) / downsample)))
    y1 = min(mask.shape[0], int(math.ceil((center_y + half) / downsample)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


def plan(args):
    output_root = Path(args.output_root)
    internal_rows = [
        row
        for row in read_csv(
            output_root / "private" / "manual_pair_inventory_internal.csv"
        )
        if row["landmark_status"] == "eligible"
    ]
    if args.max_pairs is not None:
        internal_rows = internal_rows[: args.max_pairs]

    Aslide = import_aslide(args)
    rows = []
    for pair_index, info in enumerate(internal_rows, 1):
        slide = Aslide(info["he_wsi_path"])
        try:
            width, height = map(int, slide.dimensions)
            thumbnail, downsample = planning_thumbnail(
                slide, args.planning_max_downsample
            )
            mask = tissue_mask(thumbnail)
            half_context = args.context_size // 2
            candidates = []
            for center_y in range(
                half_context, height - half_context + 1, args.stride
            ):
                for center_x in range(
                    half_context, width - half_context + 1, args.stride
                ):
                    fraction = mask_fraction(
                        mask,
                        center_x,
                        center_y,
                        args.patch_size,
                        downsample,
                    )
                    if fraction >= args.min_tissue_fraction:
                        candidates.append((center_x, center_y, fraction))

            if (
                args.max_patches_per_pair
                and len(candidates) > args.max_patches_per_pair
            ):
                selected = np.linspace(
                    0,
                    len(candidates) - 1,
                    args.max_patches_per_pair,
                    dtype=int,
                )
                candidates = [candidates[index] for index in selected]

            for center_x, center_y, fraction in candidates:
                rows.append(
                    {
                        "sample_id": sample_id(
                            info["pair_id"],
                            center_x,
                            center_y,
                            args.patch_size,
                            args.context_size,
                        ),
                        "pair_id": info["pair_id"],
                        "case_id": info["case_id"],
                        "patient_group_id": info["patient_group_id"],
                        "split": info["split"],
                        "cohort": info["cohort"],
                        "pair_type": info["pair_type"],
                        "stain": info["stain"],
                        "he_center_x": center_x,
                        "he_center_y": center_y,
                        "patch_size": args.patch_size,
                        "context_size": args.context_size,
                        "stride": args.stride,
                        "he_tissue_fraction_estimate": fraction,
                        "affine_rmse_px": info["affine_rmse_px"],
                        "transform_path": info["transform_path"],
                        "plan_status": "planned",
                    }
                )
            print(
                "[%d/%d] %s patches=%d planning_ds=%g"
                % (
                    pair_index,
                    len(internal_rows),
                    info["pair_id"],
                    len(candidates),
                    downsample,
                )
            )
        finally:
            close_slide(slide)

    variant = dataset_variant(args.patch_size, args.context_size)
    fields = list(rows[0].keys()) if rows else []
    write_csv_atomic(output_root / "patch_plan.csv", rows, fields)
    write_csv_atomic(output_root / ("patch_plan_%s.csv" % variant), rows, fields)
    print("VARIANT", variant)
    print("PATCHES", len(rows), "SPLIT", dict(Counter(row["split"] for row in rows)))


def read_padded(slide, x0, y0, width, height):
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    valid = np.zeros((height, width), dtype=np.uint8)
    slide_width, slide_height = map(int, slide.dimensions)
    source_x0 = max(0, x0)
    source_y0 = max(0, y0)
    source_x1 = min(slide_width, x0 + width)
    source_y1 = min(slide_height, y0 + height)
    if source_x1 <= source_x0 or source_y1 <= source_y0:
        return canvas, valid
    array = rgb_array(
        slide.read_region(
            (source_x0, source_y0),
            0,
            (source_x1 - source_x0, source_y1 - source_y0),
        )
    )
    destination_x = source_x0 - x0
    destination_y = source_y0 - y0
    canvas[
        destination_y : destination_y + array.shape[0],
        destination_x : destination_x + array.shape[1],
    ] = array
    valid[
        destination_y : destination_y + array.shape[0],
        destination_x : destination_x + array.shape[1],
    ] = 1
    return canvas, valid


def warp_ihc_affine(ihc_slide, matrix, he_x0, he_y0, size, margin):
    corners = np.asarray(
        [
            [he_x0, he_y0],
            [he_x0 + size - 1, he_y0],
            [he_x0, he_y0 + size - 1],
            [he_x0 + size - 1, he_y0 + size - 1],
        ],
        dtype=np.float64,
    )
    mapped = np.column_stack([corners, np.ones(4)]) @ matrix.T
    source_x0 = int(math.floor(mapped[:, 0].min())) - margin
    source_y0 = int(math.floor(mapped[:, 1].min())) - margin
    source_x1 = int(math.ceil(mapped[:, 0].max())) + margin + 1
    source_y1 = int(math.ceil(mapped[:, 1].max())) + margin + 1

    source, source_valid = read_padded(
        ihc_slide,
        source_x0,
        source_y0,
        source_x1 - source_x0,
        source_y1 - source_y0,
    )

    x_grid, y_grid = np.meshgrid(np.arange(size), np.arange(size))
    global_x = x_grid.astype(np.float64) + he_x0
    global_y = y_grid.astype(np.float64) + he_y0
    map_x = (
        matrix[0, 0] * global_x
        + matrix[0, 1] * global_y
        + matrix[0, 2]
        - source_x0
    ).astype(np.float32)
    map_y = (
        matrix[1, 0] * global_x
        + matrix[1, 1] * global_y
        + matrix[1, 2]
        - source_y0
    ).astype(np.float32)

    warped = cv2.remap(
        source,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    valid = cv2.remap(
        source_valid,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped, valid


def center_crop(array, size):
    height, width = array.shape[:2]
    if size > height or size > width:
        raise ValueError(
            "Cannot center-crop %dx%d from %dx%d" % (size, size, width, height)
        )
    x0 = (width - size) // 2
    y0 = (height - size) // 2
    return array[y0 : y0 + size, x0 : x0 + size]


def dataset_variant(patch_size, context_size):
    return "ps%d_ctx%d" % (patch_size, context_size)


def import_dhr(args):
    root = str(Path(args.dhr_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    from deeperhistreg.dhr_pipeline.in_memory import register_and_warp_arrays
    return register_and_warp_arrays


def run_dhr(source_rgb, target_rgb, source_valid_mask, args):
    register_and_warp_arrays = import_dhr(args)
    temp_root = (
        Path(args.tmp_root)
        if args.tmp_root
        else Path(args.output_root) / "tmp"
    )
    temp_root.mkdir(parents=True, exist_ok=True)
    return register_and_warp_arrays(
        source_rgb,
        target_rgb,
        preset=args.dhr_preset,
        device=args.device,
        overrides=args.dhr_overrides,
        source_valid_mask=source_valid_mask,
        temporary_root=str(temp_root),
        keep_temporary=args.keep_dhr_temporary,
        padding_mode=args.dhr_padding_mode,
        valid_mask_threshold=args.dhr_valid_mask_threshold,
        return_valid_mask=True,
        return_displacement_field=False,
    )


def dhr_qc_rejection(qc, thresholds):
    checks = [
        ("max_folding_fraction", "folding_fraction", lambda value, limit: value > limit),
        (
            "max_nonpositive_jacobian_fraction",
            "nonpositive_jacobian_fraction",
            lambda value, limit: value > limit,
        ),
        (
            "max_displacement_p95_px",
            "displacement_p95_px",
            lambda value, limit: value > limit,
        ),
        ("min_jacobian_p01", "jacobian_p01", lambda value, limit: value < limit),
    ]
    reasons = []
    for threshold_key, qc_key, predicate in checks:
        limit = thresholds.get(threshold_key)
        if limit is not None and predicate(float(qc[qc_key]), float(limit)):
            reasons.append("%s:%g" % (threshold_key, float(limit)))
    return ";".join(reasons)


def save_image_atomic(path, array, image_format, jpeg_quality):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".png" if image_format == "png" else ".jpg"
    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        dir=str(path.parent),
        prefix=path.stem + ".",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        image = Image.fromarray(array)
        if image_format == "png":
            image.save(temp_path, format="PNG")
        else:
            image.save(temp_path, format="JPEG", quality=jpeg_quality)
        os.replace(str(temp_path), str(path))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def public_error_type(exc):
    return type(exc).__name__


def safe_unlink_derived(path, output_root):
    if not path:
        return
    candidate = Path(path)
    try:
        candidate.resolve().relative_to(Path(output_root).resolve())
    except (ValueError, FileNotFoundError):
        raise RuntimeError("Refusing to delete path outside derived output root")
    if candidate.is_file():
        candidate.unlink()


def private_error_message(exc):
    return "%s: %s" % (type(exc).__name__, str(exc))


MANIFEST_FIELDS = [
    "pipeline_schema_version",
    "sample_id",
    "pair_id",
    "case_id",
    "patient_group_id",
    "split",
    "cohort",
    "pair_type",
    "stain",
    "registration_version",
    "registration_stage",
    "dataset_variant",
    "he_center_x",
    "he_center_y",
    "patch_size",
    "context_size",
    "he_tissue_fraction",
    "affine_valid_fraction",
    "valid_fraction",
    "affine_rmse_px",
    "dhr_preset",
    "dhr_registration_time_seconds",
    "dhr_working_displacement_shape",
    "dhr_displacement_mean_px",
    "dhr_displacement_median_px",
    "dhr_displacement_p95_px",
    "dhr_displacement_max_px",
    "dhr_jacobian_min",
    "dhr_jacobian_p01",
    "dhr_jacobian_p05",
    "dhr_jacobian_median",
    "dhr_jacobian_p95",
    "dhr_jacobian_max",
    "dhr_folding_fraction",
    "dhr_nonpositive_jacobian_fraction",
    "dhr_valid_weight_mean",
    "qc_rejection_reason",
    "he_patch_path",
    "ihc_patch_path",
    "pipeline_git_commit",
    "pipeline_git_dirty",
    "pipeline_script_sha256",
    "dhr_git_commit",
    "dhr_git_dirty",
    "dhr_api_sha256",
    "config_sha256",
    "status",
    "error_type",
    "attempt",
    "updated_at",
]


PRIVATE_FAILURE_FIELDS = [
    "sample_id",
    "pair_id",
    "case_id",
    "registration_stage",
    "dataset_variant",
    "error_type",
    "error_message",
    "attempt",
    "updated_at",
]


def manifest_path(args):
    variant = dataset_variant(args.patch_size, args.context_size)
    stem = "patch_manifest_%s_%s" % (args.registration, variant)
    if args.num_shards > 1:
        stem += "_shard%03d-of-%03d" % (args.shard_index, args.num_shards)
    return Path(args.output_root) / (stem + ".csv")


def failure_path(args):
    variant = dataset_variant(args.patch_size, args.context_size)
    stem = "materialization_failures_%s_%s" % (args.registration, variant)
    if args.num_shards > 1:
        stem += "_shard%03d-of-%03d" % (args.shard_index, args.num_shards)
    return Path(args.output_root) / "private" / (stem + ".csv")


def select_plan_rows(args):
    plan_path = (
        Path(args.plan)
        if args.plan
        else Path(args.output_root)
        / ("patch_plan_%s.csv" % dataset_variant(args.patch_size, args.context_size))
    )
    rows = read_csv(plan_path)
    for row in rows:
        if (
            int(row["patch_size"]) != args.patch_size
            or int(row["context_size"]) != args.context_size
        ):
            raise ValueError("Plan contains geometry inconsistent with current config")
    if args.split:
        rows = [row for row in rows if row["split"] == args.split]
    if args.sample_id:
        rows = [row for row in rows if row["sample_id"] == args.sample_id]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if args.num_shards > 1:
        rows = [
            row
            for row in rows
            if int(deterministic_hash(row["sample_id"])[:8], 16) % args.num_shards
            == args.shard_index
        ]
    return rows


def build_base_manifest_row(plan_row, args, provenance, attempt):
    return {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "sample_id": plan_row["sample_id"],
        "pair_id": plan_row["pair_id"],
        "case_id": plan_row["case_id"],
        "patient_group_id": plan_row["patient_group_id"],
        "split": plan_row["split"],
        "cohort": plan_row["cohort"],
        "pair_type": plan_row["pair_type"],
        "stain": plan_row["stain"],
        "registration_version": args.registration_version,
        "registration_stage": args.registration,
        "dataset_variant": dataset_variant(args.patch_size, args.context_size),
        "he_center_x": plan_row["he_center_x"],
        "he_center_y": plan_row["he_center_y"],
        "patch_size": args.patch_size,
        "context_size": args.context_size,
        "he_tissue_fraction": "",
        "affine_valid_fraction": "",
        "valid_fraction": "",
        "affine_rmse_px": plan_row["affine_rmse_px"],
        "dhr_preset": args.dhr_preset if args.registration == "dhr" else "",
        "dhr_registration_time_seconds": "",
        "dhr_working_displacement_shape": "",
        "dhr_displacement_mean_px": "",
        "dhr_displacement_median_px": "",
        "dhr_displacement_p95_px": "",
        "dhr_displacement_max_px": "",
        "dhr_jacobian_min": "",
        "dhr_jacobian_p01": "",
        "dhr_jacobian_p05": "",
        "dhr_jacobian_median": "",
        "dhr_jacobian_p95": "",
        "dhr_jacobian_max": "",
        "dhr_folding_fraction": "",
        "dhr_nonpositive_jacobian_fraction": "",
        "dhr_valid_weight_mean": "",
        "qc_rejection_reason": "",
        "he_patch_path": "",
        "ihc_patch_path": "",
        "pipeline_git_commit": provenance["pipeline_git_commit"],
        "pipeline_git_dirty": provenance["pipeline_git_dirty"],
        "pipeline_script_sha256": provenance["pipeline_script_sha256"],
        "dhr_git_commit": provenance["dhr_git_commit"],
        "dhr_git_dirty": provenance["dhr_git_dirty"],
        "dhr_api_sha256": provenance["dhr_api_sha256"],
        "config_sha256": provenance["config_sha256"],
        "status": "processing",
        "error_type": "",
        "attempt": attempt,
        "updated_at": utc_now(),
    }


def should_skip_existing(existing, args, provenance):
    if args.force or not existing:
        return False
    if existing.get("pipeline_schema_version") != PIPELINE_SCHEMA_VERSION:
        return False
    for key in (
        "config_sha256",
        "pipeline_script_sha256",
        "dhr_api_sha256",
    ):
        if existing.get(key, "") != str(provenance.get(key, "")):
            return False
    status = existing.get("status")
    if status == "ready":
        return (
            bool(existing.get("he_patch_path"))
            and bool(existing.get("ihc_patch_path"))
            and Path(existing["he_patch_path"]).is_file()
            and Path(existing["ihc_patch_path"]).is_file()
        )
    if status == "failed":
        return not args.retry_failed
    if status in {"low_tissue", "low_valid_fraction", "dhr_qc_rejected"}:
        return not args.retry_rejected
    return False


def flush_manifest(path, manifest_by_id):
    ordered = sorted(
        manifest_by_id.values(),
        key=lambda row: (
            row.get("split", ""),
            row.get("pair_id", ""),
            row.get("sample_id", ""),
        ),
    )
    write_csv_atomic(path, ordered, MANIFEST_FIELDS)


def flush_failures(path, failures_by_id):
    ordered = sorted(failures_by_id.values(), key=lambda row: row["sample_id"])
    write_csv_atomic(path, ordered, PRIVATE_FAILURE_FIELDS)


def materialize(args):
    if args.context_size < args.patch_size:
        raise ValueError("context_size must be >= patch_size")
    if args.num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

    output_root = Path(args.output_root)
    plan_rows = select_plan_rows(args)
    private_inventory = {
        row["pair_id"]: row
        for row in read_csv(
            output_root / "private" / "manual_pair_inventory_internal.csv"
        )
        if row["landmark_status"] == "eligible"
    }

    target_manifest = manifest_path(args)
    target_failures = failure_path(args)
    existing_rows = read_csv(target_manifest) if target_manifest.exists() else []
    manifest_by_id = {row["sample_id"]: row for row in existing_rows}
    existing_failures = read_csv(target_failures) if target_failures.exists() else []
    failures_by_id = {row["sample_id"]: row for row in existing_failures}

    dhr_api_path = Path(args.dhr_root) / "deeperhistreg" / "dhr_pipeline" / "in_memory.py"
    provenance = {
        "pipeline_git_commit": git_commit(args.repo_root),
        "pipeline_git_dirty": str(git_dirty(args.repo_root)),
        "pipeline_script_sha256": sha256_file(Path(__file__).resolve()),
        "dhr_git_commit": git_commit(args.dhr_root),
        "dhr_git_dirty": str(git_dirty(args.dhr_root)),
        "dhr_api_sha256": sha256_file(dhr_api_path),
        "config_sha256": sha256_file(args.config),
    }
    snapshot = {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "registration_version": args.registration_version,
        "dataset_variant": dataset_variant(args.patch_size, args.context_size),
        "registration_stage": args.registration,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": provenance["config_sha256"],
        "pipeline_git_commit": provenance["pipeline_git_commit"],
        "pipeline_git_dirty": provenance["pipeline_git_dirty"],
        "pipeline_script_sha256": provenance["pipeline_script_sha256"],
        "dhr_git_commit": provenance["dhr_git_commit"],
        "dhr_git_dirty": provenance["dhr_git_dirty"],
        "dhr_api_sha256": provenance["dhr_api_sha256"],
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "created_at": utc_now(),
    }
    snapshot_name = "provenance_%s_%s" % (
        args.registration,
        dataset_variant(args.patch_size, args.context_size),
    )
    if args.num_shards > 1:
        snapshot_name += "_shard%03d-of-%03d" % (args.shard_index, args.num_shards)
    write_json_atomic(output_root / (snapshot_name + ".json"), snapshot)

    Aslide = import_aslide(args)
    by_pair = defaultdict(list)
    skipped = 0
    for plan_row in plan_rows:
        existing = manifest_by_id.get(plan_row["sample_id"])
        if should_skip_existing(existing, args, provenance):
            skipped += 1
            continue
        by_pair[plan_row["pair_id"]].append(plan_row)

    print(
        "MATERIALIZE",
        "selected",
        len(plan_rows),
        "to_process",
        sum(len(rows) for rows in by_pair.values()),
        "resume_skipped",
        skipped,
        "shard",
        "%d/%d" % (args.shard_index, args.num_shards),
    )

    processed_since_flush = 0
    for pair_index, pair_id in enumerate(sorted(by_pair), 1):
        info = private_inventory[pair_id]
        he_slide = Aslide(info["he_wsi_path"])
        ihc_slide = Aslide(info["ihc_wsi_path"])
        try:
            transform = load_json(info["transform_path"])
            matrix = np.asarray(transform["matrix_2x3"], dtype=np.float64)

            for row in by_pair[pair_id]:
                sample = row["sample_id"]
                prior = manifest_by_id.get(sample)
                attempt = int(prior.get("attempt", 0) or 0) + 1 if prior else 1
                result = build_base_manifest_row(row, args, provenance, attempt)
                he_path = None
                ihc_path = None

                if prior:
                    safe_unlink_derived(prior.get("he_patch_path", ""), output_root)
                    safe_unlink_derived(prior.get("ihc_patch_path", ""), output_root)

                try:
                    patch_size = int(row["patch_size"])
                    context_size = int(row["context_size"])
                    center_x = int(row["he_center_x"])
                    center_y = int(row["he_center_y"])
                    context_x0 = int(round(center_x - context_size / 2))
                    context_y0 = int(round(center_y - context_size / 2))

                    he_context, he_valid_context = read_padded(
                        he_slide,
                        context_x0,
                        context_y0,
                        context_size,
                        context_size,
                    )
                    ihc_affine_context, ihc_affine_valid_context = warp_ihc_affine(
                        ihc_slide,
                        matrix,
                        context_x0,
                        context_y0,
                        context_size,
                        args.affine_margin,
                    )

                    he_patch = center_crop(he_context, patch_size)
                    he_valid_patch = center_crop(he_valid_context, patch_size)
                    affine_valid_patch = center_crop(
                        ihc_affine_valid_context, patch_size
                    )
                    tissue_fraction = float(tissue_mask(he_patch).mean())
                    affine_valid_fraction = float(
                        np.mean(
                            (he_valid_patch > 0)
                            & (affine_valid_patch > 0)
                        )
                    )
                    result["he_tissue_fraction"] = tissue_fraction
                    result["affine_valid_fraction"] = affine_valid_fraction

                    if tissue_fraction < args.min_tissue_fraction:
                        result["status"] = "low_tissue"
                        result["updated_at"] = utc_now()
                        manifest_by_id[sample] = result
                        failures_by_id.pop(sample, None)
                        continue

                    if affine_valid_fraction < args.min_valid_fraction:
                        result["status"] = "low_valid_fraction"
                        result["valid_fraction"] = affine_valid_fraction
                        result["updated_at"] = utc_now()
                        manifest_by_id[sample] = result
                        failures_by_id.pop(sample, None)
                        continue

                    if args.registration == "dhr":
                        ihc_registered_context, dhr_meta = run_dhr(
                            ihc_affine_context,
                            he_context,
                            ihc_affine_valid_context,
                            args,
                        )
                        dhr_valid_context = dhr_meta["valid_mask"]
                        dhr_valid_patch = center_crop(
                            dhr_valid_context.astype(np.uint8), patch_size
                        )
                        final_valid_patch = (
                            (he_valid_patch > 0) & (dhr_valid_patch > 0)
                        )
                        valid_fraction = float(final_valid_patch.mean())
                        ihc_patch = center_crop(
                            ihc_registered_context, patch_size
                        )
                        dhr_qc = dhr_meta["deformation_qc"]
                        result.update(
                            {
                                "dhr_registration_time_seconds": dhr_meta[
                                    "registration_time_seconds"
                                ],
                                "dhr_working_displacement_shape": json.dumps(
                                    dhr_meta["working_displacement_shape"]
                                ),
                                "dhr_displacement_mean_px": dhr_qc[
                                    "displacement_mean_px"
                                ],
                                "dhr_displacement_median_px": dhr_qc[
                                    "displacement_median_px"
                                ],
                                "dhr_displacement_p95_px": dhr_qc[
                                    "displacement_p95_px"
                                ],
                                "dhr_displacement_max_px": dhr_qc[
                                    "displacement_max_px"
                                ],
                                "dhr_jacobian_min": dhr_qc["jacobian_min"],
                                "dhr_jacobian_p01": dhr_qc["jacobian_p01"],
                                "dhr_jacobian_p05": dhr_qc["jacobian_p05"],
                                "dhr_jacobian_median": dhr_qc[
                                    "jacobian_median"
                                ],
                                "dhr_jacobian_p95": dhr_qc["jacobian_p95"],
                                "dhr_jacobian_max": dhr_qc["jacobian_max"],
                                "dhr_folding_fraction": dhr_qc[
                                    "folding_fraction"
                                ],
                                "dhr_nonpositive_jacobian_fraction": dhr_qc[
                                    "nonpositive_jacobian_fraction"
                                ],
                                "dhr_valid_weight_mean": dhr_qc[
                                    "valid_weight_mean"
                                ],
                            }
                        )
                        rejection = dhr_qc_rejection(
                            dhr_qc, args.dhr_qc_thresholds
                        )
                        if rejection:
                            result["status"] = "dhr_qc_rejected"
                            result["qc_rejection_reason"] = rejection
                            result["valid_fraction"] = valid_fraction
                            result["updated_at"] = utc_now()
                            manifest_by_id[sample] = result
                            failures_by_id.pop(sample, None)
                            continue
                    else:
                        ihc_patch = center_crop(
                            ihc_affine_context, patch_size
                        )
                        final_valid_patch = (
                            (he_valid_patch > 0)
                            & (affine_valid_patch > 0)
                        )
                        valid_fraction = float(final_valid_patch.mean())

                    result["valid_fraction"] = valid_fraction
                    if valid_fraction < args.min_valid_fraction:
                        result["status"] = "low_valid_fraction"
                        result["updated_at"] = utc_now()
                        manifest_by_id[sample] = result
                        failures_by_id.pop(sample, None)
                        continue

                    extension = "png" if args.image_format == "png" else "jpg"
                    variant = dataset_variant(patch_size, context_size)
                    he_path = (
                        output_root
                        / "patches"
                        / args.registration
                        / variant
                        / row["split"]
                        / "A"
                        / (sample + "." + extension)
                    )
                    ihc_path = (
                        output_root
                        / "patches"
                        / args.registration
                        / variant
                        / row["split"]
                        / "B"
                        / (sample + "." + extension)
                    )
                    save_image_atomic(
                        he_path, he_patch, args.image_format, args.jpeg_quality
                    )
                    save_image_atomic(
                        ihc_path, ihc_patch, args.image_format, args.jpeg_quality
                    )

                    result["he_patch_path"] = str(he_path)
                    result["ihc_patch_path"] = str(ihc_path)
                    result["status"] = "ready"
                    result["updated_at"] = utc_now()
                    manifest_by_id[sample] = result
                    failures_by_id.pop(sample, None)

                except Exception as exc:
                    safe_unlink_derived(he_path, output_root)
                    safe_unlink_derived(ihc_path, output_root)
                    result["status"] = "failed"
                    result["error_type"] = public_error_type(exc)
                    result["updated_at"] = utc_now()
                    manifest_by_id[sample] = result
                    failures_by_id[sample] = {
                        "sample_id": sample,
                        "pair_id": row["pair_id"],
                        "case_id": row["case_id"],
                        "registration_stage": args.registration,
                        "dataset_variant": dataset_variant(
                            args.patch_size, args.context_size
                        ),
                        "error_type": public_error_type(exc),
                        "error_message": private_error_message(exc),
                        "attempt": attempt,
                        "updated_at": utc_now(),
                    }
                    print(
                        "FAILED",
                        sample,
                        public_error_type(exc),
                        file=sys.stderr,
                    )
                    if args.keep_dhr_temporary:
                        traceback.print_exc()
                    if args.fail_fast:
                        flush_manifest(target_manifest, manifest_by_id)
                        flush_failures(target_failures, failures_by_id)
                        raise

                finally:
                    processed_since_flush += 1
                    if processed_since_flush >= args.flush_every:
                        flush_manifest(target_manifest, manifest_by_id)
                        flush_failures(target_failures, failures_by_id)
                        processed_since_flush = 0

            print(
                "[%d/%d] %s samples=%d"
                % (pair_index, len(by_pair), pair_id, len(by_pair[pair_id]))
            )
        finally:
            close_slide(he_slide)
            close_slide(ihc_slide)

    flush_manifest(target_manifest, manifest_by_id)
    flush_failures(target_failures, failures_by_id)

    selected_ids = {row["sample_id"] for row in plan_rows}
    selected_results = [
        row for sample, row in manifest_by_id.items() if sample in selected_ids
    ]
    print(
        "MANIFEST",
        target_manifest,
        "STATUS",
        dict(Counter(row["status"] for row in selected_results)),
    )


def select_plan_rows_for_merge(args):
    original_num_shards = args.num_shards
    original_shard_index = args.shard_index
    try:
        args.num_shards = 1
        args.shard_index = 0
        return select_plan_rows(args)
    finally:
        args.num_shards = original_num_shards
        args.shard_index = original_shard_index


def merge(args):
    if args.num_shards <= 1:
        raise ValueError("merge stage requires --num-shards > 1")
    output_root = Path(args.output_root)
    variant = dataset_variant(args.patch_size, args.context_size)
    stem = "patch_manifest_%s_%s" % (args.registration, variant)
    rows_by_id = {}
    missing_paths = []
    for shard_index in range(args.num_shards):
        path = output_root / (
            stem + "_shard%03d-of-%03d.csv" % (shard_index, args.num_shards)
        )
        if not path.exists():
            missing_paths.append(str(path))
            continue
        for row in read_csv(path):
            sample = row["sample_id"]
            if sample in rows_by_id:
                raise RuntimeError(
                    "Duplicate sample_id across shard manifests: %s" % sample
                )
            rows_by_id[sample] = row

    if missing_paths:
        raise RuntimeError("Missing shard manifests: %s" % ", ".join(missing_paths))

    plan_rows = select_plan_rows_for_merge(args)
    expected_ids = {row["sample_id"] for row in plan_rows}
    actual_ids = set(rows_by_id)
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    if extra_ids:
        raise RuntimeError("Shard manifests contain samples not in plan")
    if args.require_complete and missing_ids:
        raise RuntimeError(
            "Cannot merge incomplete materialization: %d planned samples missing"
            % len(missing_ids)
        )

    ordered = sorted(
        rows_by_id.values(),
        key=lambda row: (
            row.get("split", ""),
            row.get("pair_id", ""),
            row.get("sample_id", ""),
        ),
    )
    canonical = output_root / (stem + ".csv")
    write_csv_atomic(canonical, ordered, MANIFEST_FIELDS)
    print(
        "MERGED",
        canonical,
        "ROWS",
        len(ordered),
        "MISSING",
        len(missing_ids),
        "STATUS",
        dict(Counter(row["status"] for row in ordered)),
    )


def load_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Config does not exist: %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser():
    default_config = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "vsseg"
        / "registration_v0_manual.json"
    )
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(default_config))
    pre_args, _ = pre.parse_known_args()
    config = load_config(pre_args.config)

    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument(
        "--stage",
        choices=["audit", "plan", "materialize", "merge"],
        required=True,
    )
    parser.add_argument("--project-root", default=config["project_root"])
    parser.add_argument("--output-root", default=config["output_root"])
    parser.add_argument(
        "--registration-version",
        default=config.get("registration_version", REGISTRATION_VERSION),
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=sorted(PAIR_SPECS),
        default=config["families"],
    )
    parser.add_argument(
        "--min-landmarks",
        type=int,
        default=int(config["min_landmarks"]),
    )
    parser.add_argument(
        "--require-matching-names",
        action=argparse.BooleanOptionalAction,
        default=bool(config["require_matching_names"]),
    )
    parser.add_argument(
        "--ransac-threshold-px",
        type=float,
        default=float(config["ransac_threshold_px"]),
    )
    parser.add_argument(
        "--max-affine-rmse-px",
        type=float,
        default=float(config["max_affine_rmse_px"]),
    )
    parser.add_argument("--aslide-root", default=config["aslide_root"])
    parser.add_argument("--dhr-root", default=config["dhr_root"])
    parser.add_argument("--patch-size", type=int, default=int(config["patch_size"]))
    parser.add_argument(
        "--context-size",
        type=int,
        default=int(config["context_size"]),
    )
    parser.add_argument("--stride", type=int, default=int(config["stride"]))
    parser.add_argument(
        "--min-tissue-fraction",
        type=float,
        default=float(config["min_tissue_fraction"]),
    )
    parser.add_argument(
        "--max-patches-per-pair",
        type=int,
        default=int(config["max_patches_per_pair"]),
    )
    parser.add_argument(
        "--planning-max-downsample",
        type=float,
        default=float(config["planning_max_downsample"]),
    )
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--plan")
    parser.add_argument(
        "--registration",
        choices=["affine", "dhr"],
        default="dhr",
    )
    parser.add_argument(
        "--affine-margin",
        type=int,
        default=int(config["affine_margin"]),
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=float(config["min_valid_fraction"]),
    )
    parser.add_argument(
        "--image-format",
        choices=["png", "jpg"],
        default=config["image_format"],
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=int(config["jpeg_quality"]),
    )
    parser.add_argument("--dhr-preset", default=config["dhr_preset"])
    parser.add_argument(
        "--dhr-padding-mode",
        choices=["zeros", "border", "reflection"],
        default=config["dhr_padding_mode"],
    )
    parser.add_argument(
        "--dhr-valid-mask-threshold",
        type=float,
        default=float(config["dhr_valid_mask_threshold"]),
    )
    parser.add_argument("--keep-dhr-temporary", action="store_true")
    parser.add_argument("--tmp-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--sample-id")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--require-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.set_defaults(
        dhr_overrides=config.get("dhr_overrides", {}),
        dhr_qc_thresholds=config.get("dhr_qc_thresholds", {}),
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.config = str(Path(args.config).resolve())
    if args.context_size < args.patch_size:
        parser.error("--context-size must be >= --patch-size")
    if args.flush_every < 1:
        parser.error("--flush-every must be >= 1")
    stages = {
        "audit": audit,
        "plan": plan,
        "materialize": materialize,
        "merge": merge,
    }
    stages[args.stage](args)


if __name__ == "__main__":
    main()
