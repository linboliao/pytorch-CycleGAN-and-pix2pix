#!/usr/bin/env python3
"""Plan and materialize a small VS-Seg registration-v1 paired patch dataset.

The public dataset contains de-identified pair/sample IDs and derived image paths
only. Raw WSI paths are kept under manifests/private.

A candidate is accepted only if:
1) the HE level-0 context agrees with the HE component's trusted pyramid level;
2) the raw IHC level-0 source ROI required by the affine agrees with the IHC
   component's trusted pyramid level;
3) HE tissue fraction and affine/final valid fractions pass their gates;
4) DHR completes.

No lower-resolution image is upsampled and saved as a training patch.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import registration_utils as utils
import registration_v1_component as registration_v1


SCHEMA_VERSION = 1

PUBLIC_FIELDS = [
    "dataset_schema_version",
    "dataset_version",
    "dataset_variant",
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
    "component_id",
    "he_center_x_level0",
    "he_center_y_level0",
    "patch_size",
    "context_size",
    "he_trusted_level",
    "ihc_trusted_level",
    "he_trusted_downsample",
    "ihc_trusted_downsample",
    "planned_tissue_fraction",
    "he_tissue_fraction",
    "he_roi_gray_corr",
    "he_roi_edge_corr",
    "he_roi_mask_iou",
    "ihc_roi_gray_corr",
    "ihc_roi_edge_corr",
    "ihc_roi_mask_iou",
    "affine_valid_fraction",
    "valid_fraction",
    "dhr_registration_time_seconds",
    "dhr_displacement_p95_px",
    "dhr_folding_fraction",
    "dhr_nonpositive_jacobian_fraction",
    "qc_rejection_reason",
    "he_patch_path",
    "ihc_patch_path",
    "qc_path",
    "status",
    "attempt",
    "updated_at",
]

PRIVATE_FIELDS = PUBLIC_FIELDS + [
    "he_wsi_path",
    "ihc_wsi_path",
    "transform_path",
    "ihc_source_bbox_level0",
    "error_type",
    "error_message",
]

PLAN_FIELDS = [
    "sample_id",
    "pair_id",
    "case_id",
    "patient_group_id",
    "split",
    "cohort",
    "pair_type",
    "stain",
    "component_id",
    "he_center_x_level0",
    "he_center_y_level0",
    "patch_size",
    "context_size",
    "stride",
    "planned_tissue_fraction",
    "he_trusted_level",
    "ihc_trusted_level",
    "he_trusted_downsample",
    "ihc_trusted_downsample",
    "transform_path",
    "status",
]

PRIVATE_PLAN_FIELDS = PLAN_FIELDS + ["he_wsi_path", "ihc_wsi_path"]


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def deterministic_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sample_id(pair_id, component_id, x, y, patch_size, context_size):
    token = "|".join(
        map(
            str,
            [
                "registration_v1_component",
                pair_id,
                component_id,
                x,
                y,
                patch_size,
                context_size,
            ],
        )
    )
    return "PATCH_" + deterministic_hash(token)[:16].upper()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_png_atomic(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".png", dir=str(path.parent), prefix=path.stem + ".", delete=False
    ) as handle:
        tmp = Path(handle.name)
    try:
        Image.fromarray(np.asarray(array, dtype=np.uint8)).save(tmp, format="PNG")
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


def git_dirty(path):
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except Exception:
        return None


def read_inventory(path):
    return {row["pair_id"]: row for row in utils.read_csv(path)}


def trusted_level(transform, role):
    return int(transform["read_qc"]["%s_used_level" % role])


def trusted_downsample(transform, role):
    return float(transform["read_qc"]["%s_used_downsample" % role])


def approximate_patch_tissue_fraction(
    trusted_mask, component_bbox0, center_x, center_y, patch_size
):
    bx0, by0, bx1, by1 = map(float, component_bbox0)
    sx = trusted_mask.shape[1] / max(1.0, bx1 - bx0)
    sy = trusted_mask.shape[0] / max(1.0, by1 - by0)
    half = patch_size / 2.0
    x0 = int(math.floor((center_x - half - bx0) * sx))
    x1 = int(math.ceil((center_x + half - bx0) * sx))
    y0 = int(math.floor((center_y - half - by0) * sy))
    y1 = int(math.ceil((center_y + half - by0) * sy))
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(trusted_mask.shape[1], x1)
    y1 = min(trusted_mask.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((trusted_mask[y0:y1, x0:x1] > 0).mean())


def grid_centers(component_bbox0, slide_dimensions, context_size, stride):
    bx0, by0, bx1, by1 = map(int, component_bbox0)
    slide_w, slide_h = map(int, slide_dimensions)
    half = context_size // 2
    min_x = max(bx0 + half, half)
    max_x = min(bx1 - half, slide_w - half)
    min_y = max(by0 + half, half)
    max_y = min(by1 - half, slide_h - half)
    if min_x > max_x or min_y > max_y:
        return []
    start_x = int(math.ceil(min_x / float(stride)) * stride)
    start_y = int(math.ceil(min_y / float(stride)) * stride)
    return [
        (x, y)
        for y in range(start_y, max_y + 1, stride)
        for x in range(start_x, max_x + 1, stride)
    ]


def component_quotas(total, count):
    if count <= 0:
        return []
    base = total // count
    remainder = total % count
    return [base + (1 if i < remainder else 0) for i in range(count)]


def plan(args):
    output_root = Path(args.output_root)
    manifest_dir = output_root / "manifests"
    private_dir = manifest_dir / "private"
    inventory = read_inventory(args.inventory)
    Aslide = utils.import_aslide(args.aslide_root)
    public_rows = []
    private_rows = []

    for pair_id in args.pair_ids:
        info = inventory[pair_id]
        transform_dir = Path(args.registration_root) / "transforms" / pair_id
        transform_paths = sorted(transform_dir.glob("component_*.json"))
        transforms = [
            load_json(path)
            for path in transform_paths
            if load_json(path).get("confidence") in {"medium", "high"}
        ]
        if not transforms:
            raise RuntimeError("No accepted v1 component transforms for %s" % pair_id)
        quotas = component_quotas(args.candidates_per_pair, len(transforms))
        he_slide = Aslide(info["he_wsi_path"])
        try:
            for transform_path, transform, quota in zip(
                transform_paths, transforms, quotas
            ):
                component_id = int(transform["component_id"])
                bbox0 = transform["he_component_bbox_level0"]
                level = trusted_level(transform, "he")
                trusted_rgb, _ = registration_v1.read_level0_bbox_at_level(
                    he_slide, bbox0, level
                )
                trusted_mask = utils.tissue_mask(trusted_rgb)
                candidates = []
                for center_x, center_y in grid_centers(
                    bbox0, he_slide.dimensions, args.context_size, args.stride
                ):
                    tissue_fraction = approximate_patch_tissue_fraction(
                        trusted_mask,
                        bbox0,
                        center_x,
                        center_y,
                        args.patch_size,
                    )
                    if tissue_fraction < args.min_tissue_fraction:
                        continue
                    sid = sample_id(
                        pair_id,
                        component_id,
                        center_x,
                        center_y,
                        args.patch_size,
                        args.context_size,
                    )
                    candidates.append(
                        (deterministic_hash(sid), sid, center_x, center_y, tissue_fraction)
                    )
                candidates.sort(key=lambda item: item[0])
                selected = candidates[:quota]
                if len(selected) < quota:
                    print(
                        "PLAN_WARNING",
                        pair_id,
                        "component",
                        component_id,
                        "requested",
                        quota,
                        "available",
                        len(selected),
                        file=sys.stderr,
                    )
                for _, sid, center_x, center_y, tissue_fraction in selected:
                    row = {
                        "sample_id": sid,
                        "pair_id": pair_id,
                        "case_id": info["case_id"],
                        "patient_group_id": info["patient_group_id"],
                        "split": info["split"],
                        "cohort": info["cohort"],
                        "pair_type": info["pair_type"],
                        "stain": info["stain"],
                        "component_id": component_id,
                        "he_center_x_level0": center_x,
                        "he_center_y_level0": center_y,
                        "patch_size": args.patch_size,
                        "context_size": args.context_size,
                        "stride": args.stride,
                        "planned_tissue_fraction": tissue_fraction,
                        "he_trusted_level": trusted_level(transform, "he"),
                        "ihc_trusted_level": trusted_level(transform, "ihc"),
                        "he_trusted_downsample": trusted_downsample(transform, "he"),
                        "ihc_trusted_downsample": trusted_downsample(transform, "ihc"),
                        "transform_path": str(transform_path),
                        "status": "planned",
                    }
                    public_rows.append(row)
                    private_rows.append(
                        {
                            **row,
                            "he_wsi_path": info["he_wsi_path"],
                            "ihc_wsi_path": info["ihc_wsi_path"],
                        }
                    )
        finally:
            utils.close_slide(he_slide)

    public_rows.sort(
        key=lambda row: (row["pair_id"], int(row["component_id"]), row["sample_id"])
    )
    private_rows.sort(
        key=lambda row: (row["pair_id"], int(row["component_id"]), row["sample_id"])
    )
    utils.write_csv_atomic(manifest_dir / "patch_plan.csv", public_rows, PLAN_FIELDS)
    utils.write_csv_atomic(
        private_dir / "patch_plan_internal.csv", private_rows, PRIVATE_PLAN_FIELDS
    )
    print(
        "PLAN",
        len(public_rows),
        "pairs",
        dict(Counter(row["pair_id"] for row in public_rows)),
    )
    return public_rows


def affine_source_bbox(matrix, he_bbox0, margin):
    x0, y0, x1, y1 = map(float, he_bbox0)
    corners = np.asarray(
        [[x0, y0], [x1 - 1, y0], [x0, y1 - 1], [x1 - 1, y1 - 1]],
        dtype=np.float64,
    )
    mapped = np.column_stack([corners, np.ones(4)]) @ np.asarray(matrix).T
    return [
        int(math.floor(mapped[:, 0].min())) - margin,
        int(math.floor(mapped[:, 1].min())) - margin,
        int(math.ceil(mapped[:, 0].max())) + margin + 1,
        int(math.ceil(mapped[:, 1].max())) + margin + 1,
    ]


def clip_bbox(bbox, dimensions):
    x0, y0, x1, y1 = map(int, bbox)
    width, height = map(int, dimensions)
    return [max(0, x0), max(0, y0), min(width, x1), min(height, y1)]


def read_level0_padded_bbox(slide, bbox):
    x0, y0, x1, y1 = map(int, bbox)
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError("Empty requested bbox")
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    valid = np.zeros((height, width), dtype=np.uint8)
    clipped = clip_bbox(bbox, slide.dimensions)
    cx0, cy0, cx1, cy1 = clipped
    if cx1 <= cx0 or cy1 <= cy0:
        return canvas, valid
    image, _ = registration_v1.read_level0_bbox_at_level(
        slide, clipped, 0
    )
    dx = cx0 - x0
    dy = cy0 - y0
    canvas[dy : dy + image.shape[0], dx : dx + image.shape[1]] = image
    valid[dy : dy + image.shape[0], dx : dx + image.shape[1]] = 1
    return canvas, valid


def gray_corr(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size < 16 or a.size != b.size or a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def edge_magnitude(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def roi_integrity_metrics(level0_rgb, trusted_rgb):
    trusted_size = (trusted_rgb.shape[1], trusted_rgb.shape[0])
    candidate = cv2.resize(
        level0_rgb, trusted_size, interpolation=cv2.INTER_AREA
    )
    cgray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    tgray = cv2.cvtColor(trusted_rgb, cv2.COLOR_RGB2GRAY)
    cmask = utils.tissue_mask(candidate) > 0
    tmask = utils.tissue_mask(trusted_rgb) > 0
    intersection = np.logical_and(cmask, tmask).sum()
    union = np.logical_or(cmask, tmask).sum()
    return {
        "gray_corr": gray_corr(cgray, tgray),
        "edge_corr": gray_corr(edge_magnitude(candidate), edge_magnitude(trusted_rgb)),
        "mask_iou": float(intersection / max(1, union)),
        "gray_mae": float(
            np.abs(cgray.astype(np.float32) - tgray.astype(np.float32)).mean()
        ),
    }


def roi_integrity_passes(metrics, args):
    return bool(
        metrics["edge_corr"] >= args.roi_integrity_min_edge_corr
        and (
            metrics["gray_corr"] >= args.roi_integrity_min_gray_corr
            or metrics["mask_iou"] >= args.roi_integrity_min_mask_iou
        )
    )


def validate_level0_roi(slide, bbox0, trusted_level, level0_rgb, args):
    if int(trusted_level) == 0:
        return {
            "gray_corr": 1.0,
            "edge_corr": 1.0,
            "mask_iou": 1.0,
            "gray_mae": 0.0,
            "passed": True,
        }, level0_rgb
    trusted_rgb, _ = registration_v1.read_level0_bbox_at_level(
        slide, bbox0, int(trusted_level)
    )
    metrics = roi_integrity_metrics(level0_rgb, trusted_rgb)
    metrics["passed"] = roi_integrity_passes(metrics, args)
    return metrics, trusted_rgb


def warp_source_to_he(source, source_valid, source_bbox, matrix, he_bbox0):
    sx0, sy0, _, _ = map(int, source_bbox)
    hx0, hy0, hx1, hy1 = map(int, he_bbox0)
    width = hx1 - hx0
    height = hy1 - hy0
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
    global_x = x_grid.astype(np.float64) + hx0
    global_y = y_grid.astype(np.float64) + hy0
    matrix = np.asarray(matrix, dtype=np.float64)
    map_x = (
        matrix[0, 0] * global_x
        + matrix[0, 1] * global_y
        + matrix[0, 2]
        - sx0
    ).astype(np.float32)
    map_y = (
        matrix[1, 0] * global_x
        + matrix[1, 1] * global_y
        + matrix[1, 2]
        - sy0
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


def run_dhr(source_rgb, target_rgb, source_valid_mask, args):
    register = utils.import_dhr(args.dhr_root)
    Path(args.dhr_tmp_root).mkdir(parents=True, exist_ok=True)
    return register(
        source_rgb,
        target_rgb,
        preset=args.dhr_preset,
        device=args.device,
        overrides=args.dhr_overrides,
        source_valid_mask=source_valid_mask,
        temporary_root=args.dhr_tmp_root,
        keep_temporary=False,
        padding_mode=args.dhr_padding_mode,
        valid_mask_threshold=args.dhr_valid_mask_threshold,
        return_valid_mask=True,
        return_displacement_field=False,
    )


def center_crop(array, size):
    height, width = array.shape[:2]
    x0 = (width - size) // 2
    y0 = (height - size) // 2
    return array[y0 : y0 + size, x0 : x0 + size]


def add_text_header(image, lines):
    image = np.asarray(image, dtype=np.uint8)
    header_h = 18 * len(lines) + 8
    canvas = np.full((image.shape[0] + header_h, image.shape[1], 3), 255, np.uint8)
    canvas[header_h:] = image
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()
    for i, line in enumerate(lines):
        draw.text((5, 4 + i * 18), str(line), fill=(0, 0, 0), font=font)
    return np.asarray(pil)


def registration_qc_image(he_patch, affine_patch, dhr_patch, sample, pair_id):
    overlay = cv2.addWeighted(he_patch, 0.5, dhr_patch, 0.5, 0)
    panel = np.concatenate([he_patch, affine_patch, dhr_patch, overlay], axis=1)
    return add_text_header(
        panel,
        [
            "%s  %s" % (pair_id, sample),
            "HE | affine IHC | DHR IHC | HE/DHR 50:50 overlay",
        ],
    )


def integrity_qc_image(level0_rgb, trusted_rgb, title, metrics):
    trusted_size = (512, 512)
    high = cv2.resize(level0_rgb, trusted_size, interpolation=cv2.INTER_AREA)
    trusted = cv2.resize(trusted_rgb, trusted_size, interpolation=cv2.INTER_AREA)
    panel = np.concatenate([high, trusted], axis=1)
    return add_text_header(
        panel,
        [
            title,
            "level0 downsample | trusted pyramid",
            "gray=%.3f edge=%.3f mask_iou=%.3f"
            % (metrics["gray_corr"], metrics["edge_corr"], metrics["mask_iou"]),
        ],
    )


def base_manifest_row(plan_row, args, attempt):
    row = {key: "" for key in PUBLIC_FIELDS}
    row.update(
        {
            "dataset_schema_version": SCHEMA_VERSION,
            "dataset_version": args.dataset_version,
            "dataset_variant": args.dataset_variant,
            "sample_id": plan_row["sample_id"],
            "pair_id": plan_row["pair_id"],
            "case_id": plan_row["case_id"],
            "patient_group_id": plan_row["patient_group_id"],
            "split": plan_row["split"],
            "cohort": plan_row["cohort"],
            "pair_type": plan_row["pair_type"],
            "stain": plan_row["stain"],
            "registration_version": args.registration_version,
            "registration_stage": "dhr",
            "component_id": plan_row["component_id"],
            "he_center_x_level0": plan_row["he_center_x_level0"],
            "he_center_y_level0": plan_row["he_center_y_level0"],
            "patch_size": args.patch_size,
            "context_size": args.context_size,
            "he_trusted_level": plan_row["he_trusted_level"],
            "ihc_trusted_level": plan_row["ihc_trusted_level"],
            "he_trusted_downsample": plan_row["he_trusted_downsample"],
            "ihc_trusted_downsample": plan_row["ihc_trusted_downsample"],
            "planned_tissue_fraction": plan_row["planned_tissue_fraction"],
            "status": "processing",
            "attempt": attempt,
            "updated_at": utc_now(),
        }
    )
    return row


def terminal_existing(row):
    return row and row.get("status") in {
        "ready",
        "roi_integrity_rejected",
        "low_tissue",
        "low_valid_fraction",
        "failed",
    }


def materialize(args):
    output_root = Path(args.output_root)
    plan_rows = utils.read_csv(output_root / "manifests" / "patch_plan.csv")
    if args.pair_id:
        plan_rows = [row for row in plan_rows if row["pair_id"] == args.pair_id]
    if args.max_samples is not None:
        plan_rows = plan_rows[: args.max_samples]

    inventory = read_inventory(args.inventory)
    manifest_path = output_root / "manifests" / "patch_manifest.csv"
    private_manifest_path = (
        output_root / "manifests" / "private" / "patch_manifest_internal.csv"
    )
    existing = utils.read_csv(manifest_path) if manifest_path.exists() else []
    public_by_id = {row["sample_id"]: row for row in existing}
    existing_private = (
        utils.read_csv(private_manifest_path)
        if private_manifest_path.exists()
        else []
    )
    private_by_id = {row["sample_id"]: row for row in existing_private}

    Aslide = utils.import_aslide(args.aslide_root)
    grouped = defaultdict(list)
    for row in plan_rows:
        if not args.force and terminal_existing(public_by_id.get(row["sample_id"])):
            continue
        grouped[row["pair_id"]].append(row)

    for pair_id, rows in grouped.items():
        info = inventory[pair_id]
        he_slide = Aslide(info["he_wsi_path"])
        ihc_slide = Aslide(info["ihc_wsi_path"])
        transform_cache = {}
        try:
            for plan_row in rows:
                sid = plan_row["sample_id"]
                prior = public_by_id.get(sid)
                attempt = int(prior.get("attempt", 0) or 0) + 1 if prior else 1
                public = base_manifest_row(plan_row, args, attempt)
                private = {
                    **public,
                    "he_wsi_path": info["he_wsi_path"],
                    "ihc_wsi_path": info["ihc_wsi_path"],
                    "transform_path": plan_row["transform_path"],
                    "ihc_source_bbox_level0": "",
                    "error_type": "",
                    "error_message": "",
                }
                try:
                    transform_path = plan_row["transform_path"]
                    if transform_path not in transform_cache:
                        transform_cache[transform_path] = load_json(transform_path)
                    transform = transform_cache[transform_path]
                    matrix = np.asarray(
                        transform["matrix_level0_HE_to_IHC_2x3"], dtype=np.float64
                    )
                    center_x = int(plan_row["he_center_x_level0"])
                    center_y = int(plan_row["he_center_y_level0"])
                    half_context = args.context_size // 2
                    he_bbox0 = [
                        center_x - half_context,
                        center_y - half_context,
                        center_x + half_context,
                        center_y + half_context,
                    ]

                    he_context, _ = registration_v1.read_level0_bbox_at_level(
                        he_slide, he_bbox0, 0
                    )
                    he_integrity, he_trusted = validate_level0_roi(
                        he_slide,
                        he_bbox0,
                        int(plan_row["he_trusted_level"]),
                        he_context,
                        args,
                    )
                    public["he_roi_gray_corr"] = he_integrity["gray_corr"]
                    public["he_roi_edge_corr"] = he_integrity["edge_corr"]
                    public["he_roi_mask_iou"] = he_integrity["mask_iou"]

                    if not he_integrity["passed"]:
                        reason = "he_roi_integrity"
                        public["status"] = "roi_integrity_rejected"
                        public["qc_rejection_reason"] = reason
                        qc_path = (
                            output_root
                            / "qc"
                            / "rejected"
                            / pair_id
                            / (sid + "_HE_integrity.png")
                        )
                        save_png_atomic(
                            qc_path,
                            integrity_qc_image(
                                he_context,
                                he_trusted,
                                "%s HE integrity rejected" % sid,
                                he_integrity,
                            ),
                        )
                        public["qc_path"] = str(qc_path)
                        private.update(public)
                        public_by_id[sid] = public
                        private_by_id[sid] = private
                        continue

                    source_bbox = affine_source_bbox(
                        matrix, he_bbox0, args.affine_margin
                    )
                    private["ihc_source_bbox_level0"] = json.dumps(source_bbox)
                    clipped_source_bbox = clip_bbox(source_bbox, ihc_slide.dimensions)
                    if (
                        clipped_source_bbox[2] <= clipped_source_bbox[0]
                        or clipped_source_bbox[3] <= clipped_source_bbox[1]
                    ):
                        raise RuntimeError("Affine source bbox outside IHC slide")

                    ihc_level0_for_qc, _ = registration_v1.read_level0_bbox_at_level(
                        ihc_slide, clipped_source_bbox, 0
                    )
                    ihc_integrity, ihc_trusted = validate_level0_roi(
                        ihc_slide,
                        clipped_source_bbox,
                        int(plan_row["ihc_trusted_level"]),
                        ihc_level0_for_qc,
                        args,
                    )
                    public["ihc_roi_gray_corr"] = ihc_integrity["gray_corr"]
                    public["ihc_roi_edge_corr"] = ihc_integrity["edge_corr"]
                    public["ihc_roi_mask_iou"] = ihc_integrity["mask_iou"]

                    if not ihc_integrity["passed"]:
                        reason = "ihc_roi_integrity"
                        public["status"] = "roi_integrity_rejected"
                        public["qc_rejection_reason"] = reason
                        qc_path = (
                            output_root
                            / "qc"
                            / "rejected"
                            / pair_id
                            / (sid + "_IHC_integrity.png")
                        )
                        save_png_atomic(
                            qc_path,
                            integrity_qc_image(
                                ihc_level0_for_qc,
                                ihc_trusted,
                                "%s IHC integrity rejected" % sid,
                                ihc_integrity,
                            ),
                        )
                        public["qc_path"] = str(qc_path)
                        private.update(public)
                        public_by_id[sid] = public
                        private_by_id[sid] = private
                        continue

                    ihc_source, source_valid = read_level0_padded_bbox(
                        ihc_slide, source_bbox
                    )
                    ihc_affine_context, ihc_affine_valid = warp_source_to_he(
                        ihc_source,
                        source_valid,
                        source_bbox,
                        matrix,
                        he_bbox0,
                    )

                    he_patch = center_crop(he_context, args.patch_size)
                    affine_patch = center_crop(
                        ihc_affine_context, args.patch_size
                    )
                    affine_valid_patch = center_crop(
                        ihc_affine_valid, args.patch_size
                    )
                    tissue_fraction = float(utils.tissue_mask(he_patch).mean())
                    affine_valid_fraction = float(
                        (affine_valid_patch > 0).mean()
                    )
                    public["he_tissue_fraction"] = tissue_fraction
                    public["affine_valid_fraction"] = affine_valid_fraction

                    if tissue_fraction < args.min_tissue_fraction:
                        public["status"] = "low_tissue"
                        public["qc_rejection_reason"] = "he_tissue_fraction"
                        private.update(public)
                        public_by_id[sid] = public
                        private_by_id[sid] = private
                        continue
                    if affine_valid_fraction < args.min_valid_fraction:
                        public["status"] = "low_valid_fraction"
                        public["valid_fraction"] = affine_valid_fraction
                        public["qc_rejection_reason"] = "affine_valid_fraction"
                        private.update(public)
                        public_by_id[sid] = public
                        private_by_id[sid] = private
                        continue

                    dhr_context, dhr_meta = run_dhr(
                        ihc_affine_context,
                        he_context,
                        ihc_affine_valid,
                        args,
                    )
                    dhr_patch = center_crop(dhr_context, args.patch_size)
                    dhr_valid_patch = center_crop(
                        dhr_meta["valid_mask"].astype(np.uint8), args.patch_size
                    )
                    valid_fraction = float((dhr_valid_patch > 0).mean())
                    public["valid_fraction"] = valid_fraction
                    qc = dhr_meta["deformation_qc"]
                    public["dhr_registration_time_seconds"] = dhr_meta[
                        "registration_time_seconds"
                    ]
                    public["dhr_displacement_p95_px"] = qc[
                        "displacement_p95_px"
                    ]
                    public["dhr_folding_fraction"] = qc["folding_fraction"]
                    public["dhr_nonpositive_jacobian_fraction"] = qc[
                        "nonpositive_jacobian_fraction"
                    ]

                    if valid_fraction < args.min_valid_fraction:
                        public["status"] = "low_valid_fraction"
                        public["qc_rejection_reason"] = "dhr_valid_fraction"
                        private.update(public)
                        public_by_id[sid] = public
                        private_by_id[sid] = private
                        continue

                    he_path = (
                        output_root / "images" / "he" / pair_id / (sid + ".png")
                    )
                    ihc_path = (
                        output_root / "images" / "ihc" / pair_id / (sid + ".png")
                    )
                    qc_path = (
                        output_root
                        / "qc"
                        / "registration"
                        / pair_id
                        / (sid + ".png")
                    )
                    save_png_atomic(he_path, he_patch)
                    save_png_atomic(ihc_path, dhr_patch)
                    save_png_atomic(
                        qc_path,
                        registration_qc_image(
                            he_patch, affine_patch, dhr_patch, sid, pair_id
                        ),
                    )
                    public["he_patch_path"] = str(he_path)
                    public["ihc_patch_path"] = str(ihc_path)
                    public["qc_path"] = str(qc_path)
                    public["status"] = "ready"
                    public["updated_at"] = utc_now()
                    private.update(public)
                    public_by_id[sid] = public
                    private_by_id[sid] = private
                    print(
                        "READY",
                        pair_id,
                        sid,
                        "tissue=%.3f" % tissue_fraction,
                        "HEcorr=%.3f" % float(he_integrity["gray_corr"]),
                        "IHCcorr=%.3f" % float(ihc_integrity["gray_corr"]),
                        flush=True,
                    )

                except Exception as exc:
                    public["status"] = "failed"
                    public["updated_at"] = utc_now()
                    private.update(public)
                    private["error_type"] = type(exc).__name__
                    private["error_message"] = "%s: %s" % (
                        type(exc).__name__,
                        str(exc),
                    )
                    public_by_id[sid] = public
                    private_by_id[sid] = private
                    print(
                        "FAILED",
                        pair_id,
                        sid,
                        type(exc).__name__,
                        file=sys.stderr,
                        flush=True,
                    )
                    if args.fail_fast:
                        traceback.print_exc()
                        raise

                finally:
                    ordered_public = sorted(
                        public_by_id.values(),
                        key=lambda row: (row["pair_id"], row["sample_id"]),
                    )
                    ordered_private = sorted(
                        private_by_id.values(),
                        key=lambda row: (row["pair_id"], row["sample_id"]),
                    )
                    utils.write_csv_atomic(
                        manifest_path, ordered_public, PUBLIC_FIELDS
                    )
                    utils.write_csv_atomic(
                        private_manifest_path, ordered_private, PRIVATE_FIELDS
                    )
        finally:
            utils.close_slide(he_slide)
            utils.close_slide(ihc_slide)

    return list(public_by_id.values())


def contact_sheet(paths, output, max_items):
    paths = list(paths)[:max_items]
    if not paths:
        return
    thumbs = []
    for path in paths:
        image = np.asarray(Image.open(path).convert("RGB"))
        width = 1200
        scale = width / float(image.shape[1])
        height = max(1, int(round(image.shape[0] * scale)))
        thumbs.append(
            cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        )
    rows = []
    for i in range(0, len(thumbs), 2):
        pair = thumbs[i : i + 2]
        if len(pair) == 1:
            pair.append(np.full_like(pair[0], 255))
        max_h = max(item.shape[0] for item in pair)
        padded = []
        for item in pair:
            if item.shape[0] < max_h:
                extra = np.full(
                    (max_h - item.shape[0], item.shape[1], 3), 255, np.uint8
                )
                item = np.concatenate([item, extra], axis=0)
            padded.append(item)
        rows.append(np.concatenate(padded, axis=1))
    canvas = np.concatenate(rows, axis=0)
    save_png_atomic(output, canvas)


def summarize(args):
    output_root = Path(args.output_root)
    manifest_path = output_root / "manifests" / "patch_manifest.csv"
    rows = utils.read_csv(manifest_path) if manifest_path.exists() else []
    summary_rows = []
    for pair_id in args.pair_ids:
        pair_rows = [row for row in rows if row["pair_id"] == pair_id]
        counts = Counter(row["status"] for row in pair_rows)
        summary_rows.append(
            {
                "pair_id": pair_id,
                "planned": sum(
                    1
                    for row in utils.read_csv(
                        output_root / "manifests" / "patch_plan.csv"
                    )
                    if row["pair_id"] == pair_id
                ),
                "ready": counts["ready"],
                "roi_integrity_rejected": counts["roi_integrity_rejected"],
                "low_tissue": counts["low_tissue"],
                "low_valid_fraction": counts["low_valid_fraction"],
                "failed": counts["failed"],
            }
        )
        ready_qc = [
            Path(row["qc_path"])
            for row in pair_rows
            if row["status"] == "ready" and row["qc_path"]
        ]
        contact_sheet(
            ready_qc,
            output_root / "qc" / "contact_sheets" / (pair_id + ".png"),
            args.qc_contact_sheet_max_per_pair,
        )

    utils.write_csv_atomic(
        output_root / "manifests" / "materialization_summary.csv",
        summary_rows,
        [
            "pair_id",
            "planned",
            "ready",
            "roi_integrity_rejected",
            "low_tissue",
            "low_valid_fraction",
            "failed",
        ],
    )
    print("SUMMARY", summary_rows)
    return summary_rows


def write_provenance(args):
    output_root = Path(args.output_root)
    repo_root = Path(__file__).resolve().parents[2]
    dhr_api = (
        Path(args.dhr_root)
        / "deeperhistreg"
        / "dhr_pipeline"
        / "in_memory.py"
    )
    payload = {
        "dataset_schema_version": SCHEMA_VERSION,
        "dataset_version": args.dataset_version,
        "dataset_variant": args.dataset_variant,
        "registration_version": args.registration_version,
        "pair_ids": args.pair_ids,
        "created_at": utc_now(),
        "pipeline_git_commit": utils.git_commit(repo_root),
        "pipeline_git_dirty": git_dirty(repo_root),
        "pipeline_script_sha256": utils.sha256_file(Path(__file__).resolve()),
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": utils.sha256_file(args.config),
        "registration_config_sha256": utils.sha256_file(
            repo_root / "configs" / "vsseg" / "registration_v1_component.json"
        ),
        "dhr_git_commit": utils.git_commit(args.dhr_root),
        "dhr_api_sha256": utils.sha256_file(dhr_api),
    }
    utils.write_json_atomic(
        output_root / "provenance" / "dataset_provenance.json", payload
    )


def build_parser():
    default = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "vsseg"
        / "patches_registration_v1_pilot.json"
    )
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(default))
    pre_args, _ = pre.parse_known_args()
    cfg = load_json(pre_args.config)

    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument(
        "--stage", choices=["plan", "materialize", "all"], default="all"
    )
    parser.add_argument("--pair-id", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")

    for key in [
        "dataset_version",
        "dataset_variant",
        "registration_version",
        "project_root",
        "registration_root",
        "output_root",
        "inventory",
        "aslide_root",
        "dhr_root",
        "dhr_preset",
        "dhr_tmp_root",
        "dhr_padding_mode",
        "image_format",
    ]:
        parser.add_argument("--" + key.replace("_", "-"), default=cfg[key])

    for key in [
        "patch_size",
        "context_size",
        "stride",
        "candidates_per_pair",
        "affine_margin",
        "qc_contact_sheet_max_per_pair",
    ]:
        parser.add_argument(
            "--" + key.replace("_", "-"), type=int, default=cfg[key]
        )

    for key in [
        "min_tissue_fraction",
        "min_valid_fraction",
        "roi_integrity_min_gray_corr",
        "roi_integrity_min_edge_corr",
        "roi_integrity_min_mask_iou",
        "dhr_valid_mask_threshold",
    ]:
        parser.add_argument(
            "--" + key.replace("_", "-"), type=float, default=cfg[key]
        )

    parser.set_defaults(
        pair_ids=cfg["pair_ids"],
        dhr_overrides=cfg["dhr_overrides"],
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.context_size < args.patch_size:
        raise ValueError("context_size must be >= patch_size")
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    write_provenance(args)

    if args.stage in {"plan", "all"}:
        plan(args)
    if args.stage in {"materialize", "all"}:
        materialize(args)
        summarize(args)


if __name__ == "__main__":
    main()
