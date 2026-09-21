#!/usr/bin/env python3
"""Landmark-free component-aware registration prototype for VS-Seg.

Manual landmarks are never used to estimate transforms in this script. The
pipeline operates on low-resolution WSI tissue structure:

  tissue mask -> connected components -> component matching ->
  SIFT descriptor matching -> RANSAC full affine -> optional local DHR smoke.

The output is a new derived registration version. Source WSI and annotations
remain read-only.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCHEMA_VERSION = 1


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False,
        prefix=path.name + ".", suffix=".tmp"
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(str(tmp), str(path))


def write_csv_atomic(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", dir=str(path.parent),
        delete=False, prefix=path.name + ".", suffix=".tmp"
    ) as handle:
        tmp = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(tmp), str(path))


def git_commit(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_aslide(aslide_root):
    parent = str(Path(aslide_root).parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from aslide.aslide import Aslide
    return Aslide


def import_dhr(dhr_root):
    root = str(Path(dhr_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    from deeperhistreg.dhr_pipeline.in_memory import register_and_warp_arrays
    return register_and_warp_arrays


def close_slide(slide):
    try:
        slide.close()
    except Exception:
        pass


def rgb_array(image):
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    return np.array(array[..., :3], dtype=np.uint8, copy=True, order="C")


def choose_thumbnail(slide, max_side):
    dimensions = [(int(w), int(h)) for w, h in slide.level_dimensions]
    level = len(dimensions) - 1
    for index, (width, height) in enumerate(dimensions):
        if max(width, height) <= max_side:
            level = index
            break
    width, height = dimensions[level]
    image = rgb_array(slide.read_region((0, 0), level, (width, height)))
    if max(width, height) > max_side:
        scale = float(max_side) / max(width, height)
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    scale_x = image.shape[1] / float(slide.dimensions[0])
    scale_y = image.shape[0] / float(slide.dimensions[1])
    return image, scale_x, scale_y


def tissue_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = ((hsv[..., 1] > 12) | (gray < 238)).astype(np.uint8)
    size = max(rgb.shape[:2])
    open_k = max(3, int(round(size / 700.0)))
    close_k = max(5, int(round(size / 220.0)))
    if open_k % 2 == 0:
        open_k += 1
    if close_k % 2 == 0:
        close_k += 1
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8),
        iterations=2,
    )
    return mask.astype(np.uint8)


def extract_components(mask, min_canvas_fraction, min_tissue_fraction):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    canvas_area = float(mask.shape[0] * mask.shape[1])
    tissue_area = float(max(1, mask.sum()))
    components = []
    for label in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area / canvas_area < min_canvas_fraction:
            continue
        if area / tissue_area < min_tissue_fraction:
            continue
        component_mask = (labels == label).astype(np.uint8)
        cx, cy = map(float, centroids[label])
        components.append({
            "id": len(components),
            "label": label,
            "area": area,
            "area_fraction_canvas": area / canvas_area,
            "area_fraction_tissue": area / tissue_area,
            "bbox": [x, y, w, h],
            "centroid": [cx, cy],
            "aspect": float(w) / max(1.0, float(h)),
            "mask": component_mask,
        })
    components.sort(key=lambda item: item["area"], reverse=True)
    for index, component in enumerate(components):
        component["id"] = index
    return components


def rectangle_gap(a, b):
    ax, ay, aw, ah = a["bbox"]
    bx, by, bw, bh = b["bbox"]
    dx = max(0, ax - (bx + bw), bx - (ax + aw))
    dy = max(0, ay - (by + bh), by - (ay + ah))
    return math.hypot(dx, dy)


def merge_nearby_components(components, image_shape, max_gap_fraction):
    if len(components) <= 1:
        return components
    max_gap = max(image_shape[:2]) * float(max_gap_fraction)
    parent = list(range(len(components)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if rectangle_gap(components[i], components[j]) <= max_gap:
                union(i, j)

    groups = {}
    for i in range(len(components)):
        groups.setdefault(find(i), []).append(components[i])

    canvas_area = float(image_shape[0] * image_shape[1])
    total_area = float(sum(component["area"] for component in components))
    merged = []
    for group in groups.values():
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        source_ids = []
        for component in group:
            mask |= component["mask"]
            source_ids.append(component["id"])
        ys, xs = np.where(mask > 0)
        area = int(mask.sum())
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        merged.append({
            "id": len(merged),
            "source_component_ids": sorted(source_ids),
            "area": area,
            "area_fraction_canvas": area / canvas_area,
            "area_fraction_tissue": area / max(1.0, total_area),
            "bbox": [x0, y0, x1 - x0, y1 - y0],
            "centroid": [float(xs.mean()), float(ys.mean())],
            "aspect": float(x1 - x0) / max(1.0, float(y1 - y0)),
            "mask": mask,
        })
    merged.sort(key=lambda item: item["area"], reverse=True)
    for index, component in enumerate(merged):
        component["id"] = index
    return merged


def component_cost(he, ihc, he_shape, ihc_shape):
    he_h, he_w = he_shape[:2]
    ihc_h, ihc_w = ihc_shape[:2]
    he_cx = he["centroid"][0] / he_w
    he_cy = he["centroid"][1] / he_h
    ihc_cx = ihc["centroid"][0] / ihc_w
    ihc_cy = ihc["centroid"][1] / ihc_h
    centroid_distance = math.hypot(he_cx - ihc_cx, he_cy - ihc_cy)
    area_term = abs(math.log((he["area_fraction_tissue"] + 1e-6) /
                             (ihc["area_fraction_tissue"] + 1e-6)))
    aspect_term = abs(math.log((he["aspect"] + 1e-6) / (ihc["aspect"] + 1e-6)))
    return 1.6 * centroid_distance + 0.45 * area_term + 0.20 * aspect_term


def match_components(he_components, ihc_components, he_shape, ihc_shape, max_cost):
    if not he_components or not ihc_components:
        return [], list(range(len(he_components))), list(range(len(ihc_components)))
    candidates = []
    for hi, he in enumerate(he_components):
        for ii, ihc in enumerate(ihc_components):
            candidates.append((component_cost(he, ihc, he_shape, ihc_shape), hi, ii))
    candidates.sort()
    used_he = set()
    used_ihc = set()
    matches = []
    for cost, hi, ii in candidates:
        if cost > max_cost or hi in used_he or ii in used_ihc:
            continue
        used_he.add(hi)
        used_ihc.add(ii)
        matches.append((hi, ii, float(cost)))
    return (
        matches,
        [i for i in range(len(he_components)) if i not in used_he],
        [i for i in range(len(ihc_components)) if i not in used_ihc],
    )


def clahe_gray(rgb, invert=False):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if invert:
        gray = 255 - gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def bbox_initial_affine(he_component, ihc_component):
    hx, hy, hw, hh = he_component["bbox"]
    ix, iy, iw, ih = ihc_component["bbox"]
    sx = float(iw) / max(1.0, float(hw))
    sy = float(ih) / max(1.0, float(hh))
    return np.asarray([
        [sx, 0.0, ix - sx * hx],
        [0.0, sy, iy - sy * hy],
    ], dtype=np.float64)


def apply_affine(matrix, xy):
    xy = np.asarray(xy, dtype=np.float64)
    return np.column_stack([xy, np.ones(len(xy))]) @ matrix.T


def sift_matches(he_rgb, ihc_rgb, he_component, ihc_component, args):
    sift = cv2.SIFT_create(
        nfeatures=args.sift_nfeatures,
        contrastThreshold=0.01,
        edgeThreshold=15,
    )
    he_mask = (he_component["mask"] * 255).astype(np.uint8)
    ihc_mask = (ihc_component["mask"] * 255).astype(np.uint8)
    he_rep = clahe_gray(he_rgb, invert=False)
    kp_he, desc_he = sift.detectAndCompute(he_rep, he_mask)
    if desc_he is None or len(kp_he) < 3:
        return None

    initial = bbox_initial_affine(he_component, ihc_component)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    best = None
    for inverted in (False, True):
        ihc_rep = clahe_gray(ihc_rgb, invert=inverted)
        kp_ihc, desc_ihc = sift.detectAndCompute(ihc_rep, ihc_mask)
        if desc_ihc is None or len(kp_ihc) < 3:
            continue
        knn = matcher.knnMatch(desc_he, desc_ihc, k=2)
        ratio_matches = [m for m, n in knn if m.distance < args.sift_ratio * n.distance]
        if len(ratio_matches) < 3:
            continue

        he_xy = np.asarray([kp_he[m.queryIdx].pt for m in ratio_matches], dtype=np.float64)
        ihc_xy = np.asarray([kp_ihc[m.trainIdx].pt for m in ratio_matches], dtype=np.float64)
        predicted = apply_affine(initial, he_xy)
        _, _, iw, ih = ihc_component["bbox"]
        gate = args.feature_gate_fraction * math.hypot(iw, ih)
        geometric = np.linalg.norm(predicted - ihc_xy, axis=1) <= gate
        selected = [m for m, keep in zip(ratio_matches, geometric) if keep]
        he_selected = he_xy[geometric]
        ihc_selected = ihc_xy[geometric]
        if len(selected) < 3:
            continue

        matrix, inlier_mask = cv2.estimateAffine2D(
            he_selected.astype(np.float32),
            ihc_selected.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=args.ransac_threshold_thumbnail_px,
            maxIters=10000,
            confidence=0.999,
            refineIters=50,
        )
        if matrix is None:
            continue
        inliers = inlier_mask.ravel().astype(bool)
        predicted_final = apply_affine(matrix, he_selected)
        residuals = np.linalg.norm(predicted_final - ihc_selected, axis=1)
        inlier_residuals = residuals[inliers]
        inlier_count = int(inliers.sum())
        if inlier_count < 3:
            continue
        hull = cv2.convexHull(he_selected[inliers].astype(np.float32))
        hull_area = float(cv2.contourArea(hull)) if len(hull) >= 3 else 0.0
        spatial_coverage = min(
            1.0, hull_area / max(1.0, float(he_component["area"]))
        )
        result = {
            "matrix": matrix.astype(np.float64),
            "inverted_ihc_representation": inverted,
            "he_keypoints": len(kp_he),
            "ihc_keypoints": len(kp_ihc),
            "ratio_matches": len(ratio_matches),
            "geometric_matches": len(selected),
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_count / float(len(selected)),
            "inlier_median_residual_px": float(np.median(inlier_residuals)),
            "inlier_p95_residual_px": float(np.percentile(inlier_residuals, 95)),
            "spatial_coverage": float(spatial_coverage),
            "kp_he": kp_he,
            "kp_ihc": kp_ihc,
            "matches": selected,
            "he_selected": he_selected,
            "ihc_selected": ihc_selected,
            "inliers": inliers,
        }
        score = (
            result["inlier_count"],
            result["spatial_coverage"],
            -result["inlier_median_residual_px"],
        )
        if best is None or score > best[0]:
            best = (score, result)
    return None if best is None else best[1]


def confidence_level(result):
    if result is None:
        return "low"
    if (
        result["inlier_count"] >= 12
        and result["inlier_ratio"] >= 0.25
        and result["inlier_median_residual_px"] <= 8.0
        and result["spatial_coverage"] >= 0.03
    ):
        return "high"
    if (
        result["inlier_count"] >= 6
        and result["inlier_ratio"] >= 0.15
        and result["inlier_median_residual_px"] <= 15.0
    ):
        return "medium"
    if (
        result["inlier_count"] >= 5
        and result["inlier_ratio"] >= 0.40
        and result["inlier_median_residual_px"] <= 4.0
        and result["inlier_p95_residual_px"] <= 8.0
        and result["spatial_coverage"] >= 0.04
    ):
        return "medium"
    return "low"


def thumbnail_to_level0_affine(matrix, he_sx, he_sy, ihc_sx, ihc_sy):
    M = np.vstack([matrix, [0.0, 0.0, 1.0]])
    he_scale = np.diag([he_sx, he_sy, 1.0])
    ihc_inverse_scale = np.diag([1.0 / ihc_sx, 1.0 / ihc_sy, 1.0])
    out = ihc_inverse_scale @ M @ he_scale
    return out[:2]


def draw_components(rgb, components, title, matches=None):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = [
        (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
        (255, 127, 0), (166, 86, 40), (247, 129, 191), (80, 80, 80),
    ]
    draw.rectangle((0, 0, 250, 24), fill=(255, 255, 255))
    draw.text((6, 6), title, fill=(0, 0, 0), font=font)
    for component in components:
        x, y, w, h = component["bbox"]
        color = colors[component["id"] % len(colors)]
        draw.rectangle((x, y, x+w, y+h), outline=color, width=4)
        draw.text((x+4, y+4), "C%d" % component["id"], fill=color, font=font)
    return np.asarray(image)


def hstack(images, gap=12):
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images) + gap * (len(images)-1)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x = 0
    for image in images:
        canvas[:image.shape[0], x:x+image.shape[1]] = image
        x += image.shape[1] + gap
    return canvas


def add_header(rgb, lines):
    font = ImageFont.load_default()
    header = 18 * len(lines) + 12
    canvas = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + header), (255,255,255))
    canvas.paste(Image.fromarray(rgb), (0, header))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        draw.text((6, 5 + 18*index), line, fill=(0,0,0), font=font)
    return np.asarray(canvas)


def feature_visual(he_rgb, ihc_rgb, result, component_id):
    inlier_matches = [m for m, keep in zip(result["matches"], result["inliers"]) if keep]
    output = cv2.drawMatches(
        cv2.cvtColor(he_rgb, cv2.COLOR_RGB2BGR),
        result["kp_he"],
        cv2.cvtColor(ihc_rgb, cv2.COLOR_RGB2BGR),
        result["kp_ihc"],
        inlier_matches[:120],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return add_header(output, [
        "component %d automatic SIFT/RANSAC inliers" % component_id,
        "inliers=%d ratio=%.3f median_res=%.2f thumb-px coverage=%.3f" % (
            result["inlier_count"], result["inlier_ratio"],
            result["inlier_median_residual_px"], result["spatial_coverage"],
        ),
    ])


def affine_overlay(he_rgb, ihc_rgb, matrix, he_component, component_id):
    h, w = he_rgb.shape[:2]
    warped = cv2.warpAffine(
        ihc_rgb, matrix, (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255,255,255),
    )
    overlay = cv2.addWeighted(he_rgb, 0.5, warped, 0.5, 0)
    x, y, bw, bh = he_component["bbox"]
    pad = max(20, int(round(max(bw, bh) * 0.05)))
    x0=max(0,x-pad); y0=max(0,y-pad); x1=min(w,x+bw+pad); y1=min(h,y+bh+pad)
    panel = hstack([he_rgb[y0:y1,x0:x1], warped[y0:y1,x0:x1], overlay[y0:y1,x0:x1]])
    return add_header(panel, [
        "component %d: HE | auto-affine IHC | 50/50 overlay" % component_id,
    ])


def read_padded(slide, x0, y0, width, height):
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    valid = np.zeros((height, width), dtype=np.uint8)
    sw, sh = map(int, slide.dimensions)
    sx0=max(0,x0); sy0=max(0,y0); sx1=min(sw,x0+width); sy1=min(sh,y0+height)
    if sx1 <= sx0 or sy1 <= sy0:
        return canvas, valid
    tile = rgb_array(slide.read_region((sx0,sy0),0,(sx1-sx0,sy1-sy0)))
    dx=sx0-x0; dy=sy0-y0
    canvas[dy:dy+tile.shape[0],dx:dx+tile.shape[1]]=tile
    valid[dy:dy+tile.shape[0],dx:dx+tile.shape[1]]=1
    return canvas, valid


def warp_ihc_affine(slide, matrix, he_x0, he_y0, size, margin):
    corners=np.asarray([[he_x0,he_y0],[he_x0+size-1,he_y0],[he_x0,he_y0+size-1],[he_x0+size-1,he_y0+size-1]],dtype=np.float64)
    mapped=apply_affine(matrix,corners)
    sx0=int(math.floor(mapped[:,0].min()))-margin
    sy0=int(math.floor(mapped[:,1].min()))-margin
    sx1=int(math.ceil(mapped[:,0].max()))+margin+1
    sy1=int(math.ceil(mapped[:,1].max()))+margin+1
    source, source_valid=read_padded(slide,sx0,sy0,sx1-sx0,sy1-sy0)
    xg,yg=np.meshgrid(np.arange(size),np.arange(size))
    gx=xg.astype(np.float64)+he_x0; gy=yg.astype(np.float64)+he_y0
    mx=(matrix[0,0]*gx+matrix[0,1]*gy+matrix[0,2]-sx0).astype(np.float32)
    my=(matrix[1,0]*gx+matrix[1,1]*gy+matrix[1,2]-sy0).astype(np.float32)
    warped=cv2.remap(source,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    valid=cv2.remap(source_valid,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return warped,valid


def component_anchor(component):
    mask=component["mask"].astype(np.uint8)
    distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    y,x=np.unravel_index(np.argmax(distance),distance.shape)
    return float(x),float(y)


def center_crop(array,size):
    h,w=array.shape[:2]
    x=(w-size)//2; y=(h-size)//2
    return array[y:y+size,x:x+size]


def dhr_smoke(he_slide, ihc_slide, he_component, matrix_level0, he_sx, he_sy, args):
    anchor_x_thumb, anchor_y_thumb=component_anchor(he_component)
    center_x=int(round(anchor_x_thumb/he_sx)); center_y=int(round(anchor_y_thumb/he_sy))
    context=args.context_size; patch=args.patch_size
    x0=int(round(center_x-context/2)); y0=int(round(center_y-context/2))
    he_context,he_valid=read_padded(he_slide,x0,y0,context,context)
    ihc_affine,ihc_valid=warp_ihc_affine(ihc_slide,matrix_level0,x0,y0,context,args.affine_margin)
    register_and_warp_arrays=import_dhr(args.dhr_root)
    warped,meta=register_and_warp_arrays(
        ihc_affine, he_context,
        preset=args.dhr_preset,
        device=args.dhr_device,
        overrides=args.dhr_overrides,
        source_valid_mask=ihc_valid,
        temporary_root=str(Path(args.output_root)/"tmp"),
        return_valid_mask=True,
    )
    he_patch=center_crop(he_context,patch)
    affine_patch=center_crop(ihc_affine,patch)
    dhr_patch=center_crop(warped,patch)
    valid=center_crop(meta["valid_mask"].astype(np.uint8),patch)
    overlay=cv2.addWeighted(he_patch,0.5,dhr_patch,0.5,0)
    montage=hstack([he_patch,affine_patch,dhr_patch,overlay],gap=8)
    montage=add_header(montage,[
        "HE | auto-affine IHC | DHR IHC | HE/DHR overlay",
        "center=(%d,%d) valid=%.4f fold=%.6f disp_p95=%.2f px" % (
            center_x,center_y,float(valid.mean()),
            meta["deformation_qc"]["folding_fraction"],
            meta["deformation_qc"]["displacement_p95_px"],
        ),
    ])
    return montage, {
        "he_center_level0": [center_x,center_y],
        "final_patch_valid_fraction": float(valid.mean()),
        "dhr_registration_time_seconds": meta["registration_time_seconds"],
        "dhr_working_displacement_shape": meta["working_displacement_shape"],
        "dhr_qc": meta["deformation_qc"],
    }


def component_public(component):
    return {key:value for key,value in component.items() if key != "mask"}


def process_pair(row,args,Aslide):
    pair_id=row["pair_id"]
    report_dir=Path(args.report_root)/pair_id
    transform_dir=Path(args.output_root)/"transforms"/pair_id
    report_dir.mkdir(parents=True,exist_ok=True)
    transform_dir.mkdir(parents=True,exist_ok=True)
    he_slide=Aslide(row["he_wsi_path"]); ihc_slide=Aslide(row["ihc_wsi_path"])
    try:
        he_rgb,he_sx,he_sy=choose_thumbnail(he_slide,args.thumbnail_max_side)
        ihc_rgb,ihc_sx,ihc_sy=choose_thumbnail(ihc_slide,args.thumbnail_max_side)
        he_mask=tissue_mask(he_rgb); ihc_mask=tissue_mask(ihc_rgb)
        he_components=extract_components(he_mask,args.min_component_canvas_fraction,args.min_component_tissue_fraction)
        ihc_components=extract_components(ihc_mask,args.min_component_canvas_fraction,args.min_component_tissue_fraction)
        he_components=merge_nearby_components(
            he_components, he_rgb.shape, args.component_merge_gap_fraction
        )
        ihc_components=merge_nearby_components(
            ihc_components, ihc_rgb.shape, args.component_merge_gap_fraction
        )
        matches,unmatched_he,unmatched_ihc=match_components(
            he_components,ihc_components,he_rgb.shape,ihc_rgb.shape,args.component_match_max_cost
        )
        overview=hstack([
            draw_components(he_rgb,he_components,"HE components"),
            draw_components(ihc_rgb,ihc_components,"IHC components"),
        ])
        overview=add_header(overview,[
            pair_id,
            "HE components=%d IHC components=%d matched=%d" % (len(he_components),len(ihc_components),len(matches)),
        ])
        Image.fromarray(overview).save(report_dir/"01_component_overview.png")

        pair_summary={
            "schema_version":SCHEMA_VERSION,
            "registration_version":args.registration_version,
            "pair_id":pair_id,
            "manual_landmarks_used_for_transform":False,
            "he_thumbnail_scale_xy":[he_sx,he_sy],
            "ihc_thumbnail_scale_xy":[ihc_sx,ihc_sy],
            "he_components":[component_public(c) for c in he_components],
            "ihc_components":[component_public(c) for c in ihc_components],
            "unmatched_he_components":unmatched_he,
            "unmatched_ihc_components":unmatched_ihc,
            "components":[],
        }
        summary_rows=[]
        for component_index,(hi,ii,cost) in enumerate(matches):
            he_component=he_components[hi]; ihc_component=ihc_components[ii]
            result=sift_matches(he_rgb,ihc_rgb,he_component,ihc_component,args)
            confidence=confidence_level(result)
            if result is None:
                matrix_thumb=bbox_initial_affine(he_component,ihc_component)
                method="bbox_fallback"
                metrics={
                    "he_keypoints":0,"ihc_keypoints":0,"ratio_matches":0,
                    "geometric_matches":0,"inlier_count":0,"inlier_ratio":0.0,
                    "inlier_median_residual_px":None,"inlier_p95_residual_px":None,
                    "spatial_coverage":0.0,"inverted_ihc_representation":False,
                }
            else:
                matrix_thumb=result["matrix"]
                method="sift_ransac_affine"
                metrics={key:result[key] for key in [
                    "he_keypoints","ihc_keypoints","ratio_matches","geometric_matches",
                    "inlier_count","inlier_ratio","inlier_median_residual_px",
                    "inlier_p95_residual_px","spatial_coverage","inverted_ihc_representation",
                ]}
                Image.fromarray(feature_visual(he_rgb,ihc_rgb,result,component_index)).save(
                    report_dir/("02_component_%02d_feature_matches.png"%component_index)
                )
            matrix_level0=thumbnail_to_level0_affine(matrix_thumb,he_sx,he_sy,ihc_sx,ihc_sy)
            Image.fromarray(affine_overlay(he_rgb,ihc_rgb,matrix_thumb,he_component,component_index)).save(
                report_dir/("03_component_%02d_affine_overlay.png"%component_index)
            )
            dhr_info=None
            if args.run_dhr and confidence in {"high","medium"}:
                montage,dhr_info=dhr_smoke(
                    he_slide,ihc_slide,he_component,matrix_level0,he_sx,he_sy,args
                )
                Image.fromarray(montage).save(report_dir/("04_component_%02d_dhr_smoke.png"%component_index))
            payload={
                "registration_version":args.registration_version,
                "pair_id":pair_id,
                "component_id":component_index,
                "he_component_id":hi,
                "ihc_component_id":ii,
                "component_match_cost":cost,
                "coarse_method":method,
                "confidence":confidence,
                "manual_landmarks_used_for_transform":False,
                "matrix_thumbnail_HE_to_IHC_2x3":matrix_thumb.tolist(),
                "matrix_level0_HE_to_IHC_2x3":matrix_level0.tolist(),
                "feature_qc":metrics,
                "dhr_smoke":dhr_info,
            }
            write_json_atomic(transform_dir/("component_%02d.json"%component_index),payload)
            pair_summary["components"].append(payload)
            summary_rows.append({
                "pair_id":pair_id,"component_id":component_index,
                "he_component_id":hi,"ihc_component_id":ii,
                "component_match_cost":cost,"coarse_method":method,
                "confidence":confidence,"inlier_count":metrics["inlier_count"],
                "inlier_ratio":metrics["inlier_ratio"],
                "inlier_median_residual_px":metrics["inlier_median_residual_px"],
                "spatial_coverage":metrics["spatial_coverage"],
                "dhr_smoke_status":"run" if dhr_info else "not_run",
                "dhr_valid_fraction":"" if not dhr_info else dhr_info["final_patch_valid_fraction"],
                "dhr_folding_fraction":"" if not dhr_info else dhr_info["dhr_qc"]["folding_fraction"],
                "dhr_displacement_p95_px":"" if not dhr_info else dhr_info["dhr_qc"]["displacement_p95_px"],
            })
        write_json_atomic(report_dir/"review_summary.json",pair_summary)
        write_csv_atomic(
            report_dir/"component_summary.csv",summary_rows,
            list(summary_rows[0].keys()) if summary_rows else ["pair_id"]
        )
        return pair_summary,summary_rows
    finally:
        close_slide(he_slide); close_slide(ihc_slide)


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parser():
    default=Path(__file__).resolve().parents[2]/"configs"/"vsseg"/"registration_v1_auto_component.json"
    pre=argparse.ArgumentParser(add_help=False); pre.add_argument("--config",default=str(default))
    pre_args,_=pre.parse_known_args(); config=load_config(pre_args.config)
    p=argparse.ArgumentParser(parents=[pre])
    for key in ["registration_version","inventory","output_root","report_root","aslide_root","dhr_root"]:
        p.add_argument("--"+key.replace("_","-"),default=config[key])
    p.add_argument("--pair-id",action="append",required=True)
    p.add_argument("--thumbnail-max-side",type=int,default=config["thumbnail_max_side"])
    p.add_argument("--min-component-canvas-fraction",type=float,default=config["min_component_canvas_fraction"])
    p.add_argument("--min-component-tissue-fraction",type=float,default=config["min_component_tissue_fraction"])
    p.add_argument("--component-match-max-cost",type=float,default=config["component_match_max_cost"])
    p.add_argument("--component-merge-gap-fraction",type=float,default=config["component_merge_gap_fraction"])
    p.add_argument("--sift-nfeatures",type=int,default=config["sift_nfeatures"])
    p.add_argument("--sift-ratio",type=float,default=config["sift_ratio"])
    p.add_argument("--feature-gate-fraction",type=float,default=config["feature_gate_fraction"])
    p.add_argument("--ransac-threshold-thumbnail-px",type=float,default=config["ransac_threshold_thumbnail_px"])
    p.add_argument("--context-size",type=int,default=config["context_size"])
    p.add_argument("--patch-size",type=int,default=config["patch_size"])
    p.add_argument("--affine-margin",type=int,default=config["affine_margin"])
    p.add_argument("--dhr-preset",default=config["dhr_preset"])
    p.add_argument("--dhr-device",default=config["dhr_device"])
    p.add_argument("--run-dhr",action=argparse.BooleanOptionalAction,default=True)
    p.set_defaults(dhr_overrides=config["dhr_overrides"])
    return p


def main():
    args=parser().parse_args()
    inventory={row["pair_id"]:row for row in read_csv(args.inventory)}
    Aslide=import_aslide(args.aslide_root)
    all_rows=[]
    pair_results=[]
    for pair_id in args.pair_id:
        if pair_id not in inventory:
            raise KeyError("Unknown pair_id: %s"%pair_id)
        print("PROCESS",pair_id,flush=True)
        pair_summary,rows=process_pair(inventory[pair_id],args,Aslide)
        pair_results.append(pair_summary); all_rows.extend(rows)
        for row in rows:
            print(" COMPONENT",row["component_id"],row["confidence"],"inliers",row["inlier_count"],"coverage",row["spatial_coverage"],"DHR",row["dhr_smoke_status"],flush=True)
    Path(args.output_root).mkdir(parents=True,exist_ok=True)
    fields=list(all_rows[0].keys()) if all_rows else ["pair_id"]
    write_csv_atomic(Path(args.output_root)/"auto_component_summary.csv",all_rows,fields)
    provenance={
        "registration_version":args.registration_version,
        "pipeline_git_commit":git_commit(Path(__file__).resolve().parents[2]),
        "dhr_git_commit":git_commit(args.dhr_root),
        "script_sha256":sha256_file(Path(__file__).resolve()),
        "config_sha256":sha256_file(args.config),
        "pair_ids":args.pair_id,
    }
    write_json_atomic(Path(args.output_root)/"provenance.json",provenance)
    print("SUMMARY",dict(Counter(row["confidence"] for row in all_rows)),flush=True)


if __name__=="__main__":
    main()
