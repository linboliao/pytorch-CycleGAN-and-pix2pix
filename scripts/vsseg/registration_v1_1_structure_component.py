#!/usr/bin/env python3
"""Structure-first, landmark-free component registration prototype.

This is a successor to registration_v1_auto_component. Sparse SIFT matches are
kept only as one candidate source. A candidate must survive tissue-shape
refinement and multimodal structural scoring before it can reach DHR.
"""

import argparse
import contextlib
import io
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from PIL import Image

import registration_v1_auto_component as base


SCHEMA_VERSION = 1


def choose_registration_view(slide, max_side):
    """Read the closest pyramid level at/above target size, then downsample."""
    dims = [(int(w), int(h)) for w, h in slide.level_dimensions]
    above = [i for i, (w, h) in enumerate(dims) if max(w, h) >= max_side]
    if above:
        level = min(above, key=lambda i: max(dims[i]))
    else:
        level = max(range(len(dims)), key=lambda i: max(dims[i]))
    width, height = dims[level]
    image = base.rgb_array(slide.read_region((0, 0), level, (width, height)))
    if max(width, height) > max_side:
        scale = float(max_side) / max(width, height)
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    sx = image.shape[1] / float(slide.dimensions[0])
    sy = image.shape[0] / float(slide.dimensions[1])
    return image, sx, sy


def component_to_view(component, source_shape, target_shape):
    mask = cv2.resize(
        component["mask"].astype(np.uint8),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Component vanished after view resampling")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    area = int(mask.sum())
    return {
        "id": component["id"],
        "source_component_ids": component.get("source_component_ids", [component["id"]]),
        "area": area,
        "area_fraction_canvas": area / float(target_shape[0] * target_shape[1]),
        "area_fraction_tissue": component["area_fraction_tissue"],
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "centroid": [float(xs.mean()), float(ys.mean())],
        "aspect": float(x1 - x0) / max(1.0, float(y1 - y0)),
        "mask": mask,
    }


def warp_roi_to_he(image, matrix_he_to_ihc, roi, interpolation, border_value):
    x0, y0, x1, y1 = roi
    translate = np.asarray(
        [[1.0, 0.0, float(x0)], [0.0, 1.0, float(y0)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    local_matrix = (homogeneous(matrix_he_to_ihc) @ translate)[:2]
    return cv2.warpAffine(
        image,
        local_matrix.astype(np.float32),
        (x1 - x0, y1 - y0),
        flags=interpolation | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def homogeneous(matrix):
    return np.vstack([np.asarray(matrix, dtype=np.float64), [0.0, 0.0, 1.0]])


def compose(first, second):
    """Compose HE->intermediate first with intermediate->IHC second convention.

    Matrices here map fixed/output HE coordinates to moving/input IHC coordinates.
    Applying residual R in HE coordinates then base C gives C @ R.
    """
    return (homogeneous(first) @ homogeneous(second))[:2]


def component_bbox_level0(component, scale_x, scale_y, slide_dimensions, margin_fraction):
    x, y, w, h = component["bbox"]
    x0 = x / scale_x
    y0 = y / scale_y
    x1 = (x + w) / scale_x
    y1 = (y + h) / scale_y
    pad = max(x1 - x0, y1 - y0) * float(margin_fraction)
    x0 = max(0.0, x0 - pad)
    y0 = max(0.0, y0 - pad)
    x1 = min(float(slide_dimensions[0]), x1 + pad)
    y1 = min(float(slide_dimensions[1]), y1 + pad)
    return [int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))]


def slide_uses_level_coordinates(slide):
    backend = getattr(slide, "_osr", None)
    module = type(backend).__module__.lower()
    name = type(backend).__name__.lower()
    return "kfb" in module or "kfb" in name


def backend_location_from_level0(slide, location_level0, level):
    x0, y0 = map(float, location_level0)
    if slide_uses_level_coordinates(slide) and int(level) > 0:
        downsample = float(slide.level_downsamples[level])
        return (
            int(round(x0 / downsample)),
            int(round(y0 / downsample)),
        )
    return int(round(x0)), int(round(y0))


def fixed_tile_location(slide, location_level0, level, x_level, y_level):
    base_location = backend_location_from_level0(slide, location_level0, level)
    if slide_uses_level_coordinates(slide):
        return (
            int(base_location[0] + x_level),
            int(base_location[1] + y_level),
        )
    downsample = float(slide.level_downsamples[level])
    return (
        int(round(base_location[0] + x_level * downsample)),
        int(round(base_location[1] + y_level * downsample)),
    )


def read_region_resilient(slide, location_level0, level, size_level):
    backend_location = backend_location_from_level0(
        slide, location_level0, level
    )
    try:
        return base.rgb_array(
            slide.read_region(backend_location, level, size_level)
        )
    except Exception as roi_error:
        if not hasattr(slide, "read_fixed_region"):
            raise
        width, height = map(int, size_level)
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        tile_size = 256
        try:
            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    loc = fixed_tile_location(
                        slide, location_level0, level, x, y
                    )
                    with contextlib.redirect_stdout(io.StringIO()):
                        tile = base.rgb_array(
                            slide.read_fixed_region(
                                loc, level, (tile_size, tile_size)
                            )
                        )
                    copy_h = min(tile.shape[0], height - y)
                    copy_w = min(tile.shape[1], width - x)
                    canvas[y:y+copy_h, x:x+copy_w] = tile[:copy_h, :copy_w]
            return canvas
        except Exception:
            raise roi_error


def read_component_crop(slide, bbox_level0, max_side):
    x0, y0, x1, y1 = bbox_level0
    width0 = max(1, x1 - x0)
    height0 = max(1, y1 - y0)
    choices = []
    for level, downsample in enumerate(slide.level_downsamples):
        w = int(math.ceil(width0 / float(downsample)))
        h = int(math.ceil(height0 / float(downsample)))
        choices.append((level, float(downsample), w, h, max(w, h)))
    above = [item for item in choices if item[4] >= max_side]
    selected = min(above, key=lambda item: item[4]) if above else max(choices, key=lambda item: item[4])
    level, downsample, width, height, _ = selected
    crop = read_region_resilient(slide, (x0, y0), level, (width, height))
    if max(width, height) > max_side:
        factor = float(max_side) / max(width, height)
        new_w = max(64, int(round(width * factor)))
        new_h = max(64, int(round(height * factor)))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scale_x = crop.shape[1] / float(width0)
    scale_y = crop.shape[0] / float(height0)
    return crop, [x0, y0], scale_x, scale_y


def component_from_local_tissue(rgb):
    mask = base.tissue_mask(rgb)
    components = base.extract_components(mask, 0.001, 0.01)
    if not components:
        raise ValueError("No tissue component found inside component crop")
    components = base.merge_nearby_components(components, rgb.shape, 0.04)
    keep = [component for component in components if component["area_fraction_tissue"] >= 0.05]
    if not keep:
        keep = [components[0]]
    union = np.zeros(mask.shape, dtype=np.uint8)
    source_ids = []
    for component in keep:
        union |= component["mask"].astype(np.uint8)
        source_ids.extend(component.get("source_component_ids", [component["id"]]))
    ys, xs = np.where(union > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    area = int(union.sum())
    return {
        "id": 0,
        "source_component_ids": sorted(set(source_ids)),
        "area": area,
        "area_fraction_canvas": area / float(union.shape[0] * union.shape[1]),
        "area_fraction_tissue": 1.0,
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "centroid": [float(xs.mean()), float(ys.mean())],
        "aspect": float(x1 - x0) / max(1.0, float(y1 - y0)),
        "mask": union,
    }


def local_affine_to_level0(matrix_local, he_origin, he_sx, he_sy, ihc_origin, ihc_sx, ihc_sy):
    he_to_level0 = np.asarray(
        [[1.0 / he_sx, 0.0, he_origin[0]], [0.0, 1.0 / he_sy, he_origin[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    level0_to_ihc_local = np.asarray(
        [[ihc_sx, 0.0, -ihc_origin[0] * ihc_sx], [0.0, ihc_sy, -ihc_origin[1] * ihc_sy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    # matrix_local = level0_to_ihc_local @ matrix_level0 @ he_to_level0
    matrix_level0 = np.linalg.inv(level0_to_ihc_local) @ homogeneous(matrix_local) @ np.linalg.inv(he_to_level0)
    return matrix_level0[:2]


def local_point_to_level0(x, y, origin, sx, sy):
    return [origin[0] + x / sx, origin[1] + y / sy]


def pca_affine_candidates(he_component, ihc_component):
    def stats(component):
        ys, xs = np.where(component["mask"] > 0)
        xy = np.column_stack([xs, ys]).astype(np.float64)
        center = xy.mean(axis=0)
        centered = xy - center
        cov = centered.T @ centered / max(1, len(xy) - 1)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order], 1e-6)
        vectors = vectors[:, order]
        if np.linalg.det(vectors) < 0:
            vectors[:, 1] *= -1
        return center, values, vectors

    he_center, he_values, he_vectors = stats(he_component)
    ihc_center, ihc_values, ihc_vectors = stats(ihc_component)
    scales = np.sqrt(ihc_values / he_values)
    candidates = []
    for sign in (1.0, -1.0):
        target_vectors = ihc_vectors.copy()
        target_vectors[:, 0] *= sign
        target_vectors[:, 1] *= sign
        A = target_vectors @ np.diag(scales) @ he_vectors.T
        if np.linalg.det(A) <= 0:
            continue
        t = ihc_center - A @ he_center
        candidates.append(np.column_stack([A, t]))
    return candidates


def signed_distance(mask, clip_distance=96.0):
    mask = (mask > 0).astype(np.uint8)
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    signed = np.clip(inside - outside, -clip_distance, clip_distance)
    signed = (signed + clip_distance) / (2.0 * clip_distance)
    return signed.astype(np.float32)


def crop_bbox(component, shape, pad_fraction=0.15):
    x, y, w, h = component["bbox"]
    pad = int(round(max(w, h) * pad_fraction))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(shape[1], x + w + pad)
    y1 = min(shape[0], y + h + pad)
    return x0, y0, x1, y1


def local_to_global_affine(local_matrix, origin):
    ox, oy = origin
    T1 = np.asarray([[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]])
    T0 = np.asarray([[1.0, 0.0, -ox], [0.0, 1.0, -oy], [0.0, 0.0, 1.0]])
    return (T1 @ homogeneous(local_matrix) @ T0)[:2]


def warp_to_he(image, matrix_he_to_ihc, he_shape, interpolation, border_value):
    return cv2.warpAffine(
        image,
        np.asarray(matrix_he_to_ihc, dtype=np.float32),
        (he_shape[1], he_shape[0]),
        flags=interpolation | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def refine_shape_ecc(he_component, ihc_component, candidate, he_shape, args):
    roi = crop_bbox(he_component, he_shape, pad_fraction=0.25)
    x0, y0, x1, y1 = roi
    fixed_mask = he_component["mask"][y0:y1, x0:x1]
    moving_mask = warp_roi_to_he(
        ihc_component["mask"].astype(np.uint8),
        candidate,
        roi,
        cv2.INTER_NEAREST,
        0,
    )
    if min(fixed_mask.shape) < 32:
        return candidate, None
    fixed = signed_distance(fixed_mask)
    moving = signed_distance(moving_mask)
    residual = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(args.shape_ecc_iterations),
        float(args.shape_ecc_epsilon),
    )
    try:
        correlation, residual = cv2.findTransformECC(
            fixed,
            moving,
            residual,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            5,
        )
        residual_global = local_to_global_affine(residual, (x0, y0))
        refined = compose(candidate, residual_global)
        return refined, float(correlation)
    except cv2.error:
        return candidate, None


def similarity_matrix_from_sitk(transform):
    matrix = np.asarray(transform.GetMatrix(), dtype=np.float64).reshape(2, 2)
    center = np.asarray(transform.GetCenter(), dtype=np.float64)
    translation = np.asarray(transform.GetTranslation(), dtype=np.float64)
    offset = center + translation - matrix @ center
    return np.column_stack([matrix, offset])


def multimodal_mi_refine(he_rgb, ihc_rgb, he_component, ihc_component, candidate, args):
    fixed_gray_full = base.clahe_gray(he_rgb, invert=False)
    moving_gray_full = base.clahe_gray(ihc_rgb, invert=False)
    roi = crop_bbox(he_component, fixed_gray_full.shape, pad_fraction=0.25)
    x0, y0, x1, y1 = roi
    fixed_crop = fixed_gray_full[y0:y1, x0:x1]
    moving_crop = warp_roi_to_he(
        moving_gray_full, candidate, roi, cv2.INTER_LINEAR, 255
    )
    fixed_mask_crop = (he_component["mask"][y0:y1, x0:x1] > 0).astype(np.uint8)
    moving_mask_crop = warp_roi_to_he(
        ihc_component["mask"].astype(np.uint8),
        candidate, roi, cv2.INTER_NEAREST, 0
    ).astype(np.uint8)
    if min(fixed_crop.shape) < 64 or np.count_nonzero(fixed_mask_crop) < 500:
        return candidate, None

    original_h, original_w = fixed_crop.shape
    scale_x = scale_y = 1.0
    if max(original_h, original_w) > args.mi_max_side:
        scale = float(args.mi_max_side) / max(original_h, original_w)
        new_w = max(64, int(round(original_w * scale)))
        new_h = max(64, int(round(original_h * scale)))
        scale_x = new_w / float(original_w)
        scale_y = new_h / float(original_h)
        fixed_crop = cv2.resize(fixed_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        moving_crop = cv2.resize(moving_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        fixed_mask_crop = cv2.resize(fixed_mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        moving_mask_crop = cv2.resize(moving_mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    fixed = sitk.GetImageFromArray(fixed_crop.astype(np.float32))
    moving = sitk.GetImageFromArray(moving_crop.astype(np.float32))
    fixed_m = sitk.GetImageFromArray((fixed_mask_crop > 0).astype(np.uint8))
    moving_m = sitk.GetImageFromArray((moving_mask_crop > 0).astype(np.uint8))

    tx = sitk.Similarity2DTransform()
    tx.SetCenter((fixed_crop.shape[1] / 2.0, fixed_crop.shape[0] / 2.0))
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(float(args.mi_sampling_fraction), 17)
    registration.SetMetricFixedMask(fixed_m)
    registration.SetMetricMovingMask(moving_m)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=int(args.mi_iterations),
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-7,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(tx, inPlace=True)
    try:
        registration.Execute(fixed, moving)
        residual_scaled = similarity_matrix_from_sitk(tx)
        scale_matrix = np.asarray(
            [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        residual_local = (
            np.linalg.inv(scale_matrix) @ homogeneous(residual_scaled) @ scale_matrix
        )[:2]
        residual_global = local_to_global_affine(residual_local, (x0, y0))
        refined = compose(candidate, residual_global)
        return refined, float(registration.GetMetricValue())
    except RuntimeError:
        return candidate, None


def mask_boundary(mask):
    mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel)
    return ((mask > 0) & (eroded == 0)).astype(np.uint8)


def normalized_mutual_information(a, b, mask, bins=64):
    values = mask > 0
    if values.sum() < 256:
        return 0.0
    av = a[values].astype(np.float32)
    bv = b[values].astype(np.float32)
    hist, _, _ = np.histogram2d(av, bv, bins=bins, range=[[0,255],[0,255]])
    prob = hist / max(1.0, hist.sum())
    pa = prob.sum(axis=1)
    pb = prob.sum(axis=0)
    eps = 1e-12
    ha = -np.sum(pa[pa > 0] * np.log(pa[pa > 0] + eps))
    hb = -np.sum(pb[pb > 0] * np.log(pb[pb > 0] + eps))
    hab = -np.sum(prob[prob > 0] * np.log(prob[prob > 0] + eps))
    if hab <= eps:
        return 0.0
    return float((ha + hb) / hab)


def gradient_correlation(a, b, mask):
    def magnitude(image):
        image = image.astype(np.float32) / 255.0
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)
    values = mask > 0
    if values.sum() < 256:
        return 0.0
    x = magnitude(a)[values]
    y = magnitude(b)[values]
    if x.std() < 1e-6 or y.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def score_candidate(he_rgb, ihc_rgb, he_component, ihc_component, matrix):
    roi = crop_bbox(he_component, he_rgb.shape[:2], pad_fraction=0.30)
    x0, y0, x1, y1 = roi
    he_mask = (he_component["mask"][y0:y1, x0:x1] > 0).astype(np.uint8)
    warped_mask = warp_roi_to_he(
        ihc_component["mask"].astype(np.uint8),
        matrix,
        roi,
        cv2.INTER_NEAREST,
        0,
    )
    intersection = float(np.logical_and(he_mask > 0, warped_mask > 0).sum())
    denominator = float(he_mask.sum() + warped_mask.sum())
    dice = 0.0 if denominator == 0 else 2.0 * intersection / denominator

    he_edge = mask_boundary(he_mask)
    ihc_edge = mask_boundary(warped_mask)
    dt_he = cv2.distanceTransform((1 - he_edge).astype(np.uint8), cv2.DIST_L2, 5)
    dt_ihc = cv2.distanceTransform((1 - ihc_edge).astype(np.uint8), cv2.DIST_L2, 5)
    d1 = dt_ihc[he_edge > 0]
    d2 = dt_he[ihc_edge > 0]
    distances = np.concatenate([d1, d2]) if len(d1) and len(d2) else np.asarray([1e6])
    boundary_median = float(np.median(distances))
    boundary_p95 = float(np.percentile(distances, 95))
    _, _, w, h = he_component["bbox"]
    diagonal = math.hypot(w, h)
    boundary_fraction = boundary_median / max(1.0, diagonal)
    boundary_score = math.exp(-boundary_fraction / 0.025)

    he_gray_full = base.clahe_gray(he_rgb, invert=False)
    ihc_gray_full = base.clahe_gray(ihc_rgb, invert=False)
    he_gray = he_gray_full[y0:y1, x0:x1]
    warped_gray = warp_roi_to_he(
        ihc_gray_full, matrix, roi, cv2.INTER_LINEAR, 255
    )
    overlap = ((he_mask > 0) & (warped_mask > 0)).astype(np.uint8)
    nmi = normalized_mutual_information(he_gray, warped_gray, overlap)
    nmi_score = float(np.clip(nmi - 1.0, 0.0, 1.0))
    grad_corr = gradient_correlation(he_gray, warped_gray, overlap)
    grad_score = max(0.0, grad_corr)
    total = 0.55 * dice + 0.20 * boundary_score + 0.15 * nmi_score + 0.10 * grad_score
    return {
        "mask_dice": float(dice),
        "boundary_median_px": boundary_median,
        "boundary_p95_px": boundary_p95,
        "boundary_fraction": float(boundary_fraction),
        "nmi": float(nmi),
        "gradient_correlation": float(grad_corr),
        "score": float(total),
    }


def affine_plausible(matrix):
    A = np.asarray(matrix, dtype=np.float64)[:, :2]
    det = float(np.linalg.det(A))
    if not np.isfinite(det) or det <= 0:
        return False
    singular = np.linalg.svd(A, compute_uv=False)
    if singular.min() < 0.25 or singular.max() > 4.0:
        return False
    if singular.max() / singular.min() > 3.0:
        return False
    return True


def candidate_pipeline(he_rgb, ihc_rgb, he_component, ihc_component, args):
    raw = [("bbox", base.bbox_initial_affine(he_component, ihc_component))]
    for index, matrix in enumerate(pca_affine_candidates(he_component, ihc_component)):
        raw.append(("pca_%d" % index, matrix))
    sift_result = base.sift_matches(he_rgb, ihc_rgb, he_component, ihc_component, args)
    if sift_result is not None:
        raw.append(("sift_ransac", sift_result["matrix"]))

    quick = []
    for name, matrix in raw:
        if not affine_plausible(matrix):
            continue
        for stage_name, stage_matrix, ecc_value in [(name + ":raw", matrix, None)]:
            metrics = score_candidate(he_rgb, ihc_rgb, he_component, ihc_component, stage_matrix)
            quick.append({"name": stage_name, "matrix": stage_matrix, "shape_ecc": ecc_value, "mi_optimizer_metric": None, **metrics})
        shape_matrix, ecc = refine_shape_ecc(
            he_component, ihc_component, matrix, he_rgb.shape[:2], args
        )
        if affine_plausible(shape_matrix):
            metrics = score_candidate(he_rgb, ihc_rgb, he_component, ihc_component, shape_matrix)
            quick.append({"name": name + ":shape", "matrix": shape_matrix, "shape_ecc": ecc, "mi_optimizer_metric": None, **metrics})

    quick.sort(key=lambda item: item["score"], reverse=True)
    candidates = list(quick)
    for source in quick[: int(args.mi_top_k)]:
        mi_matrix, mi_metric = multimodal_mi_refine(
            he_rgb, ihc_rgb, he_component, ihc_component, source["matrix"], args
        )
        if not affine_plausible(mi_matrix):
            continue
        metrics = score_candidate(he_rgb, ihc_rgb, he_component, ihc_component, mi_matrix)
        candidates.append({
            "name": source["name"] + "+mi",
            "matrix": mi_matrix,
            "shape_ecc": source.get("shape_ecc"),
            "mi_optimizer_metric": mi_metric,
            **metrics,
        })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates, sift_result


def candidate_confidence(best, second, args):
    if best is None:
        return "low"
    if (
        best["mask_dice"] < args.candidate_min_mask_dice
        or best["boundary_fraction"] > args.candidate_max_boundary_fraction
    ):
        return "low"
    internal_ok = (
        best["nmi"] >= 1.02
        or best["gradient_correlation"] >= 0.08
    )
    if not internal_ok:
        return "low"
    margin = best["score"] - (second["score"] if second is not None else 0.0)
    if (
        best["mask_dice"] >= 0.80
        and best["boundary_fraction"] <= 0.035
        and best["nmi"] >= 1.05
        and best["gradient_correlation"] >= 0.08
        and margin >= 0.01
    ):
        return "high"
    return "medium"


def serializable_candidate(candidate):
    return {
        key: (value.tolist() if key == "matrix" else value)
        for key, value in candidate.items()
    }


def write_candidate_csv(path, candidates):
    fields = [
        "rank", "name", "score", "mask_dice", "boundary_median_px",
        "boundary_p95_px", "boundary_fraction", "nmi",
        "gradient_correlation", "shape_ecc", "mi_optimizer_metric",
    ]
    rows = []
    for rank, candidate in enumerate(candidates, 1):
        rows.append({"rank": rank, **candidate})
    base.write_csv_atomic(path, rows, fields)


def final_overlay(he_rgb, ihc_rgb, he_component, candidate, component_id):
    warped = warp_to_he(
        ihc_rgb,
        candidate["matrix"],
        he_rgb.shape[:2],
        cv2.INTER_LINEAR,
        (255,255,255),
    )
    overlay = cv2.addWeighted(he_rgb, 0.5, warped, 0.5, 0)
    x0,y0,x1,y1 = crop_bbox(he_component, he_rgb.shape[:2], 0.15)
    panel = base.hstack([
        he_rgb[y0:y1,x0:x1], warped[y0:y1,x0:x1], overlay[y0:y1,x0:x1]
    ])
    return base.add_header(panel, [
        "component %d: HE | selected structural affine IHC | overlay" % component_id,
        "%s score=%.4f dice=%.3f boundary=%.3f%% NMI=%.3f grad=%.3f" % (
            candidate["name"], candidate["score"], candidate["mask_dice"],
            100.0*candidate["boundary_fraction"], candidate["nmi"],
            candidate["gradient_correlation"],
        ),
    ])


def process_pair(row, args, Aslide):
    pair_id = row["pair_id"]
    report_dir = Path(args.report_root) / pair_id
    transform_dir = Path(args.output_root) / "transforms" / pair_id
    report_dir.mkdir(parents=True, exist_ok=True)
    transform_dir.mkdir(parents=True, exist_ok=True)
    he_slide = Aslide(row["he_wsi_path"])
    ihc_slide = Aslide(row["ihc_wsi_path"])
    try:
        he_detect, he_detect_sx, he_detect_sy = base.choose_thumbnail(
            he_slide, args.component_detection_max_side
        )
        ihc_detect, ihc_detect_sx, ihc_detect_sy = base.choose_thumbnail(
            ihc_slide, args.component_detection_max_side
        )
        he_components = base.extract_components(
            base.tissue_mask(he_detect), args.min_component_canvas_fraction,
            args.min_component_tissue_fraction
        )
        ihc_components = base.extract_components(
            base.tissue_mask(ihc_detect), args.min_component_canvas_fraction,
            args.min_component_tissue_fraction
        )
        he_components = base.merge_nearby_components(
            he_components, he_detect.shape, args.component_merge_gap_fraction
        )
        ihc_components = base.merge_nearby_components(
            ihc_components, ihc_detect.shape, args.component_merge_gap_fraction
        )
        matches, unmatched_he, unmatched_ihc = base.match_components(
            he_components, ihc_components, he_detect.shape, ihc_detect.shape,
            args.component_match_max_cost
        )
        overview = base.hstack([
            base.draw_components(he_detect, he_components, "HE components"),
            base.draw_components(ihc_detect, ihc_components, "IHC components"),
        ])
        overview = base.add_header(overview, [
            pair_id,
            "v1.1 component detection only; matched blocks=%d" % len(matches),
        ])
        Image.fromarray(overview).save(report_dir / "01_component_overview.png")

        pair_summary = {
            "schema_version": SCHEMA_VERSION,
            "registration_version": args.registration_version,
            "pair_id": pair_id,
            "manual_landmarks_used_for_transform": False,
            "unmatched_he_components": unmatched_he,
            "unmatched_ihc_components": unmatched_ihc,
            "components": [],
        }
        rows = []
        for component_id, (hi, ii, component_cost) in enumerate(matches):
            he_bbox0 = component_bbox_level0(
                he_components[hi], he_detect_sx, he_detect_sy,
                he_slide.dimensions, args.component_crop_margin_fraction
            )
            ihc_bbox0 = component_bbox_level0(
                ihc_components[ii], ihc_detect_sx, ihc_detect_sy,
                ihc_slide.dimensions, args.component_crop_margin_fraction
            )
            he_crop, he_origin, he_sx, he_sy = read_component_crop(
                he_slide, he_bbox0, args.component_crop_max_side
            )
            ihc_crop, ihc_origin, ihc_sx, ihc_sy = read_component_crop(
                ihc_slide, ihc_bbox0, args.component_crop_max_side
            )
            he_component = component_from_local_tissue(he_crop)
            ihc_component = component_from_local_tissue(ihc_crop)

            candidates, sift_result = candidate_pipeline(
                he_crop, ihc_crop, he_component, ihc_component, args
            )
            best = candidates[0] if candidates else None
            second = candidates[1] if len(candidates) > 1 else None
            confidence = candidate_confidence(best, second, args)
            write_candidate_csv(
                report_dir / ("02_component_%02d_candidates.csv" % component_id),
                candidates,
            )
            if sift_result is not None:
                Image.fromarray(base.feature_visual(
                    he_crop, ihc_crop, sift_result, component_id
                )).save(report_dir / ("02_component_%02d_sift_candidate.png" % component_id))

            dhr_info = None
            matrix_level0 = None
            if best is not None:
                Image.fromarray(final_overlay(
                    he_crop, ihc_crop, he_component, best, component_id
                )).save(report_dir / ("03_component_%02d_selected_overlay.png" % component_id))
                matrix_level0 = local_affine_to_level0(
                    best["matrix"], he_origin, he_sx, he_sy,
                    ihc_origin, ihc_sx, ihc_sy
                )
                if args.run_dhr and confidence in {"high", "medium"}:
                    anchor_x, anchor_y = base.component_anchor(he_component)
                    center0 = local_point_to_level0(
                        anchor_x, anchor_y, he_origin, he_sx, he_sy
                    )
                    montage, dhr_info = dhr_smoke_at_center(
                        he_slide, ihc_slide, center0, matrix_level0, args
                    )
                    Image.fromarray(montage).save(
                        report_dir / ("04_component_%02d_dhr_smoke.png" % component_id)
                    )

            payload = {
                "registration_version": args.registration_version,
                "pair_id": pair_id,
                "component_id": component_id,
                "he_component_id": hi,
                "ihc_component_id": ii,
                "component_match_cost": component_cost,
                "he_component_bbox_level0": he_bbox0,
                "ihc_component_bbox_level0": ihc_bbox0,
                "confidence": confidence,
                "manual_landmarks_used_for_transform": False,
                "selected_candidate": None if best is None else serializable_candidate(best),
                "runner_up_candidate": None if second is None else serializable_candidate(second),
                "matrix_level0_HE_to_IHC_2x3": None if matrix_level0 is None else matrix_level0.tolist(),
                "sift_candidate_qc": None if sift_result is None else {
                    key: sift_result[key] for key in [
                        "he_keypoints", "ihc_keypoints", "ratio_matches",
                        "geometric_matches", "inlier_count", "inlier_ratio",
                        "inlier_median_residual_px", "inlier_p95_residual_px",
                        "spatial_coverage",
                    ]
                },
                "dhr_smoke": dhr_info,
            }
            base.write_json_atomic(
                transform_dir / ("component_%02d.json" % component_id), payload
            )
            pair_summary["components"].append(payload)
            rows.append({
                "pair_id": pair_id,
                "component_id": component_id,
                "he_component_id": hi,
                "ihc_component_id": ii,
                "component_match_cost": component_cost,
                "confidence": confidence,
                "selected_candidate": "" if best is None else best["name"],
                "score": "" if best is None else best["score"],
                "mask_dice": "" if best is None else best["mask_dice"],
                "boundary_fraction": "" if best is None else best["boundary_fraction"],
                "nmi": "" if best is None else best["nmi"],
                "gradient_correlation": "" if best is None else best["gradient_correlation"],
                "dhr_status": "run" if dhr_info else "not_run",
                "dhr_valid_fraction": "" if not dhr_info else dhr_info["final_patch_valid_fraction"],
                "dhr_folding_fraction": "" if not dhr_info else dhr_info["dhr_qc"]["folding_fraction"],
                "dhr_displacement_p95_px": "" if not dhr_info else dhr_info["dhr_qc"]["displacement_p95_px"],
            })
        base.write_json_atomic(report_dir / "review_summary.json", pair_summary)
        base.write_csv_atomic(
            report_dir / "component_summary.csv", rows,
            list(rows[0].keys()) if rows else ["pair_id"]
        )
        return pair_summary, rows
    finally:
        base.close_slide(he_slide)
        base.close_slide(ihc_slide)


def dhr_smoke_at_center(he_slide, ihc_slide, center_level0, matrix_level0, args):
    matrix_level0 = np.asarray(matrix_level0, dtype=np.float64)
    center_x = int(round(center_level0[0]))
    center_y = int(round(center_level0[1]))
    context = args.context_size
    patch = args.patch_size
    x0 = int(round(center_x - context / 2))
    y0 = int(round(center_y - context / 2))
    he_context, he_valid = base.read_padded(he_slide, x0, y0, context, context)
    ihc_affine, ihc_valid = base.warp_ihc_affine(
        ihc_slide, matrix_level0, x0, y0, context, args.affine_margin
    )
    register = base.import_dhr(args.dhr_root)
    Path(args.dhr_tmp_root).mkdir(parents=True, exist_ok=True)
    warped, meta = register(
        ihc_affine,
        he_context,
        preset=args.dhr_preset,
        device=args.dhr_device,
        overrides=args.dhr_overrides,
        source_valid_mask=ihc_valid,
        temporary_root=args.dhr_tmp_root,
        return_valid_mask=True,
    )
    he_patch = base.center_crop(he_context, patch)
    affine_patch = base.center_crop(ihc_affine, patch)
    dhr_patch = base.center_crop(warped, patch)
    valid = base.center_crop(meta["valid_mask"].astype(np.uint8), patch)
    overlay = cv2.addWeighted(he_patch, 0.5, dhr_patch, 0.5, 0)
    montage = base.hstack([he_patch, affine_patch, dhr_patch, overlay], gap=8)
    montage = base.add_header(montage, [
        "HE | structure-affine IHC | DHR IHC | HE/DHR overlay",
        "center=(%d,%d) valid=%.4f fold=%.6f disp_p95=%.2f px" % (
            center_x, center_y, float(valid.mean()),
            meta["deformation_qc"]["folding_fraction"],
            meta["deformation_qc"]["displacement_p95_px"],
        ),
    ])
    return montage, {
        "he_center_level0": [center_x, center_y],
        "final_patch_valid_fraction": float(valid.mean()),
        "dhr_registration_time_seconds": meta["registration_time_seconds"],
        "dhr_working_displacement_shape": meta["working_displacement_shape"],
        "dhr_qc": meta["deformation_qc"],
    }


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser():
    default = Path(__file__).resolve().parents[2] / "configs" / "vsseg" / "registration_v1_1_structure_component.json"
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(default))
    pre_args, _ = pre.parse_known_args()
    config = load_config(pre_args.config)
    parser = argparse.ArgumentParser(parents=[pre])
    for key in ["registration_version", "inventory", "output_root", "report_root", "aslide_root", "dhr_root"]:
        parser.add_argument("--" + key.replace("_", "-"), default=config[key])
    parser.add_argument("--pair-id", action="append", required=True)
    parser.add_argument("--component-detection-max-side", type=int, default=config["component_detection_max_side"])
    parser.add_argument("--component-crop-max-side", type=int, default=config["component_crop_max_side"])
    parser.add_argument("--component-crop-margin-fraction", type=float, default=config["component_crop_margin_fraction"])
    parser.add_argument("--min-component-canvas-fraction", type=float, default=config["min_component_canvas_fraction"])
    parser.add_argument("--min-component-tissue-fraction", type=float, default=config["min_component_tissue_fraction"])
    parser.add_argument("--component-merge-gap-fraction", type=float, default=config["component_merge_gap_fraction"])
    parser.add_argument("--component-match-max-cost", type=float, default=config["component_match_max_cost"])
    parser.add_argument("--sift-nfeatures", type=int, default=config["sift_nfeatures"])
    parser.add_argument("--sift-ratio", type=float, default=config["sift_ratio"])
    parser.add_argument("--feature-gate-fraction", type=float, default=config["feature_gate_fraction"])
    parser.add_argument("--ransac-threshold-thumbnail-px", type=float, default=config["ransac_threshold_thumbnail_px"])
    parser.add_argument("--shape-ecc-iterations", type=int, default=config["shape_ecc_iterations"])
    parser.add_argument("--shape-ecc-epsilon", type=float, default=config["shape_ecc_epsilon"])
    parser.add_argument("--mi-iterations", type=int, default=config["mi_iterations"])
    parser.add_argument("--mi-sampling-fraction", type=float, default=config["mi_sampling_fraction"])
    parser.add_argument("--mi-top-k", type=int, default=config["mi_top_k"])
    parser.add_argument("--mi-max-side", type=int, default=config["mi_max_side"])
    parser.add_argument("--candidate-min-mask-dice", type=float, default=config["candidate_min_mask_dice"])
    parser.add_argument("--candidate-max-boundary-fraction", type=float, default=config["candidate_max_boundary_fraction"])
    parser.add_argument("--context-size", type=int, default=config["context_size"])
    parser.add_argument("--patch-size", type=int, default=config["patch_size"])
    parser.add_argument("--affine-margin", type=int, default=config["affine_margin"])
    parser.add_argument("--dhr-preset", default=config["dhr_preset"])
    parser.add_argument("--dhr-device", default=config["dhr_device"])
    parser.add_argument("--dhr-tmp-root", default=config["dhr_tmp_root"])
    parser.add_argument("--run-dhr", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(dhr_overrides=config["dhr_overrides"])
    return parser


def main():
    args = build_parser().parse_args()
    inventory = {row["pair_id"]: row for row in base.read_csv(args.inventory)}
    Aslide = base.import_aslide(args.aslide_root)
    all_rows = []
    for pair_id in args.pair_id:
        if pair_id not in inventory:
            raise KeyError("Unknown pair_id: %s" % pair_id)
        print("PROCESS", pair_id, flush=True)
        _, rows = process_pair(inventory[pair_id], args, Aslide)
        all_rows.extend(rows)
        for row in rows:
            print(
                " COMPONENT", row["component_id"], row["confidence"],
                row["selected_candidate"], "score", row["score"],
                "dice", row["mask_dice"], "DHR", row["dhr_status"],
                flush=True,
            )
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0].keys()) if all_rows else ["pair_id"]
    base.write_csv_atomic(Path(args.output_root) / "structure_component_summary.csv", all_rows, fields)
    provenance = {
        "registration_version": args.registration_version,
        "pipeline_git_commit": base.git_commit(Path(__file__).resolve().parents[2]),
        "dhr_git_commit": base.git_commit(args.dhr_root),
        "script_sha256": base.sha256_file(Path(__file__).resolve()),
        "config_sha256": base.sha256_file(args.config),
        "pair_ids": args.pair_id,
        "manual_landmarks_used_for_transform": False,
    }
    base.write_json_atomic(Path(args.output_root) / "provenance.json", provenance)


if __name__ == "__main__":
    main()
