#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    error = None
    for enc in ("utf-8-sig", "gbk"):
        try:
            return json.loads(Path(path).read_text(encoding=enc))
        except Exception as exc:
            error = exc
    raise RuntimeError("Could not read %s: %s" % (path, error))


def norm_name(value):
    return re.sub(r"\s+", "", value or "")


def positions(path):
    out = []
    for item in load_json(path):
        if not isinstance(item, dict) or item.get("type") != "Position":
            continue
        region = item.get("region") or {}
        if "x" not in region or "y" not in region:
            continue
        out.append({
            "name": str(item.get("name", "")),
            "norm": norm_name(str(item.get("name", ""))),
            "x": float(region["x"]),
            "y": float(region["y"]),
        })
    return out


def match_points(he, ihc):
    he_names = [p["norm"] for p in he]
    ihc_names = [p["norm"] for p in ihc]
    if he_names == ihc_names:
        return list(zip(he, ihc)), [], [], "exact_sequence"
    if len(set(he_names)) != len(he_names) or len(set(ihc_names)) != len(ihc_names):
        raise ValueError("Duplicate landmark names prevent unambiguous review")
    ihc_by_name = {p["norm"]: p for p in ihc}
    pairs = [(p, ihc_by_name[p["norm"]]) for p in he if p["norm"] and p["norm"] in ihc_by_name]
    matched = {a["norm"] for a, _ in pairs}
    he_unmatched = [p for p in he if p["norm"] not in matched]
    ihc_unmatched = [p for p in ihc if p["norm"] not in matched]
    return pairs, he_unmatched, ihc_unmatched, "name_intersection"


def fit_affine(pairs):
    src = np.asarray([[a["x"], a["y"]] for a, _ in pairs], dtype=np.float64)
    dst = np.asarray([[b["x"], b["y"]] for _, b in pairs], dtype=np.float64)
    X = np.column_stack([src, np.ones(len(src))])
    coef, _, rank, _ = np.linalg.lstsq(X, dst, rcond=None)
    if rank < 3:
        raise ValueError("Degenerate landmark geometry")
    M = coef.T
    pred = X @ M.T
    err = np.linalg.norm(pred - dst, axis=1)
    return M, err


def import_aslide(root):
    parent = str(Path(root).parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from aslide.aslide import Aslide
    return Aslide


def rgb(image):
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    arr = arr[..., :3]
    return np.array(arr, dtype=np.uint8, copy=True)


def close_slide(slide):
    try:
        slide.close()
    except Exception:
        pass


def review_thumbnail(slide, max_side=1600):
    dims = [(int(w), int(h)) for w, h in slide.level_dimensions]
    level = len(dims) - 1
    for i, (w, h) in enumerate(dims):
        if max(w, h) <= max_side:
            level = i
            break
    w, h = dims[level]
    arr = rgb(slide.read_region((0, 0), level, (w, h)))
    if max(w, h) > max_side:
        scale = float(max_side) / max(w, h)
        arr = cv2.resize(arr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    W0, H0 = map(float, slide.dimensions)
    sx = arr.shape[1] / W0
    sy = arr.shape[0] / H0
    return arr, sx, sy


def point_palette(n):
    base = [
        (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
        (255, 127, 0), (255, 255, 51), (166, 86, 40), (247, 129, 191),
        (153, 153, 153), (0, 170, 170), (100, 60, 200), (40, 180, 100),
    ]
    return [base[i % len(base)] for i in range(n)]


def annotate_points(arr, matched_points, unmatched_points, sx, sy, side_label):
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    colors = point_palette(len(matched_points))
    radius = max(5, int(round(max(img.size) / 250)))
    font = ImageFont.load_default()
    draw.rectangle((0, 0, 180, 26), fill=(255, 255, 255))
    draw.text((8, 7), side_label, fill=(0, 0, 0), font=font)
    for i, p in enumerate(matched_points):
        x = int(round(p["x"] * sx)); y = int(round(p["y"] * sy))
        c = colors[i]
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=c, width=3)
        draw.text((x+radius+2, y-radius), "M%d" % (i+1), fill=c, font=font)
    for i, p in enumerate(unmatched_points):
        x = int(round(p["x"] * sx)); y = int(round(p["y"] * sy))
        draw.line((x-radius, y-radius, x+radius, y+radius), fill=(255, 0, 0), width=4)
        draw.line((x-radius, y+radius, x+radius, y-radius), fill=(255, 0, 0), width=4)
        draw.text((x+radius+2, y-radius), "U%d" % (i+1), fill=(255, 0, 0), font=font)
    return np.asarray(img)


def hstack_with_gap(images, gap=16, background=245):
    height = max(img.shape[0] for img in images)
    width = sum(img.shape[1] for img in images) + gap * (len(images)-1)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    x = 0
    for img in images:
        canvas[:img.shape[0], x:x+img.shape[1]] = img
        x += img.shape[1] + gap
    return canvas


def draw_header(arr, lines):
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    header_h = 18 * len(lines) + 14
    canvas = Image.new("RGB", (img.width, img.height + header_h), (255, 255, 255))
    canvas.paste(img, (0, header_h))
    draw2 = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw2.text((8, 6 + i*18), line, fill=(0, 0, 0), font=font)
    return np.asarray(canvas)


def warp_ihc_thumb_to_he(ihc, he_shape, M, he_sx, he_sy, ihc_sx, ihc_sy):
    h, w = he_shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    he_x0 = x / he_sx
    he_y0 = y / he_sy
    map_x = (M[0,0]*he_x0 + M[0,1]*he_y0 + M[0,2]) * ihc_sx
    map_y = (M[1,0]*he_x0 + M[1,1]*he_y0 + M[1,2]) * ihc_sy
    return cv2.remap(ihc, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))


def read_center_crop(slide, x, y, size=1024):
    half = size // 2
    x0 = int(round(x)) - half
    y0 = int(round(y)) - half
    W, H = map(int, slide.dimensions)
    sx0 = max(0, x0); sy0 = max(0, y0)
    sx1 = min(W, x0 + size); sy1 = min(H, y0 + size)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    if sx1 > sx0 and sy1 > sy0:
        tile = rgb(slide.read_region((sx0, sy0), 0, (sx1-sx0, sy1-sy0)))
        dx = sx0 - x0; dy = sy0 - y0
        canvas[dy:dy+tile.shape[0], dx:dx+tile.shape[1]] = tile
    return canvas


def crop_montage(he_slide, ihc_slide, pairs, residuals, crop_size=1024, display_size=320):
    font = ImageFont.load_default()
    rows = []
    for i, ((he_p, ihc_p), residual) in enumerate(zip(pairs, residuals), 1):
        he = read_center_crop(he_slide, he_p["x"], he_p["y"], crop_size)
        ihc = read_center_crop(ihc_slide, ihc_p["x"], ihc_p["y"], crop_size)
        he = cv2.resize(he, (display_size, display_size), interpolation=cv2.INTER_AREA)
        ihc = cv2.resize(ihc, (display_size, display_size), interpolation=cv2.INTER_AREA)
        pair_img = Image.fromarray(hstack_with_gap([he, ihc], gap=8))
        draw = ImageDraw.Draw(pair_img)
        cx = display_size // 2; cy = display_size // 2
        for offset in (0, display_size + 8):
            draw.line((offset+cx-12, cy, offset+cx+12, cy), fill=(255,0,0), width=2)
            draw.line((offset+cx, cy-12, offset+cx, cy+12), fill=(255,0,0), width=2)
        label = "M%d residual=%.1f px" % (i, residual)
        draw.rectangle((0,0,pair_img.width,22), fill=(255,255,255))
        draw.text((6,6), label, fill=(0,0,0), font=font)
        rows.append(np.asarray(pair_img))
    width = max(r.shape[1] for r in rows)
    total_h = sum(r.shape[0] for r in rows) + 8*(len(rows)-1)
    canvas = np.full((total_h, width, 3), 245, dtype=np.uint8)
    y = 0
    for row in rows:
        canvas[y:y+row.shape[0], :row.shape[1]] = row
        y += row.shape[0] + 8
    return canvas


def write_residual_csv(path, pairs, residuals):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["match_index","he_landmark_name","ihc_landmark_name","he_x","he_y","ihc_x","ihc_y","affine_residual_px"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, ((a,b), e) in enumerate(zip(pairs,residuals),1):
            w.writerow({
                "match_index": i,
                "he_landmark_name": a["name"],
                "ihc_landmark_name": b["name"],
                "he_x": a["x"], "he_y": a["y"],
                "ihc_x": b["x"], "ihc_y": b["y"],
                "affine_residual_px": float(e),
            })


def render_pair(row, output_root, Aslide):
    pair_id = row["pair_id"]
    out = Path(output_root) / pair_id
    out.mkdir(parents=True, exist_ok=True)
    he_pts = positions(row["he_landmark_json"])
    ihc_pts = positions(row["ihc_landmark_json"])
    pairs, he_unmatched, ihc_unmatched, method = match_points(he_pts, ihc_pts)
    M, residuals = fit_affine(pairs)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    median = float(np.median(residuals)); maxerr = float(np.max(residuals))

    he_slide = Aslide(row["he_wsi_path"])
    ihc_slide = Aslide(row["ihc_wsi_path"])
    try:
        he_thumb, he_sx, he_sy = review_thumbnail(he_slide)
        ihc_thumb, ihc_sx, ihc_sy = review_thumbnail(ihc_slide)
        he_match = [a for a,_ in pairs]; ihc_match = [b for _,b in pairs]
        he_ann = annotate_points(he_thumb, he_match, he_unmatched, he_sx, he_sy, "HE")
        ihc_ann = annotate_points(ihc_thumb, ihc_match, ihc_unmatched, ihc_sx, ihc_sy, "IHC")
        overview = hstack_with_gap([he_ann, ihc_ann])
        overview = draw_header(overview, [pair_id, "matching=%s matched=%d RMSE=%.1f px median=%.1f max=%.1f" % (method, len(pairs), rmse, median, maxerr), "M*=matched landmark; U*=unmatched landmark"])
        Image.fromarray(overview).save(out / "01_landmarks_overview.png")

        warped = warp_ihc_thumb_to_he(ihc_thumb, he_thumb.shape, M, he_sx, he_sy, ihc_sx, ihc_sy)
        overlay = cv2.addWeighted(he_thumb, 0.5, warped, 0.5, 0)
        aff = hstack_with_gap([he_thumb, warped, overlay])
        aff = draw_header(aff, [pair_id, "HE | affine-warped IHC | 50/50 overlay", "name-matched affine RMSE=%.1f px" % rmse])
        Image.fromarray(aff).save(out / "02_affine_overlay.png")

        crops = crop_montage(he_slide, ihc_slide, pairs, residuals)
        crops = draw_header(crops, [pair_id, "HE crop | IHC crop; red cross = annotated landmark center"])
        Image.fromarray(crops).save(out / "03_landmark_crops.png")

        write_residual_csv(out / "landmark_residuals.csv", pairs, residuals)
        summary = {
            "pair_id": pair_id,
            "audit_status": row.get("landmark_status", ""),
            "audit_reason": row.get("reason", ""),
            "matching_method": method,
            "he_landmark_count": len(he_pts),
            "ihc_landmark_count": len(ihc_pts),
            "matched_landmark_count": len(pairs),
            "unmatched_he_names": [p["name"] for p in he_unmatched],
            "unmatched_ihc_names": [p["name"] for p in ihc_unmatched],
            "affine_rmse_px": rmse,
            "affine_median_error_px": median,
            "affine_max_error_px": maxerr,
            "affine_matrix_2x3": M.tolist(),
            "files": ["01_landmarks_overview.png","02_affine_overlay.png","03_landmark_crops.png","landmark_residuals.csv"],
        }
        (out / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        (out / "README.txt").write_text(
            "Open in this order:\n"
            "1. 01_landmarks_overview.png - whole-slide point placement and unmatched points.\n"
            "2. 03_landmark_crops.png - local anatomy around every name-matched landmark.\n"
            "3. 02_affine_overlay.png - whole-slide affine alignment sanity check.\n"
            "4. landmark_residuals.csv - per-landmark numerical residual.\n",
            encoding="utf-8",
        )
        print(pair_id, "RMSE", rmse, "OUT", out)
    finally:
        close_slide(he_slide); close_slide(ihc_slide)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", default="/NAS145/linboliao/Data/VS-Seg/derived/registration_v0_manual_v2/private/manual_pair_inventory_internal.csv")
    p.add_argument("--output-root", default="/NAS145/linboliao/Data/VS-Seg/reports/dataset_audit/registration_v0_manual_v2_review")
    p.add_argument("--aslide-root", default="/NAS3/lbliao/Code-138/aslide")
    p.add_argument("--pair-id", action="append", required=True)
    args = p.parse_args()
    rows = {r["pair_id"]: r for r in read_csv(args.inventory)}
    Aslide = import_aslide(args.aslide_root)
    for pair_id in args.pair_id:
        if pair_id not in rows:
            raise KeyError("Unknown pair_id: %s" % pair_id)
        render_pair(rows[pair_id], args.output_root, Aslide)


if __name__ == "__main__":
    main()
