import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from ultralytics import YOLO

from utils import load_stalls_csv, bbox_to_polygon
from visualize import draw_polygon, put_label


CAR_CLASSES = {"car", "truck", "bus", "motorbike", "bicycle", "van"}


def collect_car_geoms(res, min_car_area: float) -> List[Polygon]:
    """Return list of car polygons (seg masks if available, else bboxes)."""
    geoms: List[Polygon] = []

    names = res.names
    cls = res.boxes.cls.cpu().numpy()
    boxes = res.boxes.xyxy.cpu().numpy()

    # Use segmentation masks if present (better overlap than bboxes)
    masks = None
    try:
        if hasattr(res, "masks") and res.masks is not None:
            masks = res.masks  # (segments to polygons)
    except Exception:
        masks = None

    if masks is not None:
        # Each mask -> Nx2 polygon array in image coordinates
        segs = masks.xy  # list[np.ndarray]
        for seg, cid in zip(segs, cls):
            if names[int(cid)] not in CAR_CLASSES:
                continue
            if seg is None or len(seg) < 3:
                continue
            poly = Polygon(seg.tolist())
            if poly.is_valid and poly.area >= min_car_area:
                geoms.append(poly)
    else:
        # Fall back to bboxes
        for box, cid in zip(boxes, cls):
            if names[int(cid)] not in CAR_CLASSES:
                continue
            poly = bbox_to_polygon(box)
            if poly.area >= min_car_area:
                geoms.append(poly)

    return geoms


def main():
    ap = argparse.ArgumentParser(description="Parking Occupancy Detector")
    ap.add_argument("--image", required=True, help="Path to lot image")
    ap.add_argument("--stalls", required=True, help="CSV with stall polygons")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO weights (seg works too)")
    ap.add_argument("--outdir", default="outputs")
    # Tunables
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="YOLO NMS IoU")
    ap.add_argument("--overlap_thresh", type=float, default=0.15, help="area(car∩stall)/area(stall)")
    ap.add_argument("--min_car_area", type=float, default=100.0, help="ignore tiny detections (px^2)")
    ap.add_argument("--stall_inset", type=float, default=0.0, help="shrink stall by N pixels before overlap")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Image not found: {args.image}")

    stalls = load_stalls_csv(args.stalls)

    # Run YOLO once
    model = YOLO(args.model)
    res = model(args.image, conf=args.conf, iou=args.iou)[0]

    # Build car geometry list (masks preferred, else bboxes)
    car_polys = collect_car_geoms(res, min_car_area=args.min_car_area)

    # Optional union for cleaner intersections
    cars_union = unary_union(car_polys) if car_polys else None

    summary = {"image": str(args.image), "total": len(stalls), "stalls": []}

    vis = img.copy()
    for s in stalls:
        sid = s["stall_id"]
        stall_poly: Polygon = s["poly"]

        # Inset the stall a bit to avoid border touches flipping status
        if args.stall_inset != 0.0 and stall_poly.is_valid:
            try:
                # Negative buffer shrinks; positive expands
                stall_poly = stall_poly.buffer(-args.stall_inset)
                if stall_poly.is_empty:
                    # If we over-shrunk, fall back to original
                    stall_poly = s["poly"]
            except Exception:
                pass

        if cars_union is None:
            overlap = 0.0
        else:
            inter_area = stall_poly.intersection(cars_union).area
            overlap = inter_area / max(stall_poly.area, 1e-6)

        is_occ = overlap >= args.overlap_thresh
        color = (0, 0, 255) if is_occ else (0, 180, 0)
        draw_polygon(vis, stall_poly, color=color, thickness=2, alpha=0.25)
        cx, cy = map(int, stall_poly.centroid.coords[0])
        put_label(vis, f"{sid}:{'O' if is_occ else 'F'}", (cx, cy))
        summary["stalls"].append({
            "stall_id": sid,
            "status": "occupied" if is_occ else "free",
            "overlap": round(overlap, 3)
        })

    summary["occupied"] = sum(1 for x in summary["stalls"] if x["status"] == "occupied")
    summary["free"] = summary["total"] - summary["occupied"]

    stem = Path(args.image).stem
    out_img = outdir / f"{stem}_annotated.png"
    out_json = outdir / f"{stem}_summary.json"
    cv2.imwrite(str(out_img), vis)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"saved {out_img}")
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
