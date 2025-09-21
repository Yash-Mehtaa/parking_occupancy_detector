import csv
from typing import List, Tuple, Dict, Optional
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

# Try Shapely 2's make_valid; fall back to buffer(0)
try:
    from shapely.validation import make_valid as _make_valid
except Exception:
    _make_valid = None

Point = Tuple[float, float]


def _parse_xy(s: str) -> Point:
    s = s.strip().lstrip("(").rstrip(")")
    x, y = s.split(",")
    return float(x), float(y)


def _to_valid_polygon(pts: List[Point]) -> Optional[BaseGeometry]:
    poly = Polygon(pts)
    if poly.is_valid and not poly.is_empty:
        return poly
    # Try to fix invalid polygons
    if _make_valid is not None:
        fixed = _make_valid(poly)
        if fixed.geom_type == "Polygon":
            poly = fixed
        elif hasattr(fixed, "geoms"):
            polys = [g for g in fixed.geoms if g.geom_type == "Polygon"]
            poly = max(polys, key=lambda p: p.area) if polys else None
        else:
            poly = None
    else:
        try:
            poly = poly.buffer(0)
        except Exception:
            poly = None
    if poly is not None and poly.is_valid and not poly.is_empty:
        return poly
    return None


def load_stalls_csv(path: str) -> List[Dict]:
    """CSV columns: stall_id, polygon="(x,y);(x,y);..."
    Returns [{"stall_id": str, "poly": Polygon}, ...]; skips invalid rows.
    """
    stalls: List[Dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("polygon") or "").strip()
            if not raw:
                continue
            pts = [_parse_xy(p) for p in raw.split(";") if p.strip()]
            if len(pts) < 3:
                continue
            poly = _to_valid_polygon(pts)
            if poly is None:
                print(f"[warn] skipping invalid stall polygon: {row.get('stall_id')}")
                continue
            stalls.append({"stall_id": row.get("stall_id", f"S{len(stalls)+1}"), "poly": poly})
    return stalls


def bbox_to_polygon(xyxy) -> Polygon:
    x1, y1, x2, y2 = map(float, xyxy)
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
