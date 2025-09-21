import cv2
import numpy as np

def draw_polygon(img, poly, color=(0, 255, 0), thickness=2, alpha=0.25):
    pts = np.array(poly.exterior.coords[:], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [pts], True, color, thickness)

def put_label(img, text, xy, color=(0, 0, 0)):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
