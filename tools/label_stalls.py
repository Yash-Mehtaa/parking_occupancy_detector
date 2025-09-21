import argparse, csv, cv2

# How to use:
#  - Left-click: add a corner point
#  - n :finish current stall and start a New one
#  - u :Undo last point in the current stall
#  - s :Save CSV and quit
#  - q :Quit without saving

stalls = []
current = []
img = None
vis = None

def draw():
    global vis
    vis = img.copy()
    # draw existing stalls (green)
    for poly in stalls:
        for i in range(len(poly)):
            cv2.line(vis, poly[i], poly[(i+1) % len(poly)], (0, 200, 0), 2)
    # draw current stall (red)
    for i, p in enumerate(current):
        cv2.circle(vis, p, 4, (0, 0, 255), -1)
        if i > 0:
            cv2.line(vis, current[i-1], p, (0, 0, 255), 2)

    cv2.putText(vis, "Click=corner | n=next | u=undo | s=save | q=quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20,20,20), 3, cv2.LINE_AA)
    cv2.putText(vis, "Click=corner | n=next | u=undo | s=save | q=quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current.append((x, y))
        draw()

def main():
    global img, vis
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to lot image")
    ap.add_argument("--out", default="data/stalls/lot1_stalls.csv")
    args = ap.parse_args()

    img0 = cv2.imread(args.image)
    if img0 is None:
        raise SystemExit(f"Image not found: {args.image}")

    img = img0
    vis = img.copy()

    cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("label", on_mouse)
    draw()

    while True:
        cv2.imshow("label", vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('n'):
            if len(current) >= 3:
                stalls.append(current[:])
            current.clear()
            draw()
        elif k == ord('u'):
            if current:
                current.pop()
                draw()
        elif k == ord('s'):
            if len(current) >= 3:
                stalls.append(current[:])
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["stall_id", "polygon"])
                for i, poly in enumerate(stalls, start=1):
                    poly_str = ";".join(f"({x},{y})" for x, y in poly)
                    w.writerow([f"S{i}", poly_str])
            print("saved", args.out)
            break
        elif k == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
