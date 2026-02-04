import cv2
import numpy as np
import pandas as pd
from pathlib import Path

IN_DIR = Path("outputs/stabilized")
OUT_DIR = Path("outputs/tracks")
DBG_DIR = Path("outputs/tracks_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DBG_DIR.mkdir(parents=True, exist_ok=True)

# --- Tuning knobs (safe defaults) ---
MIN_AREA = 25000         # ignore tiny blobs
MAX_AREA = 250000        # ignore huge blobs (like full-screen graphics)
MOTION_THRESH = 22       # binarize motion; bigger = less sensitive
MORPH_KERNEL = 7         # cleanup size

def bbox_from_contour(c):
    x, y, w, h = cv2.boundingRect(c)
    return np.array([x, y, w, h], dtype=float)

def center(b):
    x, y, w, h = b
    return np.array([x + w/2, y + h/2], dtype=float)

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0

def pick_two_boxes(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        boxes.append((area, bbox_from_contour(c)))
    if not boxes:
        return []
    boxes.sort(key=lambda x: x[0], reverse=True)
    return [b for _, b in boxes[:2]]

def assign_lr(boxes, prev_L, prev_R):
    """
    Given up to 2 boxes, decide which is Left fencer and Right fencer.
    - If we have previous boxes, match by nearest center.
    - Otherwise, decide by x position.
    """
    if len(boxes) == 0:
        return None, None
    if len(boxes) == 1:
        b = boxes[0]
        # If we had prevs, attach to closer one; else decide by x
        if prev_L is not None and prev_R is not None:
            dL = np.linalg.norm(center(b) - center(prev_L))
            dR = np.linalg.norm(center(b) - center(prev_R))
            return (b, None) if dL <= dR else (None, b)
        else:
            return (b, None) if center(b)[0] < 0.5 else (None, b)

    b1, b2 = boxes[0], boxes[1]

    if prev_L is not None and prev_R is not None:
        # choose assignment that minimizes total distance
        d11 = np.linalg.norm(center(b1) - center(prev_L))
        d12 = np.linalg.norm(center(b1) - center(prev_R))
        d21 = np.linalg.norm(center(b2) - center(prev_L))
        d22 = np.linalg.norm(center(b2) - center(prev_R))

        # option A: b1->L, b2->R
        costA = d11 + d22
        # option B: b1->R, b2->L
        costB = d12 + d21

        if costA <= costB:
            return b1, b2
        else:
            return b2, b1

    # no prev: sort by x
    return (b1, b2) if center(b1)[0] <= center(b2)[0] else (b2, b1)

def track_one_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # debug video writer
    dbg_path = DBG_DIR / f"{video_path.stem}_dbg.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dbg = cv2.VideoWriter(str(dbg_path), fourcc, fps, (w, h))

    # read first frame
    ok, prev = cap.read()
    if not ok:
        cap.release()
        dbg.release()
        raise RuntimeError("Could not read first frame")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))

    rows = []
    prev_L, prev_R = None, None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # motion mask = absdiff(prev, curr)
        diff = cv2.absdiff(gray, prev_gray)
        _, mask = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)

        # clean up
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        boxes = pick_two_boxes(mask)
        L, R = assign_lr(boxes, prev_L, prev_R)

        # if a box disappeared, keep last known for a short time (simple hold)
        if L is None and prev_L is not None:
            L = prev_L
        if R is None and prev_R is not None:
            R = prev_R

        prev_L, prev_R = L, R

        # store row (use -1 when missing)
        def pack(b):
            if b is None:
                return [-1, -1, -1, -1]
            x, y, bw, bh = b
            return [int(x), int(y), int(bw), int(bh)]

        rows.append({
            "frame": frame_idx,
            "time_sec": frame_idx / fps,
            "L_x": pack(L)[0], "L_y": pack(L)[1], "L_w": pack(L)[2], "L_h": pack(L)[3],
            "R_x": pack(R)[0], "R_y": pack(R)[1], "R_w": pack(R)[2], "R_h": pack(R)[3],
        })

        # draw debug
        vis = frame.copy()
        if L is not None:
            x, y, bw, bh = map(int, L)
            cv2.rectangle(vis, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(vis, "L", (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        if R is not None:
            x, y, bw, bh = map(int, R)
            cv2.rectangle(vis, (x, y), (x+bw, y+bh), (255, 0, 0), 2)
            cv2.putText(vis, "R", (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        dbg.write(vis)

        prev_gray = gray
        frame_idx += 1

    cap.release()
    dbg.release()

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / f"{video_path.stem}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved tracks: {out_csv}  |  Debug video: {dbg_path}")

def main():
    vids = sorted(IN_DIR.glob("*.mp4"))
    if not vids:
        print(f"No videos found in {IN_DIR.resolve()}")
        return

    for v in vids:
        print(f"Tracking: {v.name}")
        track_one_video(v)

    print("Done.")

if __name__ == "__main__":
    main()
