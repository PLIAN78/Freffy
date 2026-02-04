import cv2
import numpy as np
from pathlib import Path

VIDEO_PATH = Path("data/raw/clip_001fixed.mp4")  
OUT_PATH = Path("outputs/cuts.txt")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH.resolve()}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame")

    # resize smaller for speed + robustness
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (480, 270))

    cut_frames = []
    frame_idx = 1

    # Bigger threshold = fewer cuts found. We'll tune if needed.
    CUT_THRESHOLD = 35.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (480, 270))

        diff = cv2.absdiff(gray, prev_gray)
        score = float(np.mean(diff))

        if score > CUT_THRESHOLD:
            cut_frames.append((frame_idx, score))

        prev_gray = gray
        frame_idx += 1

    cap.release()

    lines = []
    lines.append(f"Video: {VIDEO_PATH}")
    lines.append(f"FPS: {fps:.2f}")
    lines.append(f"Detected {len(cut_frames)} potential hard cuts (threshold={CUT_THRESHOLD}).")
    lines.append("First 30 cuts:")

    for i, (f, s) in enumerate(cut_frames[:30]):
        t = f / fps
        lines.append(f"{i+1:02d}  frame={f}  time={t:.2f}s  score={s:.1f}")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[:8]))
    print(f"\nSaved details to: {OUT_PATH}")

if __name__ == "__main__":
    main()
