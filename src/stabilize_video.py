import cv2
import numpy as np
from pathlib import Path

IN_DIR = Path("data/clips")
OUT_DIR = Path("outputs/stabilized")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def stabilize_video(in_path: Path, out_path: Path, max_frames: int | None = None):
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # Read first frame as reference
    ok, prev = cap.read()
    if not ok:
        cap.release()
        writer.release()
        raise RuntimeError("Could not read first frame")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    writer.write(prev)  # first frame unchanged

    frame_idx = 1
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ok, curr = cap.read()
        if not ok:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Detect + match features between prev and curr
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            # If feature detection fails, just write frame as-is
            writer.write(curr)
            prev_gray = curr_gray
            frame_idx += 1
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Keep best matches
        good = matches[:80] if len(matches) > 80 else matches
        if len(good) < 10:
            writer.write(curr)
            prev_gray = curr_gray
            frame_idx += 1
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Estimate affine transform from curr -> prev
        M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0)

        if M is None:
            writer.write(curr)
        else:
            stabilized = cv2.warpAffine(curr, M, (w, h))
            writer.write(stabilized)

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()
    writer.release()

def main():
    clips = sorted(IN_DIR.glob("*.mp4"))
    if not clips:
        print(f"No clips found in {IN_DIR.resolve()}")
        return

    for p in clips:
        out = OUT_DIR / f"{p.stem}_stab.mp4"
        print(f"Stabilizing: {p.name} -> {out.name}")
        stabilize_video(p, out)

    print(f"Done. Stabilized {len(clips)} clip(s) into {OUT_DIR}")

if __name__ == "__main__":
    main()
