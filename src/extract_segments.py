import cv2
import numpy as np
from pathlib import Path

VIDEO_PATH = Path("data/raw/clip_001fixed.mp4") 
OUT_DIR = Path("data/clips")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def segment_score(cap, start_f, end_f, sample_step=5):
    """
    Returns a simple score for how 'consistent' the segment is:
    - high motion spikes everywhere often means crowd/replay/graphics
    - moderate, structured motion often means fencing camera
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    ok, prev = cap.read()
    if not ok:
        return None

    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_g = cv2.resize(prev_g, (320, 180))

    scores = []
    f = start_f + 1

    while f <= end_f:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (320, 180))

        diff = cv2.absdiff(g, prev_g)
        scores.append(float(np.mean(diff)))
        prev_g = g
        f += sample_step

    if not scores:
        return None

    # Return both mean and variability; we’ll use them for filtering
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    return mean, std

def write_segment(in_path, out_path, start_f, end_f):
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError("Could not reopen video for writing segment")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    for _ in range(start_f, end_f + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)

    writer.release()
    cap.release()

def detect_cuts(cap, threshold=35.0):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame for cut detection")

    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_g = cv2.resize(prev_g, (480, 270))

    cuts = [0]
    frame_idx = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (480, 270))
        score = float(np.mean(cv2.absdiff(g, prev_g)))
        if score > threshold:
            cuts.append(frame_idx)
        prev_g = g
        frame_idx += 1

    cuts.append(frame_idx - 1)
    return cuts

def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH.resolve()}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cuts = detect_cuts(cap, threshold=35.0)
    cap.release()

    # Convert cut points into segments
    segments = []
    for i in range(len(cuts) - 1):
        a = cuts[i]
        b = cuts[i + 1] - 1
        if b > a:
            segments.append((a, b))

    print(f"Total frames (metadata): {total_frames}")
    print(f"Found {len(segments)} segments between cuts.")

    # Filter segments:
    # Keep segments that are 1.5s to 12s long (good for phrases)
    # and have reasonable motion statistics.
    kept = []
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    for (a, b) in segments:
        dur = (b - a + 1) / (fps if fps > 0 else 30.0)
        if dur < 1.5 or dur > 12.0:
            continue

        ms = segment_score(cap, a, b, sample_step=5)
        if ms is None:
            continue
        mean, std = ms

        # Heuristic:
        # - crowd/replay/graphics often has large mean diff and/or very high variability
        # - piste shot tends to be moderate
        if mean < 3.0:
            continue
        if mean > 20.0:
            continue
        if std > 12.0:
            continue

        kept.append((a, b, dur, mean, std))

    cap.release()

    print(f"Kept {len(kept)} likely-fencing segments.")
    kept = sorted(kept, key=lambda x: x[2], reverse=True)[:10]  # top 10 by duration

    # Export them as clips
    for i, (a, b, dur, mean, std) in enumerate(kept, start=1):
        out = OUT_DIR / f"seg_{i:02d}_{dur:.1f}s_f{a}-{b}.mp4"
        print(f"[{i:02d}] {dur:.1f}s  mean={mean:.1f} std={std:.1f}  -> {out.name}")
        write_segment(VIDEO_PATH, out, a, b)

    print(f"Done. Wrote {len(kept)} clips to {OUT_DIR}")

if __name__ == "__main__":
    main()
