import cv2
import numpy as np
import pandas as pd
from pathlib import Path

STAB_DIR = Path("outputs/stabilized")
TRACK_DIR = Path("outputs/tracks")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Tuning knobs (start here, tweak later) ---
MOTION_THRESH = 18         # threshold on per-frame motion energy (normalized)
SMOOTH_WIN = 7             # moving average window (frames)
SUSTAIN_FRAMES = 8         # must stay above threshold this many frames (~0.25s at 30fps)
MIN_BOX_AREA = 2000        # ignore tiny boxes

def moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")

def safe_crop(gray, box):
    x, y, w, h = box
    x = int(max(0, x)); y = int(max(0, y))
    w = int(max(0, w)); h = int(max(0, h))
    if w <= 1 or h <= 1:
        return None
    if x + w > gray.shape[1] or y + h > gray.shape[0]:
        w = min(w, gray.shape[1] - x)
        h = min(h, gray.shape[0] - y)
    if w <= 1 or h <= 1:
        return None
    return gray[y:y+h, x:x+w]

def motion_energy(prev_gray, curr_gray, box):
    crop1 = safe_crop(prev_gray, box)
    crop2 = safe_crop(curr_gray, box)
    if crop1 is None or crop2 is None:
        return 0.0
    if crop1.size < MIN_BOX_AREA or crop2.size < MIN_BOX_AREA:
        return 0.0
    diff = cv2.absdiff(crop2, crop1)
    # normalized mean abs diff
    return float(np.mean(diff)) / 255.0

def first_sustained_crossing(signal: np.ndarray, thresh: float, sustain: int):
    """Return first index where signal stays >= thresh for sustain frames."""
    n = len(signal)
    for i in range(0, n - sustain):
        if np.all(signal[i:i+sustain] >= thresh):
            return i
    return None

def process_one(stab_video: Path, track_csv: Path):
    df = pd.read_csv(track_csv)

    cap = cv2.VideoCapture(str(stab_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {stab_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    ok, prev = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    L_energy = []
    R_energy = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # align with df rows (use min length)
        if frame_idx >= len(df):
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        row = df.iloc[frame_idx]
        L_box = (row["L_x"], row["L_y"], row["L_w"], row["L_h"])
        R_box = (row["R_x"], row["R_y"], row["R_w"], row["R_h"])

        L_energy.append(motion_energy(prev_gray, curr_gray, L_box))
        R_energy.append(motion_energy(prev_gray, curr_gray, R_box))

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()

    L_energy = np.array(L_energy, dtype=float)
    R_energy = np.array(R_energy, dtype=float)

    L_s = moving_avg(L_energy, SMOOTH_WIN)
    R_s = moving_avg(R_energy, SMOOTH_WIN)

    L_i = first_sustained_crossing(L_s, MOTION_THRESH, SUSTAIN_FRAMES)
    R_i = first_sustained_crossing(R_s, MOTION_THRESH, SUSTAIN_FRAMES)

    # Decide winner
    if L_i is None and R_i is None:
        first = "unknown"
        t = None
    elif R_i is None or (L_i is not None and L_i < R_i):
        first = "L"
        t = L_i / fps
    else:
        first = "R"
        t = R_i / fps

    return {
        "clip": stab_video.stem,
        "fps": float(fps),
        "frames_used": int(min(len(df), frame_idx)),
        "first_action": first,
        "first_action_time_sec": (None if t is None else float(t)),
        "L_peak": float(np.max(L_s)) if len(L_s) else None,
        "R_peak": float(np.max(R_s)) if len(R_s) else None,
        "threshold": float(MOTION_THRESH),
        "sustain_frames": int(SUSTAIN_FRAMES),
        "smooth_win": int(SMOOTH_WIN),
    }

def main():
    vids = sorted(STAB_DIR.glob("*.mp4"))
    if not vids:
        print(f"No stabilized videos found in {STAB_DIR.resolve()}")
        return

    rows = []
    for v in vids:
        csv = TRACK_DIR / f"{v.stem}.csv"
        if not csv.exists():
            print(f"Missing track CSV for {v.name}: expected {csv}")
            continue

        print(f"First-action baseline: {v.name}")
        rows.append(process_one(v, csv))

    out = OUT_DIR / "first_action_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
