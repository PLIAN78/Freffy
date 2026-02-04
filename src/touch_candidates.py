import numpy as np
import pandas as pd
from pathlib import Path

STAB_DIR = Path("outputs/stabilized")
TRACK_DIR = Path("outputs/tracks")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Tuning knobs ---
SMOOTH_WIN = 9
TOP_K = 3                 # return top K candidates per clip
MIN_VALID_BOX = 10        # bbox width/height must be >= this

def moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")

def center_xy(row, prefix):
    x, y, w, h = row[f"{prefix}_x"], row[f"{prefix}_y"], row[f"{prefix}_w"], row[f"{prefix}_h"]
    if w < MIN_VALID_BOX or h < MIN_VALID_BOX or x < 0 or y < 0:
        return None
    return np.array([x + w/2.0, y + h/2.0], dtype=float)

def main():
    vids = sorted(STAB_DIR.glob("*.mp4"))
    if not vids:
        print(f"No stabilized videos found in {STAB_DIR.resolve()}")
        return

    rows_out = []

    for v in vids:
        csv = TRACK_DIR / f"{v.stem}.csv"
        if not csv.exists():
            print(f"Missing track CSV for {v.name}")
            continue

        df = pd.read_csv(csv)

        # distance between fencers over time
        d = []
        for _, r in df.iterrows():
            Lc = center_xy(r, "L")
            Rc = center_xy(r, "R")
            if Lc is None or Rc is None:
                d.append(np.nan)
            else:
                d.append(float(np.linalg.norm(Lc - Rc)))

        d = np.array(d, dtype=float)
        # fill gaps
        if np.all(np.isnan(d)):
            print(f"{v.stem}: no valid distance measurements")
            continue
        # simple interpolation for NaNs
        nans = np.isnan(d)
        if np.any(nans):
            idx = np.arange(len(d))
            d[nans] = np.interp(idx[nans], idx[~nans], d[~nans])

        d_s = moving_avg(d, SMOOTH_WIN)

        # "closing speed" = negative derivative (bigger => closing faster)
        closing = -(np.gradient(d_s))
        closing_s = moving_avg(closing, SMOOTH_WIN)

        # score: emphasize strong closing events
        score = closing_s
        # pick top K peaks (basic)
        peak_idx = np.argsort(score)[::-1][:TOP_K]

        fps = float(df["time_sec"].iloc[1] - df["time_sec"].iloc[0]) if len(df) > 1 else (1/30.0)
        # careful: df already has time_sec; use it
        for rank, i in enumerate(peak_idx, start=1):
            t = float(df["time_sec"].iloc[i]) if "time_sec" in df.columns else float(i * fps)
            rows_out.append({
                "clip": v.stem,
                "candidate_rank": rank,
                "candidate_frame": int(df["frame"].iloc[i]) if "frame" in df.columns else int(i),
                "candidate_time_sec": t,
                "score": float(score[i]),
                "distance_px": float(d_s[i]),
            })

        print(f"{v.stem}: touch candidates @ frames {sorted([int(df['frame'].iloc[i]) for i in peak_idx])}")

    out = OUT_DIR / "touch_candidates.csv"
    pd.DataFrame(rows_out).to_csv(out, index=False)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
