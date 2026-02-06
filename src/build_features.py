import numpy as np
import pandas as pd
from pathlib import Path

TRACK_DIR = Path("outputs/tracks")
LABELS_PATH = Path("labels/phrases.csv")
OUT_PATH = Path("outputs/train_features.csv")

PRE_WINDOW = 15  # frames before first action (~0.5s at 30fps)

def center(row, prefix):
    x, y, w, h = row[f"{prefix}_x"], row[f"{prefix}_y"], row[f"{prefix}_w"], row[f"{prefix}_h"]
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        return None
    return np.array([x + w/2.0, y + h/2.0], dtype=float)

def mean_max_speed(seg: pd.DataFrame, prefix: str):
    pts = []
    for _, r in seg.iterrows():
        c = center(r, prefix)
        if c is not None:
            pts.append(c)
    if len(pts) < 2:
        return 0.0, 0.0
    pts = np.array(pts)
    vel = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(np.mean(vel)), float(np.max(vel))

def main():
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_PATH}")

    labels = pd.read_csv(LABELS_PATH)

    rows = []
    for _, lab in labels.iterrows():
        clip = str(lab["clip"]).strip()
        track_path = TRACK_DIR / f"{clip}.csv"
        if not track_path.exists():
            print(f"Skipping (no tracks): {clip}")
            continue

        if pd.isna(lab.get("first_action_frame")) or pd.isna(lab.get("first_action_fencer")):
            print(f"Skipping (missing label): {clip}")
            continue

        first_frame = int(lab["first_action_frame"])
        fencer = str(lab["first_action_fencer"]).strip().upper()
        y = 0 if fencer == "L" else 1 if fencer == "R" else None
        if y is None:
            print(f"Skipping (bad first_action_fencer): {clip}")
            continue

        df = pd.read_csv(track_path)

        start = max(0, first_frame - PRE_WINDOW)
        end = min(len(df), first_frame)
        seg = df.iloc[start:end]
        if len(seg) < 5:
            print(f"Skipping (too short pre-window): {clip}")
            continue

        L_mean, L_max = mean_max_speed(seg, "L")
        R_mean, R_max = mean_max_speed(seg, "R")

        # distance closing (positive means they got closer)
        dists = []
        for _, r in seg.iterrows():
            Lc = center(r, "L")
            Rc = center(r, "R")
            if Lc is not None and Rc is not None:
                dists.append(np.linalg.norm(Lc - Rc))
        if len(dists) >= 2:
            distance_closing = float(dists[0] - dists[-1])
        else:
            distance_closing = 0.0

        rows.append({
            "clip": clip,
            "y_first_action_is_R": y,  # 0 = L, 1 = R
            "L_speed_mean": L_mean,
            "L_speed_max": L_max,
            "R_speed_mean": R_mean,
            "R_speed_max": R_max,
            "distance_closing": distance_closing,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved training features to: {OUT_PATH} (rows={len(out_df)})")

if __name__ == "__main__":
    main()
