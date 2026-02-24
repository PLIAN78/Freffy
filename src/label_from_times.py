import csv
from pathlib import Path
import cv2

CLIPS_DIR = Path("outputs/stabilized")
LABELS_PATH = Path("labels/phrases.csv")
LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

HEADER = [
    "clip","fps","y_first_action_is_R",
    "first_action_frame","touch_frame",
    "first_action_time_sec","touch_time_sec",
    "notes"
]

def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps if fps and fps > 0 else 30.0)

def ensure_csv():
    if not LABELS_PATH.exists():
        with open(LABELS_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(HEADER)

def already_labeled() -> set[str]:
    if not LABELS_PATH.exists():
        return set()
    with open(LABELS_PATH, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return set(row["clip"].strip() for row in r if row.get("clip"))

def append_row(row: dict):
    ensure_csv()
    with open(LABELS_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([row.get(h, "") for h in HEADER])

def main():
    ensure_csv()
    labeled = already_labeled()
    clips = sorted(CLIPS_DIR.glob("*.mp4"))

    if not clips:
        print(f"No clips found in {CLIPS_DIR.resolve()}")
        return

    print("Label by TIMES (seconds) while watching in VLC.")
    print("For each clip, enter:")
    print("  y_first_action_is_R: 0 (Left initiated) or 1 (Right initiated)")
    print("  first_action_time_sec, touch_time_sec (e.g. 1.23)")
    print("Notes: tags like good_clean,zoom_in,track_swap\n")

    for p in clips:
        clip = p.stem
        if clip in labeled:
            continue

        fps = get_fps(p)
        print(f"\n--- {clip} ---  fps={fps:.2f}")
        print(f"Open this file in VLC: {p}")

        y = input("Initiator? (0=Left, 1=Right, Enter to skip): ").strip()
        if y == "":
            print("Skipped.")
            continue
        if y not in ("0", "1"):
            print("Invalid. Skipped.")
            continue

        fa_t = input("First action time (sec): ").strip()
        touch_t = input("Touch time (sec): ").strip()
        notes = input("Notes/tags (optional): ").strip()

        try:
            fa_t_f = float(fa_t)
            touch_t_f = float(touch_t)
        except ValueError:
            print("Bad time input. Skipped.")
            continue

        fa_frame = int(round(fa_t_f * fps))
        touch_frame = int(round(touch_t_f * fps))

        row = {
            "clip": clip,
            "fps": fps,
            "y_first_action_is_R": int(y),
            "first_action_frame": fa_frame,
            "touch_frame": touch_frame,
            "first_action_time_sec": fa_t_f,
            "touch_time_sec": touch_t_f,
            "notes": notes,
        }
        append_row(row)
        labeled.add(clip)
        print(f"Saved -> {LABELS_PATH}  (fa_frame={fa_frame}, touch_frame={touch_frame})")

    print("\nDone.")

if __name__ == "__main__":
    main()