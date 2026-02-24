from pathlib import Path
import cv2

RAW_VIDEO = Path("data/raw/clip_001fixed.mp4")
CUTS_FILE = Path("outputs/cuts.txt")
OUT_DIR = Path("data/clips")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# More permissive settings
MIN_SEC = 2.0
MAX_SEC = 18.0

def read_cuts():
    cuts = []
    if not CUTS_FILE.exists():
        return cuts
    for line in CUTS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Expected format contains "frame=###"
        if "frame=" in line:
            try:
                part = line.split("frame=")[1]
                frame = int(part.split()[0])
                cuts.append(frame)
            except:
                pass
    cuts = sorted(set(cuts))
    return cuts

def main():
    cap = cv2.VideoCapture(str(RAW_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {RAW_VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cuts = read_cuts()
    # Add boundaries
    boundaries = [0] + cuts + [total]
    segments = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        dur = (b - a) / fps
        if MIN_SEC <= dur <= MAX_SEC:
            segments.append((a, b, dur))

    print(f"FPS={fps:.2f}  total_frames={total}")
    print(f"Found {len(segments)} candidate segments between cuts (MIN={MIN_SEC}s MAX={MAX_SEC}s).")

    # Write each segment out
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for i, (a, b, dur) in enumerate(segments, start=1):
        out_path = OUT_DIR / f"cand_{i:03d}_{dur:.1f}s_f{a}-{b}.mp4"
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        ok, frame0 = cap.read()
        if not ok:
            continue
        h, w = frame0.shape[:2]
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        writer.write(frame0)

        for _ in range(a + 1, b):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)

        writer.release()
        print(f"Wrote {out_path.name}")

    cap.release()
    print(f"Done. Candidate clips in {OUT_DIR}")

if __name__ == "__main__":
    main()