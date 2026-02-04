import cv2
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("outputs/frames")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_NAME = "clip_001fixed.mp4"
VIDEO_PATH = RAW_DIR / VIDEO_NAME

def save_jpg(stem: str, idx: int, frame):
    out_path = OUT_DIR / f"{stem}_frame_{idx}.jpg"
    cv2.imwrite(str(out_path), frame)
    print(f"Saved: {out_path}")

def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Could not find: {VIDEO_PATH.resolve()}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the video. (Bad path or unsupported codec)")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps and fps > 0 else None

    print("--- Video Info ---")
    print(f"File: {VIDEO_PATH}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames (metadata): {frame_count}")
    if duration_sec is not None:
        print(f"Duration (metadata): {duration_sec:.2f} seconds")

    stem = VIDEO_PATH.stem

    # 1) Save frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read frame 0.")
    save_jpg(stem, 0, frame0)

    # 2) Save middle frame (only if metadata looks valid)
    if frame_count and frame_count > 0:
        mid = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, framem = cap.read()
        if ok:
            save_jpg(stem, mid, framem)
        else:
            print(f"Could not read middle frame {mid} (will skip).")
    else:
        print("Frame count metadata invalid; skipping middle frame.")

    # 3) Save last readable frame by scanning forward (robust)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    last_ok_frame = None
    last_ok_idx = -1
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last_ok_frame = frame
        last_ok_idx = idx
        idx += 1

    if last_ok_frame is None:
        raise RuntimeError("Could not read any frames while scanning.")

    save_jpg(stem, last_ok_idx, last_ok_frame)
    cap.release()

    print(f"Done. Saved 3 frames (0, mid, last={last_ok_idx}).")

if __name__ == "__main__":
    main()
