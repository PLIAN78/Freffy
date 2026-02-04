import cv2
import pandas as pd
from pathlib import Path

CLIPS_DIR = Path("outputs/stabilized")   # label stabilized clips (recommended)
LABELS_PATH = Path("labels/phrases.csv")
LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

HELP = """
Controls:
  Space      : play/pause
  A / D      : step back/forward 1 frame
  J / L      : jump back/forward 10 frames
  1          : mark FIRST ACTION frame
  2          : mark TOUCH frame
  Q          : quit (saves current clip row if marks exist)

Tips:
- First action frame = first committed move forward / attack initiation (not tiny blade twitch)
- Touch frame = moment of contact / light (best guess)
"""

def load_labels():
    if LABELS_PATH.exists():
        return pd.read_csv(LABELS_PATH)
    return pd.DataFrame(columns=[
        "clip", "fps",
        "first_action_frame", "touch_frame",
        "first_action_time_sec", "touch_time_sec",
        "notes"
    ])

def save_labels(df):
    df.to_csv(LABELS_PATH, index=False)

def main():
    print(HELP)

    clips = sorted(CLIPS_DIR.glob("*.mp4"))
    if not clips:
        print(f"No videos found in {CLIPS_DIR.resolve()}")
        return

    labels = load_labels()
    labeled_set = set(labels["clip"].astype(str).tolist()) if len(labels) else set()

    print(f"Found {len(clips)} clip(s). Already labeled: {len(labeled_set)}")

    for clip_path in clips:
        clip_name = clip_path.stem
        if clip_name in labeled_set:
            print(f"Skipping (already labeled): {clip_name}")
            continue

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            print(f"Could not open {clip_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\nLabeling: {clip_name}  (frames={total}, fps={fps:.2f})")

        first_action_frame = None
        touch_frame = None

        paused = True
        frame_idx = 0

        def read_frame(idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            return ok, frame

        ok, frame = read_frame(frame_idx)
        if not ok:
            cap.release()
            continue

        while True:
            # overlay info
            vis = frame.copy()
            cv2.putText(vis, f"{clip_name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(vis, f"frame {frame_idx}/{max(total-1,0)}  time {frame_idx/fps:.2f}s",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if first_action_frame is not None:
                cv2.putText(vis, f"FIRST ACTION: {first_action_frame} ({first_action_frame/fps:.2f}s)",
                            (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            if touch_frame is not None:
                cv2.putText(vis, f"TOUCH: {touch_frame} ({touch_frame/fps:.2f}s)",
                            (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)

            cv2.putText(vis, "Space play/pause | A/D step | J/L +/-10 | 1 first | 2 touch | Q quit+save",
                        (20, vis.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Label Helper", vis)

            key = cv2.waitKey(30 if not paused else 0) & 0xFF

            if key == ord(' '):  # pause/play
                paused = not paused

            elif key == ord('a'):  # back 1
                frame_idx = max(0, frame_idx - 1)
                ok, frame = read_frame(frame_idx)

            elif key == ord('d'):  # forward 1
                frame_idx = min(total - 1, frame_idx + 1)
                ok, frame = read_frame(frame_idx)

            elif key == ord('j'):  # back 10
                frame_idx = max(0, frame_idx - 10)
                ok, frame = read_frame(frame_idx)

            elif key == ord('l'):  # forward 10
                frame_idx = min(total - 1, frame_idx + 10)
                ok, frame = read_frame(frame_idx)

            elif key == ord('1'):  # mark first action
                first_action_frame = frame_idx
                print(f"Marked FIRST ACTION at frame {first_action_frame} ({first_action_frame/fps:.2f}s)")

            elif key == ord('2'):  # mark touch
                touch_frame = frame_idx
                print(f"Marked TOUCH at frame {touch_frame} ({touch_frame/fps:.2f}s)")

            elif key == ord('q') or key == 27:  # quit or ESC
                # save row if at least one mark exists
                if first_action_frame is not None or touch_frame is not None:
                    row = {
                        "clip": clip_name,
                        "fps": float(fps),
                        "first_action_frame": (None if first_action_frame is None else int(first_action_frame)),
                        "touch_frame": (None if touch_frame is None else int(touch_frame)),
                        "first_action_time_sec": (None if first_action_frame is None else float(first_action_frame / fps)),
                        "touch_time_sec": (None if touch_frame is None else float(touch_frame / fps)),
                        "notes": ""
                    }
                    labels = pd.concat([labels, pd.DataFrame([row])], ignore_index=True)
                    save_labels(labels)
                    print(f"Saved labels for {clip_name} -> {LABELS_PATH}")
                else:
                    print(f"No marks set for {clip_name}; nothing saved.")
                break

            # if playing, advance
            if not paused and key == 255:
                frame_idx += 1
                if frame_idx >= total:
                    frame_idx = total - 1
                    paused = True
                ok, frame = read_frame(frame_idx)
                if not ok:
                    paused = True

        cap.release()

    cv2.destroyAllWindows()
    print("\nDone labeling.")

if __name__ == "__main__":
    main()
