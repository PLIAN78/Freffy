import cv2
import pandas as pd
from pathlib import Path

CLIPS_DIR = Path("outputs/tracks_debug")  # label stabilized clips (recommended)
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
import cv2
import pandas as pd
from pathlib import Path

CLIPS_DIR = Path("outputs/stabilized")
LABELS_PATH = Path("labels/phrases.csv")
LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

HELP = """
Controls:
  Space      : play/pause
  A / D      : step back/forward 1 frame
  J / L      : jump back/forward 10 frames
  1          : mark FIRST ACTION frame
  2          : mark TOUCH frame
  [          : set initiator = LEFT  (y=0)
  ]          : set initiator = RIGHT (y=1)

Notes tags (toggle on/off):
  G          : good_clean
  M          : messy_but_ok
  K          : skip_training
  Z          : zoom_in
  P          : pan_fast
  H          : shake
  O          : overlay
  E          : replay
  R          : ref_in_frame
  S          : track_swap
  F          : track_flicker
  U          : first_action_unclear
  T          : touch_unclear

  Q          : quit this clip (saves if you set any marks/label)
"""

TAG_KEYS = {
    ord('g'): "good_clean",
    ord('m'): "messy_but_ok",
    ord('k'): "skip_training",
    ord('z'): "zoom_in",
    ord('p'): "pan_fast",
    ord('h'): "shake",
    ord('o'): "overlay",
    ord('e'): "replay",
    ord('r'): "ref_in_frame",
    ord('s'): "track_swap",
    ord('f'): "track_flicker",
    ord('u'): "first_action_unclear",
    ord('t'): "touch_unclear",
}

def load_labels():
    if LABELS_PATH.exists():
        return pd.read_csv(LABELS_PATH)
    return pd.DataFrame(columns=[
        "clip", "fps", "y_first_action_is_R",
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

        # Skip already labeled clips
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
        y_first_action_is_R = None  # 0 or 1
        notes = set()

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
            vis = frame.copy()

            # Overlay
            cv2.putText(vis, f"{clip_name}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(vis, f"frame {frame_idx}/{max(total-1,0)}  time {frame_idx/fps:.2f}s",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if y_first_action_is_R is None:
                init_txt = "Initiator: (unset)  press [ for L, ] for R"
                init_color = (0, 200, 255)
            else:
                init_txt = f"Initiator: {'R' if y_first_action_is_R==1 else 'L'} (y={y_first_action_is_R})"
                init_color = (255,255,255)

            cv2.putText(vis, init_txt, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, init_color, 2)

            if first_action_frame is not None:
                cv2.putText(vis, f"FIRST ACTION: {first_action_frame} ({first_action_frame/fps:.2f}s)",
                            (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            if touch_frame is not None:
                cv2.putText(vis, f"TOUCH: {touch_frame} ({touch_frame/fps:.2f}s)",
                            (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)

            notes_txt = "notes: " + (",".join(sorted(notes)) if notes else "(none)")
            cv2.putText(vis, notes_txt[:120], (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

            cv2.putText(vis, "Space play/pause | A/D step | J/L +/-10 | 1 first | 2 touch | [ L | ] R | Q save+next",
                        (20, vis.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("Label Helper", vis)
            key = cv2.waitKey(30 if not paused else 0) & 0xFF

            if key == ord(' '):
                paused = not paused

            elif key == ord('a'):
                frame_idx = max(0, frame_idx - 1)
                ok, frame = read_frame(frame_idx)

            elif key == ord('d'):
                frame_idx = min(total - 1, frame_idx + 1)
                ok, frame = read_frame(frame_idx)

            elif key == ord('j'):
                frame_idx = max(0, frame_idx - 10)
                ok, frame = read_frame(frame_idx)

            elif key == ord('l'):
                frame_idx = min(total - 1, frame_idx + 10)
                ok, frame = read_frame(frame_idx)

            elif key == ord('1'):
                first_action_frame = frame_idx
                print(f"Marked FIRST ACTION @ frame {first_action_frame} ({first_action_frame/fps:.2f}s)")

            elif key == ord('2'):
                touch_frame = frame_idx
                print(f"Marked TOUCH @ frame {touch_frame} ({touch_frame/fps:.2f}s)")

            elif key == ord('['):  # Left initiates
                y_first_action_is_R = 0
                print("Initiator set to LEFT (y=0)")

            elif key == ord(']'):  # Right initiates
                y_first_action_is_R = 1
                print("Initiator set to RIGHT (y=1)")

            elif key in TAG_KEYS:
                tag = TAG_KEYS[key]
                if tag in notes:
                    notes.remove(tag)
                    print(f"Removed note tag: {tag}")
                else:
                    notes.add(tag)
                    print(f"Added note tag: {tag}")

            elif key == ord('q') or key == 27:
                # Save if anything meaningful is set
                if (first_action_frame is not None) or (touch_frame is not None) or (y_first_action_is_R is not None) or notes:
                    row = {
                        "clip": clip_name,
                        "fps": float(fps),
                        "y_first_action_is_R": (None if y_first_action_is_R is None else int(y_first_action_is_R)),
                        "first_action_frame": (None if first_action_frame is None else int(first_action_frame)),
                        "touch_frame": (None if touch_frame is None else int(touch_frame)),
                        "first_action_time_sec": (None if first_action_frame is None else float(first_action_frame / fps)),
                        "touch_time_sec": (None if touch_frame is None else float(touch_frame / fps)),
                        "notes": ",".join(sorted(notes))
                    }
                    labels = pd.concat([labels, pd.DataFrame([row])], ignore_index=True)
                    save_labels(labels)
                    print(f"Saved labels for {clip_name} -> {LABELS_PATH}")
                else:
                    print(f"No labels set for {clip_name}; nothing saved.")
                break

            # If playing, advance
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
