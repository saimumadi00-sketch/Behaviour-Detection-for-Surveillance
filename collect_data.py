"""
collect_data.py  —  Keypoint Data Collection Helper
Run this to build your training dataset.
  Usage:
      python collect_data.py --action FALLING --samples 100
Records `seq_len` frames per sample, saves to data/<action>/<idx>.npy

FIXES APPLIED
─────────────
  FIX-7  Removed unused `import time`
  FIX-8  Sequence always saved with exactly SEQ_LEN frames (pad with zeros if
          camera drops frames mid-recording instead of saving short arrays)
  FIX-9  cap.release() now inside finally block — always runs even on KeyboardInterrupt
  FIX-10 Skeleton overlay shown during recording so user can verify pose is detected
"""
import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
from main import extract_keypoints, CONFIG

SEQ_LEN = CONFIG["sequence_length"]
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles


def collect(action: str, num_samples: int):
    os.makedirs(f"data/{action}", exist_ok=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print(f"[INFO] Collecting '{action}' — press SPACE to start each sample, Q to quit.")

    # FIX-9: release camera even if user Ctrl-C's
    try:
        with mp_pose.Pose(model_complexity=1) as pose:
            sample_idx = 0
            while sample_idx < num_samples:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)

                # Show skeleton on preview
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = pose.process(rgb)
                rgb.flags.writeable = True
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                    )

                cv2.putText(frame, f"Action: {action}  Sample {sample_idx+1}/{num_samples}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 100), 2)
                cv2.putText(frame, "Press SPACE to record",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
                cv2.imshow("Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    seq = []
                    for _ in range(SEQ_LEN):
                        ret, f = cap.read()
                        if not ret:
                            # FIX-8: pad with zeros on dropped frame
                            seq.append(np.zeros(33 * 4))
                            continue
                        f   = cv2.flip(f, 1)
                        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

                        # FIX-10: show skeleton during recording
                        results_rec = pose.process(rgb)
                        if results_rec.pose_landmarks:
                            mp_drawing.draw_landmarks(
                                f, results_rec.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                            )
                        cv2.putText(f, "RECORDING...", (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 60, 255), 2)
                        cv2.imshow("Data Collection", f)
                        cv2.waitKey(1)

                        kp = extract_keypoints(results_rec)
                        seq.append(kp)

                    # FIX-8: guarantee exact shape (SEQ_LEN, 132)
                    seq_array = np.array(seq)
                    if len(seq_array) < SEQ_LEN:
                        pad = np.zeros((SEQ_LEN - len(seq_array), 33 * 4))
                        seq_array = np.vstack([seq_array, pad])

                    np.save(f"data/{action}/{sample_idx}.npy", seq_array)
                    print(f"  Saved sample {sample_idx + 1}  shape={seq_array.shape}")
                    sample_idx += 1

    finally:
        # FIX-9: always release
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--action",  required=True)
    p.add_argument("--samples", type=int, default=50)
    args = p.parse_args()
    collect(args.action, args.samples)
