"""
collect_data.py  —  Keypoint Data Collection Helper
Run this to build your training dataset.
  Usage:
      python collect_data.py --action FALLING --samples 100
Records `seq_len` frames per sample, saves to data/<action>.npy
"""
import cv2, mediapipe as mp, numpy as np, argparse, os, time
from main import extract_keypoints, CONFIG

SEQ_LEN = CONFIG["sequence_length"]
mp_pose = mp.solutions.pose

def collect(action: str, num_samples: int):
    os.makedirs(f"data/{action}", exist_ok=True)
    cap = cv2.VideoCapture(0)
    print(f"[INFO] Collecting '{action}' — press SPACE to start each sample, Q to quit.")

    with mp_pose.Pose(model_complexity=1) as pose:
        sample_idx = 0
        while sample_idx < num_samples:
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)

            cv2.putText(frame, f"Action: {action}  Sample {sample_idx+1}/{num_samples}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,100), 2)
            cv2.putText(frame, "Press SPACE to record",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord(" "):
                seq = []
                for _ in range(SEQ_LEN):
                    ret, f = cap.read()
                    if not ret: continue
                    rgb = cv2.cvtColor(cv2.flip(f, 1), cv2.COLOR_BGR2RGB)
                    kp  = extract_keypoints(pose.process(rgb))
                    seq.append(kp)
                np.save(f"data/{action}/{sample_idx}.npy", np.array(seq))
                print(f"  Saved sample {sample_idx+1}")
                sample_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--action",  required=True)
    p.add_argument("--samples", type=int, default=50)
    args = p.parse_args()
    collect(args.action, args.samples)
