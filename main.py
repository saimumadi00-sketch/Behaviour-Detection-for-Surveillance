"""
=============================================================
 Real-Time Human Action Detection for Public Safety
=============================================================
 Course  : 400-Level ML — Behavior Detection (Neural Networks + CV)
 Author  : [Your Name]

 PIPELINE
 ────────
   Webcam → Preprocessing → MediaPipe Pose (33 keypoints)
       → Feature Extraction → Sequence Buffer (sliding window)
       → Action Classifier (LSTM — your model goes here)
       → Alert Engine → Annotated Live Feed

 FIXES APPLIED
 ─────────────
   FIX-1  Classifier throttled — runs every CLASSIFY_EVERY frames, not every frame
   FIX-2  Webcam disconnect handled — retry limit + clean exit instead of infinite loop
   FIX-3  FPS measured before classification to reflect true capture rate
   FIX-4  cap.set() results verified with cap.get() and warned if ignored
   FIX-5  try/except wraps the main loop — errors log instead of crashing the stream
   FIX-6  Alert cooldown added — alert clears after ALERT_COOLDOWN_FRAMES frames
   FIX-7  Unused import `time` removed from collect_data.py
=============================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Uncomment when you plug in your trained model:
# import torch
# from model import ActionLSTM


# ─────────────────────────────────────────────────────────
#  CONFIG  — edit these as your project grows
# ─────────────────────────────────────────────────────────
CONFIG = {
    "camera_index":          0,       # 0 = default webcam
    "frame_width":           640,
    "frame_height":          480,
    "sequence_length":       30,      # frames fed into the classifier
    "detection_confidence":  0.6,
    "tracking_confidence":   0.6,
    "alert_threshold":       0.80,    # confidence to trigger alert

    # FIX-1: Only classify every N frames (reduces CPU load ~10x)
    "classify_every":        5,

    # FIX-6: How many frames the alert banner stays visible after triggering
    "alert_cooldown_frames": 60,      # ~2 seconds at 30 FPS

    # FIX-2: Max consecutive failed frame reads before giving up
    "max_read_failures":     30,

    # Classes your classifier will predict
    "action_labels": [
        "IDLE", "WALKING", "RUNNING",
        "FALLING", "FIGHTING", "WAVING",
    ],

    # BGR colours
    "color_safe":  (0, 220, 100),    # green
    "color_alert": (0, 60, 255),     # red
    "color_info":  (255, 200, 0),    # amber
}


# ─────────────────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles


# ─────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────
def extract_keypoints(results) -> np.ndarray:
    """
    Returns a flat (132,) array: x, y, z, visibility for each
    of the 33 BlazePose landmarks.
    Extend this with joint angles, inter-joint distances, etc.
    """
    if results.pose_landmarks:
        lms = results.pose_landmarks.landmark
        return np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in lms]).flatten()
    return np.zeros(33 * 4)


def joint_angle(a, b, c) -> float:
    """Angle in degrees at joint B given 2-D points A, B, C."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# ─────────────────────────────────────────────────────────
#  PLACEHOLDER CLASSIFIER
#  TODO: Replace with your trained LSTM / ST-GCN model
# ─────────────────────────────────────────────────────────
class PlaceholderClassifier:
    """
    Randomly samples class probabilities so the full pipeline
    runs before your real model is ready.

    SWAP with something like:
        self.model = ActionLSTM()
        self.model.load_state_dict(torch.load("action_lstm.pt"))
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(sequence).unsqueeze(0).float())
            probs  = torch.softmax(logits, dim=-1).squeeze().numpy()
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def predict(self, sequence: np.ndarray):
        """sequence: (seq_len, feature_dim) → (label_idx, confidence)"""
        probs     = np.random.dirichlet(np.ones(self.num_classes))
        label_idx = int(np.argmax(probs))
        return label_idx, float(probs[label_idx])


# ─────────────────────────────────────────────────────────
#  HUD OVERLAY
# ─────────────────────────────────────────────────────────
def draw_hud(frame, label, conf, fps, alert, cfg):
    h, w = frame.shape[:2]
    color = cfg["color_alert"] if alert else cfg["color_safe"]

    # Top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 52), (10, 10, 20), -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"ACTION: {label}  ({conf*100:.1f}%)",
                (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:.1f}",
                (w - 110, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                cfg["color_info"], 1, cv2.LINE_AA)

    # Alert banner (FIX-6: driven by cooldown counter, not raw bool)
    if alert:
        ban = frame.copy()
        cv2.rectangle(ban, (0, h - 58), (w, h), (0, 0, 160), -1)
        cv2.addWeighted(ban, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"!! ALERT — {label} DETECTED !!",
                    (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────
def main():
    cfg        = CONFIG
    seq_len    = cfg["sequence_length"]
    labels     = cfg["action_labels"]
    buffer     = deque(maxlen=seq_len)
    clf        = PlaceholderClassifier(len(labels))

    cap = cv2.VideoCapture(cfg["camera_index"])

    # FIX-4: Verify cap.set() was actually applied
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["frame_height"])
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if actual_w != cfg["frame_width"] or actual_h != cfg["frame_height"]:
        print(f"[WARN] Camera resolution {actual_w:.0f}x{actual_h:.0f} "
              f"(requested {cfg['frame_width']}x{cfg['frame_height']} — camera ignored it)")

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam — check camera_index in CONFIG.")
        return

    print("[INFO] Running. Press Q to quit.")

    prev_t         = time.time()
    action         = "DETECTING..."
    conf           = 0.0
    alert          = False
    alert_cooldown = 0            # FIX-6: countdown frames
    frame_count    = 0            # FIX-1: throttle counter
    read_failures  = 0            # FIX-2: disconnect counter

    with mp_pose.Pose(
        min_detection_confidence=cfg["detection_confidence"],
        min_tracking_confidence=cfg["tracking_confidence"],
        model_complexity=1,
    ) as pose:

        while True:
            # FIX-3: Measure FPS at capture, before classification
            now    = time.time()
            fps    = 1.0 / (now - prev_t + 1e-6)
            prev_t = now

            ret, frame = cap.read()

            # FIX-2: Handle webcam disconnect gracefully
            if not ret:
                read_failures += 1
                if read_failures >= cfg["max_read_failures"]:
                    print("[ERROR] Too many consecutive read failures — camera disconnected.")
                    break
                continue
            read_failures = 0   # reset on successful read

            frame_count += 1
            frame = cv2.flip(frame, 1)

            # FIX-5: Wrap processing in try/except so one bad frame doesn't kill the stream
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = pose.process(rgb)
                rgb.flags.writeable = True

                # Draw skeleton
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                    )

                # Buffer keypoints every frame (cheap)
                buffer.append(extract_keypoints(results))

                # FIX-1: Classify only every N frames
                if len(buffer) == seq_len and frame_count % cfg["classify_every"] == 0:
                    idx, conf = clf.predict(np.array(buffer))
                    action    = labels[idx]

                    # FIX-6: Trigger cooldown instead of raw True/False
                    if (conf >= cfg["alert_threshold"]
                            and action in ("FALLING", "FIGHTING")):
                        alert_cooldown = cfg["alert_cooldown_frames"]

            except Exception as e:
                print(f"[WARN] Frame processing error: {e}")

            # FIX-6: Decrement cooldown; alert active while counter > 0
            if alert_cooldown > 0:
                alert_cooldown -= 1
            alert = alert_cooldown > 0

            frame = draw_hud(frame, action, conf, fps, alert, cfg)
            cv2.imshow("Public Safety — Action Detection  [Q = quit]", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
