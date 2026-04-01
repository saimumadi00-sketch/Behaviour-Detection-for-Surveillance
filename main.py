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
=============================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Uncomment when you plug in your trained model:
# import torch
# from model import ActionClassifier   # <-- you will create this


# ─────────────────────────────────────────────────────────
#  CONFIG  — edit these as your project grows
# ─────────────────────────────────────────────────────────
CONFIG = {
    "camera_index":          0,      # 0 = default webcam
    "frame_width":           640,
    "frame_height":          480,
    "sequence_length":       30,     # frames fed into the classifier
    "detection_confidence":  0.6,
    "tracking_confidence":   0.6,
    "alert_threshold":       0.80,   # confidence to trigger alert

    # Classes your classifier will predict
    "action_labels": [
        "IDLE", "WALKING", "RUNNING",
        "FALLING", "FIGHTING", "WAVING",
    ],

    # BGR colours
    "color_safe":  (0, 220, 100),   # green
    "color_alert": (0, 60, 255),    # red
    "color_info":  (255, 200, 0),   # amber
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
        self.model = torch.load("action_lstm.pt")
        probs = self.model(torch.tensor(sequence).unsqueeze(0))
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

    # Alert banner
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
    cfg       = CONFIG
    seq_len   = cfg["sequence_length"]
    labels    = cfg["action_labels"]
    buffer    = deque(maxlen=seq_len)
    clf       = PlaceholderClassifier(len(labels))

    cap = cv2.VideoCapture(cfg["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["frame_height"])

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam — check camera_index in CONFIG.")
        return

    print("[INFO] Running. Press Q to quit.")
    prev_t, action, conf, alert = time.time(), "DETECTING...", 0.0, False

    with mp_pose.Pose(
        min_detection_confidence=cfg["detection_confidence"],
        min_tracking_confidence=cfg["tracking_confidence"],
        model_complexity=1,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

            # Buffer keypoints
            buffer.append(extract_keypoints(results))

            # Classify
            if len(buffer) == seq_len:
                idx, conf = clf.predict(np.array(buffer))
                action    = labels[idx]
                alert     = conf >= cfg["alert_threshold"] and \
                            action in ("FALLING", "FIGHTING")

            # FPS
            now    = time.time()
            fps    = 1.0 / (now - prev_t + 1e-6)
            prev_t = now

            frame = draw_hud(frame, action, conf, fps, alert, cfg)
            cv2.imshow("Public Safety — Action Detection  [Q = quit]", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
