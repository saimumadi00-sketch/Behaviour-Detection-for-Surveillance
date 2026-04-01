# Real-Time Human Action Detection — Public Safety
**400-Level ML Course | Neural Networks & Computer Vision**

## Project Structure
```
action_detection/
├── main.py            ← entry point — run this to start the webcam feed
├── model.py           ← LSTM classifier architecture (fill in your training)
├── collect_data.py    ← data collection helper (record your own keypoints)
├── requirements.txt   ← dependencies
└── data/              ← auto-created when you run collect_data.py
```

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the live detection
python main.py
```

## Development Roadmap
1. **Run main.py** — confirm webcam + MediaPipe skeleton works
2. **Collect data** — `python collect_data.py --action FALLING --samples 100`
3. **Train model** — use `model.py` ActionLSTM on your collected data
4. **Plug model in** — swap `PlaceholderClassifier` in `main.py` with your trained `ActionLSTM`
5. **Evaluate** — confusion matrix, frame-level F1, latency benchmarks

## Pipeline
```
Webcam → MediaPipe BlazePose (33 joints × 4 values = 132 features)
       → Sliding window buffer (30 frames)
       → LSTM classifier  →  Alert Engine  →  Annotated display
```

## Key Files to Edit
- `CONFIG` dict in `main.py` — change labels, thresholds, camera index
- `extract_keypoints()` in `main.py` — add joint angles, velocities
- `ActionLSTM` in `model.py` — tune hidden size, layers, dropout
