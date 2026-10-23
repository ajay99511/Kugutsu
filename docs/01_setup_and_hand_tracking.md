# Progress Log: Milestone 1 – Repo Setup & Webcam Hand Tracking

This document records what has been built so far and why, so the project is easy to follow and reproduce.

## 1. Goal

Control a simulated robot hand/gripper using hand gestures captured from a webcam. The plan is:

1. Track the hand (MediaPipe) -> 2. Simulate a robot (MuJoCo + Gymnasium) -> 3. Bridge hand to robot action -> 4. Record demos (LeRobot) -> 5. Train an ACT policy.

Only steps 1 and part of 3 (skeleton) exist so far.

## 2. Repository Structure

```
airobot/
├── teleop_input/          # Hand-tracking input drivers
│   ├── __init__.py        # Placeholder HandTracker interface (read/close)
│   └── hand_tracker.py    # Working webcam MediaPipe demo
├── sim_env/               # Simulation
│   ├── __init__.py        # Placeholder Gymnasium RobotEnv
│   └── assets/            # (empty) MuJoCo XML scenes go here
├── policies/              # Learning
│   ├── README.md
│   ├── scripts/           # (empty) LeRobot dataset scripts
│   └── checkpoints/       # (empty) ACT checkpoints, git-ignored
├── docs/                  # Documentation (this file)
├── main_teleop_loop.py    # Skeleton: hand state -> robot action loop
├── README.md              # Project overview + milestone checklist
└── .gitignore             # Ignores .venv, caches, checkpoints, datasets, .env
```

A git repository was initialised with `git init`. **Nothing has been committed yet.**

### Component status

| Component | Status |
|---|---|
| `teleop_input/hand_tracker.py` | Working demo (standalone script) |
| `teleop_input/__init__.py` | Placeholder class, not implemented |
| `sim_env/__init__.py` | Placeholder class, not implemented |
| `main_teleop_loop.py` | Skeleton; `hand_to_action()` raises `NotImplementedError` |
| `policies/` | Empty folders and README only |

## 3. Hand Tracker (`teleop_input/hand_tracker.py`)

The script opens the webcam, mirrors the image, and runs MediaPipe Hands (1 hand, detection confidence 0.7). For each frame with a detected hand it computes a 4-value control vector `[x, y, z, gripper]`:

- **XYZ target:** landmark 9 (middle-finger base, the palm centre). `x` and `y` are normalised image coordinates (0–1); `z` is MediaPipe's relative depth.
- **Gripper openness:** distance between thumb tip (landmark 4) and index tip (landmark 8), mapped as `clip((dist - 0.03) / 0.12, 0, 1)`. `0.0` = pinched/grabbing, `1.0` = open.
- **Smoothing:** exponential filter `state = ALPHA * raw + (1 - ALPHA) * state` with `ALPHA = 0.3` (lower = smoother but laggier).
- **Overlay:** draws landmarks, the XYZ vector, and the gripper value with status (`GRABBING` if below 0.2, otherwise `OPEN`). Press `q` to quit.

Run it from the project root:

```powershell
.\.venv\Scripts\python.exe teleop_input\hand_tracker.py
```

The last run exited cleanly (code 0).

## 4. Environment Setup

- Python **3.10.19** virtual environment at `.venv/`.
- The venv initially had no `pip`; it was added with `python -m ensurepip --upgrade`.
- Packages: `mediapipe==0.10.14`, `opencv-contrib-python==4.11.0.86`, `numpy` 2.2.6 (plus dependencies such as `jax`, `protobuf` 4.25.9).

### Problems hit and how they were fixed

| Problem | Cause | Fix |
|---|---|---|
| `No module named pip` | The venv was created without pip | `ensurepip --upgrade` |
| `mp.solutions` missing | Latest `mediapipe` (1.1.1) removed the legacy `solutions` API the script uses | Pinned `mediapipe==0.10.14` |
| `cv2.imshow` error ("not implemented... GTK+ 2.x or Cocoa") | `opencv-python`, `opencv-python-headless` and `opencv-contrib-python` were all installed. They share one `cv2` module and overwrote each other, leaving a build with no GUI | Uninstalled all OpenCV packages and installed only `opencv-contrib-python==4.11.0.86` (Win32 UI confirmed) |

> **Warning:** Do not `pip install opencv-python` in this venv. It will clash with `opencv-contrib-python` (required by MediaPipe) and bring the GUI error back.

### Harmless console messages

These appear on each run and can be ignored:
- `Created TensorFlow Lite XNNPACK delegate for CPU`
- `Feedback manager requires a model with a single signature inference`
- protobuf `SymbolDatabase.GetPrototype() is deprecated`

## 5. Known Limitations / Notes

- `hand_tracker.py` runs its loop at import time, so `main_teleop_loop.py` cannot import it yet. It needs refactoring into a `HandTracker` class with `read()` returning the 4-value state and `close()` releasing the camera.
- The legacy `mp.solutions` API is pinned to an old MediaPipe release. A later migration to the MediaPipe Tasks `HandLandmarker` is recommended.
- `x`/`y` are image-space values, not yet mapped to robot workspace coordinates.
- No `requirements.txt` exists yet, so the environment is not reproducible from the repo alone.

## 6. Next Steps

1. Refactor `hand_tracker.py` into a reusable `HandTracker` class.
2. Add `requirements.txt` (`mediapipe==0.10.14`, `opencv-contrib-python==4.11.0.86`, `numpy`, then `mujoco`, `gymnasium`, `lerobot`).
3. Build a MuJoCo scene in `sim_env/assets/` and implement `RobotEnv`.
4. Implement `hand_to_action()` (workspace scaling, gripper mapping) and run the full teleop loop.
5. Make the first git commit and push to GitHub.
6. Record demonstrations with LeRobot and train an ACT policy.
