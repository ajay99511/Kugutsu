# ai-hand-robotics

Hand-tracking teleoperation of a simulated robot hand, built in public.

```
├── teleop_input/        # Webcam MediaPipe & future ESP32 IMU drivers
├── sim_env/             # MuJoCo XML scenes (assets/) & Gymnasium wrappers
├── policies/            # LeRobot dataset scripts & ACT model checkpoints
└── main_teleop_loop.py  # Bridge: hand state -> robot action
```

## Milestones
- [ ] Webcam hand tracking (MediaPipe)
- [ ] MuJoCo scene + Gymnasium env
- [ ] Teleop loop end-to-end
- [ ] LeRobot dataset recording
- [ ] ACT policy training
