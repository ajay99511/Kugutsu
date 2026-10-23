"""Bridge: hand state (teleop_input) -> robot action (sim_env)."""

import numpy as np

from teleop_input import HandTracker
from sim_env import RobotEnv


def hand_to_action(hand_state: np.ndarray) -> np.ndarray:
    """Maps normalized [x, y, z, grip_open] (0..1) to robot workspace meters."""
    norm_x, norm_y, norm_z, grip_open = hand_state

    # Map screen X (0..1) -> Robot X (-0.25m to +0.25m left/right)
    robot_x = (norm_x - 0.5) * 0.5
    # Map palm depth Z (0..1) -> Robot Y (-0.20m to +0.20m forward/back)
    robot_y = (0.5 - norm_z) * 0.4
    # Map screen Y (0..1, top is 0) -> Robot Height Z (0.03m table to 0.35m high)
    robot_z = np.clip((1.0 - norm_y) * 0.35, 0.035, 0.35)

    # Invert gripper openness so 1.0 = closed fingers, 0.0 = open fingers
    grip_close = 1.0 - grip_open

    return np.array([robot_x, robot_y, robot_z, grip_close], dtype=np.float32)


def main():
    tracker = HandTracker(alpha=0.35, show_hud=True)
    env = RobotEnv()

    print("Teleop active! Move your hand to position the gripper, pinch to grab "
          "the red cube. Press 'q' on the camera window to quit.")
    try:
        while True:
            hand_state, tracker_ok = tracker.read()
            if not tracker_ok:
                break

            action = hand_to_action(hand_state)
            if not env.step(action):
                break
    finally:
        tracker.close()
        env.close()


if __name__ == "__main__":
    main()
