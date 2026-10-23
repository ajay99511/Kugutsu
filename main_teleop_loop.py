"""Bridge: hand state (teleop_input) -> robot action (sim_env)."""

from teleop_input import HandTracker
from sim_env import RobotEnv


def hand_to_action(hand_state):
    """Map a hand state to a robot action. TODO: implement retargeting."""
    raise NotImplementedError


def main():
    tracker = HandTracker()
    env = RobotEnv()
    obs, _ = env.reset()
    try:
        while True:
            hand_state = tracker.read()
            action = hand_to_action(hand_state)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        tracker.close()
        env.close()


if __name__ == "__main__":
    main()
