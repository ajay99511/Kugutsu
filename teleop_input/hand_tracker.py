import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, camera_idx: int = 0, alpha: float = 0.3, show_hud: bool = True):
        self.cap = cv2.VideoCapture(camera_idx)
        self.alpha = alpha
        self.show_hud = show_hud
        
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        # Default neutral state: centered [0.5, 0.5, 0.5] and open gripper [1.0]
        self.smoothed_state = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)

    def read(self) -> tuple[np.ndarray, bool]:
        """Returns (smoothed_state [x, y, z, grip], is_running)."""
        success, frame = self.cap.read()
        if not success:
            return self.smoothed_state, False

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            if self.show_hud:
                self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

            # 1. Palm center (Landmark 9) for X and Y
            palm = hand.landmark[9]
            wrist = hand.landmark[0]

            # 2. Apparent hand size (Wrist to Middle Base distance) as a rock-solid Z proxy
            palm_size = np.linalg.norm([palm.x - wrist.x, palm.y - wrist.y])
            z_norm = np.clip((palm_size - 0.10) / 0.20, 0.0, 1.0)

            # 3. Gripper pinch (Thumb Tip 4 to Index Tip 8)
            thumb = np.array([hand.landmark[4].x, hand.landmark[4].y])
            index = np.array([hand.landmark[8].x, hand.landmark[8].y])
            pinch_dist = np.linalg.norm(thumb - index)
            gripper_openness = np.clip((pinch_dist - 0.03) / 0.12, 0.0, 1.0)

            raw_state = np.array([palm.x, palm.y, z_norm, gripper_openness], dtype=np.float32)
            self.smoothed_state = (self.alpha * raw_state) + ((1.0 - self.alpha) * self.smoothed_state)

        if self.show_hud:
            x, y, z, grip = self.smoothed_state
            status = "GRABBING" if grip < 0.25 else "OPEN"
            cv2.putText(frame, f"XYZ: [{x:.2f}, {y:.2f}, {z:.2f}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Grip: {grip:.2f} ({status})", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("AeroBot - Hand Teleop HUD", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return self.smoothed_state, False

        return self.smoothed_state.copy(), True

    def close(self):
        self.hands.close()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = HandTracker()
    while True:
        state, running = tracker.read()
        if not running:
            break
    tracker.close()