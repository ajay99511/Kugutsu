import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Smoothing factor (0.1 = very smooth/laggy, 0.9 = very responsive/jittery)
ALPHA = 0.3
smoothed_state = np.zeros(4) # [x, y, z, pinch]

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip image for mirror-like teleoperation feel
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # 1. Extract Wrist/Palm root (Landmark 0 & 9) for XYZ position
            palm = hand.landmark[9]
            
            # 2. Extract Thumb Tip (4) and Index Tip (8) for Gripper Pinch
            thumb = np.array([hand.landmark[4].x, hand.landmark[4].y])
            index = np.array([hand.landmark[8].x, hand.landmark[8].y])
            pinch_dist = np.linalg.norm(thumb - index)
            
            # Normalize pinch: 0.0 (closed/grabbing) to 1.0 (open)
            gripper_openness = np.clip((pinch_dist - 0.03) / 0.12, 0.0, 1.0)

            raw_state = np.array([palm.x, palm.y, palm.z, gripper_openness])
            smoothed_state = (ALPHA * raw_state) + ((1 - ALPHA) * smoothed_state)

            # Display the live robotics control vector on screen
            x, y, z, grip = smoothed_state
            status = "GRABBING" if grip < 0.2 else "OPEN"
            cv2.putText(frame, f"Target XYZ: [{x:.2f}, {y:.2f}, {z:.2f}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Gripper: {grip:.2f} ({status})", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("AI Robotics Hand Teleop Node", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
