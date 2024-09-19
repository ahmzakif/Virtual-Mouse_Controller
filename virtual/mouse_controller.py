import cv2
import mediapipe as mp
import pyautogui
import numpy as np

class MouseController:
    def __init__(self, sensitivity=1.0):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.screen_width, self.screen_height = pyautogui.size()
        self.cap = cv2.VideoCapture(0)
        self.previous_finger_tip_distance = None
        self.click_distance_threshold = 20
        self.sensitivity = sensitivity

    def adjust_sensitivity(self, value):
        self.sensitivity = max(0.1, min(2.0, self.sensitivity + value))
        print(f"Sensitivity adjusted to: {self.sensitivity}")

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return frame, results

    def calculate_distance(self, point1, point2, frame_width, frame_height):
        x1, y1 = point1.x * frame_width, point1.y * frame_height
        x2, y2 = point2.x * frame_width, point2.y * frame_height
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def detect_click(self, distance):
        if distance < self.click_distance_threshold:
            if self.previous_finger_tip_distance and self.previous_finger_tip_distance >= self.click_distance_threshold:
                pyautogui.click()
            self.previous_finger_tip_distance = distance
        else:
            self.previous_finger_tip_distance = distance

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, results = self.process_frame(frame)
            frame_height, frame_width, _ = frame.shape

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

                    cursor_x = min(int(index_finger_tip.x * self.screen_width * self.sensitivity), self.screen_width - 1)
                    cursor_y = min(int(index_finger_tip.y * self.screen_height * self.sensitivity), self.screen_height - 1)

                    pyautogui.moveTo(cursor_x, cursor_y)

                    distance = self.calculate_distance(index_finger_tip, thumb_tip, frame_width, frame_height)
                    self.detect_click(distance)

            cv2.imshow('Virtual Mouse', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):
                self.adjust_sensitivity(0.1)
            elif key == ord('-'):
                self.adjust_sensitivity(-0.1)

        self.cleanup()

    def cleanup(self):
        self.hands.close()
        self.cap.release()
        cv2.destroyAllWindows()