import cv2
import math
import mediapipe as mp
from pynput.mouse import Button, Controller
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mouse = Controller()

def is_pinched(hand_landmarks):
    if hand_landmarks is not None:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        
        distance_thumb_index = math.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
        distance_thumb_middle = math.sqrt(
            (thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2)
        distance_thumb_ring = math.sqrt(
            (thumb_tip.x - ring_tip.x) ** 2 + (thumb_tip.y - ring_tip.y) ** 2)
        distance_thumb_pinky = math.sqrt(
            (thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2)
        
        if (distance_thumb_index < 0.02 and
            distance_thumb_middle < 0.02 and
            distance_thumb_ring < 0.02 and
            distance_thumb_pinky < 0.02):
            return True
    return False

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                if is_pinched(landmarks):
                    screenshot = pyautogui.screenshot()
                    screenshot.save('screenshot.png')
                    print("Fingers pinched! Screenshot taken.")
        
        cv2.imshow('Hand Tracking', frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
