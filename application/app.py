import cv2
import mediapipe as mp
import time
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import pyautogui

# Load the pre-trained model and scaler
model = load_model('model/hand_model.h5')
scaler = joblib.load('model/scaler.pkl')  # Corrected extension

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Prepare for FPS calculation
prev_time = 0

# Define a custom threshold for predictions
prediction_threshold = 0.99

# Screen size for mapping hand coordinates
screen_width, screen_height = pyautogui.size()

# Initialize mouse control variables
mouse_is_down = False

cap = cv2.VideoCapture(1)

with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.3) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB, flip the image for a laterally correct view, and process it with MediaPipe Hands
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get landmark 8 (index finger tip) coordinates
                lm8 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                lm8_x, lm8_y = int(lm8.x * screen_width), int(lm8.y * screen_height)
                
                # Move the mouse to the position of landmark 8
                pyautogui.moveTo(lm8_x, lm8_y)

                # Extract landmarks for prediction
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                # Normalize the landmarks
                landmarks = scaler.transform([landmarks])
                # Make a prediction
                prediction = model.predict(landmarks)
                
                # Use threshold to determine the label and control the mouse click
                if prediction[0][0] > prediction_threshold and not mouse_is_down:
                    pyautogui.mouseDown()
                    mouse_is_down = True
                elif prediction[0][0] <= prediction_threshold and mouse_is_down:
                    pyautogui.mouseUp()
                    mouse_is_down = False

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Display the image
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
