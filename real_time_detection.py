import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
from collections import deque

# Load trained model
model = tf.keras.models.load_model('sign_language_model.h5')
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Smoothing predictions
prediction_queue = deque(maxlen=10)

# Set webcam resolution
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Extract hand landmarksq
def extract_landmarks(results):
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
    return np.array(landmarks)

# Draw UI
def draw_ui(frame, prediction, confidence):
    # Prediction display
    cv2.putText(frame, f'Prediction: {prediction}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if confidence >= 75 else (0, 0, 255), 2)

    # Confidence bar
    bar_width = min(int((confidence / 100) * 400), 400)
    cv2.rectangle(frame, (20, 100), (20 + bar_width, 130), (0, 255, 0), -1)
    cv2.rectangle(frame, (20, 100), (420, 130), (255, 255, 255), 2)
    cv2.putText(frame, f'Confidence: {confidence:.2f}%', (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    smoothed_prediction = "No prediction"
    confidence = 0

    # Hand detection and prediction
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = extract_landmarks(results)
        if landmarks.shape == (21, 3):
            landmarks = landmarks.reshape(1, 21, 3)
            predictions = model.predict(landmarks)
            confidence = np.max(predictions) * 100
            predicted_label = labels[np.argmax(predictions)]

            # Confidence threshold for predictions
            if confidence >= 75:
                prediction_queue.append(predicted_label)
                smoothed_prediction = max(set(prediction_queue), key=prediction_queue.count)

                # Announce high-confidence predictions
                engine.say(smoothed_prediction)
                engine.runAndWait()

    draw_ui(frame, smoothed_prediction, confidence)
    cv2.imshow('Sign2Speak: Real-Time Hand Gesture Recognition', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
