import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Path to save data
data_dir = 'data'
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Create folders for each label
for label in labels:
    label_dir = os.path.join(data_dir, label)
    os.makedirs(label_dir, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

print("Available labels: ", labels)
sign = input("Enter the sign (A-Z) you want to collect data for: ").upper()
if sign not in labels:
    print("Invalid sign. Please enter a letter from A to Z.")
    cap.release()
    exit()

print(f"Collecting data for sign: {sign}")

# Frame counter for file names
frame_count = 0
max_frames = 1000

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Collect hand landmarks as a list of (x, y, z) points
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            # Save landmarks to a file
            np.save(os.path.join(data_dir, sign, f'{frame_count}.npy'), landmarks)
            frame_count += 1

    # Show webcam feed
    cv2.putText(frame, f'Collecting: {sign} ({frame_count}/{max_frames})', (10, 30),
                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Data Collection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Data collection for {sign} completed: {frame_count} frames saved.")
cap.release()
cv2.destroyAllWindows()
