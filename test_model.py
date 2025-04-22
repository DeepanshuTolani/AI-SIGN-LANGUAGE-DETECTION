import numpy as np
import tensorflow as tf
import os

# Load the trained model
model = tf.keras.models.load_model('sign_language_model.h5')

# Define labels for A to Z
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Path to test data
test_data_path = 'data'
X_test, y_test = [], []

# Load test data for all letters
for label, sign in enumerate(labels):
    sign_path = os.path.join(test_data_path, sign)
    if os.path.exists(sign_path):
        for file in os.listdir(sign_path):
            if file.endswith('.npy'):
                data = np.load(os.path.join(sign_path, file))
                X_test.append(data)
                y_test.append(label)

# Convert to numpy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'\nTest Accuracy: {test_accuracy * 100:.2f}%')

# Predictions
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Sample predictions
print("\nSample Predictions:")
for i in range(10):
    print(f'Actual: {labels[y_test[i]]}, Predicted: {labels[predicted_labels[i]]}')

# Display counts of correct predictions per letter
print("\nCorrect predictions per letter:")
correct_counts = {label: 0 for label in labels}
for actual, predicted in zip(y_test, predicted_labels):
    if actual == predicted:
        correct_counts[labels[actual]] += 1

for letter, count in correct_counts.items():
    print(f'{letter}: {count} correct predictions')
