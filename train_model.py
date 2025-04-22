import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the preprocessed data
X = np.load("X.npy")
y = np.load("y.npy")

# Normalize data (optional but recommended)
X = X / np.max(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(21, 3)),  # Adjust input shape if needed
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(26, activation='softmax')  # 26 output classes (A-Z)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Save the trained model
model.save("sign_language_model.h5")

print("Model training complete and saved as 'sign_language_model.h5' 🚀")

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
