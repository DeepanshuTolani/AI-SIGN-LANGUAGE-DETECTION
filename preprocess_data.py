import os
import numpy as np

# Define labels for A-Z
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
data_path = "data"

X = []
y = []

# Load data from all 26 folders
for idx, label in enumerate(labels):
    folder_path = os.path.join(data_path, label)
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' not found. Skipping.")
        continue
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    print(f"Loading {len(files)} samples for {label}")

    for file in files:
        file_path = os.path.join(folder_path, file)
        data = np.load(file_path)
        X.append(data)
        y.append(idx)  # Assign numeric label (A=0, B=1, ..., Z=25)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"Final dataset size: {X.shape}, {y.shape}")

# Save processed data
np.save("X.npy", X)
np.save("y.npy", y)

print("Data preprocessing complete! 🚀")
