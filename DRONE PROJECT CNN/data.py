import cv2
import os
import time

# --- CONFIG ---
DATA_DIR = "dataset"
IMG_SIZE = 128  # We will resize images to this for the CNN
SAMPLES_PER_CLASS = 700

# Define your classes (0-7)
LABELS = {
    0: "rest",      # No command / Background
    1: "take_off",  # Open Palm
    2: "land",      # Fist
    3: "up",        # Index Up
    4: "down",      # Thumbs Down
    5: "left",      # Index Pointing Left
    6: "right",     # Index Pointing Right
    7: "flip",       
    8: "capture"
}

# Create directories
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
for label in LABELS.values():
    path = os.path.join(DATA_DIR, label)
    if not os.path.exists(path):
        os.makedirs(path)

cap = cv2.VideoCapture(0)

print(LABELS)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Mirror the frame (Selfie view)
    frame = cv2.flip(frame, 1)
    
    # Define the Region of Interest (ROI) box
    # We only want to save the HAND, not your face/room
    cv2.rectangle(frame, (300, 50), (600, 350), (0, 255, 0), 2)
    
    # Extract the ROI
    roi = frame[50:350, 300:600]
    
    cv2.imshow("Collector", frame)
    cv2.imshow("ROI (What the CNN sees)", roi)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    
    # Check if a number key (0-7) was pressed
    if 48 <= key <= 56:  # ASCII for '0'-'7'
        class_idx = key - 48
        class_name = LABELS[class_idx]
        
        save_path = os.path.join(DATA_DIR, class_name)
        count = len(os.listdir(save_path))
        
        if count < SAMPLES_PER_CLASS:
            # Resize and Save
            img_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            # Create unique filename
            filename = f"{class_name}_{int(time.time())}_{count}.jpg"
            cv2.imwrite(os.path.join(save_path, filename), img_resized)
            print(f"Saved {class_name}: {count+1}/{SAMPLES_PER_CLASS}")
        else:
            print(f"Class {class_name} is full!")

cap.release()
cv2.destroyAllWindows()