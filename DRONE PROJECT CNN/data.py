import cv2
import os
import time


DATA_DIR = "dataset"
IMG_SIZE = 128  
SAMPLES_PER_CLASS = 700


LABELS = {
    0: "rest",      
    1: "take_off",  
    2: "land",      
    3: "up",        
    4: "down", 
    5: "left",      
    6: "right",   
    7: "flip",       
    8: "capture"
}


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

    frame = cv2.flip(frame, 1)
    

    cv2.rectangle(frame, (300, 50), (600, 350), (0, 255, 0), 2)
    

    roi = frame[50:350, 300:600]
    
    cv2.imshow("Collector", frame)
    cv2.imshow("ROI (What the CNN sees)", roi)
    
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break

    if 48 <= key <= 56:  
        class_idx = key - 48
        class_name = LABELS[class_idx]
        
        save_path = os.path.join(DATA_DIR, class_name)
        count = len(os.listdir(save_path))
        
        if count < SAMPLES_PER_CLASS:

            img_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))

            filename = f"{class_name}_{int(time.time())}_{count}.jpg"
            cv2.imwrite(os.path.join(save_path, filename), img_resized)
            print(f"Saved {class_name}: {count+1}/{SAMPLES_PER_CLASS}")
        else:
            print(f"Class {class_name} is full!")

cap.release()
cv2.destroyAllWindows()