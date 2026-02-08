import cv2
import numpy as np
import tensorflow as tf

# --- CONFIG ---
MODEL_PATH = "drone_cnn.keras"
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.6 # Lowered slightly to let commands through
CLASSES = ['capture', 'down', 'flip', 'land', 'left', 'rest', 'right', 'take_off', 'up']

print("Loading Model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model Loaded!")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # Mirror the camera
    
    # --- ROI CONFIGURATION ---
    # Right Hand Side
    x_start, y_start = 340, 50
    x_end, y_end = 640, 350

    # Draw the Box (Visual Only)
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    
    # 1. Get the Image (The Data)
    roi = frame[y_start:y_end, x_start:x_end]
    
    # 2. DEBUG VIEW: SHOW WHAT THE AI SEES
    # If this box is Black/Blue/Wrong, the AI fails.
    cv2.imshow("AI Vision", roi) 

    # 3. PREPROCESS (STRICT MATCH TO TRAINING)
    # A. Convert BGR -> RGB (Critical!)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # B. Resize to 128x128
    img = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
    
    # C. Normalize (0 to 1) - MATCHING YOUR TRAINING SCRIPT
    img = img.astype('float32') / 255.0
    
    # D. Add Batch Dimension (1, 128, 128, 3)
    img = np.expand_dims(img, axis=0)
    
    # 4. PREDICT
    predictions = model.predict(img, verbose=0)
    score = np.max(predictions)
    class_idx = np.argmax(predictions)
    predicted_label = CLASSES[class_idx]
    
    # --- VISUALIZE PROBABILITIES (THE BAR CHART) ---
    y_pos = 100
    for i, label in enumerate(CLASSES):
        prob = predictions[0][i]
        
        # Color Logic: Green if Winner, Gray if Loser
        color = (0, 255, 0) if i == class_idx else (100, 100, 100)
        
        # Draw Bar
        cv2.putText(frame, f"{label}: {int(prob*100)}%", (10, y_pos), 
                   cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
        cv2.rectangle(frame, (120, y_pos-10), (120+int(prob*150), y_pos), color, -1)
        y_pos += 20

    # --- FINAL COMMAND DISPLAY ---
    if score > CONFIDENCE_THRESHOLD:
        cmd_text = predicted_label.upper()
        cmd_color = (0, 255, 0)
    else:
        cmd_text = "UNCERTAIN"
        cmd_color = (0, 0, 255)
        
    cv2.putText(frame, f"CMD: {cmd_text}", (340, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, cmd_color, 2)
    
    cv2.imshow("Drone Control", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()