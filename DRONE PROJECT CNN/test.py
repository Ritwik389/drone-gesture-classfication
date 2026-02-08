import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = "drone_cnn.keras"
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.85 
CLASSES = ['capture', 'down', 'flip', 'land', 'left', 'rest', 'right', 'take_off', 'up']

print("Loading Model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model Loaded!")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) 

    x_start, y_start = 1200, 50
    x_end, y_end = 1500, 350


    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    roi = frame[y_start:y_end, x_start:x_end]
    

    cv2.imshow("AI Vision", roi) 

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    

    img = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
    

    img = img.astype('float32') / 255.0

    img = np.expand_dims(img, axis=0)
    

    predictions = model.predict(img, verbose=0)
    score = np.max(predictions)
    class_idx = np.argmax(predictions)
    predicted_label = CLASSES[class_idx]

    y_pos = 100
    for i, label in enumerate(CLASSES):
        prob = predictions[0][i]

        color = (0, 255, 0) if i == class_idx else (100, 100, 100)

        cv2.putText(frame, f"{label}: {int(prob*100)}%", (10, y_pos), 
                   cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
        cv2.rectangle(frame, (120, y_pos-10), (120+int(prob*150), y_pos), color, -1)
        y_pos += 20

    if score > CONFIDENCE_THRESHOLD:
        cmd_text = predicted_label.upper()
        cmd_color = (0, 255, 0)
    else:
        cmd_text = "REST"
        cmd_color = (0, 0, 255)
        
    cv2.putText(frame, f"CMD: {cmd_text}", (340, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, cmd_color, 2)
    
    cv2.imshow("Drone Control", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()