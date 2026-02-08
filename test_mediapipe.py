import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def get_finger_status(lm):
    fingers = []
    

    if lm[4].x > lm[3].x: 
        fingers.append(True)
    else:
        fingers.append(False)
    

    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):

        if lm[tip].y < lm[pip].y:
            fingers.append(True)
        else:
            fingers.append(False)
    return fingers


def determine_gesture(fingers, lm):
    thumb, index, middle, ring, pinky = fingers
    count = fingers.count(True)

    if count == 5:
        return "TAKE_OFF", (0, 255, 0) 

    if index and middle and not ring and not pinky:
        return "CAPTURE", (128, 84, 231)

    if index and pinky and not middle and not ring and not thumb:
        return "FLIP", (255, 0, 0) 

    if index and not middle and not ring and not pinky:

        x_diff = abs(lm[8].x - lm[5].x)

        is_vertical = x_diff < 0.1 

        if is_vertical:

            if lm[8].y < lm[5].y: 
                return "UP", (0, 255, 0)
        else:

            if lm[8].x < lm[5].x:
                return "LEFT", (0, 255, 0)
    if not index and not middle and not ring and not pinky:
        if lm[4].x > lm[3].x:
             return "RIGHT", (0, 255, 0)


    if not index and not middle and not ring and not pinky:

        if lm[4].y > lm[3].y:
            return "DOWN", (0, 255, 0)


    if count == 0:

        if lm[4].y < lm[3].y: 
            return "LAND", (0, 255, 255)


    return "DEAD", (0, 0, 255) 


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1) 
    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    command = "DEAD"
    color = (0, 0, 255)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            finger_states = get_finger_status(lm)
            command, color = determine_gesture(finger_states, lm)
            
    cv2.putText(frame, f"CMD: {command}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.imshow("Drone Control", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
