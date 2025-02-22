import cv2
import mediapipe as mp

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Initialize the hand detection model
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB (MediaPipe uses RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(frame_rgb)
        
        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                print("hand_landmarks : ",hand_landmarks.landmark)
                
                # Display each landmark's coordinates
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    print("idx : ",idx)
                    h, w, c = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.putText(frame, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Show the frame
        cv2.imshow("Hand Landmarks", frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
