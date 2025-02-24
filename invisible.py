import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Capture the first frame as background reference
    if snapshot is None:
        snapshot = frame.copy()
    
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for detecting pink color
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    mask_inv = cv2.bitwise_not(mask)
    
    # Apply masks to extract foreground and background
    foreground = cv2.bitwise_and(frame, frame, mask=mask_inv)
    background = cv2.bitwise_and(snapshot, snapshot, mask=mask)
    
    # Combine the masked foreground and background
    result = cv2.add(background, foreground)
    
    # Display the result
    cv2.imshow('Pink Color Filter', result)
    
    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()