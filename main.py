import math
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)
fw, fh = 1280, 720
cap.set(3, fw)
cap.set(4, fh)

detector = HandDetector(maxHands=1, detectionCon=0.8)

# Define canvas
class DrawCanva():
    def __init__(self, color=(0, 0, 255), canvas_width=500, canvas_height=500):
        self.color = color
        self.cx = 0
        self.cy = 0
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.canvas_pos_x, self.canvas_pos_y = 0, 0
        self.canvas = np.zeros((self.canvas_height, self.canvas_width, 3), np.uint8)
        self.reset = True
        self.previous_center_point = 0

    def draw(self, lmList):
        p1 = lmList[8]
        p2 = lmList[12]
        self.cx, self.cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        if self.reset:
            self.previous_center_point = (self.cx, self.cy)
            self.reset = False
            return
        cv2.line(self.canvas, self.previous_center_point, (self.cx, self.cy), self.color, 10)
        self.previous_center_point = (self.cx, self.cy)

    def loadCanvas(self, image=None):
        canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, canvas_binary = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
        canvas_binary = cv2.cvtColor(canvas_binary, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_or(image, self.canvas)

# Define board
class Board():
    def __init__(self, posX=0, posY=0, width=500, height=500, color=(255, 255, 255)):
        self.image = None
        self.posX = posX
        self.posY = posY
        self.width = width
        self.height = height
        self.color = color

    def createBoard(self, borderColor=(255, 0, 0), thickness=5, dark_bg=False):
        overlay = self.image.copy()
        cv2.rectangle(overlay, (self.posX, self.posY), (self.posX + self.width, self.posY + self.height), self.color, cv2.FILLED)
        return overlay

# Initialize drawing components
drawCanvas = DrawCanva(canvas_width=600, canvas_height=400)
board = Board(posX=100, posY=100, width=600, height=400, color=(255, 255, 255))

# Button for capturing canvas
button_x, button_y, button_size = 1100, 50, 50
def capture_and_send_to_gemini():
    snapshot = drawCanvas.canvas.copy()
    print("Captured Canvas and Sent to Gemini")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hand = detector.findHands(img, draw=False)
    board.image = img
    img = board.createBoard(borderColor=(204, 33, 53), thickness=5, dark_bg=True)
    img = drawCanvas.loadCanvas(img)

    # Draw button
    cv2.rectangle(img, (button_x, button_y), (button_x + button_size, button_y + button_size), (0, 255, 0), -1)
    cv2.putText(img, "AI", (button_x + 10, button_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if hand[0]:
        lmList = hand[0][0]["lmList"]
        p1 = lmList[8][0], lmList[8][1]
        p2 = lmList[12][0], lmList[12][1]
        cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        
        # Hover detection for the button
        if button_x < cx < button_x + button_size and button_y < cy < button_y + button_size:
            cv2.rectangle(img, (button_x, button_y), (button_x + button_size, button_y + button_size), (0, 255, 255), 3)
            capture_and_send_to_gemini()
    
    cv2.imshow("Image", img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
