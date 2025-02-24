import math
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import tkinter as tk
import threading
import requests
import base64
import tkinter as tk
from tkinter import scrolledtext
import threading
import google.generativeai as genai
import requests
import threading
from PIL import Image
from io import BytesIO


API_KEY = "AIzaSyCGG63veC7HT6B60X6UMCtKSWIk8oJ4hDE"  
genai.configure(api_key=API_KEY)

def convert_to_pil(canvas_img):
    """Converts a NumPy image (OpenCV) to a PIL image."""
    return Image.fromarray(cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB))

def get_image_description(pil_img):
    """Sends the PIL image to Gemini API and retrieves the description."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([pil_img, "Describe this image in short with easy way , if it is a maths problem solve it in simple steps "])
        return response.text if response else "No description available."
    except Exception as e:
        return f"Error: {str(e)}"

def solve_function():
    """Main function to process the image and get the description."""
    canvas_img = drawCanvas.canvas  # Assuming `drawCanvas.canvas` is a valid NumPy image array
    pil_img = convert_to_pil(canvas_img)  # Convert to PIL Image
    description = get_image_description(pil_img)
    # Run UI update in a separate thread
    threading.Thread(target=show_description, args=(description,)).start()


import tkinter as tk
from tkinter import ttk, scrolledtext

def show_description(description):
    """Displays the image description in a modern UI."""
    # Preprocess text: Remove asterisks
    description = description.replace("*", "")

    root = tk.Tk()
    root.title("Image Description")
    root.geometry("600x400")  # Increased size for better readability
    root.configure(bg="#f0f0f0")  # Light gray background

    # Use themed styling for modern UI
    style = ttk.Style()
    style.configure("TFrame", background="#ffffff", padding=10)
    style.configure("TLabel", background="#ffffff", font=("Arial", 14), foreground="#333")
    
    # Create main frame
    frame = ttk.Frame(root, style="TFrame")
    frame.pack(expand=True, fill="both", padx=20, pady=20)

    # Title Label
    label = ttk.Label(frame, text="Image Description", style="TLabel", font=("Arial", 16, "bold"))
    label.pack(pady=10)

    # Scrollable Text Widget
    text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Arial", 14), bg="#ffffff", fg="#333333")
    text.insert(tk.END, description)
    text.config(state=tk.DISABLED)  # Make text read-only
    text.pack(expand=True, fill="both")

    root.mainloop()


# Initialize video capture
cap = cv2.VideoCapture(0)
fw, fh = 1300, 720
cap.set(3, fw)
cap.set(4, fh)

# Utility function to calculate distance between two points
def findDistance(p1, p2, img, color=(0, 0, 0)):
    disColor = (255, 146, 51)
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = math.hypot(x2 - x1, y2 - y1)
    info = (x1, y1, x2, y2, cx, cy)
    if img is not None:
        cv2.circle(img, (x1, y1), 5, disColor, cv2.FILLED)
        cv2.circle(img, (x2, y2), 5, disColor, cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), disColor, 3)
        cv2.circle(img, (cx, cy), 15, color, cv2.FILLED)
    return length, info, img

# Drawing canvas class
class DrawCanva:
    def __init__(self, color=(0, 0, 255), canvas_width=500, canvas_height=500):
        self.color = color
        self.cx = 0
        self.cy = 0
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.canvas_pos_x, self.canvas_pos_y = 0, 0
        self.canvas = np.zeros((self.canvas_height, self.canvas_width, 3), np.uint8)
        self.image = None
        self.reset = True
        self.previous_center_point = None
        self.is_draw = False

    def centroid(self, p1, p2):
        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
        return (x1 + x2) // 2, (y1 + y2) // 2

    def draw(self, lmList):
        p1 = lmList[8]
        p2 = lmList[12]
        self.cx, self.cy = self.centroid(p1, p2)
        
        if self.is_draw:
            self.cx -= self.canvas_pos_x
            self.cy -= self.canvas_pos_y
        else:
            self.previous_center_point = None

        if self.reset or self.previous_center_point is None:
            self.previous_center_point = (self.cx, self.cy)
            self.reset = False
            return "drawing canvas started"
        
        cv2.line(self.canvas, self.previous_center_point, (self.cx, self.cy), self.color, 10)
        self.previous_center_point = (self.cx, self.cy)

    def moveCanvas(self, posX, posY):
        self.canvas_pos_x = posX
        self.canvas_pos_y = posY

    def loadCanvas(self, image=None):
        px, py, cw, ch = self.visible_area()
        sub_image = image[py:py + ch, px:px + cw, :]
        if sub_image.shape != (ch, cw, 3):
            return image
        canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, canvas_binary = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
        canvas_binary = cv2.cvtColor(canvas_binary, cv2.COLOR_GRAY2BGR)
        sub_image = cv2.bitwise_and(sub_image, canvas_binary)
        sub_image = cv2.bitwise_or(sub_image, self.canvas)
        image[py:py + ch, px:px + cw, :] = sub_image
        return image

    def visible_area(self):
        c = [self.canvas_pos_x, self.canvas_pos_y, self.canvas_width, self.canvas_height]
        if self.canvas_pos_x < 0:
            c[0] = 0
            c[2] += c[0]
        if self.canvas_pos_y < 0:
            c[1] = 0
            c[3] += c[1]
        return c

# Board class for the drawing area
class Board:
    def __init__(self, posX=0, posY=0, width=500, height=500, color=(255, 255, 255), alpha=0.4):
        self.image = None
        self.posX = posX
        self.posY = posY
        self.width = width
        self.height = height
        self.color = color
        self.alpha = alpha

    def createBoard(self, borderColor=(255, 0, 0), thickness=5, dark_bg=False):
        bg_canvas = np.zeros((self.image.shape[0], self.image.shape[1], 3), np.uint8)
        overlay = bg_canvas if dark_bg else self.image.copy()
        length = 50
        corners = [
            [self.posX, self.posY, self.posX + length, self.posY],
            [self.posX, self.posY, self.posX, self.posY + length],
            [self.posX, self.posY + self.height, self.posX + length, self.posY + self.height],
            [self.posX, self.posY + self.height, self.posX, self.posY + self.height - length],
            [self.posX + self.width, self.posY, self.posX + self.width - length, self.posY],
            [self.posX + self.width, self.posY, self.posX + self.width, self.posY + length],
            [self.posX + self.width, self.posY + self.height, self.posX + self.width - length, self.posY + self.height],
            [self.posX + self.width, self.posY + self.height, self.posX + self.width, self.posY + self.height - length]
        ]
        cv2.rectangle(overlay, (self.posX, self.posY), (self.posX + self.width, self.posY + self.height), self.color, cv2.FILLED)
        overlay = cv2.addWeighted(overlay, self.alpha, self.image, 1 - self.alpha, 0)
        for c in corners:
            cv2.line(overlay, (c[0], c[1]), (c[2], c[3]), borderColor, thickness, cv2.LINE_AA)
        return overlay

    def moveBoard(self, lmList):
        cood = [5, 9, 13, 17, 0]
        cx, cy = 0, 0
        for i in cood:
            cx += lmList[i][0]
            cy += lmList[i][1]
        cx = cx / len(cood)
        cy = cy / len(cood)
        self.posX = int(cx - self.width / 2)
        self.posY = int(cy - self.height / 2)
        return self.posX, self.posY

    def findGuesture(self, lmList):
        px1, py1 = lmList[4][:2]
        px2, py2 = lmList[20][:2]
        hx1, hy1 = lmList[12][:2]
        hx2, hy2 = lmList[0][:2]
        pd = math.hypot(px1 - px2, py2 - py1)
        hd = math.hypot(hx1 - hx2, hy2 - hy1)
        return pd / hd
    
    def finger_distance(self, lmList):
        px1, py1 = lmList[8][:2]
        px2, py2 = lmList[12][:2]
        hx1, hy1 = lmList[12][:2]
        hx2, hy2 = lmList[0][:2]
        pd = math.hypot(px1 - px2, py2 - py1)
        hd = math.hypot(hx1 - hx2, hy2 - hy1)
        return pd / hd

# Color selection rectangle class
class ColorRect:
    def __init__(self, x, y, color, thickness=-1):
        self.x = x
        self.y = y
        self.color = color
        self.size = 50
        self.thickness = thickness
        self.selected = False

    def click(self, hx=None, hy=None, img=None):
        cv2.rectangle(img, (self.x, self.y), (self.x + self.size, self.y + self.size), self.color, self.thickness)
        start_point = (self.x, self.y)
        end_point = (self.x + self.size, self.y + self.size)
        cv2.rectangle(img, start_point, end_point, (255, 255, 255), 2)
        if hx is None:
            self.selected = False
            return self.selected
        if self.x < hx < self.x + self.size and self.y < hy < self.y + self.size:
            self.selected = True
            self.drawBorder(img)
        else:
            self.selected = False
        return self.selected
        
    def drawBorder(self, img, color=(66, 96, 245)):
        if self.selected:
            start_point = (self.x, self.y)
            end_point = (self.x + self.size, self.y + self.size)
            cv2.rectangle(img, start_point, end_point, color, 5)

class Button:
    def __init__(self, x, y,text="",color=(55,44,66)):
        self.x = x
        self.y = y
        self.w_size = 100
        self.h_size = 50
        self.hove = False  
        self.text = text
        self.color = color

    def draw(self, img):
        # Draw button
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w_size, self.y + self.h_size),self.color, cv2.FILLED)
        cv2.putText(img,self.text, (self.x + 20, self.y + 30), cv2.FORMATTER_FMT_PYTHON, 0.7, (255, 255, 255), 2)
        # Draw hover effect
        if self.hove:
            cv2.rectangle(img, (self.x, self.y), (self.x + self.w_size, self.y + self.h_size), (255, 255, 255), 3)

    def is_hover(self, hx, hy):
        self.hove = self.x <= hx <= self.x + self.w_size and self.y <= hy <= self.y + self.h_size
        return self.hove


# Initialize colors and objects
colors = [
    (255, 99, 71),    # Tomato Red
    (30, 144, 255),   # Dodger Blue
    (50, 205, 50),    # Lime Green
    (255, 165, 0),    # Orange
    (186, 85, 211),   # Medium Orchid
    (64, 224, 208),   # Turquoise
    (255, 20, 147)    # Deep Pink
]

eraser_color = (44, 44, 44)
obj_colors = []
x_start = 10
size = 55
for col in colors:
    obj_colors.append(ColorRect(x_start, 10, col))
    x_start += size
# obj_colors.append(ColorRect(x_start, 10, eraser_color))  # Add eraser

# Initialize hand detector and board
detector = HandDetector(maxHands=1, detectionCon=0.8)

board_width, board_height = 800, 600

drawCanvas = DrawCanva(canvas_width=board_width, canvas_height=board_height)
board = Board(posX=100, posY=100, width=board_width, height=board_height, color=(255, 255, 255))

solve_button = Button(1000, 10,"Solve",(87,87,222))
erase_btn = Button(1120, 10,"erase",(87, 152, 222))

was_hovering_save = False
was_solving = False
save_count = 0

def solve_function_thread():
    solve_function()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hand = detector.findHands(img, draw=False)

    board.image = img
    img = board.createBoard(borderColor=(204, 33, 53), thickness=1, dark_bg=True)
    img = drawCanvas.loadCanvas(img)

    for oc in obj_colors:
        oc.click(img=img)
    
    solve_button.draw(img)
    erase_btn.draw(img=img)

    if hand:
        lmList = hand[0]
        if hand[0]:
            lmList = hand[0][0]["lmList"]
            p1 = lmList[8][:2]
            p2 = lmList[12][:2]
            _, info = findDistance(p1, p2, img, color=drawCanvas.color)[:2]
            cx, cy = info[4:]

            pame_dist = board.findGuesture(lmList)
            finger_dist = board.finger_distance(lmList)

            for oc in obj_colors:
                oc.click(hx=cx, hy=cy, img=img)
                if oc.selected:
                    if oc.color == eraser_color:
                        drawCanvas.canvas = np.zeros((drawCanvas.canvas_height, drawCanvas.canvas_width, 3), np.uint8)
                    else:
                        drawCanvas.color = oc.color

            if pame_dist > 0.7:
                drawCanvas.reset = True
                posx, posy = board.moveBoard(lmList)
                drawCanvas.moveCanvas(posx, posy)
            if finger_dist < 0.18:
                drawCanvas.is_draw = True
                drawCanvas.draw(lmList)
            else:
                drawCanvas.is_draw = False
                drawCanvas.previous_center_point = None

            # Save button logic (ensuring function runs only once per hover and doesn't block UI)
            if solve_button.is_hover(cx, cy):
                if not was_hovering_save:
                    was_hovering_save = True
                    threading.Thread(target=solve_function_thread, daemon=True).start()
            else:
                was_hovering_save = False  # Reset when not hovering

            if erase_btn.is_hover(cx,cy):
                drawCanvas.canvas = np.zeros((drawCanvas.canvas_height, drawCanvas.canvas_width, 3), np.uint8)

    cv2.imshow("Image", img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
