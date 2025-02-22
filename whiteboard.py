import math
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)

fw = 1280
fh = 720

cap.set(3, fw)
cap.set(4, fh)
First = True


def findDistance(p1, p2, img,color=(0,0,0)):
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
    else:
        return length, info


class DrawCanva():
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
        self.previous_center_point = 0
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
            print("breakinnggg")
            self.is_draw = False


        if self.reset:
            self.previous_center_point = (self.cx, self.cy)
            self.reset = False
            return "drawing canvas started"
        cv2.line(self.canvas, self.previous_center_point, (self.cx, self.cy),self.color, 10)
        self.previous_center_point = (self.cx, self.cy)

    def moveCanvas(self, posX, posY):
        self.canvas_pos_x = posX
        self.canvas_pos_y = posY

    def loadCanvas1(self, image=None):
        self.image = image
        canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, canvas_binary = cv2.threshold(canvas_gray, 20, 255,
                                         cv2.THRESH_BINARY_INV)
        canvas_binary = cv2.cvtColor(canvas_binary, cv2.COLOR_GRAY2BGR)
        self.image = cv2.bitwise_and(self.image, canvas_binary)
        return cv2.bitwise_or(self.image, self.canvas)

    def visible_area(self):
        c = [self.canvas_pos_x, self.canvas_pos_y, self.canvas_width, self.canvas_height]
        if self.canvas_pos_x < 0:
            c[0] = 0
            c[2] += c[0]
        if self.canvas_pos_y < 0:
            c[1] = 0
            c[3] += c[1]
        return c

    def loadCanvas(self, image=None):
        px, py, cw, ch = self.visible_area()
        sub_image = image[py:py + ch, px:px + cw, :]
        if sub_image.shape != (ch, cw, 3):
            return image
        canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, canvas_binary = cv2.threshold(canvas_gray, 20, 255,
                                         cv2.THRESH_BINARY_INV)
        canvas_binary = cv2.cvtColor(canvas_binary, cv2.COLOR_GRAY2BGR)
        sub_image = cv2.bitwise_and(sub_image, canvas_binary)
        sub_image = cv2.bitwise_or(sub_image, self.canvas)
        image[py:py + ch, px:px + cw, :] = sub_image
        return image


class Board():
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
        leftTopX = [self.posX, self.posY, self.posX + length, self.posY]
        leftTopY = [self.posX, self.posY, self.posX, self.posY + length]

        leftBottomY = [self.posX, self.posY + self.height, self.posX, self.posY + self.height - length]
        leftBottomX = [self.posX, self.posY + self.height, self.posX + length, self.posY + self.height]

        rightTopX = [self.posX + self.width, self.posY, self.posX + self.width - length, self.posY]
        rightTopY = [self.posX + self.width, self.posY, self.posX + self.width, self.posY + length]

        rightBottomX = [self.posX + self.width, self.posY + self.height, self.posX + self.width - length,
                        self.height + self.posY]
        rightBottomY = [self.posX + self.width, self.posY + self.height, self.posX + self.width,
                        self.height + self.posY - length]

        cv2.rectangle(overlay, (self.posX, self.posY), (self.posX + self.width, self.posY + self.height), self.color,
                      cv2.FILLED)
        overlay = cv2.addWeighted(overlay, self.alpha, self.image, 1 - self.alpha, 0)
        corners = [leftTopX, leftTopY, leftBottomX, leftBottomY, rightTopX, rightTopY, rightBottomX, rightBottomY]
        for c in corners:
            cv2.line(overlay, (c[0], c[1]), (c[2], c[3]), borderColor, thickness)
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

    def getPos(self):
        return self.posX, self.posY

    def findGuesture(self, lmList):
        # px1, py1 = lmList[20][:2]
        # px2, py2 = lmList[0][:2]

        # hx1, hy1 = lmList[5][:2]
        # hx2, hy2 = lmList[17][:2]

        # pd = math.hypot(px1 - px2, py2 - py1)
        # hd = math.hypot(hx1 - hx2, hy2 - hy1)
        # dist = pd / hd

        px1, py1 = lmList[4][:2]
        px2, py2 = lmList[20][:2]

        hx1, hy1 = lmList[12][:2]
        hx2, hy2 = lmList[0][:2]
        pd = math.hypot(px1 - px2, py2 - py1)
        hd = math.hypot(hx1 - hx2, hy2 - hy1)
        dist = pd / hd

        return dist
    
    def finger_distance(se,lmlist):
        px1, py1 = lmList[8][:2]
        px2, py2 = lmList[12][:2]

        hx1, hy1 = lmList[12][:2]
        hx2, hy2 = lmList[0][:2]
        pd = math.hypot(px1 - px2, py2 - py1)
        hd = math.hypot(hx1 - hx2, hy2 - hy1)
        dist = pd / hd
        return dist


class ColorRect():
    def __init__(self,x,y,color,thickness=-1):
        self.x = x
        self.y = y
        self.color = color
        self.size = 50
        self.thickness = thickness
        self.selected = False
        self.clicked = False
        self.inside = False
    def click(self,hx=None,hy=None,img=None):
        cv2.rectangle(img,(self.x,self.y),(self.x+self.size,self.y+self.size),self.color, self.thickness)
        start_point = (self.x, self.y)
        end_point = (self.x + self.size, self.y + self.size)
        cv2.rectangle(img, start_point, end_point, (255,255,255), 2)
        if hx == None:
            self.selected = False
            return self.selected
        if self.x<hx<self.x+self.size and self.y<hy<self.y+self.size:
            self.selected = True
            self.drawBorder(img)
        else:
            self.selected = False
        return self.selected
        
    def drawBorder(self, img,color=(66,96,245)):
        if self.selected:
            start_point = (self.x, self.y)
            end_point = (self.x + self.size, self.y + self.size)
            cv2.rectangle(img, start_point, end_point,border_selected_color, 5)


colors = [(141,245,66),(245,167,66),(0,0,255),(245,66,167),(66,191,245),(66,239,245),(188,66,245)]
eraser_color  = (44,44,44) 

obj_colors = []
x_start = 10
size = 55
border_selected_color = (190, 255, 0)
border_hover_color_color = (66,66,245)
for col in colors:
    obj_colors.append(ColorRect(x_start,10,col))
    x_start+=size

detector = HandDetector(maxHands=1, detectionCon=0.8)

board_width = 600
board_height = 400

drawCanvas = DrawCanva(canvas_width=board_width, canvas_height=board_height)
board = Board(posX=100, posY=100, width=board_width, height=board_height, color=(255, 255, 255))
draw = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hand = detector.findHands(img, draw=False)

    board.image = img
    img = board.createBoard(borderColor=(204, 33, 53), thickness=5, dark_bg=True)
    img = drawCanvas.loadCanvas(img)

    for oc in obj_colors:
        oc.click(img=img)

    if hand[0]:
        lmList = hand[0][0]["lmList"]
        # print("handmark : ",lmList)

        p1 = lmList[8][0], lmList[8][1]
        p2 = lmList[12][0], lmList[12][1]
        _, info = findDistance(p1, p2, img,color=drawCanvas.color)[:2]
        cx, cy = info[4:]

        pame_dist = board.findGuesture(lmList)
        finger_dist = board.finger_distance(lmList)

        # print("finger dist : ",finger_dist)
        if pame_dist > 0.7:
            # print(f"moving canva... {pame_dist}")
            drawCanvas.reset = True
            posx, posy = board.moveBoard(lmList)
            drawCanvas.moveCanvas(posx, posy)
        if finger_dist < 0.18:
            # print(f"drawing...{finger_dist} . ({finger_dist < 1.1})")
            drawCanvas.draw(lmList)
            drawCanvas.is_draw = True
            for oc in obj_colors:
                oc.click(hx=cx,hy=cy,img=img)
                if oc.selected:
                    drawCanvas.color = oc.color
        else:
            drawCanvas.is_draw = False
            print("breaking the drawing")



    cv2.imshow("Image", img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
