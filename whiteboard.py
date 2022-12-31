import math
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)

fw = 1280
fh = 720

cap.set(3,fw)
cap.set(4,fh)
First = True

class DrawCanva():
    def __init__(self,color=(0,0,255),canvas_width=500,canvas_height=500):
        self.color = color
        self.cx = 0
        self.cy = 0
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.canvas_pos_x ,self.canvas_pos_y = 0,0
        self.canvas = np.zeros((self.canvas_height, self.canvas_width, 3), np.uint8)
        self.image = None
        self.reset = True
        self.previous_center_point = 0

    def centroid(self,p1,p2):
        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
        return (x1 + x2) // 2, (y1 + y2) // 2

    def draw(self,lmList):
        p1 = lmList[8]
        p2 = lmList[12]
        self.cx, self.cy = self.centroid(p1,p2)
        if self.reset:
            self.previous_center_point = (self.cx,self.cy)
            self.reset = False
            return "drawing canvas started"
        cv2.line(self.canvas, self.previous_center_point, (self.cx, self.cy), (0, 0, 255), 10)
        self.previous_center_point = (self.cx, self.cy)

    def moveCanvas(self,posX,posY):
        self.canvas_pos_x = posX
        self.canvas_pos_y = posY


    def loadCanvas(self,image=None):
        self.image = image
        canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, canvas_binary = cv2.threshold(canvas_gray, 20, 255,
                                         cv2.THRESH_BINARY_INV)
        canvas_binary = cv2.cvtColor(canvas_binary, cv2.COLOR_GRAY2BGR)
        self.image = cv2.bitwise_and(self.image, canvas_binary)
        # return cv2.bitwise_or(self.image[self.canvas_pos_x:self.canvas_pos_x+self.canvas_height,self.canvas_pos_y:self.canvas_pos_y+self.canvas_width,:],
        #                       self.canvas[self.canvas_pos_x:self.canvas_pos_x+self.canvas_height,self.canvas_pos_y:self.canvas_pos_y+self.canvas_width,:])
        return cv2.bitwise_or(self.image,self.canvas)



class Board():
    def __init__(self,posX=0,posY=0,width=0,height=0,color=(255,255,255),alpha=0.4,drawCanvas=None):
        self.image = None
        self.posX = posX
        self.posY = posY
        self.width = width
        self.height = height
        self.color = color
        self.alpha = alpha
        self.drawCanvas = drawCanvas

    def createBoard(self,borderColor=(255,0,0),thickness=5):
        length = 50
        leftTopX = [self.posX, self.posY,self.posX+length, self.posY]
        leftTopY = [self.posX, self.posY,self.posX, self.posY+length]

        leftBottomY = [self.posX,self.posY+self.height,self.posX,self.posY+self.height-length]
        leftBottomX = [self.posX,self.posY+self.height,self.posX+length,self.posY+self.height]

        rightTopX = [self.posX+self.width,self.posY,self.posX+self.width-length,self.posY]
        rightTopY = [self.posX+self.width,self.posY,self.posX+self.width,self.posY+length]

        rightBottomX = [self.posX+self.width, self.posY+self.height,self.posX+self.width-length, self.height+self.posY]
        rightBottomY = [self.posX+self.width, self.posY+self.height,self.posX+self.width, self.height+self.posY-length]

        # overlay = self.image.copy()
        overlay = self.image.copy() if self.drawCanvas == None else self.drawCanvas.canvas.copy()


        cv2.rectangle(overlay,(self.posX,self.posY),(self.posX+self.width,self.posY+self.height),self.color, cv2.FILLED)
        overlay = cv2.addWeighted(overlay,self.alpha,self.image, 1 - self.alpha, 0)
        corners = [leftTopX,leftTopY,leftBottomX,leftBottomY,rightTopX,rightTopY,rightBottomX,rightBottomY]
        for c in corners:
            cv2.line(overlay, (c[0], c[1]), (c[2], c[3]), borderColor, thickness)
        return overlay


    def moveBoard(self,lmList):
        cood = [5,9,13,17,0]
        cx,cy = 0,0
        for i in cood:
            cx += lmList[i][0]
            cy += lmList[i][1]
        cx = cx/len(cood)
        cy = cy/len(cood)

        self.posX = int(cx-self.width/2)
        self.posY = int(cy-self.height/2)
        return self.posX,self.posY

    def findGuesture(self,lmList):
        global First
        px1,py1 = lmList[20][:2]
        px2,py2 = lmList[0][:2]

        hx1,hy1 = lmList[5][:2]
        hx2,hy2 = lmList[17][:2]

        pd = math.hypot(px1-px2,py2-py1)
        hd = math.hypot(hx1-hx2,hy2-hy1)
        dist = pd/hd
        return dist





def findDistance(p1,p2,img):
    disColor = (255,146,51)
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = math.hypot(x2 - x1, y2 - y1)
    info = (x1, y1, x2, y2, cx, cy)
    if img is not None:
        cv2.circle(img, (x1, y1), 5, disColor, cv2.FILLED)
        cv2.circle(img, (x2, y2), 5,disColor, cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2),disColor, 3)
        cv2.circle(img, (cx, cy), 15,disColor, cv2.FILLED)
        return length, info, img
    else:
        return length, info


detector = HandDetector(maxHands=1,detectionCon=0.8)

board_width = fw
board_height = fh
drawCanvas = DrawCanva(canvas_width=board_width,canvas_height=board_height)
board = Board(posX=10,posY=10,width=700,height=500,color=(255,255,255),drawCanvas=drawCanvas)


while True:
    success,img = cap.read()
    img = cv2.flip(img,1)
    hand = detector.findHands(img,draw=False)

    board.image = img
    img = board.createBoard(borderColor=(204,33,53),thickness=5)
    img = drawCanvas.loadCanvas(img)
    if hand:
        lmList = hand[0]["lmList"]
        values = findDistance(lmList[8], lmList[12], img)
        dist = board.findGuesture(lmList)
        print(dist)
        if dist>2.1:
            drawCanvas.reset = True
            posx,posy = board.moveBoard(lmList)
            drawCanvas.moveCanvas(posx,posy)
        if dist<1.7:
            drawCanvas.draw(lmList)







    cv2.imshow("Image",img)
    # cv2.imshow("Image2",canvas)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break