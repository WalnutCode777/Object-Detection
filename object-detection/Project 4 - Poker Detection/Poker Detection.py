import math

from ultralytics import YOLO
import cv2
import cvzone

capture = cv2.VideoCapture("../../workshopProject/Videos/ppe-1.mp4")

model = YOLO("")

classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
              'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
              'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

defaultColor = (0, 0, 255)

while True:
    success, img = capture.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            boundingbox = int(x1), int(y1), int(w), int(h)

            cvzone.cornerRect(img, boundingbox)

            confidence_value = math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
            if confidence_value > 0.45:
                if currentClass == 'NO-Hardhat' or currentClass == 'NO-Mask' or currentClass == 'NO-Safety Vest':
                    defaultColor = (0, 0, 255)
                elif currentClass == 'Hardhat' or currentClass == 'Mask' or currentClass == 'Safety Vest':
                    defaultColor = (0, 255, 0)
                else:
                    defaultColor = (255, 0, 0)
                # colorB = text background color, colorT = text color
                cvzone.putTextRect(img, f'{classNames[cls]} {confidence_value}', (max(0, x1), max(40, y1)), scale=2, thickness=2,
                               colorB=defaultColor, colorT=(255, 255, 255), offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), defaultColor, 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)