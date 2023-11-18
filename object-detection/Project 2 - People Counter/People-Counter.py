import math

from ultralytics import YOLO
import cv2
import cvzone
from sort import *

capture = cv2.VideoCapture("../../workshopProject/Videos/people.mp4")

model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask2.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

counterUp = []
counterDown = []

while True:
    success, img = capture.read()
    detectRegion = cv2.bitwise_and(img, mask)
    results = model(detectRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            boundingbox = int(x1), int(y1), int(w), int(h)

            confidence_value = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])

            currentClass = classNames[cls]
            if currentClass == 'person' and confidence_value > 0.3:
                cvzone.putTextRect(img, f'{classNames[cls]} {confidence_value}', (max(0, x1), max(40, y1)),
                                   scale=0.7, thickness=1, offset=3)

                currentArray = np.array([x1, y1, x2, y2, confidence_value])

                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        boundingbox = int(x1), int(y1), int(w), int(h)

        cvzone.cornerRect(img, boundingbox, l=5, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(40, y1)),
                           scale=1, thickness=1, offset=5)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if counterUp.count(id) == 0:
                counterUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if counterDown.count(id) == 0:
                counterDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

    cvzone.putTextRect(img, f'Up: {len(counterUp)}', (929, 345))
    cvzone.putTextRect(img, f'Down: {len(counterDown)}', (929, 395))

    cv2.imshow("Image", img)
    # cv2.imshow("DetectRegion", detectRegion)
    cv2.waitKey(0)
