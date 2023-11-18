import math

from ultralytics import YOLO
import cv2
import cvzone

# 1. Set up camera
# 2. Set up the model you want to use
# 3. Combine the model and video streaming

# Set up webcam, same for Pi Camera
capture = cv2.VideoCapture(0)
# Prop ID for capture width is 3, and 4 for height
# Can use 1280, 720, but 640, 480 is most commonly used for IoT, or less computational power
capture.set(3, 1280)
capture.set(4, 720)

model = YOLO("../Yolo-Weights/yolov8n.pt")

# These are classes that it can detect for this model, but it will only show the ID
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

# -------------------------------------------------------------------------------------------------------------
# Part 2
# Using l version will be a lot slower, because we are using CPU to run the model, later we will configure to use GPU
# model = YOLO("../Yolo-Weights//yolov8l.pt")
# capture = cv2.VideoCapture("../Videos/people.mp4") # For video

while True:
    success, img = capture.read()
    # stream=True, which means it will use generators, more efficient compare to without it
    results = model(img, stream=True)
    for r in results:
        # Get bounding boxes for each results
        boxes = r.boxes
        for box in boxes:
            # # Bounding Box
            # # Can use box.xyxy or box.xywh, but xyxy is better since it is makes it easier for opencv to input directly
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # # Draw the bounding boxes
            # # rectangle(source, point1, point2, box color, thickness of the box line)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # To let it label the mAP on the bounding box the easier way, use the below method
            # x1, y1, w, h for cvzone, tho doesnt make any difference for cvzone
            w, h = x2-x1, y2-y1
            boundingbox = int(x1), int(y1), int(w), int(h)
            # Can click on the function to see what you can do with it, ctrl + click
            cvzone.cornerRect(img, boundingbox)

            # Confidence value
            # To round off the value, use math.ceiling or math.floor
            # confidence_value = math.ceil((box.confidence_value[0] * 100)) / 100 can use this with same name or not, but this is the best practice
            confidence_value = math.ceil((box.conf[0]*100))/100
            # print(confidence_value)
            # So to put the confidence value text in above the box, and when the object go too near the edge of the window, it wont disappear
            # opencv will have the problem of text not centered with the bounding box properly, and difficult to deal with the problem mentioned
            # y1-20 is because we want to push y1 a bit up, *rmb to explain the window coordinates on whiteboard*
            # cvzone.putTextRect(img, f'{confidence_value}', (max(0, x1), max(0, y1-20)))

            # Class Name
            # Dont use class as variable name, because it is a syntax
            # cls = box.cls[0]
            # cvzone.putTextRect(img, f'{cls} {confidence_value}', (max(0, x1), max(40, y1)))

            cls = int(box.cls[0])
            # scale = 0.5 means will scale down the text size to 2, default is 3,
            # changing scale size will need to change text thickness or else will clump tgt
            cvzone.putTextRect(img, f'{classNames[cls]} {confidence_value}', (max(0, x1), max(40, y1)), scale=2, thickness=2)

    cv2.imshow("Image", img)
    # 1 second delay
    cv2.waitKey(1)