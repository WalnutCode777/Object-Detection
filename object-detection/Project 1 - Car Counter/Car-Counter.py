import math

from ultralytics import YOLO
import cv2
import cvzone
from sort import *

capture = cv2.VideoCapture("../../workshopProject/Videos/cars.mp4")

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

# ----------------------------------------------------------------------------------------------------------------------
# Part 2
# Import mask
mask = cv2.imread("Project 1 - Masks.png")

# ----------------------------------------------------------------------------------------------------------------------
# Part 3
# Tracker instance
# max_age = if the object labeled with an ID is lost, the max number of frames that it will wait until the object
# comeback before treating as new object
# min_hits =
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# ----------------------------------------------------------------------------------------------------------------------
# Part 4
# Set the limits for the line (2 coordinates), adjust yourself until you find the correct coordinates, all the coordinates are stored in this array
# x1, y1, x2, y2
limits = [400, 297, 673, 297]

# ----------------------------------------------------------------------------------------------------------------------
# Part 5
# totalCount = 0
totalCount = []

while True:
    success, img = capture.read()
    # ----------------------------
    # Part 2
    # cv2.bitwise_and means overlay the mask on the source images sliced from the video
    # Mask and source must be the same size, or else wont work
    detectRegion = cv2.bitwise_and(img, mask)
    # ----------------------------
    # results = model(img, stream=True)
    # ----------------------------
    # Part 2
    # Instead of img, we will pass in detectRegion
    results = model(detectRegion, stream=True)
    # ----------------------------
    # Part 3
    # Declare the detections list for tracker after getting the results
    detections = np.empty((0, 5))
    # ----------------------------
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1, y2-y1
            boundingbox = int(x1), int(y1), int(w), int(h)
            # l = 8, means the thickness of green border, we make it smaller because it is too thick which block the class
            # cvzone.cornerRect(img, boundingbox)
            # cvzone.cornerRect(img, boundingbox, l=5)

            # Confidence value
            confidence_value = math.ceil((box.conf[0]*100))/100

            # Class name
            cls = int(box.cls[0])


            # cvzone.putTextRect(img, f'{classNames[cls]} {confidence_value}', (max(0, x1), max(40, y1)), scale=2, thickness=2)
            # Since the scale is still too big and text thickness is too thick which block the labels,
            # we adjust it smaller
            # Offset is the text box space that is not covered with text, default is 10
            # cvzone.putTextRect(img, f'{classNames[cls]} {confidence_value}', (max(0, x1), max(40, y1)),
            #                    scale=0.7, thickness=1, offset=3)

            # Count vehicles
            # We only want to detect and count the vehicles, which are the car, bus, truck, motorbike
            currentClass = classNames[cls]
            if currentClass == 'car' or currentClass == 'bus' or currentClass == 'truck' or currentClass == 'motorbike' \
                    and confidence_value > 0.3:
                cvzone.putTextRect(img, f'{classNames[cls]} {confidence_value}', (max(0, x1), max(40, y1)),
                                   scale=0.7, thickness=1, offset=3)
                # cvzone.cornerRect(img, boundingbox, l=5, rt=2) # Comment this out in part 4 as we only need one box

                # Part 3
                # Since the furthest and nearest part of the vehicles on the road is not detected clearly due to distance
                # Hence, we need to locate the region where the vehicles are identified correctly and mask the other regions
                # Can do this by going to Canva, click create a design, choose youtube thumbnail because the video window is 1280*720
                # Press 'r' to get a rectangle and align the rectangle with the left dotted part of the road, and right side, and back and forth
                # After masking with black rectangles, delete the video, and you will get a white region
                # Then change the img parameter in the model to detectRegion
                # Now, we need to determine at which part of the region where cars pass by is considered as counted
                # We also need to find a way to track the car, because in first frame after the car passes, the model
                # dont know where the car goes in the second frame, so we need a tracking ID

                # Part 4
                # We need to determine a line where the car is counted only after passing that line

                # Only after the object detected fulfill the condition above, we save the tracking information into detections numpy list
                currentArray = np.array([x1, y1, x2, y2, confidence_value])
                # Normally when we want to store several lists in an array or list, we will use append which concatenate the list behind the last one
                # But in numpy, we use stack
                detections = np.vstack((detections, currentArray))
                # ----------------------------
    # Part 3
    # To run the tracker, we only need to update the tracker with bunch of detections (listed 5 parameters in the method)
    # The 5 parameters are x1, y1, x2, y2, ID number, these are store in a numpy list or array
    resultsTracker = tracker.update(detections)
    # ----------------------------
    # Part 4 in part 3
    # Draw a line in the video to act as the counting line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    # ----------------------------
    # To show the tracker and details
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        boundingbox = int(x1), int(y1), int(w), int(h)
        # Blue line (255, 0, 0) is the tracker detection
        cvzone.cornerRect(img, boundingbox, l=5, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(40, y1)),
                           scale=1, thickness=1, offset=5)
        #### Rmb if the id is in random, it is ok, as long as the id remains same for the same car

        # Part 5
        # Determine the center point of the vehicle where only when that point pass through, then it is counted
        cx, cy = x1+w//2, y1+h//2
        # Draw center point
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        # So now we want to make the line to become a detector to detect the car after the central point passes through
        # So the first part is the horizontal region, the second is the vertical region, and since we dont want to make it a pixel
        # because it might have issue that there is any delay in the system
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            # This will have a problem, run the code and show them, where the counter will increase as long as the car is in the region
            # totalCount += 1
            # So if the id is not present in the list, then we will add into the list
            if totalCount.count(id) == 0:
                totalCount.append(id)
                # After the car passby we can let the line to turn green to indicate the counter is working by overlapping another line on top
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f'Counter: {totalCount}', (50, 50))
    # We will count have many id is registered in the list, which is the total number of car added to the list
    cvzone.putTextRect(img, f'Counter: {len(totalCount)}', (50, 50))

    cv2.imshow("Image", img)
    # ----------------------------
    # Part 2
    # cv2.imshow("DetectRegion", detectRegion)
    # ----------------------------
    # 0, means need to press any key to play the video
    cv2.waitKey(0)
    # When want to show the actual code working, use this
    # cv2.waitKey(1)