from ultralytics import YOLO
import cv2

# Weight of the model, n = Nano version, this will download nano version (Faster version, less accurate), m = medium (slower version, more accurate)
# l = large (Slow, accurate)
# model = YOLO('yolov8n.pt')
# model = YOLO('../Yolo-Weights/yolov8n.pt')
model = YOLO('../Yolo-Weights/yolov8n.pt')
# Pass the source into the model, "show=True" is to show the image result in the end
# n = less accurate, smaller size, l = more accurate and even found skateboard, greater size
# results = model("Images/1.png", show=True)

# results = model("Images/2.png", show=True)

results = model("Images/3.png", show=True)

# 0 means unless there are inputs or else dont do anything
cv2.waitKey(0)