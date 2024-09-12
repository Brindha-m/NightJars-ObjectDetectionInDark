import cv2 as cv
import torch
from ultralytics import YOLO

# setting parameters
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5

# colors for object detected
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
PINK = (147, 20, 255)
fonts = cv.FONT_HERSHEY_COMPLEX

# Load the YOLOv8 model
model_select = "yolov8xcdark.pt"
model = YOLO(model_select,'conf=0.45')
# Verify that the model loaded correctly
if not model:
    raise ValueError("Failed to load YOLOv8 model. Check the file path or model integrity.")

# reading class names from YOLOv8 model
class_names = model.names  # This will get the class names from the loaded YOLOv8 model

def ObjectDetector(image):
    # Resize the frame to fit the input size of YOLOv8
    image_resized = cv.resize(image, (640, 640))  # YOLOv8 works well with 640x640 resolution
    image_rgb = cv.cvtColor(image_resized, cv.COLOR_BGR2RGB)  # YOLO expects RGB images

    # Perform detection
    results = model(image_rgb)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Get confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Get class IDs
        
        for (box, score, class_id) in zip(boxes, scores, class_ids):
            if score > CONFIDENCE_THRESHOLD:  # Apply confidence threshold
                color = COLORS[int(class_id) % len(COLORS)]
                label = f"{class_names[int(class_id)]} : {score:.2f}"
                
                # Draw rectangle and label on original image (not the resized one)
                cv.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv.putText(image, label, (int(box[0]), int(box[1])-10), fonts, 0.5, color, 2)

# setting up camera
camera = cv.VideoCapture(0)
counter = 0
capture = False
number = 0

while True:
    ret, frame = camera.read()
    
    if not ret:
        break

    original = frame.copy()
    ObjectDetector(frame)
    
    # Show original frame with detections
    cv.imshow('frame', frame)
    
    if capture and counter < 10:
        counter += 1
        cv.putText(frame, f"Capturing Img No: {number}", (30, 30), fonts, 0.6, PINK, 2)
    else:
        counter = 0

    # Show original frame (before detection)
    cv.imshow('original', original)

    key = cv.waitKey(1)
    
    if key == ord('c'):
        capture = True
        number += 1
        cv.imwrite(f'ReferenceImages/image{number}.png', original)
    if key == ord('q'):
        break

camera.release()
cv.destroyAllWindows()
