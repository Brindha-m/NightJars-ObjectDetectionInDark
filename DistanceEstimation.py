import cv2 as cv
import numpy as np
from tts import *
from ultralytics import YOLO

# Distance constants
KNOWN_DISTANCE = 45  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES
CHAIR_WIDTH = 20.0  # INCHES
LAPTOP_WIDTH = 12  # INCHES

text1 = ""
text2 = ""

# Object detector constants
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors for detected objects
COLORS = [(151, 157, 255),(56, 56, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72), (23, 204, 146),
          (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100),
          (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
# Defining fonts
FONTS = cv.FONT_HERSHEY_PLAIN

# Load the YOLOv8 model
model_select = "yolov8xcdark.pt"
model = YOLO(model_select,'conf=0.45')  # You can replace it with 'yolov8-tiny.pt' if you want a smaller version

# Get class names from the YOLO model
class_names = model.names


# Object detector function 
def object_detector(image):
    results = model(image)
    data_list = []

    # Dictionary to store object center positions to avoid duplicates
    detected_objects = {}

    for result in results:
        for box, score, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            height, width, _ = image.shape

            # Check if the object is already detected based on its center position
            if (center_x, center_y) in detected_objects:
                continue  # Skip the duplicate object
            else:
                detected_objects[(center_x, center_y)] = True

            # Determine object position
            W_pos = "left" if center_x <= width / 3 else "center" if center_x <= 2 * width / 3 else "right"
            H_pos = "top" if center_y <= height / 3 else "mid" if center_y <= 2 * height / 3 else "bottom"

            text1, text2 = W_pos, H_pos
            color = COLORS[int(class_id) % len(COLORS)]
            label = f"{class_names[int(class_id)]} : {score:.2f}"

            # Draw bounding box and label
            cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv.putText(image, label, (x1, y1 - 10), FONTS, 0.5, color, 2)

            # Append relevant data
            if int(class_id) in [0, 67, 56, 72]:  # person, mobile, chair, laptop
                data_list.append([class_names[int(class_id)], x2 - x1, (x1, y1 - 2), text1, text2])

    return data_list

# Focal length and distance functions
def focal_length_finder(measured_distance, real_width, width_in_rf):
    return (width_in_rf * measured_distance) / real_width

def distance_finder(focal_length, real_object_width, width_in_frame):
    return (real_object_width * focal_length) / width_in_frame

# Reading reference images
ref_person = cv.imread('ReferenceImages/image14.png')
ref_mobile = cv.imread('ReferenceImages/image4.png')
ref_chair = cv.imread('ReferenceImages/image22.png')
ref_laptop = cv.imread('ReferenceImages/image2.png')

# Get reference widths
person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[0][1]

chair_data = object_detector(ref_chair)
chair_width_in_rf = chair_data[0][1]

laptop_data = object_detector(ref_laptop)
# laptop_width_in_rf = laptop_data[0][1]

# Calculate focal lengths
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
focal_chair = focal_length_finder(KNOWN_DISTANCE, CHAIR_WIDTH, chair_width_in_rf)
# focal_laptop = focal_length_finder(KNOWN_DISTANCE, LAPTOP_WIDTH, laptop_width_in_rf)

# Function to process each frame and write to the output text file
def get_frame_output(frame, frame_cnt):
    output_text_file = open('output_text.txt', 'w')
    data = object_detector(frame)

    for d in data:
        if d[0] == 'person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
        elif d[0] == 'cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
        elif d[0] == 'chair':
            distance = distance_finder(focal_chair, CHAIR_WIDTH, d[1])
        # elif d[0] == 'laptop':
        #     distance = distance_finder(focal_laptop, LAPTOP_WIDTH, d[1])

        x, y = d[2]
        text1, text2 = d[3], d[4]


        # Overlay distance information on the frame
        cv.rectangle(frame, (x+2, y+4), (x + 150, y + 20), BLACK, -1)
        cv.putText(frame, f'Distance: {round(distance, 2)} inches', (x + 7, y + 17), FONTS, 0.58, WHITE, 1)

        # Generate output text based on position and distance
        OUTPUT_TEXT = ""
        if distance > 100:
            OUTPUT_TEXT = "Get closer"
        elif 50 < round(distance) <= 100 and text2 == "mid":
            OUTPUT_TEXT = "Go straight"
        else:
            OUTPUT_TEXT = f"{d[0]} {int(round(distance))} inches, take left or right"

        output_text_file.write(OUTPUT_TEXT + "\n")

    output_text_file.close()
    return frame


def get_live_frame_output(frame, result_list_json):
    output_text_file = open('output_text.txt', 'w')

    print("Im here are get live frame")
    # Iterate over the detection results in result_list_json
    for result in result_list_json:
        class_name = result['class']
        box = result['bbox']
        x1, y1, x2, y2 = box['x_min'], box['y_min'], box['x_max'], box['y_max']
        width = x2 - x1

        distance = None

        # Determine the distance based on the detected object class
        if class_name == 'person':
            distance = distance_finder(focal_person, PERSON_WIDTH, width)
        elif class_name == 'cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, width)
        elif class_name == 'chair':
            distance = distance_finder(focal_chair, CHAIR_WIDTH, width)
        # elif class_name == 'laptop':
        #     distance = distance_finder(focal_laptop, LAPTOP_WIDTH, width)

        # Calculate the object's center and positional text (W_pos and H_pos)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        height, frame_width, _ = frame.shape
        W_pos = "left" if center_x <= frame_width / 3 else "center" if center_x <= 2 * frame_width / 3 else "right"
        H_pos = "top" if center_y <= height / 3 else "mid" if center_y <= 2 * height / 3 else "bottom"
        text1, text2 = W_pos, H_pos

        # Overlay distance information on the frame
        cv.rectangle(frame, (x1 + 2, y1 + 4), (x1 + 150, y1 + 20), BLACK, -1)
        cv.putText(frame, f'Distance: {round(distance, 2)} inches', (x1 + 7, y1 + 17), FONTS, 0.58, WHITE, 1)

        print(distance)


        # Generate output text based on position and distance
        OUTPUT_TEXT = ""
        if distance > 100:
            OUTPUT_TEXT = "Get closer"
        elif 50 < round(distance) <= 100 and text2 == "mid":
            OUTPUT_TEXT = "Go straight"
        else:
            OUTPUT_TEXT = f"{class_name} {int(round(distance))} inches, take left or right"

        # Write the output text to a file
        output_text_file.write(OUTPUT_TEXT + "\n")

    output_text_file.close()
    return frame
