import cv2 as cv 
import numpy as np
from tts import *
import warnings
warnings.filterwarnings("ignore")

# Distance constants 
KNOWN_DISTANCE = 45 #INCHES
PERSON_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES
CHAIR_WIDTH = 20.0
LAPTOP_WIDTH = 16
CUP_WIDTH = 3
KEYBOARD_WIDTH = 4

text1 = ""
text2 = ""

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(255, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0),(0, 0, 0),(255, 0, 255)]
GREEN =(255,255,255)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_DUPLEX

# getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov8-tiny.weights', 'yolov8-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, 0.4, 0.3)
    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        x1,y1,x2,y2 = box
        center_x, center_y =  ( x1 + x2 ) / 2, ( y1 + y2 ) / 2
        height, width, channels = image.shape
        # print(x1,y1,x2,y2)
        # define color of each, object based on its class id 

        if center_x <= width/3:
            W_pos = "left"
        elif center_x <= (width/3 * 2):
            W_pos = "center"
        else:
            W_pos = "right"
        
        if center_y <= height/3:
            H_pos = "top"
        elif center_y <= (height/3 * 2):
            H_pos = "mid"
        else:
            H_pos = "bottom"

        text1 = W_pos
        text2 = H_pos
        color= COLORS[int(classid) % len(COLORS)]

    
        label = "%s : %f" % (class_names[classid], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 1)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 1)

         # getting the data 
        # 1: class name  
        # 2: object width in pixels, 
        # 3: position where have to draw text(distance)

        # print("objects identified status")
        # print("person identified : ",classid == 0)
        # print("mobile identified : ",classid == 67)
        # print("chair identified : ",classid == 56)
        # print("bicycle identified : ",classid == 1)
        # print("car identified : ",classid == 2)
        # print("motorbike identified : ",classid == 3)
        # print("Laptop identified : ",classid == 72)
        # print("Keyboard identified : ",classid == 75)


        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 0: # person class id 
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2),text1,text2])
        elif classid == 1:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2),text1,text2])
        elif classid == 2: #car
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2),text1,text2])
        elif classid == 3:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2),text1,text2])
        elif classid == 67:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2),text1,text2])
        elif classid == 56:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2),text1,text2])
        elif classid == 72:
             data_list.append([class_names[classid], box[2], (box[0], box[1]-2),text1,text2])
        elif classid == 75:
             data_list.append([class_names[classid], box[2], (box[0], box[1]-2),text1,text2])
        elif classid == 41: #cup
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2),text1,text2])
        elif classid ==66: #keyboard
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2),text1,text2])
        
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data. 
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

# reading the reference image from dir 
ref_person = cv.imread('ReferenceImages/person.jpg')
ref_mobile = cv.imread('ReferenceImages/mobile.jpg')
ref_chair = cv.imread('ReferenceImages/image22.png')
ref_laptop = cv.imread('ReferenceImages/image2.png')
cup_image_path = cv.imread('ReferenceImages/cup.jpg')
kb_image_path =cv.imread('ReferenceImages/keyboard.jpg')



person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

chair_data = object_detector(ref_person)
chair_width_in_rf = chair_data[0][1]

lap_data = object_detector(ref_laptop)
lap_width_in_rf = person_data[0][1]

keyboard_data = object_detector(kb_image_path)
#print(keyboard_data)
keyboard_width_in_rf = keyboard_data[1][1]

cup_data = object_detector(cup_image_path)
#print(cup_data)
cup_width_in_rf = cup_data[1][1]


# print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
focal_chair = focal_length_finder(KNOWN_DISTANCE, CHAIR_WIDTH, chair_width_in_rf)
focal_latop = focal_length_finder(KNOWN_DISTANCE, LAPTOP_WIDTH, lap_width_in_rf)
focal_cup = focal_length_finder(KNOWN_DISTANCE, CUP_WIDTH, cup_width_in_rf)
focal_kb = focal_length_finder(KNOWN_DISTANCE, KEYBOARD_WIDTH, keyboard_width_in_rf)

#d[]

def get_frame_output(frame, frame_cnt):
    output_text_file = open('output_text.txt','w') 
    data = object_detector(frame)
    for d in data:
        if d[0] =='person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
        elif d[0] =='cell phone':
            distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'chair':
            distance = distance_finder (focal_chair, CHAIR_WIDTH, d[1])
            x, y = d[2]
        
        elif d[0] == 'laptop':
            distance = distance_finder (focal_latop, LAPTOP_WIDTH, d[1])
            x, y = d[2]
        
        elif d[0] =='cup':
            distance = distance_finder (focal_cup, CUP_WIDTH, d[1])
            x, y = d[2]  
        elif d[0] =='keyboard':
            distance = distance_finder (focal_kb, KEYBOARD_WIDTH, d[1])
            x, y = d[2]
        
        text1,text2=d[3],d[4]

        cv.rectangle(frame, (x, y-8), (x+250, y+29),BLACK, -1)
        cv.putText(frame, f'Distance - {round(distance,2)} inch', (x+5,y+13), FONTS, 0.58, GREEN, 1)
        
        OUTPUTtEXT=""

        if distance > 100:
            OUTPUTtEXT = "Get closer"

        elif (round(distance) > 50) and (text2 == "mid"):
            OUTPUTtEXT="Go straight"
        else:
            OUTPUTtEXT = (str(d[0]) + " " + str(int(round(distance,1))) +" inches"+" take left or right")
                
        output_text_file.write(OUTPUTtEXT)

        output_text_file.write("\n")
    
    output_text_file.close()
    
    return frame
