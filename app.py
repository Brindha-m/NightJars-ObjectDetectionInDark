import os
os.environ["MY_ENV_VARIABLE"] = "True"
import cv2
import json
import subprocess
import numpy as np
import pandas as pd
from _collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort
from stqdm import stqdm
from collections import Counter
import time
from ultralytics import YOLO
from ultralytics.engine.results import Results
import json
from model_utils import get_yolo, get_system_stat
from streamlit_webrtc import RTCConfiguration, VideoTransformerBase, webrtc_streamer
from DistanceEstimation import *
from streamlit_autorefresh import st_autorefresh
import streamlit as st
import av
from tts import *
import torch
import intel_extension_for_pytorch as ipex
import openvino.runtime as ov
import gc
from pathlib import Path
import gdown
import requests
import os
import zipfile

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# colors for visualization for image visualization
COLORS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72), (23, 204, 146),
          (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100),
          (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]



def result_to_json(result: Results, tracker=None):
    """
    Convert result from ultralytics YOLOv8 prediction to json format
    Parameters:
        result: Results from ultralytics YOLOv8 prediction
        tracker: DeepSort tracker
    Returns:
        result_list_json: detection result in json format
    """
    len_results = len(result.boxes)
    result_list_json = [
        {
            'class_id': int(result.boxes.cls[idx]),
            'class': result.names[int(result.boxes.cls[idx])],
            'confidence': float(result.boxes.conf[idx]),
            'bbox': {
                'x_min': int(result.boxes.xyxy[idx][0]),
                'y_min': int(result.boxes.xyxy[idx][1]),
                'x_max': int(result.boxes.xyxy[idx][2]),
                'y_max': int(result.boxes.xyxy[idx][3]),
                
                # 'x_min': int(result.boxes.boxes[idx][0]),
                # 'y_min': int(result.boxes.boxes[idx][1]),
                # 'x_max': int(result.boxes.boxes[idx][2]),
                # 'y_max': int(result.boxes.boxes[idx][3]),
            },
        } for idx in range(len_results)
    ]
    if result.masks is not None:
        for idx in range(len_results):
            # result_list_json[idx]['mask'] = cv2.resize(result.masks.data[idx], (width, height))
            result_list_json[idx]['mask'] = cv2.resize(result.masks.data[idx].cpu().numpy(), (result.orig_shape[1], result.orig_shape[0])).tolist()
            result_list_json[idx]['segments'] = result.masks.segments[idx].tolist()
    if tracker is not None:
        bbs = [
            (
                [
                    result_list_json[idx]['bbox']['x_min'],
                    result_list_json[idx]['bbox']['y_min'],
                    result_list_json[idx]['bbox']['x_max'] - result_list_json[idx]['bbox']['x_min'],
                    result_list_json[idx]['bbox']['y_max'] - result_list_json[idx]['bbox']['y_min']
                ],
                result_list_json[idx]['confidence'],
                result_list_json[idx]['class'],
            ) for idx in range(len_results)
        ]
        tracks = tracker.update_tracks(bbs, frame=result.orig_img)
        for idx in range(len(result_list_json)):
            track_idx = next((i for i, track in enumerate(tracks) if track.det_conf is not None and np.isclose(track.det_conf, result_list_json[idx]['confidence'])), -1)
            if track_idx != -1:
                result_list_json[idx]['object_id'] = int(tracks[track_idx].track_id)
    return result_list_json



def view_result_ultralytics(result: Results, result_list_json, centers=None):
    """
    Visualize result from ultralytics YOLOv8 prediction using ultralytics YOLOv8 built-in visualization function
    Parameters:
        result: Results from ultralytics YOLOv8 prediction
        result_list_json: detection result in json format
        centers: list of deque of center points of bounding boxes
    Returns:
        result_image_ultralytics: result image from ultralytics YOLOv8 built-in visualization function
    """
    result_image_ultralytics = result.plot()
    for result_json in result_list_json:
        class_color = COLORS[result_json['class_id'] % len(COLORS)]
        if 'object_id' in result_json and centers is not None:
            centers[result_json['object_id']].append((int((result_json['bbox']['x_min'] + result_json['bbox']['x_max']) / 2), int((result_json['bbox']['y_min'] + result_json['bbox']['y_max']) / 2)))
            for j in range(1, len(centers[result_json['object_id']])):
                if centers[result_json['object_id']][j - 1] is None or centers[result_json['object_id']][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(result_image_ultralytics, centers[result_json['object_id']][j - 1], centers[result_json['object_id']][j], class_color, thickness)
    return result_image_ultralytics


def view_result_default(result: Results, result_list_json, centers=None):
    """
    Visualize result from ultralytics YOLOv8 prediction using default visualization function
    Parameters:
        result: Results from ultralytics YOLOv8 prediction
        result_list_json: detection result in json format
        centers: list of deque of center points of bounding boxes
    Returns:
        result_image_default: result image from default visualization function
    """
    ALPHA = 0.5
    image = result.orig_img
    for result in result_list_json:
        class_color = COLORS[result['class_id'] % len(COLORS)]
        # fontFace = "/content/drive/MyDrive/Yolov8_Nightjars/models/ahronbd.ttf"
        fontScale = 1
        if 'mask' in result:
            image_mask = np.stack([np.array(result['mask']) * class_color[0], np.array(result['mask']) * class_color[1], np.array(result['mask']) * class_color[2]], axis=-1).astype(np.uint8)
            image = cv2.addWeighted(image, 1, image_mask, ALPHA, 0)
        text = f"{result['class']} {result['object_id']}: {result['confidence']:.2f}" if 'object_id' in result else f"{result['class']}: {result['confidence']:.2f}"
        cv2.rectangle(image, (result['bbox']['x_min'], result['bbox']['y_min']), (result['bbox']['x_max'], result['bbox']['y_max']), class_color, 1)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.85, 1)
        cv2.rectangle(image, (result['bbox']['x_min'], result['bbox']['y_min'] - text_height - baseline), (result['bbox']['x_min'] + text_width, result['bbox']['y_min']), class_color, -1)
        cv2.putText(image, text , (result['bbox']['x_min'], result['bbox']['y_min'] - baseline), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 1)
        if 'object_id' in result and centers is not None:
            centers[result['object_id']].append((int((result['bbox']['x_min'] + result['bbox']['x_max']) / 2), int((result['bbox']['y_min'] + result['bbox']['y_max']) / 2)))
            for j in range(1, len(centers[result['object_id']])):
                if centers[result['object_id']][j - 1] is None or centers[result['object_id']][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(image, centers[result['object_id']][j - 1], centers[result['object_id']][j], class_color, thickness)
    return image


def image_processing(frame, model, image_viewer=view_result_default, tracker=None, centers=None):
    """
    Process image frame using ultralytics YOLOv8 model and possibly DeepSort tracker if it is provided
    Parameters:
        frame: image frame
        model: ultralytics YOLOv8 model
        image_viewer: function to visualize result, default is view_result_default, can be view_result_ultralytics
        tracker: DeepSort tracker
        centers: list of deque of center points of bounding boxes
    Returns:
        result_image: result image with bounding boxes, class names, confidence scores, object masks, and possibly object IDs
        result_list_json: detection result in json format
    """
    results = model.predict(frame)
    result_list_json = result_to_json(results[0], tracker=tracker)
    result_image = image_viewer(results[0], result_list_json, centers=centers)
    return result_image, result_list_json

# @st.cache_data
def video_processing(video_file, model, image_viewer=view_result_default, tracker=None, centers=None):
    """
    Process video file using ultralytics YOLOv8 model and possibly DeepSort tracker if it is provided
    Parameters:
        video_file: video file
        model: ultralytics YOLOv8 model
        image_viewer: function to visualize result, default is view_result_default, can be view_result_ultralytics
        tracker: DeepSort tracker
        centers: list of deque of center points of bounding boxes
    Returns:
        video_file_name_out: name of output video file
        result_video_json_file: file containing detection result in json format
    """
    results = model.predict(video_file)
    model_name = model.ckpt_path.split('/')[-1].split('.')[0]
    output_folder = os.path.join('output_videos', video_file.split('.')[0])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video_file_name_out = os.path.join(output_folder, f"{video_file.split('.')[0]}_{model_name}_output.mp4")
    if os.path.exists(video_file_name_out):
        os.remove(video_file_name_out)
    result_video_json_file = os.path.join(output_folder, f"{video_file.split('.')[0]}_{model_name}_output.json")
    if os.path.exists(result_video_json_file):
        os.remove(result_video_json_file)
    json_file = open(result_video_json_file, 'a')
    temp_file = 'temp.mp4'
    video_writer = cv2.VideoWriter(temp_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (results[0].orig_img.shape[1], results[0].orig_img.shape[0]))
    json_file.write('[\n')
    for result in stqdm(results, desc=f"Processing video"):
        result_list_json = result_to_json(result, tracker=tracker)
        result_image = image_viewer(result, result_list_json, centers=centers)
        video_writer.write(result_image)
        json.dump(result_list_json, json_file, indent=2)
        json_file.write(',\n')
    json_file.write(']')
    video_writer.release()
    subprocess.call(args=f"ffmpeg -i {os.path.join('.', temp_file)} -c:v libx264 {os.path.join('.', video_file_name_out)}".split(" "))
    os.remove(temp_file)
    return video_file_name_out, result_video_json_file


# @st.cache_resource
# def load_model(model_path):
#     # Load and return the YOLO model
#     return YOLO(model_path)




st.set_page_config(page_title="NightJars YOLOv8 ", layout="wide", page_icon="detective.ico")
st.title("Intel Custom YOLOv8 Dark Object Detection 📸🕵🏻‍♀️")


# Global OpenVINO core instance
core = ov.Core()

# Function to compile OpenVINO models
@st.cache_resource
def compile_model(det_model_path, device):
    det_ov_model = core.read_model(det_model_path)

    # OpenVINO configuration
    ov_config = {}
    if device != "CPU":
        det_ov_model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}

    det_compiled_model = core.compile_model(det_ov_model, device, ov_config)
    return det_compiled_model

# Function to load YOLO model and integrate OpenVINO
@st.cache_resource
def load_openvino_model(model_dir, device):
    # Define paths to OpenVINO files
    det_model_path = Path(model_dir) / "yolovc8x.xml"  # Adjust for your actual file name if necessary
    compiled_model = compile_model(det_model_path, device)

    # Initialize YOLO with OpenVINO
    det_model = YOLO(model_dir, task="detect")

    if det_model.predictor is None:
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # Default arguments
        args = {**det_model.overrides, **custom}
        det_model.predictor = det_model._smart_load("predictor")(overrides=args, _callbacks=det_model.callbacks)
        det_model.predictor.setup_model(model=det_model.model)

    det_model.predictor.model.ov_compiled_model = compiled_model
    return det_model
    

device = "CPU"  # Change environment: "GPU", "AUTO", etc.

# Function to download file from Google Drive
def download_file_from_gdrive(file_id, local_filename):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    if not os.path.exists(local_filename):
        gdown.download(url, local_filename, quiet=False)
    else:
        print(f"File {local_filename} already exists locally. Skipping download.")

# Define file IDs and local paths
file_id_detection = '1hE6iWo6RmrH5i-z7H2yfvzYi8kh8dMlC'
local_filename_detection = 'yolovc8x_openvino_model.zip'
# file_id_segmentation = 'your_file_id_for_segmentation_model'  # Replace with actual file ID
# local_filename_segmentation = 'yolov8xcdark_openvino_model.zip'

# Download model files
download_file_from_gdrive(file_id_detection, local_filename_detection)
download_file_from_gdrive(file_id_segmentation, local_filename_segmentation)

# Extract the zip files
import zipfile
with zipfile.ZipFile(local_filename_detection, 'r') as zip_ref:
    zip_ref.extractall("yolovc8x_openvino_model")
with zipfile.ZipFile(local_filename_segmentation, 'r') as zip_ref:
    zip_ref.extractall("yolovc8xcdark_openvino_model")

# Load models
model_dir = "yolovc8x_openvino_model"
model_seg_dir = "yolov8xcdark_openvino_model"

model = load_openvino_model(Path(model_dir) / "yolovc8x.xml", device)
# model1 = load_openvino_model(Path(model_seg_dir) / "model.xml", device)

st.write("Models loaded successfully!")

# Cache seg model paths
model1= YOLO("yolov8xcdark-seg.pt")


source = ("Image Detection📸", "Video Detections📽️", "Live Camera Detection🤳🏻","RTSP","MOBILE CAM")
source_index = st.sidebar.selectbox("Select Input type", range(
        len(source)), format_func=lambda x: source[x])


# Image detection section

if source_index == 0:
    st.header("Image Processing using YOLOv8")
    image_file = st.file_uploader("Upload an image 🔽", type=["jpg", "jpeg", "png"])
    process_image_button = st.button("Detect")
    process_seg_button = st.button("Click here for Segmentation result")

    with st.spinner("Detecting with 💕"):
        if image_file is None and process_image_button:
            st.warning("Please upload an image file to be processed!")
        if image_file is not None and process_image_button:
            st.write(" ")
            st.sidebar.success("Successfully uploaded")
            st.sidebar.image(image_file, caption="Uploaded image")
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
            
            # For detection with bounding boxes
            print(f"Used Custom reframed YOLOv8 model: {model_select}")
            
            img, result_list_json = image_processing(img, model)
            st.success("✅ Task Detect : Detection using custom-trained v8 model")
            st.image(img, caption="Detected image", channels="BGR")     
            # Current number of classes
            detected_classes = [item['class'] for item in result_list_json]
            class_fq = Counter(detected_classes)
            
            # Create a DataFrame for class frequency
            df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
            
            # Display class frequency count as a table
            st.write("Class Frequency:")
            st.dataframe(df_fq)  # Display the class frequency DataFrame
            
               
           
 

# Video & Live cam section
if source_index == 1:

    st.header("Video & Live Cam Processing using YOLOv8")
    video_file = st.file_uploader("Upload a video", type=["mp4"])
    process_video_button = st.button("Process Video")
    if video_file is None and process_video_button:
        st.warning("Please upload a video file to be processed!")
    if video_file is not None and process_video_button:
        with st.spinner(text='Detecting with 💕...'):
            tracker = DeepSort(max_age=5)
            centers = [deque(maxlen=30) for _ in range(10000)]
            open(video_file.name, "wb").write(video_file.read())
            video_file_out, result_video_json_file = video_processing(video_file.name, model, tracker=tracker, centers=centers)
            os.remove(video_file.name)
            # print(json.dumps(result_video_json_file, indent=2))
            video_bytes = open(video_file_out, 'rb').read()
            st.video(video_bytes)

    

if source_index == 2:
    st.header("Live Stream Processing using YOLOv8")
    tab_webcam = st.tabs(["Webcam Detections"])
    p_time = 0

    st.sidebar.title('Settings')
    # Choose the model
    model_type = "YOLOv8"
    sample_img = cv2.imread('detective.png')
    FRAME_WINDOW = st.image(sample_img, channels='BGR')
    cap = None


    if not model_type == 'YOLO Model':
        if model_type == 'YOLOv8':
            # GPU
            gpu_option = st.sidebar.radio(
                'Choose between:', ('CPU', 'GPU'))
            # Model
            if gpu_option == 'CPU':
                model = model
            if gpu_option == 'GPU':
                model = model

        

        # Load Class names
        class_labels = model.names


        # Confidence
        confidence = st.sidebar.slider(
            'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

        # Draw thickness
        draw_thick = st.sidebar.slider(
            'Draw Thickness:', min_value=1,
            max_value=20, value=3
        )
        
        # Inference Mode
        # Web-cam
        
        cam_options = st.selectbox('Webcam Channel',
                                        ('Select Channel', '0', '1', '2', '3'))
    
        if not cam_options == 'Select Channel':
            pred = st.checkbox(f'Predict Using {model_type}')
            cap = cv2.VideoCapture(int(cam_options))
            if (cap != None) and pred:
                stframe1 = st.empty()
                stframe2 = st.empty()
                stframe3 = st.empty()
                tracker = DeepSort(max_age=5)
                centers = [deque(maxlen=30) for _ in range(10000)]
                
                
                while True:
                    success, img = cap.read()
                    if not success:
                        st.error(
                            f" NOT working\nCheck {cam_options} properly!!",
                            icon="🚨"
                        )
                        break
                              

 
                    # # Call get_yolo to get detections
                    # img, current_no_class = get_yolo(img, model_type, model, confidence, class_labels, draw_thick)

                    # Call DeepSort for tracking
                    img, result_list_json = image_processing(img, model, image_viewer=view_result_default, tracker=tracker, centers=centers)

                    # # Call get_frame_output to overlay distance information
                    processed_frame = get_live_frame_output(img, result_list_json)
                    

                  
                    # Display the processed frame
                    FRAME_WINDOW.image(processed_frame, channels='BGR')
                    st.cache_data.clear()

                    
                    # FPS
                    c_time = time.time()
                    fps = 1 / (c_time - p_time)
                    p_time = c_time
                        
                    # Current number of classes
                    # Current number of classes
                    detected_classes = [item['class'] for item in result_list_json]
                    class_fq = Counter(detected_classes)
                    df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])

                        # Updating Inference results
                    get_system_stat(stframe1, stframe2, stframe3, fps, df_fq)
                    
                    


if source_index == 3:
    st.header("Live Stream Processing using YOLOv8")
    tab_rtsp = st.tabs(["RTSP Detections"])
    p_time = 0

    st.sidebar.title('Settings')
    # Choose the model
    model_type = "YOLOv8"
    sample_img = cv2.imread('detective.png')
    FRAME_WINDOW = st.image(sample_img, channels='BGR')
    cap = None

    if not model_type == 'YOLO Model':
        
        if model_type == 'YOLOv8':
            # GPU
            gpu_option = st.sidebar.radio(
                'Choose between:', ('CPU', 'GPU'))

        
            # Model
            if gpu_option == 'CPU':
                model = model
                # model = custom(path_or_model=path_model_file)
            if gpu_option == 'GPU':
                model = model
                # model = custom(path_or_model=path_model_file, gpu=True)

        

        # Load Class names
        class_labels = model.names


        # Confidence
        confidence = st.sidebar.slider(
            'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

        # Draw thickness
        draw_thick = st.sidebar.slider(
            'Draw Thickness:', min_value=1,
            max_value=20, value=3
        )
        


        
        rtsp_url = st.text_input(
            'RTSP URL:',
            'eg: rtsp://admin:name6666@198.162.1.58/cam/realmonitor?channel=0&subtype=0'
        )
        pred1 = st.checkbox(f'Predict Using rtsp {model_type}')
        cap = cv2.VideoCapture(rtsp_url)

        if (cap != None) and pred1:
                stframe1 = st.empty()
                stframe2 = st.empty()
                stframe3 = st.empty()
                
                tracker = DeepSort(max_age=5)
                centers = [deque(maxlen=30) for _ in range(10000)]

                while True:
                    success, img = cap.read()
                    if not success:
                        st.error(
                            f" NOT working\nCheck {cam_options} properly!!",
                            icon="🚨"
                        )
                        break

                    
                    # Call DeepSort for tracking
                    img, result_list_json = image_processing(img, model, image_viewer=view_result_default, tracker=tracker, centers=centers)

                    # # Call get_frame_output to overlay distance information
                    processed_frame = get_live_frame_output(img, result_list_json)
                    
                  
                    # Display the processed frame
                    FRAME_WINDOW.image(processed_frame, channels='BGR')

                    # FPS
                    c_time = time.time()
                    fps = 1 / (c_time - p_time)
                    p_time = c_time
                        
                    # Current number of classes
                    detected_classes = [item['class'] for item in result_list_json]
                    class_fq = Counter(detected_classes)
                    df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                   
                        # Updating Inference results
                    get_system_stat(stframe1, stframe2, stframe3, fps, df_fq)
                                    
                    


class VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        super().__init__()
        self.frame_count = 0
        self.tracker = DeepSort(max_age=5)  # Initialize the DeepSort tracker
        self.centers = [deque(maxlen=30) for _ in range(10000)]  # Initialize centers deque

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process the frame using image_processing
        img, result_list_json = image_processing(img, model, image_viewer=view_result_default, tracker=self.tracker, centers=self.centers)
        
        # Call get_frame_output to overlay distance information
        processed_frame = get_live_frame_output(img, result_list_json)
        
        return processed_frame

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        new_image = self.transform(frame)
        return av.VideoFrame.from_ndarray(new_image, format="bgr24")


# Streamlit application
if source_index == 4:
    st.header("Live Stream Processing using YOLOv8")
    webcam_st = st.tabs(["St webcam"])
    p_time = 0

    # RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    RTC_CONFIGURATION = RTCConfiguration({
              "iceServers": [
                  {"urls": ["stun:openrelay.metered.ca:80"]},  # Free public STUN server
                  {"urls": ["turn:openrelay.metered.ca:80"], "username": "user", "credential": "pass"}  # Example TURN server
              ]
          })
          
    count = st_autorefresh(interval=4500, limit=1000000, key="fizzbuzzcounter")
    try:
      webrtc_streamer(
        key="test",
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoTransformer
    )
    except Exception as e:
      st.error(f"Error initializing WebRTC: {e}")
    # webrtc_streamer(
    #     key="test",
    #     media_stream_constraints={"video": True, "audio": False},
    #     video_processor_factory=VideoTransformer
    # )
    st.cache_data.clear()
