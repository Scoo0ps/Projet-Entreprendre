import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort


class SingleHornetTracker:
    def __init__(self, model):
        self.model = model
        self.tracker = Sort()
        self.target_id = None  # ID du frelon suivi
    
    def detect(self, frame):
        results = self.model(frame)
        detections = []
        for r in results[0].boxes:
            # filtre : détecter uniquement la classe "frelon" (ici cls=1, à adapter selon ton modèle)
            if int(r.cls[0]) == 1:
                detections.append({
                    "bbox": r.xyxy[0].tolist(),
                    "conf": float(r.conf[0]),
                    "class": int(r.cls[0])
                })
        return detections

    def pick_target(self, detections, frame_shape):
        h, w = frame_shape[:2]
        cx, cy = w // 2, h // 2
        min_dist = float('inf')
        chosen = None
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            fx = (x1 + x2) / 2
            fy = (y1 + y2) / 2
            dist = np.sqrt((fx - cx)**2 + (fy - cy)**2)
            if dist < min_dist:
                min_dist = dist
                chosen = d
        return chosen

    def update(self, detections, frame_shape):
        if self.target_id is None:
            target = self.pick_target(detections, frame_shape)
            if target is None:
                return []
            dets = [target["bbox"] + [target["conf"]]]
            tracks = self.tracker.update(dets)
            if len(tracks) == 0:
                return []
            self.target_id = int(tracks[0][4])
            return [{
                "track_id": self.target_id,
                "bbox": [float(x) for x in tracks[0][:4]]
            }]
        else:
            dets = [d["bbox"] + [d["conf"]] for d in detections]
            if len(dets) == 0:
                return []
            tracks = self.tracker.update(dets)
            for xmin, ymin, xmax, ymax, track_id in tracks:
                if int(track_id) == self.target_id:
                    return [{
                        "track_id": int(track_id),
                        "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
                    }]
            return []


class DroneSimulator:
    def __init__(self, frame_shape):
        self.h, self.w = frame_shape[:2]
        self.cx = self.w / 2
        self.cy = self.h / 2

    def move_toward(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        fx = (x_min + x_max) / 2
        fy = (y_min + y_max) / 2

        dx = fx - self.cx
        dy = fy - self.cy
        tolerance = 40

        # Mouvements horizontaux
        if dx > tolerance:
            print(" Drone tourne à DROITE")
        elif dx < -tolerance:
            print(" Drone tourne à GAUCHE")

        # Mouvements verticaux
        if dy > tolerance:
            print(" Drone DESCEND")
        elif dy < -tolerance:
            print(" Drone MONTE")

        # Distance selon la taille de la bbox
        box_width = x_max - x_min
        if box_width < self.w * 0.2:
            print(" Drone AVANCE (frelon trop loin)")
        elif box_width > self.w * 0.4:
            print(" Drone recule (frelon trop proche)")

################################################################
# Lecture vidéo + détection + tracking
################################################################

import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO


# focal = 50mm for most cameras

# THE SCRIPT USES CM FOR KNOWN WIDTH AND CALIB DISTANCE

# define drone simulator instance

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')
parser.add_argument('--known-width-cm', type=float, help='Known object width in cm for distance estimation (required with --focal or --calib-image)')

parser.add_argument('--focal-mm', type=float, help='Camera focal length in millimeters (e.g., 50mm lens)')
parser.add_argument('--sensor-width-mm', type=float, help='Camera sensor width in millimeters (e.g., 36mm for full frame, 23.5mm for APS-C)')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

known_width_cm = args.known_width_cm
focal_px = None

focal_mm = args.focal_mm
sensor_width_mm = args.sensor_width_mm

# Convert focal length from mm to pixels if sensor width is provided
if focal_mm and sensor_width_mm and user_res:
    image_width_px = int(user_res.split('x')[0])
    focal_px = (focal_mm * image_width_px) / sensor_width_mm
    print(f'Converted focal length: {focal_mm}mm -> {focal_px:.2f}px (sensor: {sensor_width_mm}mm, image: {image_width_px}px)')
elif focal_mm and sensor_width_mm:
    print('WARNING: --focal-mm and --sensor-width provided but --resolution not specified. Will compute focal_px after loading first frame.')

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Source vidéo
if args.source.isdigit():
    cap = cv2.VideoCapture(int(args.source))
else:
    cap = cv2.VideoCapture(args.source)

ret, frame = cap.read()
if not ret:
    print("Impossible de lire la source")
    sys.exit(0)

# Drone simulator
drone = DroneSimulator(frame.shape)

# Begin inference loop
while True:

    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder': # If source is image or image folder, load the image using its filename
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video': # If source is a video, load next frame from video file
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb': # If source is a USB camera, grab frame from camera
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    elif source_type == 'picamera': # If source is a Picamera, grab frames using picamera interface
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if (frame is None):
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Convert focal length from mm to pixels if not done yet (for sources without --resolution)
    if focal_mm and sensor_width_mm and focal_px is None:
        image_width_px = frame.shape[1]
        focal_px = (focal_mm * image_width_px) / sensor_width_mm
        print(f'Converted focal length: {focal_mm}mm -> {focal_px:.2f}px (sensor: {sensor_width_mm}mm, image: {image_width_px}px)')

    # Resize frame to desired display resolution
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # Run inference on frame
    results = model(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize variable for basic object counting example
    object_count = 0

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):

        # Get bounding box coordinates
        # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > 0.5:

            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

            # Basic example: count the number of objects in the image
            object_count = object_count + 1

            # Compute distance to detection center
            if (known_width_cm is not None) and (focal_px is not None):
                bbox_width_px = xmax - xmin
                if bbox_width_px > 0:
                    distance_cm = (known_width_cm * focal_px) / bbox_width_px
                    
                    # Get detection center coordinates
                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2
                    
                    # Draw center point
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
                    
                    # Display distance at center
                    dist_label = f'{distance_cm:.1f}cm'
                    cv2.putText(frame, dist_label, (int(center_x) - 30, int(center_y) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    #simulate drone
                    bbox = [xmin, ymin, xmax, ymax]
                    drone.move_toward(bbox)
                    


    # Calculate and draw framerate (if using video, USB, or Picamera source)
    if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw framerate
    
    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw total number of detected objects
    cv2.imshow('YOLO detection results',frame) # Display image
    if record: recorder.write(frame)

    # If inferencing on individual images, wait for user keypress before moving to next image. Otherwise, wait 5ms before moving to next frame.
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)
    
    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)


# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()