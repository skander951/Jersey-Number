import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

import torch
import numpy as np
import random


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


ellipse_annotator = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']), thickness=2)
label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']), text_color=sv.Color.from_hex('#000000'), text_position=sv.Position.BOTTOM_CENTER)
triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex('#FFD700'), base=25, height=21, outline_thickness=1)


def round_any(d, rounded_values=4):
    if type(d) == dict:
        for key in d:
            d[key] = round_any(d[key], rounded_values)
    elif type(d) in [list, set, tuple]:
        for i in range(len(d)):
            d[i] = round_any(d[i], rounded_values)
    else:
        try:
            d = rounded(d, rounded_values=rounded_values)
        except:
            pass
    return d


def rounded(number, n=0, rounded_values=2):
    if number is None:
        return None
    if abs(number) >= 1 or n > 10:
        return round(number / (10 ** n), rounded_values + n)
    else:
        return rounded(number * 10, n + 1, rounded_values)

class PlayersTracker:
    def __init__(self, base_path, track_thresh=0.25,track_buffer=30,match_thresh=0.8,frame_rate=25, minimum_consecutive_frames=1):
        self.tracker = sv.ByteTrack(
                        track_activation_threshold =track_thresh,       # Confidence threshold for tracking
                        lost_track_buffer =track_buffer,       # Number of frames to keep a track alive after losing the object
                        minimum_matching_threshold =match_thresh,       # Threshold for matching detections to tracks
                        frame_rate =frame_rate,         # Frame rate of the video (adjust if you're skipping frames)
                        minimum_consecutive_frames = minimum_consecutive_frames         )
        self.base_path=base_path
        self.yolo_model = YOLO(f'{base_path}results/weights/yolov8n.pt', task='detect')


    
    def track_with_yolo(self, frame, persist=True, conf=0.35, iou=0.25):
        detections = []
        results = self.yolo_model.track([frame, frame], persist=persist, tracker="src\\tracking\\botsort.yaml", conf=conf, iou=iou, device='cuda', save=False, classes=[0], half=True)
        tracker_ids = results[-1].boxes.id.cpu().numpy().astype(int) if results[-1].boxes.id is not None else []
        for frame_idx, result in enumerate(results):
            if frame_idx == 1: 
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                for idx, box in enumerate(boxes):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    confidence = confidences[idx]
                    detection = (x_min, y_min,x_max,y_max, width, height, confidence, 'player')
                    detections.append(detection)

        return detections, tracker_ids
    
