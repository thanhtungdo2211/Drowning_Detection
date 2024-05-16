import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import pyttsx3, threading, os

from my_utils import get_fps, resize_img
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class GridTracker:
    def __init__(self, source='001.mp4', NUM_TRACK_FRAME=15, weights=['best-5data-flip.pt']):
        # PARAMS FOR DETECTION
        self.agnostic_nms=False
        self.augment=False 
        self.classes=None 
        self.devi='cpu'
        self.exist_ok=False 
        self.img_size=640 
        self.imgsz = self.img_size
        self.name='exp' 
        self.trace=True 
        self.nosave=True 
        self.project='runs/detect' 
        self.save_conf=False 
        self.save_txt=False 
        self.update=False 

        self.view_img=True 
        self.weights=weights
        self.source = source

        self.conf_thres=0.7
        self.iou_thres=0.5 

        self.visual_algorithm = False

        # PARAMS FOR RECOGZITION
        self.GRID_MAP = {}
        self.NUM_WIDTH_GRID = 16
        self.NUM_HEIGH_GRID = 16
        
        self.RESET_TIME = 5
        self.NUM_TRACK_FRAME = NUM_TRACK_FRAME

        self.IS_SHOW_DETECT_BBOX = True
        self.IS_BLINK_WARNING_GRIDS = True
        self.IS_SAVE_DROWNE = False
        self.IS_WARNING_TALK = False
        self.IS_SHOW_TRACK_LINE = False 

        # LED
        self.IS_BLINK_LED = False

        self.build_detector()
        self.build_tracker()

    def build_detector(self):
        webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Initialize
        set_logging()
        self.device = select_device(self.devi)
        self.half = self.device.type != 'cpu'

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)
        stride = int(self.model.stride.max())
        self.imgsz = check_img_size(self.imgsz, s=stride)

        if self.trace:
            self.model = TracedModel(self.model, self.device, self.img_size)
        if self.half:
            self.model.half() 
        
        # Set Dataloader
        if webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  
            self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=stride)
        else:
            self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=stride)

        self.names = self.module.names if hasattr(self, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

    def build_tracker(self):
        # define grid 
        for i in range(self.NUM_WIDTH_GRID):
            for j in range(self.NUM_HEIGH_GRID):
                self.GRID_MAP[(i, j)] = [0, 0]

        self.p_time = time.time()
        self.p_fps_time = time.time()
        self.t_start = time.time()
        self.warning_grids = []

        if self.IS_WARNING_TALK:
            self.WARNING_TEXT = 'Có nguời chet duoi'

        if self.IS_BLINK_LED:
            import Jetson.GPIO as GPIO
            self.LED_PIN = 22
            self.LED_BUZZER = 24
            GPIO.setmode(GPIO.BOARD) 
            GPIO.setup(self.LED_PIN, GPIO.OUT, initial=GPIO.HIGH) 
            GPIO.setup(self.LED_BUZZER, GPIO.OUT, initial=GPIO.HIGH)
            GPIO.output(self.LED_PIN, GPIO.LOW)
            GPIO.output(self.LED_BUZZER, GPIO.LOW)
    
    def get_time(self):
        self.p_time = time.time()
    
    def preprocess_img(self, img):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  
        img /= 255.0 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect_img(self, img, im0s):
        img = self.preprocess_img(img)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t3 = time_synchronized()

        if self.visual_algorithm:
            self.draw_grid(im0s)
            self.draw_grid_count(im0s)

        self.grid_ids = []
        self.grid_ids_bbox = []

        for i, det in enumerate(pred):
            s, im0 = '', im0s

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum() 
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "


                grid_id = self.check_get_point_grid_id(im0, det[0])
                self.grid_ids.append(grid_id)
                self.grid_ids_bbox.append(det[0])

                if self.IS_SHOW_DETECT_BBOX:
                    for *xyxy, conf, cls in reversed(det):
                        if self.view_img: 
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)

            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        return im0

    def recognize_img(self, im0):
        if not self.IS_BLINK_WARNING_GRIDS:
            self.p_time = self.update_grid_map(self.grid_ids, self.grid_ids_bbox, self.p_time, im0)

        if self.IS_BLINK_WARNING_GRIDS:
            self.p_time, new_warning_grids = self.update_warning_grid_map(self.grid_ids, self.grid_ids_bbox, self.p_time, im0)

            self.warning_grids += new_warning_grids
            self.warning_grids = self.check_show_grid_warning(im0, self.warning_grids)
            
        return im0

    def draw_grid(self, img):
        w, h = img.shape[1], img.shape[0]

        # draw width grid
        w_scale = w // self.NUM_WIDTH_GRID
        for i in range(1, self.NUM_WIDTH_GRID):
            start_point = (i * w_scale, 0)
            end_point = (i * w_scale, h)
            cv2.line(img, start_point, end_point, (25,255,200), 2)

        # draw heigh grid
        h_scale = h // self.NUM_HEIGH_GRID
        for i in range(1, self.NUM_HEIGH_GRID):
            start_point = (0, i * h_scale)
            end_point = (w, i * h_scale)
            cv2.line(img, start_point, end_point, (25,255,200), 2)

    def draw_grid_count(self, img):
        w, h = img.shape[1], img.shape[0]

        w_scale = w // self.NUM_WIDTH_GRID
        h_scale = h // self.NUM_HEIGH_GRID
        
        for i in range(self.NUM_WIDTH_GRID):
            for j in range(self.NUM_HEIGH_GRID):
                cv2.putText(img, f'ID=({i},{j})', (w_scale * i + int(w_scale * 0.1), h_scale * j + int(h_scale * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
                cv2.putText(img, f'cnt={self.GRID_MAP[(i, j)][0]}', (w_scale * i + int(w_scale * 0.1), h_scale * j + int(h_scale * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
                cv2.putText(img, f'time={round(self.GRID_MAP[(i, j)][1], 2)}', (w_scale * i + int(w_scale * 0.1), h_scale * j + int(h_scale * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)

    def check_get_point_grid_id(self, img, bbox):
        w, h = img.shape[1], img.shape[0]
        center_point = (w/2, h/2)

        object_point = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)

        # draw and focus
        if self.IS_SHOW_TRACK_LINE:
            cv2.circle(img, (int(object_point[0]), int(object_point[1])), radius=3, color=(0, 0, 255), thickness=3)
            cv2.line(img, (int(object_point[0]), int(object_point[1])), (int(center_point[0]), int(center_point[1])), (0,255,255), 2)
        
        # find
        w_scale = w // self.NUM_WIDTH_GRID
        h_scale = h // self.NUM_HEIGH_GRID

        i = int(object_point[0] / w_scale)
        j = int(object_point[1] / h_scale)

        # validation
        if i < 0:
            i = 0
        elif i > self.NUM_WIDTH_GRID - 1:
            i = self.NUM_WIDTH_GRID - 1
        
        if j < 0:
            j = 0
        elif j > self.NUM_HEIGH_GRID - 1:
            j = self.NUM_HEIGH_GRID - 1

        return (i, j)

    def update_grid_map(self, grid_ids, grid_ids_bbox, p_time, img):
        # check all grid cells have object
        for index,grid_id in enumerate(grid_ids):
            if self.GRID_MAP[grid_id][0] < self.NUM_TRACK_FRAME:
                self.GRID_MAP[grid_id][0] += 1
                self.GRID_MAP[grid_id][1] = self.RESET_TIME
            else:
                # draw bbox
                w_delta = (grid_ids_bbox[index][2] - grid_ids_bbox[index][0]) * 0.2
                h_delta = (grid_ids_bbox[index][3] - grid_ids_bbox[index][1]) * 0.2
                cv2.rectangle(
                    img, 
                    (int(grid_ids_bbox[index][0] - w_delta), int(grid_ids_bbox[index][1] - h_delta)), 
                    (int(grid_ids_bbox[index][2] + w_delta), int(grid_ids_bbox[index][3] + h_delta)), 
                    (0,0,255), 
                    8
                )

                # save
                if self.IS_SAVE_DROWNE:
                    cv2.imwrite(os.path.join('tmp', f'{time.time()}.png'), img)

                # reset
                self.GRID_MAP[grid_id][0] = 0
                self.GRID_MAP[grid_id][1] = 0

                # # talk warning voice
                if self.IS_WARNING_TALK:
                    task2 = threading.Thread(target=self.talk_warning_voice, args=('text book',))
                    task2.start()



        # change remain time all grid cells don't have object
        for i in range(self.NUM_WIDTH_GRID):
            for j in range(self.NUM_HEIGH_GRID):
                if (i, j) not in grid_ids:
                    c_time = time.time()
                    self.GRID_MAP[(i, j)][1] -= (c_time - p_time)
                    if self.GRID_MAP[(i, j)][1] <= 0:
                        self.GRID_MAP[(i, j)][0] = 0
                        self.GRID_MAP[(i, j)][1] = 0

        p_time = c_time
        return p_time

    def update_warning_grid_map(self, grid_ids, grid_ids_bbox, p_time, img):
        new_warning_grids = []

        # check all grid cells have object
        for index,grid_id in enumerate(grid_ids):
            if self.GRID_MAP[grid_id][0] < self.NUM_TRACK_FRAME:
                self.GRID_MAP[grid_id][0] += 1
                self.GRID_MAP[grid_id][1] = self.RESET_TIME
            else:
                # draw bbox
                w_delta = (grid_ids_bbox[index][2] - grid_ids_bbox[index][0]) * 0.2
                h_delta = (grid_ids_bbox[index][3] - grid_ids_bbox[index][1]) * 0.2
                cv2.rectangle(
                    img, 
                    (int(grid_ids_bbox[index][0] - w_delta), int(grid_ids_bbox[index][1] - h_delta)), 
                    (int(grid_ids_bbox[index][2] + w_delta), int(grid_ids_bbox[index][3] + h_delta)), 
                    (0,0,255), 
                    8
                )

                # save
                if self.IS_SAVE_DROWNE:
                    cv2.imwrite(os.path.join('tmp', f'{time.time()}.png'), img)
                
                # reset
                self.GRID_MAP[grid_id][0] = 0
                self.GRID_MAP[grid_id][1] = 0

                # add warning
                new_warning_grids.append([grid_id, 30])

                # talk warning voice
                if self.IS_WARNING_TALK:
                    task2 = threading.Thread(target=self.talk_warning_voice, args=('text book',))
                    task2.start()

        # change remain time all grid cells don't have object
        for i in range(self.NUM_WIDTH_GRID):
            for j in range(self.NUM_HEIGH_GRID):
                if (i, j) not in grid_ids:
                    c_time = time.time()
                    self.GRID_MAP[(i, j)][1] -= (c_time - p_time)
                    if self.GRID_MAP[(i, j)][1] <= 0:
                        self.GRID_MAP[(i, j)][0] = 0
                        self.GRID_MAP[(i, j)][1] = 0

        p_time = c_time
        return p_time, new_warning_grids

    def check_show_grid_warning(self, img, warning_grids):
        w, h = img.shape[1], img.shape[0]
        w_scale = w // self.NUM_WIDTH_GRID
        h_scale = h // self.NUM_HEIGH_GRID
        len_warning_grid = len(warning_grids)


        for i in range(len_warning_grid-1, -1, -1):
            if warning_grids[i][1] > 0:
                # show box
                start_point = (warning_grids[i][0][0] * w_scale, warning_grids[i][0][1] * h_scale)
                end_point = ((warning_grids[i][0][0] + 1) * w_scale, (warning_grids[i][0][1] + 1) * h_scale)
                cv2.rectangle(img, start_point, end_point, (255,0,0), 8)

                warning_grids[i][1] -= 1
            else:
                warning_grids.pop(i)
        
        return warning_grids

    def talk_warning_voice(self, text):
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[65].id)
        engine.setProperty('rate', 150)
        GPIO.output(self.LED_PIN, GPIO.HIGH)
        for i in range(2):
            engine.say(self.WARNING_TEXT)
            engine.runAndWait()

        GPIO.output(self.LED_PIN, GPIO.LOW)
