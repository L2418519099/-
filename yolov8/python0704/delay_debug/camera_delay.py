# 导入所需的库
from flask import Flask, request, jsonify,Response
from flask_cors import CORS
import cv2
#from myyolov8 import usr_yolov8
from yolov80627 import usr_yolov8
import time
import json


import sys
sys.path.append('../yolo5_debug')
from mask_test import ModelTest
from message_test import SmsService

# 创建一个视频捕获对象
video_capture = cv2.VideoCapture(0)
myyolov8 = usr_yolov8()

def generate_frames():
    global video_capture
    while True:
        start_time = time.time()
        # 从摄像头读取一帧
        ret, frame = video_capture.read()
        if not ret:
            #print('获取失败')
            continue
        frame_src = frame.copy()
        res = False
        res,frame_new = myyolov8.usr_yolo_run(frame)
        if res != True:
            frame_new = frame_src
        
        text = "FPS:%.2f"%(1/(time.time()-start_time))
        AddText = frame_new.copy()
        cv2.putText(AddText, text, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 5)
        cv2.imshow('Video', AddText)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
      

if __name__ == '__main__':
    generate_frames()