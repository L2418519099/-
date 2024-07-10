
from flask import Flask, Response
import cv2

from mask_test import ModelTest


app = Flask(__name__)


video_capture = cv2.VideoCapture('./test.mp4')

ret, frame = video_capture.read()

#myyolov8 = usr_yolov8()
model = ModelTest()
warnflag = False


def generate_frames():
    global warnflag
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print('123')
            break
        #frame = myyolov8.usr_yolo_run(frame)
        frame , labels = model.predict(frame)
        for i in labels:
            if i == "warning":
                warnflag = True
        print(warnflag)

if __name__ == "__main__":
	generate_frames()