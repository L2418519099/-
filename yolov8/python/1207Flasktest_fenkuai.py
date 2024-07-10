# 导入所需的库
from flask import Flask, Response
import cv2
from myyolov8 import usr_yolov8
#from yolov80627 import usr_yolov8
import time

# 创建一个flask应用
app = Flask(__name__)

# 创建一个视频捕获对象
video_capture = cv2.VideoCapture(0)

ret, frame = video_capture.read()
if not ret:
    print('获取失败')

myyolov8 = usr_yolov8()



# 定义一个生成器函数，用yield语句返回每个视频帧
def generate_frames():
    while True:
        start_time = time.time()
        # 从摄像头读取一帧
        ret, frame = video_capture.read()
        if not ret:
            print('获取失败')
            continue
        frame_src = frame.copy()
        res,frame_new = myyolov8.usr_yolo_run(frame)
        if res != True:
            frame_new = frame_src
        
        text = "FPS:%.2f"%(1/(time.time()-start_time))
        AddText = frame_new.copy()
        cv2.putText(AddText, text, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 5)
            # 将帧转换为JPEG格式
        encoded_frame = cv2.imencode('.jpg', AddText)[1].tobytes()
        # 用分隔符和换行符包装帧
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')

# 定义一个路由，用于发送视频流
@app.route('/video')
def video():
    # 创建一个flask的Response对象，将生成器函数作为参数传入，同时指定mimetype为'image/jpeg'
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='192.168.137.34', port=8001)