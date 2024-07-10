# 导入所需的库
from flask import Flask, Response
import cv2
from myyolov8 import usr_yolov8

# 创建一个flask应用
app = Flask(__name__)

# 创建一个视频捕获对象
video_capture = cv2.VideoCapture(0)

ret, frame = video_capture.read()

#myyolov8 = usr_yolov8()

# 定义一个生成器函数，用yield语句返回每个视频帧
def generate_frames():
    while True:
        # 从摄像头读取一帧
        ret, frame = video_capture.read()
        if not ret:
            print('获取失败')
            break
        #frame = myyolov8.usr_yolo_run(frame)

        # 将帧转换为JPEG格式
        encoded_frame = cv2.imencode('.jpg', frame)[1].tobytes()
        # 用分隔符和换行符包装帧
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')

# 定义一个路由，用于发送视频流
@app.route('/video')
def video():
    # 创建一个flask的Response对象，将生成器函数作为参数传入，同时指定mimetype为'image/jpeg'
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)