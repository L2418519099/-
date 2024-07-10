from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import time
import subprocess
import threading

# 创建 Flask 应用 1
app1 = Flask(__name__)
CORS(app1, supports_credentials=True, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# 使用 v4l2-ctl 命令选择第一个视频流
subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--set-fmt-video=width=1280,height=720,pixelformat=0"])

# 创建一个 VideoCapture 对象，用于访问第一个摄像头

# 定义一个生成器函数，用于生成第一个摄像头的视频帧
def generate_frames_1():
    while True:
        video_capture_1 = cv2.VideoCapture(0)
        start = time.time()
        ret, frame = video_capture_1.read()
        video_capture_1.release()
        if not ret:
            continue
        text = "FPS:%.2f" % (1 / (time.time() - start))
        AddText = frame.copy()
        cv2.putText(AddText, text, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 5)

        encoded_frame = cv2.imencode('.jpg', AddText)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')

# 定义一个路由，用于发送第一个摄像头的视频流
@app1.route('/video1')
def video1():
    return Response(generate_frames_1(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app1.route('/api/save_coordinates1', methods=['POST'])
def save_coordinates1():
    coordinates = request.get_json()
    print(coordinates)
    return jsonify({"status": "success", "message": "Coordinates saved."}), 200

# 创建 Flask 应用 2
app2 = Flask(__name__)
CORS(app2, supports_credentials=True, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# 使用 v4l2-ctl 命令选择第二个视频流
subprocess.run(["v4l2-ctl", "-d", "/dev/video2", "--set-fmt-video=width=1280,height=720,pixelformat=0"])

# 创建一个 VideoCapture 对象，用于访问第二个摄像头

# 定义一个生成器函数，用于生成第二个摄像头的视频帧
def generate_frames_2():
    while True:
        video_capture_2 = cv2.VideoCapture(2)
        start = time.time()
        ret, frame = video_capture_2.read()
        video_capture_2.release()
        if not ret:
            print('video2 error ')
            continue
        text = "FPS:%.2f" % (1 / (time.time() - start))
        AddText = frame.copy()
        cv2.putText(AddText, text, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 5)

        encoded_frame = cv2.imencode('.jpg', AddText)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')

# 定义一个路由，用于发送第二个摄像头的视频流
@app2.route('/video2')
def video2():
    return Response(generate_frames_2(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app2.route('/api/save_coordinates2', methods=['POST'])
def save_coordinates2():
    coordinates = request.get_json()
    print(coordinates)
    return jsonify({"status": "success", "message": "Coordinates saved."}), 200

# 创建一个线程，用于运行 Flask 应用 1
thread1 = threading.Thread(target=app1.run, kwargs={'host': '192.168.137.254', 'port': 8001})
thread1.start()

# 创建一个线程，用于运行 Flask 应用 2
thread2 = threading.Thread(target=app2.run, kwargs={'host': '192.168.137.254', 'port': 8002})
thread2.start()

# 等待两个线程运行完成
thread1.join()
thread2.join()