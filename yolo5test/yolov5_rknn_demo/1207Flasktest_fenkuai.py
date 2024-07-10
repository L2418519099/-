# 导入所需的库
from flask import Flask, request, jsonify,Response
from flask_cors import CORS
import cv2
import time
from mask_test import ModelTest

# 创建一个flask应用
app = Flask(__name__)
CORS(app, supports_credentials=True, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# 创建一个视频捕获对象
video_capture = cv2.VideoCapture(0)

ret, frame = video_capture.read()
if not ret:
    print('获取失败')

#myyolov8 = usr_yolov8()
model = ModelTest()
warnflag = False
# 定义一个生成器函数，用yield语句返回每个视频帧
def generate_frames():
    global warnflag
    while True:
        start_time = time.time()
        # 从摄像头读取一帧
        ret, frame = video_capture.read()
        if not ret:
            print('获取失败')
            continue
        frame_src = frame.copy()
        frame_new , labels = model.predict(frame)
        for i in labels:
            if i == "warning":
                warnflag = True
        #print(warnflag)
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

@app.route('/api/save_coordinates', methods=['POST'])
def save_coordinates():
    coordinates = request.get_json()
    # 这里处理coordinates数据，比如保存到数据库
    print(coordinates)
    return jsonify({"status": "success", "message": "Coordinates saved."}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002)