# 导入所需的库
from flask import Flask, request, jsonify,Response
from flask_cors import CORS
import cv2
#from myyolov8 import usr_yolov8
from yolov80627 import usr_yolov8
import time
import json
import datetime


import sys
sys.path.append('../yolo5_debug')
from mask_test import ModelTest
from message_test import SmsService

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
video_capture = cv2.VideoCapture(2)

ret, frame = video_capture.read()
if not ret:
    print('获取失败')

myyolov8 = usr_yolov8()

model = ModelTest()
warnflag = False

def expand_and_combine_json(dic, padding):
    def expand_rectangle(rectangle_coords, padding):
        # 确保矩形在合法范围内
        clipped_top_left = [max(0, rectangle_coords[0][0] - padding), max(0, rectangle_coords[0][1] - padding)]
        clipped_top_right = [min(640-1, rectangle_coords[1][0] + padding), max(0, rectangle_coords[1][1] - padding)]
        clipped_bottom_right = [min(640-1, rectangle_coords[2][0] + padding), min(480-1, rectangle_coords[2][1] + padding)]
        clipped_bottom_left = [max(0, rectangle_coords[3][0] - padding), min(480-1, rectangle_coords[3][1] + padding)]

        # 返回扩张后的矩形坐标
        return [clipped_top_left, clipped_top_right, clipped_bottom_right, clipped_bottom_left]

    def process_zone(zone):
        corrected_zone = {
            'topLeft': {'x': max(0, min(zone['topLeft']['x'], 640-1)), 'y': max(0, min(zone['topLeft']['y'], 480-1))},
            'topRight': {'x': max(0, min(zone['topRight']['x'], 640-1)), 'y': max(0, min(zone['topRight']['y'], 480-1))},
            'bottomRight': {'x': max(0, min(zone['bottomRight']['x'], 640-1)), 'y': max(0, min(zone['bottomRight']['y'], 480-1))},
            'bottomLeft': {'x': max(0, min(zone['bottomLeft']['x'], 640-1)), 'y': max(0, min(zone['bottomLeft']['y'], 480-1))}
        }
        rectangle_coords = [
            [corrected_zone['topLeft']['x'], corrected_zone['topLeft']['y']],
            [corrected_zone['topRight']['x'], corrected_zone['topRight']['y']],
            [corrected_zone['bottomRight']['x'], corrected_zone['bottomRight']['y']],
            [corrected_zone['bottomLeft']['x'], corrected_zone['bottomLeft']['y']]
        ]
        expanded_rectangle = expand_rectangle(rectangle_coords, padding)
        return rectangle_coords, expanded_rectangle

    # 将JSON字符串转换为Python对象
    data = dic

    # 处理每个安全区域
    rectangle_coords_list, expanded_rectangle_list = [], []
    for zone in data['safeZones']:
        rect, expanded_rect = process_zone(zone)
        rectangle_coords_list.append(rect)
        expanded_rectangle_list.append(expanded_rect)

    # 返回原始和扩展的矩形坐标列表
    return rectangle_coords_list, expanded_rectangle_list

sms_service = SmsService()
last_call_time = 0  # 初始化上次调用时间为0
def some_function():
    """示例警告处理函数"""
    global last_call_time
    sms_service.send_warning()
    print("Warning detected and function called.")

def handle_warning():
    global last_call_time, warnflag
    current_time = time.time()
    
    # 检查是否已经过了5分钟
    if warnflag and (current_time - last_call_time) >= 300:  # 300秒等于5分钟
        some_function()  # 调用处理函数
        last_call_time = current_time  # 更新最后调用时间为当前时间
    warnflag = False  # 重置warnflag，根据实际需求调整此行逻辑


index_flag = 0

# 定义一个生成器函数，用yield语句返回每个视频帧
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
        res,frame_new = myyolov8.usr_yolo_run(frame)
        if res != True:
            frame_new = frame_src
        
        text = "FPS:%.2f"%(1/(time.time()-start_time))
        AddText = frame_new.copy()
        cv2.putText(AddText, text, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 5)
        
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S") + '.' + str(current_time.microsecond)[:3]

        # 在帧上添加时间戳
        cv2.putText(AddText, formatted_time, (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 5)
        
        # 将帧转换为JPEG格式
        
        encoded_frame = cv2.imencode('.jpg', AddText)[1].tobytes()
        # 用分隔符和换行符包装帧
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')

def generate_frames2():
    global warnflag
    while True:
        start_time = time.time()
        # 从摄像头读取一帧
        ret, frame = video_capture.read()
        if not ret:
            #print('获取失败')
            continue
        frame_src = frame.copy()
        frame_new , labels = model.predict(frame)
        for i in labels:
            if i == "warning":
                current_time = datetime.datetime.now()
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S") + '.' + str(current_time.microsecond)[:3]
                print("warning,current time:",formatted_time)
                warnflag = True
                handle_warning()
        #print(warnflag)
        text = "FPS:%.2f"%(1/(time.time()-start_time))
        AddText = frame_new.copy()
        cv2.putText(AddText, text, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 5)
        
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S") + '.' + str(current_time.microsecond)[:3]

        # 在帧上添加时间戳
        cv2.putText(AddText, formatted_time, (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 5)
        
        # 将帧转换为JPEG格式
        encoded_frame = cv2.imencode('.jpg', AddText)[1].tobytes()
        # 用分隔符和换行符包装帧
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')


# 定义一个路由，用于发送视频流
@app.route('/video1')
def video1():
    # 创建一个flask的Response对象，将生成器函数作为参数传入，同时指定mimetype为'image/jpeg'
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video2')
def video2():
    # 创建一个flask的Response对象，将生成器函数作为参数传入，同时指定mimetype为'image/jpeg'
    return Response(generate_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/video_change',methods=['POST'])
def video_change():
    global video_capture
    global index_flag
    video_change = request.get_json()
    video_change_index = video_change.get('video_index')
    print(video_change_index)
    index_flag = video_change_index
    print('video_change')
    if video_change_index == 0:
        video_capture.release()
        video_capture = cv2.VideoCapture(2)
    else :
        video_capture.release()
        video_capture = cv2.VideoCapture(0)
    print(video_capture.isOpened(),'video_capture.isOpened()')
    return jsonify({"status": "success", "message": "video_change saved."}), 200

@app.route('/api/save_coordinates', methods=['POST'])
def save_coordinates():
    coordinates = request.get_json()
    # 这里处理coordinates数据，比如保存到数据库
    padding = 20
    rect_coords, combined_coords = expand_and_combine_json(coordinates, padding)
    model.Inline_points = rect_coords[0]
    model.Midline_points = combined_coords[0]
    print(rect_coords,combined_coords)
    print(coordinates)
    return jsonify({"status": "success", "message": "Coordinates saved."}), 200


if __name__ == '__main__':
    app.run(host='192.168.137.34', port=8001)