import cv2
import zmq
import base64
import numpy as np

def test1():


    # 创建一个视频捕捉对象，参数为0表示使用第一个摄像头
    cap = cv2.VideoCapture(0)
    # 循环获取和显示每一帧图像
    while True:
        # 读取一帧图像，返回一个布尔值和一个图像矩阵
        ret, frame = cap.read()
        # 如果读取成功，显示图像
        if ret:
            cv2.imshow('camera1', frame)
        # 等待20毫秒，如果按下ESC键，退出循环
        key = cv2.waitKey(1)
        if key == 27:
            break


    # 释放摄像头资源
    cap.release()
    # 关闭所有的窗口
    cv2.destroyAllWindows()


def test2():
    IP = '192.168.137.1'  # 视频接受端的IP地址
    # 创建一个VideoCapture对象，用于从摄像头中捕获视频帧
    cap = cv2.VideoCapture(0)
    # 设置视频帧的格式为bgr
    cap.set(cv2.CAP_PROP_FORMAT, cv2.COLOR_BGR2RGB)
    # 设置视频帧的大小为640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,30)

    """实例化用来发送帧的zmq对象"""
    contest = zmq.Context()
    """zmq对象使用TCP通讯协议"""
    footage_socket = contest.socket(zmq.PAIR)
    """zmq对象和视频接收端建立TCP通讯协议"""
    footage_socket.connect('tcp://%s:5555' % IP)
    print(IP)

    while True:
        # 从摄像头中读取一帧图像，返回一个布尔值和一个图像矩阵
        ret, frame = cap.read()
        # 如果读取成功，继续处理
        if ret:
            # 将图像矩阵转换为字节流，用于发送
            frame_image = frame

            encoded, buffer = cv2.imencode('.jpg', frame_image)
            jpg_as_test = base64.b64encode(buffer)
            # 先发送长度数据，再发送图像数据
            footage_socket.send(jpg_as_test)  # 把编码后的流数据发送给视频的接收端

            cv2.imshow('video', frame)
            if cv2.waitKey(1) == 27:
                break
        # 如果读取失败，退出循环
        else:
            break

    # 释放VideoCapture对象
    cap.release()
    # 关闭连接对象
    footage_socket.disconnect('tcp://%s:5555' % IP)
    footage_socket.close()
    # 关闭socket对象
    contest.term()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    test1();
