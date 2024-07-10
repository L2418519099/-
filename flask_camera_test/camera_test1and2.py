import cv2
import threading
from PyQt5.QtWidgets import QApplication

def capture_camera(camera_index, window_name):
    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication([])  # 初始化 Qt 环境

    thread1 = threading.Thread(target=capture_camera, args=(0, 'Camera 1'))
    thread2 = threading.Thread(target=capture_camera, args=(2, 'Camera 2'))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    app.exec_()  # 运行 Qt 事件循环