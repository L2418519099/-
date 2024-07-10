import os
import sys
import urllib
import time
import traceback
import numpy as np
import cv2
import platform
from rknnlite.api import RKNNLite 
from scipy.special import softmax

start_time = time.time()
sum_time = 0
# 默认的RKNN模型路径和ONNX模型路径
DEFAULT_RKNN_PATH = '../model/resnet50-v2-7.rknn'
DEFAULT_ONNX_PATH = '../model/resnet50-v2-7.onnx'
CLASS_LABEL_PATH = '../model/synset.txt'
DEFAULT_QUANT = True
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

# 兼容的NPU设备类型
RKNPU1_TARGET = ['rk1808', 'rv1109', 'rv1126']

# 各个设备对应的模型文件名
#resnet50_95.0200_fp.rknn    resnet50_95.0200_u8.rknn
#resnet18_94.8100_fp.rknn resnet18_94.8100_u8.rknn  
RK3566_RK3568_RKNN_MODEL = 'resnet50_95.0200.rknn'
RK3588_RKNN_MODEL = '../model/resnet18_94.8100_fp.rknn'
RK3562_RKNN_MODEL = 'resnet18_for_rk3562.rknn'
RK3576_RKNN_MODEL = 'resnet18_for_rk3576.rknn'

# 将速度转换为可读格式的函数
def readable_speed(speed):
    speed_bytes = float(speed)
    speed_kbytes = speed_bytes / 1024
    if speed_kbytes > 1024:
        speed_mbytes = speed_kbytes / 1024
        if speed_mbytes > 1024:
            speed_gbytes = speed_mbytes / 1024
            return "{:.2f} GB/s".format(speed_gbytes)
        else:
            return "{:.2f} MB/s".format(speed_mbytes)
    else:
        return "{:.2f} KB/s".format(speed_kbytes)

# 显示下载进度的函数
def show_progress(blocknum, blocksize, totalsize):
    speed = (blocknum * blocksize) / (time.time() - start_time)
    speed_str = " Speed: {}".format(readable_speed(speed))
    recv_size = blocknum * blocksize

    f = sys.stdout
    progress = (recv_size / totalsize)
    progress_str = "{:.2f}%".format(progress * 100)
    n = round(progress * 50)
    s = ('#' * n).ljust(50, '-')
    f.write(progress_str.ljust(8, ' ') + '[' + s + ']' + speed_str)
    f.flush()
    f.write('\r\n')

# 检查并下载原始的ONNX模型文件
def check_and_download_origin_model():
    global start_time
    if not os.path.exists(DEFAULT_ONNX_PATH):
        print('--> Download {}'.format(DEFAULT_ONNX_PATH))
        url = 'https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx'
        download_file = DEFAULT_ONNX_PATH
        try:
            start_time = time.time()
            urllib.request.urlretrieve(url, download_file, show_progress)
        except:
            print('Download {} failed.'.format(download_file))
            print(traceback.format_exc())
            exit(-1)
        print('done')

# 解析命令行参数
def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} [onnx_model_path] [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk3562,rk3566,rk3568,rk3588,rk1808,rv1109,rv1126]")
        print("       dtype choose from [i8, fp] for [rk3562,rk3566,rk3568,rk3588]")
        print("       dtype choose from [u8, fp] for [rk1808,rv1109,rv1126]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['u8', 'i8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    return model_path, platform, do_quant, output_path

# 自定义的ResNet类
class usr_resnet:
    def __init__(self):
        # 定义情绪类别
        self.usr_resnet_emotion = ['angry','disgust','fear','happy','neutral','sad','surprise']

        rknn_model = RK3588_RKNN_MODEL
        self.rknn_lite = RKNNLite()
        # 加载RKNN模型
        print('--> Load RKNN model')
        ret = self.rknn_lite.load_rknn(rknn_model)
        if ret != 0:
            print('Load RKNN model failed')
            exit(ret)
        print('done')

        # 初始化运行时环境
        ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

    # 运行模型进行推理
    def usr_resnet_run(self, img):
        global sum_time

        res = False
        if(img is None):
            return res, None
        # 图像预处理：BGR转RGB、调整大小、增加维度
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, 0)

        # 模型推理
        #print('--> Running model')
        start_time = time.time()
        outputs = self.rknn_lite.inference(inputs=[img])
        sum_time += (time.time() - start_time)
        if(outputs is None):
            return res, None
        scores = softmax(outputs[0])
        # 打印前五个推理结果
        scores = np.squeeze(scores)
        a = np.argsort(scores)[::-1]
        #print('-----TOP 5-----')
        # for i in a[0:5]:
        #     print('[%d] score=%.2f "' % (i, scores[i]))
        # print('done')
        res = True
        return res, self.usr_resnet_emotion[a[0]]

    # 释放RKNN资源
    def usr_resnet_release(self):
        self.rknn_lite.release()

# 评估模型在测试数据集上的准确率
def evaluate_accuracy(test_data_folder):
    # 获取所有子文件夹，即表情类别
    emotion_folders = os.listdir(test_data_folder)
    emotion_dict = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    
    total_images = 0
    correct_predictions = 0
    res_sum = 0
    my_resnet = usr_resnet()

    # 遍历每个表情类别文件夹
    for emotion in emotion_folders:
        emotion_path = os.path.join(test_data_folder, emotion)
        if os.path.isdir(emotion_path):
            images = os.listdir(emotion_path)
            for image_name in images:
                image_path = os.path.join(emotion_path, image_name)
                img = cv2.imread(image_path)
                if img is not None:
                    # 对每张图片进行推理
                    res, prediction = my_resnet.usr_resnet_run(img)
                    if res:
                        total_images += 1
                        # 判断预测结果是否正确
                        if emotion_dict[emotion] == emotion_dict[prediction]:
                            correct_predictions += 1
                    else:
                        res_sum += 1
    
    # 计算并打印准确率
    accuracy = correct_predictions / total_images if total_images > 0 else 0
    print('resanet18_fp')
    print("Total images:", total_images)
    #print("Correct predictions:", correct_predictions)
    print("Accuracy:",accuracy)
    print("Average time:", sum_time / total_images)
    print('\n')
    print('\n')
    print('res_error',res_sum)
    my_resnet.usr_resnet_release()

if __name__ == '__main__':
    test_data_folder = '../test_data/'  # 测试数据文件夹路径

    evaluate_accuracy(test_data_folder)
