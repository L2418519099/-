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
DEFAULT_RKNN_PATH = '../model/resnet50-v2-7.rknn'  # 默认RKNN模型路径
DEFAULT_ONNX_PATH = '../model/resnet50-v2-7.onnx'  # 默认ONNX模型路径
CLASS_LABEL_PATH = '../model/synset.txt'  # 分类标签路径
DEFAULT_QUANT = True  # 默认量化设置
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'  # 设备兼容节点

RKNPU1_TARGET = ['rk1808', 'rv1109', 'rv1126']  # RKNPU1目标设备

RK3566_RK3568_RKNN_MODEL = 'resnet50_95.0200.rknn'  # RK3566和RK3568的RKNN模型路径
RK3588_RKNN_MODEL = '../../model/resnet0703false.rknn'  # RK3588的RKNN模型路径
RK3562_RKNN_MODEL = 'resnet18_for_rk3562.rknn'  # RK3562的RKNN模型路径
RK3576_RKNN_MODEL = 'resnet18_for_rk3576.rknn'  # RK3576的RKNN模型路径

# 转换速度为可读格式的函数
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

# 检查并下载原始模型的函数
def check_and_download_origin_model():
    global start_time
    if not os.path.exists(DEFAULT_ONNX_PATH):  # 如果ONNX模型文件不存在
        print('--> Download {}'.format(DEFAULT_ONNX_PATH))  # 打印下载信息
        url = 'https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx'
        download_file = DEFAULT_ONNX_PATH
        try:
            start_time = time.time()
            urllib.request.urlretrieve(url, download_file, show_progress)  # 下载文件
        except:
            print('Download {} failed.'.format(download_file))  # 下载失败信息
            print(traceback.format_exc())  # 打印错误信息
            exit(-1)
        print('done')  # 下载完成

# 解析命令行参数的函数
def parse_arg():
    if len(sys.argv) < 3:  # 参数数量检查
        print("Usage: python3 {} [onnx_model_path] [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk3562,rk3566,rk3568,rk3588,rk1808,rv1109,rv1126]")
        print("       dtype choose from [i8, fp] for [rk3562,rk3566,rk3568,rk3588]")
        print("       dtype choose from [u8, fp] for [rk1808,rv1109,rv1126]")
        exit(1)

    model_path = sys.argv[1]  # 模型路径
    platform = sys.argv[2]  # 平台类型

    do_quant = DEFAULT_QUANT  # 量化设置
    if len(sys.argv) > 3:
        model_type = sys.argv[3]  # 模型类型
        if model_type not in ['u8', 'i8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))  # 无效模型类型错误
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]  # 输出路径
    else:
        output_path = DEFAULT_RKNN_PATH  # 默认输出路径

    return model_path, platform, do_quant, output_path

# 自定义的ResNet类
class usr_resnet:
    def __init__(self):
        self.usr_resnet_emotion = ['angry','disgust','fear','happy','neutral','sad','surprise']  # 情感分类标签

        rknn_model = RK3588_RKNN_MODEL  # RKNN模型路径
        self.rknn_lite = RKNNLite()  # 创建RKNNLite对象
        # 加载RKNN模型
        print('--> Load RKNN model')
        ret = self.rknn_lite.load_rknn(rknn_model)
        if ret != 0:
            print('Load RKNN model failed')  # 加载失败信息
            exit(ret)
        print('done')  # 加载成功信息

        # 初始化运行时环境
        ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
        if ret != 0:
            print('Init runtime environment failed')  # 初始化失败信息
            exit(ret)
        print('done')  # 初始化成功信息

    # 推理函数
    def usr_resnet_run(self,img):
        res = False
        if(img is None):
            return res,None  # 如果图像为空，返回失败
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像颜色
        img = cv2.resize(img, (224, 224))  # 调整图像大小
        img = np.expand_dims(img, 0)  # 扩展图像维度

        # 运行模型
        #print('--> Running model')
        outputs = self.rknn_lite.inference(inputs=[img])
        if(outputs is None):
            return res,None  # 推理失败返回
        scores = softmax(outputs[0])  # 计算softmax得分
        scores = np.squeeze(scores)  # 去除多余维度
        a = np.argsort(scores)[::-1]  # 按得分排序
        #print('-----TOP 5-----')
        #for i in a[0:5]:
            #print('[%d] score=%.2f "' % (i, scores[i]))  # 打印前5个分类结果
        #print('done')
        #res = True
        return self.usr_resnet_emotion[a[0]]  # 返回推理结果和最高得分的情感标签

    # 释放资源函数
    def usr_resnet_release(self):
        self.rknn_lite.release()  # 释放RKNNLite对象资源

# 主程序入口
if __name__ == '__main__':
    my_resnet = usr_resnet()  # 创建ResNet对象
    img = cv2.imread('../model/smile.jpg')  # 读取测试图像
    print('判断空',img is not None)  # 判断图像是否为空
    motion = my_resnet.usr_resnet_run(img)  # 运行推理
    print('推理结果',motion)  # 打印推理结果
    my_resnet.usr_resnet_release()  # 释放资源
