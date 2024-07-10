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

DEFAULT_RKNN_PATH = '../model/resnet50-v2-7.rknn'
DEFAULT_ONNX_PATH = '../model/resnet50-v2-7.onnx'
CLASS_LABEL_PATH = '../model/synset.txt'
DEFAULT_QUANT = True
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

RKNPU1_TARGET = ['rk1808', 'rv1109', 'rv1126']

RK3566_RK3568_RKNN_MODEL = 'resnet50_95.0200.rknn'
RK3588_RKNN_MODEL = '../model/resnet50_95.0200.rknn'
RK3562_RKNN_MODEL = 'resnet18_for_rk3562.rknn'
RK3576_RKNN_MODEL = 'resnet18_for_rk3576.rknn'


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

class usr_resnet:
    def __init__(self):
        self.usr_resnet_emotion = ['angry','disgust','fear','happy','neutral','sad','surprise']

        rknn_model = RK3588_RKNN_MODEL
        self.rknn_lite = RKNNLite()
            # Load RKNN model
        print('--> Load RKNN model')
        ret = self.rknn_lite.load_rknn(rknn_model)
        if ret != 0:
            print('Load RKNN model failed')
            exit(ret)
        print('done')

        ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

    def usr_resnet_run(self,img):
        res = False
        if(img is None):
            return res,None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, 0)

        # Inference
        print('--> Running model')
        outputs = self.rknn_lite.inference(inputs=[img])
        if(outputs is None):
            return res,None
        scores = softmax(outputs[0])
        # print the top-5 inferences class
        scores = np.squeeze(scores)
        a = np.argsort(scores)[::-1]
        print('-----TOP 5-----')
        for i in a[0:5]:
            print('[%d] score=%.2f "' % (i, scores[i]))
        print('done')
        res = True
        return res,self.usr_resnet_emotion[a[0]]

    def usr_resnet_release(self):
        self.rknn_lite.release()

if __name__ == '__main__':
    my_resnet = usr_resnet()
    # Set inputs
    img = cv2.imread('../model/smile.jpg')
    print('判断空',img is not None)
    res,motion = my_resnet.usr_resnet_run(img)
    print('推理结果',res,motion)
    my_resnet.usr_resnet_release()
