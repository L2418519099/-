import os
import cv2
import sys
import argparse
from argparse import Namespace
from datetime import datetime
from usr_resnet import usr_resnet
# 添加路径
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

from py_utils.coco_utils import COCO_test_helper
import numpy as np

OBJ_THRESH = 0.25  # 目标阈值
NMS_THRESH = 0.45  # 非极大值抑制阈值

# 以下两个参数用于MAP测试
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 640)  # 图像大小 (宽度, 高度)，例如 (1280, 736)

# 类别名称
CLASSES = ("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
           "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
           "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

# COCO数据集类别ID列表
coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

# 过滤框函数
def filter_boxes(boxes, box_confidences, box_class_probs):
    """根据目标阈值过滤框。"""
    box_confidences = box_confidences.reshape(-1)  # 将置信度调整为一维数组
    candidate, class_num = box_class_probs.shape  # 获取候选框数量和类别数量

    class_max_score = np.max(box_class_probs, axis=-1)  # 获取每个框的最大类别得分
    classes = np.argmax(box_class_probs, axis=-1)  # 获取每个框的类别索引

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)  # 过滤符合阈值的框
    scores = (class_max_score * box_confidences)[_class_pos]  # 获取符合条件的得分

    boxes = boxes[_class_pos]  # 获取符合条件的框
    classes = classes[_class_pos]  # 获取符合条件的类别

    return boxes, classes, scores  # 返回框、类别和得分

# 非极大值抑制函数
def nms_boxes(boxes, scores):
    """抑制非极大值框。
    # 返回
        keep: ndarray，有效框的索引。
    """
    x = boxes[:, 0]  # 获取每个框的左上角x坐标
    y = boxes[:, 1]  # 获取每个框的左上角y坐标
    w = boxes[:, 2] - boxes[:, 0]  # 计算每个框的宽度
    h = boxes[:, 3] - boxes[:, 1]  # 计算每个框的高度

    areas = w * h  # 计算每个框的面积
    order = scores.argsort()[::-1]  # 按得分降序排序

    keep = []  # 用于保存保留的框索引
    while order.size > 0:
        i = order[0]  # 获取得分最高的框索引
        keep.append(i)  # 保留该框索引

        # 计算得分最高框与其他框的交集
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        # 计算交集区域的宽度和高度
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1  # 计算交集面积

        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # 计算交并比
        inds = np.where(ovr <= NMS_THRESH)[0]  # 获取符合NMS阈值的框索引
        order = order[inds + 1]  # 更新排序后的索引列表
    keep = np.array(keep)  # 转换为数组
    return keep  # 返回保留的框索引

# DFL损失函数
def dfl(position):
    # Distribution Focal Loss (DFL)
    import torch
    x = torch.tensor(position)  # 将位置转换为张量
    n, c, h, w = x.shape  # 获取张量的形状
    p_num = 4  # 每个框的坐标数量
    mc = c // p_num  # 计算每个坐标的通道数
    y = x.reshape(n, p_num, mc, h, w)  # 重塑张量
    y = y.softmax(2)  # 对每个坐标应用softmax
    acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)  # 创建精度矩阵
    y = (y * acc_metrix).sum(2)  # 计算加权和
    return y.numpy()  # 返回numpy数组

# 框处理函数
def box_process(position):
    grid_h, grid_w = position.shape[2:4]  # 获取网格的高度和宽度
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))  # 创建网格
    col = col.reshape(1, 1, grid_h, grid_w)  # 重塑列网格
    row = row.reshape(1, 1, grid_h, grid_w)  # 重塑行网格
    grid = np.concatenate((col, row), axis=1)  # 合并网格
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)  # 计算步幅

    position = dfl(position)  # 应用DFL
    box_xy = grid + 0.5 - position[:, 0:2, :, :]  # 计算左上角坐标
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]  # 计算右下角坐标
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)  # 合并坐标

    return xyxy  # 返回框坐标

# 后处理函数
def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch = 3  # 默认分支数量
    pair_per_branch = len(input_data) // defualt_branch  # 每个分支的对数

    # 处理每个分支的数据
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    # 展平函数
    def sp_flatten(_in):
        ch = _in.shape[1]  # 获取通道数
        _in = _in.transpose(0, 2, 3, 1)  # 转置
        return _in.reshape(-1, ch)  # 重塑为二维数组

    # 展平数据
    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)  # 合并所有框
    classes_conf = np.concatenate(classes_conf)  # 合并所有类别置信度
    scores = np.concatenate(scores)  # 合并所有得分

    # 根据阈值过滤框
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # 非极大值抑制
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)  # 获取当前类别的索引
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)  # 进行NMS

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)  # 合并所有框
    classes = np.concatenate(nclasses)  # 合并所有类别
    scores = np.concatenate(nscores)  # 合并所有得分

    return boxes, classes, scores  # 返回框、类别和得分

# 绘制检测结果
def draw(image, boxes, scores, classes,emotion):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]  # 获取框的坐标
        #print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)  # 绘制矩形框
        cv2.putText(image, emotion,
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # 绘制标签

# 设置模型
def setup_model(args):
    model_path = args.model_path  # 获取模型路径
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'  # 平台为pytorch
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'  # 平台为rknn
        from py_utils.rknn_executor import RKNN_model_container
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'  # 平台为onnx
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)  # 不支持的模型格式
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform  # 返回模型和平台

# 检查图像文件类型
def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']  # 支持的图像类型
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

# 自定义YOLOv8类
class usr_yolov8:
    def __init__(self):
        self.args = Namespace(
            model_path='../../model/yolov8n.rknn',
            target='rk3566',
            device_id=None,
            img_show=False,
            img_save=False,
            anno_json='../../../datasets/COCO/annotations/instances_val2017.json',
            coco_map_test=False
        )
        self.model, self.platform = setup_model(self.args)  # 设置模型
        self.co_helper = COCO_test_helper(enable_letter_box=True)  # 创建COCO测试助手
        self.my_resnet = usr_resnet()  # 创建ResNet对象

    # 运行YOLOv8推理
    def usr_yolo_run(self, img_src):
        res = False
        img_copy=img_src.copy()
        img = self.co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
                                        pad_color=(0, 0, 0))  # 图像预处理
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色空间
        img = np.expand_dims(img, 0)  # 扩展维度
        input_data = img
        # 推理
        outputs = self.model.run([input_data])
        boxes, classes, scores = post_process(outputs)  # 后处理

        if boxes is not None:
            results = []  # 用于存储框坐标和对应的ResNet50推理结果

            for box in boxes:
                # 获取框的坐标
                top, left, right, bottom = [int(_b) for _b in box]
                # 扩大裁剪框，例如各边向外扩5个单位
                expand_size = 10
                top = max(0, top - expand_size)  # 确保top边界不小于0
                bottom = min(img_copy.shape[0], bottom)  # 确保bottom边界不超过图像高度
                left = max(0, left - expand_size)  # 确保left边界不小于0
                right = min(img_copy.shape[1], right + expand_size)  # 确保right边界不超过图像宽度

                cropped_img = img_copy[left:bottom, top:right]  # 裁剪框内的图像

                # 送入resnet50模型进行推理
                if cropped_img.size > 0:  # 检查裁剪后的图像是否为空

                    resnet_result = self.my_resnet.usr_resnet_run(cropped_img)  # 运行ResNet50模型
                    if resnet_result[0]:  # 如果推理成功
                        results.append((box, resnet_result))  # 保存框坐标和表情分类结果

            # 在图像上绘制检测框和表情分类结果
            for box, emotion in results:
                top, left, right, bottom = [int(_b) for _b in box]
                draw(img_copy, self.co_helper.get_real_box(boxes), scores, classes,emotion)
                #cv2.putText(img_copy, emotion, (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # 绘制表情分类结果

            res = True
            return res ,img_copy  # 返回绘制后的图像
        else:
            return res,None


# 主函数
if __name__ == '__main__':
    myyolov8 = usr_yolov8()  # 创建YOLOv8对象
    my_resnet = usr_resnet()  # 创建ResNet对象
    img = cv2.imread('../model/smile.jpg')  # 读取图像
    new_img = myyolov8.usr_yolo_run(img)  # 运行YOLOv8模型进行推理
    cv2.imshow('Result', new_img)  # 显示结果图像
    cv2.waitKey(0)  # 等待按键
'''
    if boxes is not None:
        results = []  # 用于存储框坐标和对应的ResNet50推理结果

        for box in boxes:
            # 获取框的坐标
            top, left, right, bottom = [int(_b) for _b in box]
            cropped_img = img[left:bottom, top:right]  # 裁剪框内的图像

            # 送入resnet50模型进行推理
            if cropped_img.size > 0:  # 检查裁剪后的图像是否为空

                resnet_result = my_resnet.usr_resnet_run(cropped_img)  # 运行ResNet50模型
                if resnet_result[0]:  # 如果推理成功
                    results.append((box, resnet_result))  # 保存框坐标和表情分类结果

        # 在图像上绘制检测框和表情分类结果
        for box, emotion in results:
            top, left, right, bottom = [int(_b) for _b in box]
            cv2.rectangle(img, (top, left), (right, bottom), (255, 0, 0), 2)  # 绘制检测框
            cv2.putText(img, emotion, (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # 绘制表情分类结果
'''


