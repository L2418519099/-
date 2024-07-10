import cv2
import time
import random
import numpy as np
from rknnlite.api import RKNNLite


"""
RK3588 yolov5s交通标志模型
"""

def get_max_scale(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    return scale

def get_new_size(img, scale):
    return tuple(map(int, np.array(img.shape[:2][::-1]) * scale))

def show_top5(result):
    output = result[0].reshape(-1)
    # Softmax
    output = np.exp(output) / np.sum(np.exp(output))
    # Get the indices of the top 5 largest values
    output_sorted_indices = np.argsort(output)[::-1][:5]
    top5_str = 'resnet18\n-----TOP 5-----\n'
    for i, index in enumerate(output_sorted_indices):
        value = output[index]
        if value > 0:
            topi = '[{:>3d}] score:{:.6f} class:""\n'.format(index, value)
        else:
            topi = '-1: 0.0\n'
        top5_str += topi
    print(top5_str)


class AutoScale:
    def __init__(self, img, max_w, max_h):
        self._src_img = img
        self.scale = get_max_scale(img, max_w, max_h)
        self._new_size = get_new_size(img, self.scale)
        self.__new_img = None

    @property
    def size(self):
        return self._new_size

    @property
    def new_img(self):
        if self.__new_img is None:
            self.__new_img = cv2.resize(self._src_img, self._new_size)
        return self.__new_img

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def filter_boxes(boxes, box_confidences, box_class_probs, conf_thres):
    box_scores = box_confidences * box_class_probs  # 条件概率， 在该cell存在物体的概率的基础上是某个类别的概率
    box_classes = np.argmax(box_scores, axis=-1)  # 找出概率最大的类别索引
    box_class_scores = np.max(box_scores, axis=-1)  # 最大类别对应的概率值
    pos = np.where(box_class_scores >= conf_thres)  # 找出概率大于阈值的item
    # pos = box_class_scores >= OBJ_THRESH  # 找出概率大于阈值的item
    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]
    return boxes, classes, scores


def nms_boxes(boxes, scores, iou_thres):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def letterbox(img, new_wh=(416, 416), color=(114, 114, 114)):
    a = AutoScale(img, *new_wh)
    new_img = a.new_img
    h, w = new_img.shape[:2]
    new_img = cv2.copyMakeBorder(new_img, 0, new_wh[1] - h, 0, new_wh[0] - w, cv2.BORDER_CONSTANT, value=color)
    return new_img, (new_wh[0] / a.scale, new_wh[1] / a.scale)

def load_model_npu(PATH, npu_id):
    rknn = RKNNLite()
    devs = rknn.list_devices()
    device_id_dict = {}
    for index, dev_id in enumerate(devs[-1]):
        if dev_id[:2] != 'TS':
            device_id_dict[0] = dev_id
        if dev_id[:2] == 'TS':
            device_id_dict[1] = dev_id
    print('-->loading model : ' + PATH)
    rknn.load_rknn(PATH)
    print('--> Init runtime environment on: ' + device_id_dict[npu_id])
    ret = rknn.init_runtime(device_id=device_id_dict[npu_id])
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn


def load_rknn_model(PATH):
    rknn = RKNNLite()
    ret = rknn.load_rknn(PATH)
    if ret != 0:
        print('load rknn model failed')
        exit(ret)
    print('done')
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn


class RKNNDetector:
    def __init__(self, model, wh, masks, anchors, names):
        self.wh = wh
        self._masks = masks
        self._anchors = anchors
        self.names = names
        if isinstance(model, str):
            model = load_rknn_model(model)
        self._rknn = model
        self.draw_box = True

    def _predict(self, img_src, _img, gain, conf_thres=0.4, iou_thres=0.45):
        src_h, src_w = img_src.shape[:2]
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        _img = np.expand_dims(_img, 0)
        # 调用NPU进行推理
        pred_onx = self._rknn.inference(inputs=[_img])
        #print("Inference result:", pred_onx,type(pred_onx))
        #print("inference time:\t", time.time() - t0)
        #show_top5(pred_onx)
        #print("----------------------------------------")
        # 处理推理结果
        boxes, classes, scores = [], [], []
        for t in range(3):
            input0_data = sigmoid(pred_onx[t][0])
            input0_data = np.transpose(input0_data, (1, 2, 0, 3))
            grid_h, grid_w, channel_n, predict_n = input0_data.shape
            anchors = [self._anchors[i] for i in self._masks[t]]
            box_confidence = input0_data[..., 4]
            box_confidence = np.expand_dims(box_confidence, axis=-1)
            box_class_probs = input0_data[..., 5:]
            box_xy = input0_data[..., :2]
            box_wh = input0_data[..., 2:4]
            col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
            row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)
            col = col.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            row = row.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            grid = np.concatenate((col, row), axis=-1)
            box_xy = box_xy * 2 - 0.5 + grid
            box_wh = (box_wh * 2) ** 2 * anchors
            box_xy /= (grid_w, grid_h)  # 计算原尺寸的中心
            box_wh /= self.wh  # 计算原尺寸的宽高
            box_xy -= (box_wh / 2.)  # 计算原尺寸的中心
            box = np.concatenate((box_xy, box_wh), axis=-1)
            res = filter_boxes(box, box_confidence, box_class_probs, conf_thres)
            boxes.append(res[0])
            classes.append(res[1])
            scores.append(res[2])
        boxes, classes, scores = np.concatenate(boxes), np.concatenate(classes), np.concatenate(scores)
        #print('inference:        ',len(classes))
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = nms_boxes(b, s, iou_thres)
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
        if len(nboxes) < 1:
            return [], []
        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        label_list = []
        box_list = []
        new_img = ''
        for (x, y, w, h), score, cl in zip(boxes, scores, classes):
            x *= gain[0]
            y *= gain[1]
            w *= gain[0]
            h *= gain[1]
            x1 = max(0, np.floor(x).astype(int))
            y1 = max(0, np.floor(y).astype(int))
            x2 = min(src_w, np.floor(x + w + 0.5).astype(int))
            y2 = min(src_h, np.floor(y + h + 0.5).astype(int))
            if cl>5:
                cl = 5
            label_list.append(self.names[cl])
            box_list.append((x1, y1, x2, y2))
            # if self.draw_box:
            #     new_img = plot_one_box((x1, y1, x2, y2), img_src, label=self.names[cl])
        return label_list, box_list

    def predict_resize(self, img_src, conf_thres=0.4, iou_thres=0.45):
        """
        预测一张图片，预处理使用resize
        return: labels,boxes
        """
        _img = cv2.resize(img_src, self.wh)
        gain = img_src.shape[:2][::-1]
        return self._predict(img_src, _img, gain, conf_thres, iou_thres, )

    def predict(self, img_src, conf_thres=0.4, iou_thres=0.45):
        """
        预测一张图片，预处理保持宽高比
        return: labels,boxes
        """
        _img, gain = letterbox(img_src, self.wh)
        return self._predict(img_src, _img, gain, conf_thres, iou_thres)

    def close(self):
        self._rknn.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


### 线段和射线的特殊关系判断
def isRayIntersectsSegment(poi, s_poi, e_poi):  # [x,y] [lng,lat]
    # 输入：判断点，边起点，边终点，都是[lng,lat]格式数组
    if s_poi[1] == e_poi[1]:  # 排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[1] > poi[1] and e_poi[1] > poi[1]:  # 线段在射线上边
        return False
    if s_poi[1] < poi[1] and e_poi[1] < poi[1]:  # 线段在射线下边
        return False
    if s_poi[1] == poi[1] and e_poi[1] > poi[1]:  # 交点为下端点，对应spoint
        return False
    if e_poi[1] == poi[1] and s_poi[1] > poi[1]:  # 交点为下端点，对应epoint
        return False
    if s_poi[0] < poi[0] and e_poi[0] < poi[0]:  # 线段在射线左边
        return False

    xseg = e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1])  # 求交
    if xseg < poi[0]:  # 交点在射线起点的左侧
        return False
    return True  # 排除上述情况之后


### 射线法判断点是否在区域内
def isPoiWithinPoly(poi, poly):
    global entrance,count
    sinsc = 0  # 交点个数
    # for epoly in poly:  # 循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
    #     for i in range(len(epoly) - 1):  # [0,len-1]
    #         s_poi = epoly[i]
    #         e_poi = epoly[i + 1]
    #         if isRayIntersectsSegment(poi, s_poi, e_poi):
    #             sinsc += 1  # 有交点就加1
    for i in range(len(poly)):  # [0,len-1]
        s_poi = poly[i]
        if i == len(poly)-1:
            e_poi = poly[0]
        else:
            e_poi = poly[i + 1]
        if isRayIntersectsSegment(poi, s_poi, e_poi):
            sinsc += 1  # 有交点就加1
    if sinsc % 2 == 1:
        return True

'''
### 区域检测框
def poltLine(im0):
    # 1,2,3,4 分别对应左上，右上，右下，左下四个点
    hl1 = 435 / 1114  # 监测区域高度距离图片顶部比例
    wl1 = 949 / 1979  # 监测区域高度距离图片左部比例
    hl2 = 523 / 1114  # 监测区域高度距离图片顶部比例
    wl2 = 1661 / 1979  # 监测区域高度距离图片左部比例
    hl3 = 800 / 1114  # 监测区域高度距离图片顶部比例
    wl3 = 1452 / 1979  # 监测区域高度距离图片左部比例
    hl4 = 582 / 1114  # 监测区域高度距离图片顶部比例
    wl4 = 580 / 1979  # 监测区域高度距离图片左部比例


    pts = np.array([[int(im0.shape[1] * wl1), int(im0.shape[0] * hl1)],  # pts1
                    [int(im0.shape[1] * wl2), int(im0.shape[0] * hl2)],  # pts2
                    [int(im0.shape[1] * wl3), int(im0.shape[0] * hl3)],  # pts3
                    [int(im0.shape[1] * wl4), int(im0.shape[0] * hl4)]], np.int32)  # pts4
    # pts = pts.reshape((-1, 1, 2))
    # zeros = np.zeros((im0.shape), dtype=np.uint8)
    #
    # mask = cv2.fillPoly(zeros, [pts], color=(0, 165, 255))
    # im0 = cv2.addWeighted(im0, 1, mask, 0.2, 0)
    cv2.polylines(im0, [pts], True, (255, 255, 0), 3)
    return im0
'''

class ModelTest:
    def __init__(self, RKNN_MODEL_PATH="./model.rknn"):
        self.RKNN_MODEL_PATH = RKNN_MODEL_PATH
        self.SIZE = (640, 640)
        self.CLASSES = ('person', '2', '3', '4', '5', '6')
        self.MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ANCHORS = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
        self.model = load_rknn_model(self.RKNN_MODEL_PATH)
        self.detector = RKNNDetector(self.model,
                                     self.SIZE,
                                     self.MASKS,
                                     self.ANCHORS,
                                     self.CLASSES
                                     )
        self.Inline_points = [[0,0],[640-1,0],[640-1,480-1],[0,480-1]]
        self.Midline_points = [[0,0],[640-1,0],[640-1,480-1],[0,480-1],[0,0],[640-1,0],[640-1,480-1],[0,480-1]]

    def predict(self, img):
        labels, boxes = self.detector.predict(img)
        label_list = []
        img = self.plotPolygon(img,self.Inline_points)  # 区域目标检测框
        for (x1, y1, x2, y2) in boxes:

            zs = [x1, y1]  # 左上点
            ys = [x2, y1]  # 右上点
            zx = [x1, y2]  # 左下点
            yx = [x2, y2]  # 右下点

            # w1 为检测框的宽，h1为检测框的高
            w1 = x1 - x2
            h1 = y1 - y2

            ployInline = self.Inline_points
            ployMidLine = self.Midline_points

            if isPoiWithinPoly([int((zx[0] + yx[0]) / 2), int((zx[1] + yx[1]) / 2)], ployInline):
                img = plot_one_box((x1, y1, x2, y2), img, label="inline", color=(183, 123, 65), line_thickness=3)
                label_list.append("inline")
            elif isPoiWithinPoly([int((zx[0] + yx[0]) / 2), int((zx[1] + yx[1]) / 2)], ployMidLine):
                img = plot_one_box((x1, y1, x2, y2), img, label="warning", color=(65, 65, 65), line_thickness=3)
                label_list.append("warning")
            else:
                img = plot_one_box((x1, y1, x2, y2), img, label="outline", color=(238, 211, 27), line_thickness=3)
                label_list.append("outline")
        return img, label_list
    def plotPolygon(self,im0, polygon_coords):
        # 确保polygon_coords是一个至少包含三个点的列表，每个点为[x, y]格式
        if len(polygon_coords) < 3:
            print(polygon_coords)
            raise ValueError("polygon_coords must contain at least three [x, y] points.")

        # 使用polygon_coords中的坐标绘制多边形
        pts = np.array(polygon_coords, np.int32)
        pts = pts.reshape((-1, 1, 2))  # Reshape to (-1, 1, 2) format required by OpenCV functions

        # 绘制多边形边框
        cv2.polylines(im0, [pts], True, (255, 255, 0), 3)  # Yellow color with thickness 3

        return im0
    def line_set(self, Inline_points_set,Midline_points_set):
        self.Inline_points = Inline_points_set
        self.Midline_points = Midline_points_set

        
if __name__ == '__main__':
    img = cv2.imread("./test.jpg")
    # img = cv2.resize(img, dsize=(640, 640))

    model = ModelTest(RKNN_MODEL_PATH=r"./model.rknn")
    new_img, labels = model.predict(img)

    cv2.imwrite("./result.jpg", new_img)

    print(labels)






    #RKNN_MODEL_PATH = r"./yolov5s-traffic_light.rknn"
    # if len(new_img) > 0:
    #     cv2.imshow('result', new_img)
    #     while True:
    #         if cv2.waitKey(100)== ord('q'):  # 特定的100ms
    #             break
    #     cv2.destroyAllWindows()

