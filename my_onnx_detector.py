# -------------------------------------------------------
# Face-Detector-1MB-with-landmark
# Copyright (c) 2019 biubug6
# Licensed under The MIT License
# -------------------------------------------------------

import os
import time
import cv2
import numpy as np
from math import ceil
from itertools import product
import onnxruntime as ort


from PIL import Image, ImageDraw
from PIL import ImageFont

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (numpy): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (numpy): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (numpy): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (numpy): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), 1)
    return landms

def get_result_bboxes_landmarks(loc, conf, landms, im_height, im_width, scale, resize,
                                conf_threshold=0.02, top_k=5000, nms_threshold=0.4, keep_top_k=750, vis_thres=0.6):
    priorbox = PriorBox(image_size=(im_height, im_width))
    priors = priorbox.forward()
    
    boxes = decode(loc[0], priors, [0.1, 0.2])
    boxes = boxes * scale / resize
    scores = conf[0][:, 1]
    landms = decode_landm(landms[0], priors, [0.1, 0.2])
    scale1 = np.array([im_width, im_height, im_width, im_height,
                            im_width, im_height, im_width, im_height,
                            im_width, im_height])
    
    landms = landms * scale1 / resize

    # ignore low scores
    inds = np.where(scores > conf_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    bboxes, lms = [], []
    for b in dets:
        if b[4] < vis_thres:
            continue
        b = list(map(int, b))
        bboxes.append(np.array([b[0], b[1], b[2], b[3]]).reshape(-1))
        lm = np.array([b[k] for k in range(5, 15)]).reshape(5, 2)
        lms.append(lm)
    return bboxes, lms


def load_onnx_model(onnx_path):
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    return sess

def inference_detection(onnx_sess, image_data, down_factor=1):
    resize = 1.
    
    # 支持传入文件路径(str)或numpy数组
    if isinstance(image_data, (str, os.PathLike)):
        # 使用 Pillow 读取图像
        try:
            with Image.open(image_data) as pil_img:
                pil_img = pil_img.convert("RGB")  # 转换为 RGB 模式
                img_raw = np.array(pil_img)  # 转换为 numpy 数组
        except Exception as e:
            raise ValueError(f"Failed to read image: {image_data}. Error: {e}")
    elif isinstance(image_data, np.ndarray):
        # 如果是numpy数组，确保是RGB格式
        img_raw = image_data.copy()
        if len(img_raw.shape) == 2:
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
        elif img_raw.shape[2] == 4:
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"Unsupported image type: {type(image_data)}")
    
    raw_h, raw_w = img_raw.shape[:2]
    assert raw_w % down_factor ==0 and raw_h % down_factor ==0, '降采样倍数必须能被图像宽度和高度整除'
    img = cv2.resize(img_raw, (raw_w//down_factor, raw_h//down_factor))
    
    img = np.float32(img)
    im_height, im_width, _ = img.shape
    scale = np.array([im_width, im_height, im_width, im_height])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
    
    # 模型输入输出名称
    input_name = onnx_sess.get_inputs()[0].name
    output_names = [o.name for o in onnx_sess.get_outputs()]
    start = time.time()
    outputs = onnx_sess.run(output_names, {input_name: img})
    inference_time = time.time() - start
    print(f"Inference time: {inference_time * 1000:.1f} ms")

    loc, conf, landms = outputs
    bboxes, lms = get_result_bboxes_landmarks(loc, conf, landms, im_height, im_width, scale, resize)
    # 还原坐标到原图尺寸
    new_bboxes = []
    new_lms = []
    ratio_w = raw_w // im_width
    ratio_h = raw_h // im_height
    for b in bboxes:
        b[0::2] = b[0::2] * ratio_w
        b[1::2] = b[1::2] * ratio_h
        new_bboxes.append(b)
    for l in lms:
        l[:, 0] = l[:, 0] * ratio_w
        l[:, 1] = l[:, 1] * ratio_h
        new_lms.append(l)

    return new_bboxes, new_lms

def get_aligned_faces(image_data, bboxes, lms, output_size):
    
    aligned_faces = []
    if len(bboxes)>0:
        for i in range(len(bboxes)):
            aligned_face = align_face(image_data, lms[i].reshape(-1), (output_size,output_size), sample_type="LINEAR")
            if type(aligned_face)==np.ndarray:
                # cv2.imwrite(os.path.join(aligned_dir, f'aligned_face_{align_size}_{i}.png'), aligned_face)
                aligned_faces.append(aligned_face)
            else:
                print('one face align failed')
    return aligned_faces

def detect_faces(onnx_model, image_path, output_face_size, down_factor=1):
    '''
    返回检测并对齐后的人脸列表[np.array((output_face_size,output_face_size,3), dtype=np.uint8),...]
    '''
    bboxes, lms = inference_detection(onnx_model, image_path, down_factor=down_factor)
    
    # 使用 Pillow 读取图像
    try:
        with Image.open(image_path) as pil_img:
            pil_img = pil_img.convert("RGB")  # 转换为 RGB 模式
            image_data = np.array(pil_img)  # 转换为 numpy 数组
    except Exception as e:
        raise ValueError(f"Failed to read image: {image_path}. Error: {e}")
    
    faces = get_aligned_faces(image_data, bboxes, lms, output_face_size)
    return faces

class PriorBox(object):
    def __init__(self, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            output = output.clip(max=1, min=0)
        return output

if __name__ == '__main__':
    # onnx_path = './FaceDetector.onnx'
    # test_image = './test_face2.png'
    # onnx_model = load_onnx_model(onnx_path)
    # faces = detect_faces(onnx_model, test_image, down_factor=3, output_face_size=112)
    # if faces:
    #     print(f'{len(faces)} faces are detected')
    # else:
    #     print('no face detected')
    
    ###对整个文件夹的图片进行处理
    onnx_path = './FaceDetector.onnx'
    input_folder = '/userdata/lwk/FaceFakeDetection/datasets/DF40/Celeb-DF-v2'
    output_folder = '/userdata/lwk/FaceFakeDetection/datasets/DF40_aligned'
    os.makedirs(output_folder, exist_ok=True)
    onnx_model = load_onnx_model(onnx_path)
    
    image_files = []
    for f in os.listdir(input_folder):
        file_path = os.path.join(input_folder, f)
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    image_files.append(file_path)
            except Exception:
                print(f"Skip invalid image file: {f}")
    
    for img_path in image_files:
                
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]
        try:
            faces = detect_faces(onnx_model, img_path, 256, down_factor=1)
            if faces:
                print(f'{len(faces)} faces are detected in {file_name}')
            
                if len(faces) == 1:
                    face_bgr = cv2.cvtColor(faces[0], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_folder, f'{base_name}.jpg'), face_bgr)
            else:
                print(f'no face detected in {file_name}')
        except ValueError as ve:
            print(f"ValueError for image {file_name}: {ve}")