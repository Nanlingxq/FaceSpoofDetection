# -*- coding: utf-8 -*-
# @Time : 2024
# @Author : Demo
# @File : realtime_demo.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import warnings
import torch
import torch.nn.functional as F

from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from src.data_io import transform as trans
from src.generate_patches import CropImage
from src.anti_spoof_predict import Detection
from src.utility import parse_model_name, get_kernel

warnings.filterwarnings('ignore')

MODEL_DIR = "./resources/anti_spoof_models"

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}


class OptimizedAntiSpoofPredict(Detection):
    """优化版：模型只加载一次，支持多个模型并行推理"""
    
    def __init__(self, device_id=0):
        super(OptimizedAntiSpoofPredict, self).__init__()
        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")
        self.models = []  # 存储加载的模型及其配置
        
    def load_models(self, model_dir):
        """预加载所有模型，只执行一次"""
        for model_name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_name)
            if not os.path.isfile(model_path):
                continue
                
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            kernel_size = get_kernel(h_input, w_input)
            
            # 创建模型并加载权重
            model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            
            # 处理多GPU训练的模型
            keys = list(state_dict.keys())
            if keys and 'module.' in keys[0]:
                new_state_dict = {k[7:]: v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)
            
            model.eval()
            
            self.models.append({
                'model': model,
                'scale': scale,
                'out_w': w_input,
                'out_h': h_input,
                'model_path': model_path
            })
        print(f"Loaded {len(self.models)} models")
        
    def predict(self, img):
        """预测，不再重新加载模型"""
        test_transform = trans.Compose([trans.ToTensor()])
        img = test_transform(img).unsqueeze(0).to(self.device)
        prediction = np.zeros((1, 3))
        
        with torch.no_grad():
            for item in self.models:
                result = item['model'].forward(img)
                prediction += F.softmax(result).cpu().numpy()
        
        return prediction


def main():
    # 初始化优化版预测器
    model_test = OptimizedAntiSpoofPredict(device_id=0)
    image_cropper = CropImage()
    
    # 预加载所有模型（关键优化！）
    model_test.load_models(MODEL_DIR)

    # 打开摄像头（尝试使用DShow后端解决兼容性问题）
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    target_width = 1280
    target_height = 720

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

    if not cap.isOpened():
        # 如果DShow失败，尝试默认后端
        print("DShow后端失败，尝试默认后端...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头！")
            return

    # 获取摄像头实际使用的分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头已打开，分辨率: {width}x{height}，按 'q' 退出")
    frame_count = 0
    start_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        # 获取人脸检测框
        try:
            image_bbox = model_test.get_bbox(frame)
            x, y, w, h = image_bbox
            
            # 确保检测框在图像范围内
            x, y = max(0, x), max(0, y)
            w = min(frame.shape[1] - x, w)
            h = min(frame.shape[0] - y, h)

            if w > 0 and h > 0:
                prediction = np.zeros((1, 3))
                
                # 对每个模型进行预测
                for item in model_test.models:
                    param = {
                        "org_img": frame,
                        "bbox": image_bbox,
                        "scale": item['scale'],
                        "out_w": item['out_w'],
                        "out_h": item['out_h'],
                        "crop": True,
                    }
                    if item['scale'] is None:
                        param["crop"] = False
                    
                    img = image_cropper.crop(**param)
                    prediction += model_test.predict(img)

                label = np.argmax(prediction)
                value = prediction[0][label] / len(model_test.models)

                if label == 1:
                    result_text = f"Real Face: {value:.2f}"
                    color = (0, 255, 0)
                else:
                    result_text = f"Fake Face: {value:.2f}"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, result_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(frame, "No face detected", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        except Exception as e:
            cv2.putText(frame, "Detection error", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 计算并显示FPS
        frame_count += 1
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Face Anti-Spoofing Demo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()