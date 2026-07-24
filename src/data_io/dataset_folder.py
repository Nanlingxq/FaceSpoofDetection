# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午4:04
# @Author : zhuying
# @Company : Minivision
# @File : dataset_folder.py
# @Software : PyCharm

import cv2
import torch
import numpy as np
import json
import os
import collections
import random

from typing import Iterable, List, Union
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
from torchvision import datasets

def opencv_loader(path):
    img = cv2.imread(path)
    return img

def find_nearest_data_parent(path_str):
    path = Path(path_str)
    
    # 只要当前 path 下不存在 Data 文件夹，就一直循环
    while not (path / 'Data').exists():
        
        # 安全机制：防止到达系统根目录后陷入死循环
        if path.parent == path:
            print("已经到达根目录，未找到包含 Data 的文件夹！")
            return None # 建议这里返回 None，表示彻底找不到了
            
        # 往上一级退
        path = path.parent
        
    # 如果能活着走出 while 循环，说明当前的 path 下一定有 Data
    print(f"找到了！包含 Data 的路径是: {path}")
    return path


class DatasetFolderFT(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 ft_width=10, ft_height=10, loader=opencv_loader, conf=None, is_train=True,
                 depth_root=None, depth_aux_enabled=False, depth_target_mode="class_conditioned"):
        super(DatasetFolderFT, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            allow_empty=True
        )
        self.root = root
        self.ft_width = ft_width
        self.ft_height = ft_height
        self.conf = conf
        self.is_train = is_train
        self.depth_root = Path(depth_root) if depth_root else None
        self.depth_aux_enabled = bool(depth_aux_enabled)
        self.depth_target_mode = depth_target_mode
        if self.conf.num_classes == 2:
            self._remap_labels()
        # 进行重新采样
        if self.is_train and getattr(self.conf, 'enable_resample', False):
            self._resample_minority_classes()

        self._calculate_class_weights()

    def _resample_minority_classes(self):
        """
        按照多数类的样本数量，对少数类进行随机有放回重采样。
        """
        num_classes = len(self.classes)
        # 将样本按类别分类存储
        class_samples = {i: [] for i in range(num_classes)}
        for path, target in self.samples:
            if target in class_samples:
                class_samples[target].append((path, target))
        
        # 找到样本最多的类的数量
        max_count = max(len(samples) for samples in class_samples.values())
        
        new_samples = []
        new_targets = []
        
        for cls_idx in range(num_classes):
            samples = class_samples[cls_idx]
            count = len(samples)
            if count == 0:
                continue
                
            # 首先把该类原有的样本加入列表
            new_samples.extend(samples)
            new_targets.extend([cls_idx] * count)
            
            # 计算需要补齐的缺口数量
            shortfall = max_count - count
            if shortfall > 0:
                # 随机有放回地抽取需要补齐的样本
                extra_samples = random.choices(samples, k=shortfall)
                new_samples.extend(extra_samples)
                new_targets.extend([cls_idx] * shortfall)
                
        # 覆盖父类原本的 samples 和 targets
        self.samples = new_samples
        self.targets = new_targets
        
        print("\n" + "*"*50)
        print(f"⚖️  [Resampling Enabled] 类别样本已配平。所有类别样本数对齐为最大值: {max_count}")
        print("*"*50)
    def _remap_labels(self):
        """
        当配置为二分类时，将标签 2 映射为 0，标签 0 和 1 保持不变。
        """
        new_samples = []
        new_targets = []
        
        # 遍历原始加载的所有样本
        for path, target in self.samples:
            # 映射逻辑：如果是 2 则变成 0，否则保持不变 (0还是0, 1还是1)
            new_target = 0 if target == 2 else target
            
            new_samples.append((path, new_target))
            new_targets.append(new_target)
            
        # 覆盖原始属性
        self.samples = new_samples
        self.targets = new_targets
        
        # 必须截断 self.classes，否则后续 _calculate_class_weights 中
        # len(self.classes) 依然是 3，会导致下标越界或权重计算错误
        if len(self.classes) > 2:
            self.classes = self.classes[:2]
            
        print("=> [Label Mapping] conf.num_class=2 detected. Merged class 2 into class 0.")

    def _calculate_class_weights(self):
        # 1. 获取所有样本的类别标签
        # datasets.ImageFolder 默认会将标签保存在 self.targets 中
        targets = self.targets
        
        # 2. 统计每个类别的样本数量
        class_counts = collections.Counter(targets)
        num_classes = len(self.classes)
        total_samples = len(targets)
        self.label_counts = [class_counts.get(i, 0) for i in range(num_classes)]

        # 3. 按照类别索引 (0 到 num_classes-1) 计算权重
        weights = []
        for i in range(num_classes):
            count = class_counts.get(i, 0)
            if count > 0:
                # 使用 scikit-learn 的 'balanced' 权重计算逻辑: N / (C * N_c)
                weight = total_samples / (num_classes * count)
            else:
                weight = 0.0 # 防止除零错误，如果某个类完全没有样本
            weights.append(weight)
        
        # 4. 转换为 torch.Tensor 并保存为实例属性
        self.class_weights_tensor = torch.tensor(weights, dtype=torch.float)
        print(f"Dataset Initialized. Class weights: {self.class_weights_tensor}")

        print("\n" + "="*50)
        print(f"📊 Dataset Loaded from: {self.root}")
        print(f"Total samples: {total_samples}")
        print("-" * 50)
        
        # 遍历 self.classes (类别名) 和 self.label_counts (数量)
        for idx, (class_name, count) in enumerate(zip(self.classes, self.label_counts)):
            # 计算该类别占总样本的百分比
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            # 打印 类别名、数量、占比、以及对应的权重
            print(f"  - Class {idx} [{class_name}]: {count} samples ({percentage:.1f}%) | Weight: {weights[idx]:.4f}")
            
        print("="*50 + "\n")
        

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        depth_target = depth_mask = None
        if self.depth_aux_enabled:
            if self.depth_root is None:
                raise RuntimeError("depth_aux_enabled=True but depth_root is not configured")
            relative_path = Path(path).relative_to(Path(self.root))
            depth_path = self.depth_root / relative_path
            depth_image = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
            if depth_image is None:
                raise FileNotFoundError(f"Missing depth target for {path}: {depth_path}")
            depth_target = depth_image.astype(np.float32) / 255.0
            depth_mask = (depth_image > 0).astype(np.uint8)
            if self.depth_target_mode == "class_conditioned" and target != 1:
                depth_target = np.zeros_like(depth_target, dtype=np.float32)

        # generate the FT picture of the sample
        ft_sample = generate_FT(sample)
        if sample is None:
            print('image is None --> ', path)
        if ft_sample is None:
            print('FT image is None -->', path)
        assert sample is not None

        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)

        if self.depth_aux_enabled:
            if not getattr(self.transform, "paired", False):
                raise TypeError("Depth auxiliary training requires a paired RGB/depth transform")
            sample, depth_target, depth_mask = self.transform(sample, depth_target, depth_mask)
        elif self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, path)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.depth_aux_enabled:
            return sample, ft_sample, depth_target, depth_mask, target
        return sample, ft_sample, target


def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift)+1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn+1) / (maxx - minn+1)
    return fimg