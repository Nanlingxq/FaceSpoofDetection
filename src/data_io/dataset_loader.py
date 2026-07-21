# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午3:40
# @Author : zhuying
# @Company : Minivision
# @File : dataset_loader.py
# @Software : PyCharm


import torch

from torch.utils.data import DataLoader, ConcatDataset
from src.data_io.dataset_folder import DatasetFolderFT
from src.data_io import transform as trans
from pathlib import Path
from torchvision import transforms as T

def get_train_loader(conf):
    train_transform = trans.Compose([
        trans.ToPILImage(),
        trans.RandomResizedCrop(size=tuple(conf.input_size),
                                scale=(0.9, 1.1)),
        trans.ColorJitter(brightness=0.4,
                          contrast=0.4, saturation=0.4, hue=0.1),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(),
        trans.ToTensor()
    ])
    # root_path = conf.train_root_path
    root_path = '{}/{}'.format(conf.train_root_path, conf.patch_info)
    trainset = DatasetFolderFT(root_path, train_transform, None, conf.ft_width, conf.ft_height, conf=conf, is_train=True)
    # trainset = CelebASpoof(Path(conf.train_root_path), Path(conf.train_json_path), train_transform)

    additional_train_json_path = getattr(conf, 'additional_train_json_path', None) 
    
    if additional_train_json_path is not None:
        additional_root = '{}/{}'.format(additional_train_json_path, conf.patch_info)
        additional_trainset = DatasetFolderFT(additional_root, train_transform, None, conf.ft_width, conf.ft_height, conf=conf, is_train=True)
        # 将基础数据集和额外数据集合并
        final_dataset = ConcatDataset([trainset, additional_trainset])
        merged_counts = [
            t_count + a_count 
            for t_count, a_count in zip(trainset.label_counts, additional_trainset.label_counts)
        ]

        total_samples = sum(merged_counts)
        num_classes = len(merged_counts)

        # 重新计算全局权重
        weights = []
        for count in merged_counts:
            if count > 0:
                weights.append(total_samples / (num_classes * count))
            else:
                weights.append(0.0)

        # 将新权重赋予 ConcatDataset
        final_dataset.class_weights_tensor = torch.tensor(weights, dtype=torch.float)
        print("\n" + "🚀"*25)
        print("🌍 [Global] Merged Dataset Statistics")
        print(f"Total Combined Samples: {total_samples}")
        print("-" * 50)
        
        for idx, (class_name, count) in enumerate(zip(trainset.classes, merged_counts)):
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"  - Class {idx} [{class_name}]: {count} samples ({percentage:.1f}%) | Global Weight: {weights[idx]:.4f}")
            
        print("🚀"*25 + "\n")
    else:
        # 如果没有额外数据，就只用基础数据集
        final_dataset = trainset

    train_loader = DataLoader(
        final_dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16)
    
    return train_loader

def get_test_loader(conf):
    test_transform = trans.Compose([
        trans.ToPILImage(),
        trans.ToTensor()
    ])
    # root_path = conf.test_root_path
    root_path = '{}/{}'.format(conf.test_root_path, conf.patch_info)
    testset = DatasetFolderFT(root_path, test_transform, None, conf.ft_width, conf.ft_height, conf=conf, is_train=False)
    # testset = CelebASpoof(root_dir=Path(conf.test_root_path), json_path=Path(conf.test_json_path), transform=test_transform,
    #                       is_train=False)
    test_loader = DataLoader(
        testset,
        batch_size=conf.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=16)
    
    return test_loader