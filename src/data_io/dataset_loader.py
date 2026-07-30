# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午3:40
# @Author : zhuying
# @Company : Minivision
# @File : dataset_loader.py
# @Software : PyCharm


import torch
import os

from torch.utils.data import DataLoader, ConcatDataset
from src.data_io.dataset_folder import DatasetFolderFT
from src.data_io import transform as trans
from pathlib import Path
from torchvision import transforms as T

def get_train_loader(conf):
    depth_enabled = bool(getattr(conf, "depth_aux_enabled", False))
    loader_workers = int(getattr(conf, "depth_num_workers", 4)) if depth_enabled else 16
    loader_pin_memory = bool(getattr(conf, "depth_pin_memory", False)) if depth_enabled else True
    loader_batch_size = int(getattr(conf, "depth_batch_size", 256)) if depth_enabled else conf.batch_size
    if depth_enabled:
        train_transform = trans.PairedTrainTransform(size=tuple(conf.input_size), depth_size=tuple(getattr(conf, "depth_target_size", conf.input_size)))
    else:
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
    depth_root = None
    if depth_enabled:
        depth_root = os.path.join(conf.depth_root_path, conf.patch_info)
        if not os.path.isdir(depth_root):
            raise FileNotFoundError(f"深度数据目录不存在: {depth_root}")
    trainset = DatasetFolderFT(
        root_path, train_transform, None, conf.ft_width, conf.ft_height,
        conf=conf, is_train=True, depth_root=depth_root,
        depth_aux_enabled=depth_enabled,
        depth_target_mode=getattr(conf, "depth_target_mode", "class_conditioned")
    )
    # trainset = CelebASpoof(Path(conf.train_root_path), Path(conf.train_json_path), train_transform)

    all_datasets = [trainset]

    # ---------------------------------------------------------
    # 2. 获取附加数据集路径
    #    同时兼容 None、单个字符串、列表和元组
    # ---------------------------------------------------------
    additional_paths = getattr(
        conf,
        "additional_train_json_path",
        []
    )

    if additional_paths is None:
        additional_paths = []

    # 防止单个字符串被逐字符遍历
    if isinstance(additional_paths, (str, os.PathLike)):
        additional_paths = [additional_paths]

    if not isinstance(additional_paths, (list, tuple)):
        raise TypeError(
            "conf.additional_train_json_path 必须是 "
            "字符串、列表、元组或者 None，"
            f"当前类型为: {type(additional_paths)}"
        )

    additional_depth_paths = getattr(conf, "additional_depth_root_paths", [])
    if additional_depth_paths is None:
        additional_depth_paths = []
    if isinstance(additional_depth_paths, (str, os.PathLike)):
        additional_depth_paths = [additional_depth_paths]
    if depth_enabled and len(additional_depth_paths) != len(additional_paths):
        raise ValueError("additional_depth_root_paths 必须与 additional_train_json_path 一一对应")

    # ---------------------------------------------------------
    # 3. 依次加载所有附加训练集
    # ---------------------------------------------------------
    for dataset_index, additional_path in enumerate(
        additional_paths,
        start=1
    ):
        if additional_path is None:
            continue

        additional_path = str(additional_path).strip()

        if not additional_path:
            continue

        additional_root = os.path.join(
            additional_path,
            conf.patch_info
        )

        if not os.path.isdir(additional_root):
            raise FileNotFoundError(
                f"附加数据集目录不存在: {additional_root}"
            )

        print(
            f"\n[Additional Dataset {dataset_index}] "
            f"Loading from: {additional_root}"
        )

        additional_depth_root = None
        if depth_enabled:
            additional_depth_root = os.path.join(
                str(additional_depth_paths[dataset_index - 1]), conf.patch_info
            )
            if not os.path.isdir(additional_depth_root):
                raise FileNotFoundError(f"附加深度数据目录不存在: {additional_depth_root}")

        additional_trainset = DatasetFolderFT(
            root=additional_root,
            transform=train_transform,
            target_transform=None,
            ft_width=conf.ft_width,
            ft_height=conf.ft_height,
            conf=conf,
            is_train=True,
            depth_root=additional_depth_root,
            depth_aux_enabled=depth_enabled,
            depth_target_mode=getattr(conf, "depth_target_mode", "class_conditioned")
        )

        # 检查不同数据集的类别定义是否一致
        if additional_trainset.classes != trainset.classes:
            raise ValueError(
                "不同数据集的类别定义不一致：\n"
                f"基础数据集: {trainset.classes}\n"
                f"附加数据集: {additional_trainset.classes}\n"
                f"附加路径: {additional_root}"
            )

        all_datasets.append(additional_trainset)

    # ---------------------------------------------------------
    # 4. 合并数据集
    # ---------------------------------------------------------
    if len(all_datasets) == 1:
        final_dataset = all_datasets[0]
    else:
        final_dataset = ConcatDataset(all_datasets)

    # ---------------------------------------------------------
    # 5. 合并所有数据集的类别数量
    # ---------------------------------------------------------
    num_classes = conf.num_classes
    merged_counts = [0] * num_classes

    for dataset in all_datasets:
        if len(dataset.label_counts) != num_classes:
            raise ValueError(
                f"数据集类别数量异常。"
                f"期望 {num_classes} 类，"
                f"实际统计为 {dataset.label_counts}"
            )

        for class_index, count in enumerate(dataset.label_counts):
            merged_counts[class_index] += count

    total_samples = sum(merged_counts)

    # ---------------------------------------------------------
    # 6. 计算合并后全局类别权重
    # ---------------------------------------------------------
    weights = []

    for count in merged_counts:
        if count > 0:
            weight = total_samples / (num_classes * count)
        else:
            weight = 0.0

        weights.append(weight)

    final_dataset.class_weights_tensor = torch.tensor(
        weights,
        dtype=torch.float
    )

    # ConcatDataset 默认没有 classes 和 label_counts，
    # 可以手动添加，方便其他代码访问
    final_dataset.classes = trainset.classes
    final_dataset.label_counts = merged_counts

    # ---------------------------------------------------------
    # 7. 打印合并后的统计信息
    # ---------------------------------------------------------
    print("\n" + "🚀" * 25)
    print("🌍 [Global] Merged Dataset Statistics")
    print(f"Number of datasets: {len(all_datasets)}")
    print(f"Total combined samples: {total_samples}")
    print("-" * 60)

    for index, count in enumerate(merged_counts):
        class_name = trainset.classes[index]

        percentage = (
            count / total_samples * 100
            if total_samples > 0
            else 0.0
        )

        print(
            f"  - Class {index} [{class_name}]: "
            f"{count} samples "
            f"({percentage:.1f}%) | "
            f"Global Weight: {weights[index]:.4f}"
        )

    print("🚀" * 25 + "\n")

    # ---------------------------------------------------------
    # 8. 创建 DataLoader
    # ---------------------------------------------------------
    train_loader = DataLoader(
        final_dataset,
        batch_size=loader_batch_size,
        shuffle=True,
        pin_memory=loader_pin_memory,
        num_workers=loader_workers
    )

    return train_loader

def get_test_loader(conf):
    depth_enabled = bool(getattr(conf, "depth_aux_enabled", False))
    loader_workers = int(getattr(conf, "depth_num_workers", 4)) if depth_enabled else 16
    loader_pin_memory = bool(getattr(conf, "depth_pin_memory", False)) if depth_enabled else True
    loader_batch_size = int(getattr(conf, "depth_batch_size", 256)) if depth_enabled else conf.batch_size
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
        batch_size=loader_batch_size,
        shuffle=False,
        pin_memory=loader_pin_memory,
        num_workers=loader_workers)
    
    return test_loader