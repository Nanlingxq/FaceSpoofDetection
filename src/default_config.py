# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:12
# @Author : zhuying
# @Company : Minivision
# @File : default_config.py
# @Software : PyCharm
# --*-- coding: utf-8 --*--
"""
default config for training
"""

import torch
from datetime import datetime
from easydict import EasyDict
from src.utility import make_if_not_exist, get_width_height, get_kernel


def get_default_config():
    conf = EasyDict()

    # ----------------------training---------------
    conf.lr = 0.1
    # [9, 13, 15]
    conf.milestones = [10, 15, 22]  # down learing rate
    conf.gamma = 0.1
    conf.epochs = 100
    conf.momentum = 0.9
    conf.batch_size = 1024
    # conf.batch_size = 256
    # model
    conf.num_classes = 2
    conf.input_channel = 3
    conf.embedding_size = 128
    # conf.pretrain_model_src = '/userdata/lwk/FaceFakeDetection/SpoofDetection/Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth' #存在则加载预训练模型

    # dataset
    # conf.train_root_path = '/userdata/lwk/FaceFakeDetection/datasets/CelebA_Spoof/CelebA_Spoof/'
    # conf.train_json_path = '/userdata/lwk/FaceFakeDetection/datasets/CelebA_Spoof/CelebA_Spoof/metas/intra_test/train_label.json'
    # conf.test_root_path = '/userdata/lwk/FaceFakeDetection/datasets/CelebA_Spoof/CelebA_Spoof/'
    # conf.test_json_path = '/userdata/lwk/FaceFakeDetection/datasets/CelebA_Spoof/CelebA_Spoof/metas/intra_test/test_label.json'
    
    conf.enable_resample = False
    conf.test_root_path = '/userdata/lwk/FaceFakeDetection/datasets/CelebA_Spoof/CelebA_Spoof_Crop/test'
    conf.train_root_path = '/userdata/lwk/FaceFakeDetection/datasets/CelebA_Spoof/CelebA_Spoof_Crop/train'
    conf.additional_train_json_path = [
        "/userdata/lwk/FaceFakeDetection/datasets/CVPR23-FAS-WILD/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/Train_Crop",
        "/userdata/lwk/FaceFakeDetection/datasets/DISFA_Crop/train",
        "/userdata/lwk/FaceFakeDetection/datasets/ff_plus_Crop/train",
    ]

    # Optional depth auxiliary supervision. False preserves the original
    # classification + Fourier reconstruction training strategy.
    conf.depth_aux_enabled = False
    # Depth supervision target size; RGB input remains 80x80.
    conf.depth_target_size = (40, 40)
    conf.depth_target_mode = "class_conditioned"
    conf.depth_loss_weight = 0.1
    conf.depth_gradient_weight = 0.1
    conf.depth_loss_warmup_epochs = 5
    conf.depth_root_path = "/userdata/lwk/FaceFakeDetection/datasets/CelebA_Spoof/CelebA_Spoof_Depth/train"
    conf.additional_depth_root_paths = [
        "/userdata/lwk/FaceFakeDetection/datasets/CVPR23-FAS-WILD/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/Train_Depth",
        "/userdata/lwk/FaceFakeDetection/datasets/DISFA_Depth/train",
        "/userdata/lwk/FaceFakeDetection/datasets/ff_plus_Depth/train",
    ]
    conf.text_log_root = "/userdata/lwk/FaceFakeDetection/SpoofDetection/Silent-Face-Anti-Spoofing-master/saved_logs"
    # save file path
    conf.snapshot_dir_path = './saved_logs/snapshot'

    # log path
    conf.log_path = './saved_logs/jobs'
    # tensorboard
    conf.board_loss_every = 10
    # save model/iter
    conf.save_every = 900

    # test
    conf.model_dir = '/userdata/lwk/FaceFakeDetection/SpoofDetection/Silent-Face-Anti-Spoofing-master/saved_logs/snapshot/Anti_Spoofing_1_80x80/2026-07-13-22-42'
    return conf


def update_config(args, conf):
    conf.devices = args.devices
    conf.patch_info = args.patch_info
    w_input, h_input = get_width_height(args.patch_info)
    conf.input_size = [h_input, w_input]
    conf.kernel_size = get_kernel(h_input, w_input)
    conf.device = "cuda:{}".format(conf.devices[0]) if torch.cuda.is_available() else "cpu"

    # resize fourier image size
    conf.ft_height = 2*conf.kernel_size[0]
    conf.ft_width = 2*conf.kernel_size[1]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    job_name = 'Anti_Spoofing_{}'.format(args.patch_info)
    log_path = '{}/{}/{} '.format(conf.log_path, job_name, current_time)
    snapshot_dir = '{}/{}'.format(conf.snapshot_dir_path, job_name)

    make_if_not_exist(snapshot_dir)
    make_if_not_exist(log_path)

    conf.model_path = snapshot_dir
    conf.log_path = log_path
    conf.job_name = job_name
    return conf
