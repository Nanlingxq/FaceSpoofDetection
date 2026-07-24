# -*- coding: utf-8 -*-
# @Time : 20-6-3 下午5:39
# @Author : zhuying
# @Company : Minivision
# @File : train.py
# @Software : PyCharm
# python train.py --device_ids 3 --patch_info 2.7_80x80
# nohup python train.py --device_ids 3 --patch_info 2.7_80x80 &

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import os
import swanlab
from src.train_main import TrainMain
from src.default_config import get_default_config, update_config


def parse_args():
    """parsing and configuration"""
    desc = "Silence-FAS"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--device_ids", type=str, default="1", help="which gpu id, 0123")
    parser.add_argument("--patch_info", type=str, default="1_80x80",
                        help="[org_1_80x60 / 1_80x80 / 2.7_80x80 / 4_80x80]")
    parser.add_argument(
        "--enable_depth_aux", action="store_true",
        help="Enable depth auxiliary supervision; default keeps the original strategy.",
    )
    parser.add_argument(
        "--num_classes", type=int, default=None, choices=(2, 3),
        help="Override the classifier class count (2 or 3).",
    )
    # parser.add_argument("--test_pretrained_only", default= False, help="only test the pretrained model")
    return parser.parse_args()

args = parse_args()
cuda_devices = [int(elem) for elem in args.device_ids]
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuda_devices))
args.devices = [x for x in range(len(cuda_devices))]

# 2. 环境变量生效后，再导入核心包和自定义网络结构
import swanlab
from src.train_main import TrainMain
from src.default_config import get_default_config, update_config

if __name__ == "__main__":
    conf = get_default_config()
    conf = update_config(args, conf)
    if args.num_classes is not None:
        conf.num_classes = args.num_classes
    if args.enable_depth_aux:
        conf.depth_aux_enabled = True
    run = swanlab.init(project="FaceSpoofDetection", experiment_name=f'Training_in_{args.patch_info}', config=conf)
    trainer = TrainMain(conf)
    trainer.train_model()