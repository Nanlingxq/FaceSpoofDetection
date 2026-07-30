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
from urllib.parse import urlparse


def _remove_unreachable_local_proxy():
    """Prevent SSH-local proxy settings from breaking cloud metric uploads.

    The server can reach SwanLab directly, but the interactive SSH environment
    may export a loopback proxy (for example, 127.0.0.1:17891). That proxy is
    tied to the client session and becomes unreachable after disconnecting.
    Remove only loopback proxy values; a deliberately configured remote proxy
    is preserved. Set FACE_SPOOF_KEEP_LOCAL_PROXY=1 to opt out.
    """
    if os.environ.get("FACE_SPOOF_KEEP_LOCAL_PROXY") == "1":
        return

    proxy_keys = (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "WSS_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "wss_proxy",
    )
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    removed = []
    for key in proxy_keys:
        value = os.environ.get(key)
        if not value:
            continue
        try:
            host = urlparse(value).hostname
        except ValueError:
            host = None
        if host in local_hosts:
            removed.append(key)
            os.environ.pop(key, None)
    if removed:
        print(
            "[Network] Removed unreachable loopback proxy variables: "
            + ", ".join(sorted(set(removed)))
        )


# Must run before importing SwanLab or any module that creates its HTTP client.
_remove_unreachable_local_proxy()


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
    parser.add_argument(
        "--depth_target_mode", choices=("raw", "class_conditioned"), default=None,
        help="Override depth target mode for this run."
    )
    parser.add_argument(
        "--no_depth_labels", type=str, default=None,
        help="Comma-separated labels whose depth targets are zeroed in class_conditioned mode."
    )
    parser.add_argument(
        "--depth_batch_size", type=int, default=None,
        help="Override the depth-training batch size for this run."
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
    if args.depth_target_mode is not None:
        conf.depth_target_mode = args.depth_target_mode
    if args.no_depth_labels is not None:
        try:
            conf.no_depth_labels = [int(label.strip()) for label in args.no_depth_labels.split(",") if label.strip()]
        except ValueError as exc:
            raise ValueError("--no_depth_labels must be a comma-separated list of integers") from exc
    if args.depth_batch_size is not None:
        if args.depth_batch_size <= 0:
            raise ValueError("--depth_batch_size must be positive")
        conf.depth_batch_size = args.depth_batch_size
    run = swanlab.init(project="FaceSpoofDetection", experiment_name=f'Training_in_{args.patch_info},Depth:{conf.depth_aux_enabled},Class:{conf.num_classes}', config=conf)
    trainer = TrainMain(conf)
    trainer.train_model()
