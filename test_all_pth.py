# -*- coding: utf-8 -*-
# @Time : [当前时间]
# @Author : Gemini
# @File : test_main.py
# @Software : PyCharm
# CUDA_VISIBLE_DEVICES=1 python test_all_pth.py
# nohup env CUDA_VISIBLE_DEVICES=1 python test_all_pth.py &

import os
import glob
import torch
import swanlab
from tqdm import tqdm
from train import parse_args
from src.model_lib.MultiFTNet import MultiFTNet
# 假设你有一个用于获取测试集 DataLoader 的函数
from src.data_io.dataset_loader import get_test_loader 
from src.default_config import get_default_config, update_config

class TestMain:
    def __init__(self, conf):
        self.conf = conf
        # 获取测试集 DataLoader
        self.test_loader = get_test_loader(self.conf)
        # 存放模型权重的文件夹路径
        self.model_dir = self.conf.model_path 
        
        # 记录测试结果
        self.best_acc = 0.0
        self.best_model_name = ""
        self.results = {}

    def run_test(self):
        # 1. 查找目录下所有的 .pth 文件
        pth_files = glob.glob(os.path.join(self.model_dir, '*.pth'))
        if not pth_files:
            print(f"在 {self.model_dir} 下未找到任何 .pth 文件！")
            return

        print(f"共找到 {len(pth_files)} 个模型，准备开始测试...")

        # 2. 初始化模型架构
        self.model = self._define_network()

        # 3. 遍历并测试每一个模型
        for pth_path in pth_files:
            self._evaluate_single_model(pth_path)

        # 4. 打印最终汇总结果
        self._print_final_results()

    def _evaluate_single_model(self, pth_path):
        model_name = os.path.basename(pth_path)
        print(f"\n[{model_name}] 正在加载并测试...")
        
        # 加载权重
        # 注意: 训练时使用了 DataParallel，所以 state_dict 的 key 带有 'module.' 前缀
        # 由于我们在 _define_network 中也使用了 DataParallel，这里可以直接 load
        state_dict = torch.load(pth_path, map_location=self.conf.device)
        self.model.load_state_dict(state_dict)
        
        # 设置为验证/测试模式 (关闭 Dropout, BatchNorm 使用全局统计量)
        self.model.eval()

        total_acc = 0.0
        total_samples = 0

        # 不计算梯度，节省显存并加速
        with torch.no_grad():
            # 假设测试集的 loader 也会返回 sample, ft_sample(可能为空或不需要), target
            for sample, target in tqdm(iter(self.test_loader), desc="Testing"):
                sample = sample.to(self.conf.device)
                target = target.to(self.conf.device)
                batch_size = target.size(0)

                # 前向传播 (仅提取分类用的 embeddings)
                embeddings = self.model(sample)

                # 获取 batch 的准确率 (利用你原有的方法)
                # _get_accuracy 返回的是 [correct_ratio]，需要提取并还原为正确的样本数
                acc_ratio = self._get_accuracy(embeddings, target)[0].item()
                
                total_acc += acc_ratio * batch_size
                total_samples += batch_size

        # 计算整个测试集的平均准确率
        avg_acc = total_acc / total_samples if total_samples > 0 else 0
        self.results[model_name] = avg_acc

        print(f"[{model_name}] 测试完成，准确率: {avg_acc * 100:.2f}%")

        # 更新最佳模型记录
        if avg_acc > self.best_acc:
            self.best_acc = avg_acc
            self.best_model_name = model_name

    def _define_network(self):
        """与训练代码保持一致的网络定义"""
        param = {
            'num_classes': self.conf.num_classes,
            'img_channel': self.conf.input_channel,
            'embedding_size': self.conf.embedding_size,
            'conv6_kernel': self.conf.kernel_size}

        model = MultiFTNet(**param).to(self.conf.device)
        model = torch.nn.DataParallel(model, self.conf.devices)
        model.to(self.conf.device)
        return model

    def _get_accuracy(self, output, target, topk=(1,)):
        """复用训练代码中的准确率计算方法"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret

    def _print_final_results(self):
        print("\n" + "="*40)
        print("所有模型测试完毕！结果总结：")
        print("="*40)
        
        # 按准确率从高到低排序打印
        sorted_results = sorted(self.results.items(), key=lambda item: item[1], reverse=True)
        for name, acc in sorted_results:
            print(f"{name}: {acc * 100:.2f}%")
            
        print("-" * 40)
        print(f"🏆 表现最好的模型是: {self.best_model_name}")
        print(f"✨ 最高准确率为: {self.best_acc * 100:.2f}%")
        print("="*40)


if __name__ == "__main__":
    # 这里的 Conf 需要你根据自己的实际环境进行替换或导入
    args = parse_args()
    conf = get_default_config()
    update_config(args,conf)
    run = swanlab.init(project="FaceSpoofDetection", experiment_name=f'Test_in_{args.patch_info}', config=conf)
    tester = TestMain(conf)
    tester.run_test()
    pass