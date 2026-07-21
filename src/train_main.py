# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:59
# @Author : zhuying
# @Company : Minivision
# @File : train_main.py
# @Software : PyCharm

import os
import torch
import swanlab
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from src.utility import get_time
from src.model_lib.MultiFTNet import MultiFTNet
from src.data_io.dataset_loader import get_train_loader, get_test_loader
from torchvision.utils import save_image
from AnaysisTools.confuion_matrix import print_confusion_matrix


class TrainMain:
    def __init__(self, conf):
        self.conf = conf
        self.board_loss_every = conf.board_loss_every
        self.save_every = conf.save_every
        self.step = 0
        self.start_epoch = 0
        self.train_loader = get_train_loader(self.conf)
        self.test_loader = get_test_loader(self.conf)
        self.start_time = get_time()
        self.save_dir = os.path.join(self.conf.model_path, self.start_time)
        # self.test_pretrained_only = conf.test_pretrained_only
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.debug_img_dir = os.path.join(self.save_dir, "debug_images")
        if not os.path.exists(self.debug_img_dir):
            os.makedirs(self.debug_img_dir)
        
        self._save_init_sample_images()
        self.best_val_acc = 0.0
        self.scaler = GradScaler()

    def _save_init_sample_images(self):
        print("\n[Debug] 正在从 DataLoader 提取初始 Batch 用于保存预览图...")
        
        # --- 保存训练集样本 ---
        try:
            train_batch = next(iter(self.train_loader))
            sample, ft_sample, target = train_batch 
            
            # 【关键修改】：将 [B, C, H, W] 中的 C 维度进行 [2, 1, 0] 反转，即 BGR 转 RGB
            sample_rgb = sample[:, [2, 1, 0], :, :]
            
            # 保存转换后的 RGB 图像
            save_image(sample_rgb[:16], os.path.join(self.debug_img_dir, 'init_train_rgb_batch.png'), normalize=True)
            save_image(ft_sample[:16], os.path.join(self.debug_img_dir, 'init_train_ft_batch.png'), normalize=True)
            print(f"[Debug] 训练集预览图已保存至: {self.debug_img_dir}")
        except Exception as e:
            print(f"[Debug] 保存训练集预览图失败: {e}")

        # --- 保存测试集样本 ---
        try:
            test_batch = next(iter(self.test_loader))
            if len(test_batch) == 3:
                sample, _, target = test_batch
            else:
                sample, target = test_batch
                
            # 【关键修改】：同理，反转测试集图像的通道
            sample_rgb = sample[:, [2, 1, 0], :, :]
            
            save_image(sample_rgb[:16], os.path.join(self.debug_img_dir, 'init_test_rgb_batch.png'), normalize=True)
            print(f"[Debug] 测试集预览图已保存至: {self.debug_img_dir}\n")
        except Exception as e:
            print(f"[Debug] 保存测试集预览图失败: {e}\n")

    def train_model(self):
        self._init_model_param()
        self._train_stage()

    def _init_model_param(self):
        class_weights = self.train_loader.dataset.class_weights_tensor.to(self.conf.device)
        self.cls_criterion = CrossEntropyLoss(weight=class_weights)
        self.ft_criterion = MSELoss()
        self.model = self._define_network()

        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.conf.lr,
                                   weight_decay=5e-4,
                                   momentum=self.conf.momentum)

        # self.schedule_lr = optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, self.conf.milestones, self.conf.gamma, - 1)

        self.schedule_lr = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.conf.epochs, eta_min=1e-6
        )
        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)

    def _train_stage(self):
        self.model.train()
        running_loss = 0.
        running_acc = 0.
        running_loss_cls = 0.
        running_loss_ft = 0.
        is_first = True

        for e in range(self.start_epoch, self.conf.epochs):
                
            if is_first:
                val_acc = self._validate_stage(epoch=e)
                self.writer = SummaryWriter(self.conf.log_path)
                is_first = False

            print('epoch {} started'.format(e))
            print("lr: ", self.schedule_lr.get_lr())

            for sample, ft_sample, target in tqdm(iter(self.train_loader)):

                imgs = [sample, ft_sample]
                labels = target

                loss, acc, loss_cls, loss_ft = self._train_batch_data(imgs, labels)
                running_loss_cls += loss_cls
                running_loss_ft += loss_ft
                running_loss += loss
                running_acc += acc

                self.step += 1

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    acc_board = running_acc / self.board_loss_every
                    lr = self.optimizer.param_groups[0]['lr']
                    loss_cls_board = running_loss_cls / self.board_loss_every
                    loss_ft_board = running_loss_ft / self.board_loss_every

                    # 3. 使用 swanlab.log 一次性提交一个字典，代替多行 writer.add_scalar
                    swanlab.log({
                        'Training/Loss': loss_board,
                        'Training/Acc': acc_board,
                        'Training/Learning_rate': lr,
                        'Training/Loss_cls': loss_cls_board,
                        'Training/Loss_ft': loss_ft_board
                    }, step=self.step)

                    running_loss = 0.
                    running_acc = 0.
                    running_loss_cls = 0.
                    running_loss_ft = 0.
                if self.step % self.save_every == 0 and self.step != 0:
                    self._save_state(extra=self.conf.job_name)

            val_acc = self._validate_stage(epoch=e)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                print(f"🌟 新的最佳验证集精度: {self.best_val_acc * 100:.2f}%! 正在保存模型...")
                self._save_state(extra="best_model", best=True)

            self.schedule_lr.step()

        time_stamp = get_time()
        self._save_state(extra=self.conf.job_name)
        self.writer.close()

    def _train_batch_data(self, imgs, labels):
        self.optimizer.zero_grad()
        labels = labels.to(self.conf.device)
        # embeddings, feature_map = self.model.forward(imgs[0].to(self.conf.device))
        embeddings, feature_map = self.model(imgs[0].to(self.conf.device))

        loss_cls = self.cls_criterion(embeddings, labels)
        loss_fea = self.ft_criterion(feature_map, imgs[1].to(self.conf.device))

        loss = 0.5*loss_cls + 0.5*loss_fea
        acc = self._get_accuracy(embeddings, labels)[0]
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc, loss_cls.item(), loss_fea.item()
    
    def _validate_stage(self, epoch):
        self.model.eval()  # 切换到验证模式
        val_loss = 0.
        val_acc = 0.
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad(): # 不计算梯度，节省显存加速推理
            for data in tqdm(iter(self.test_loader), desc=f"Epoch {epoch} Validating"):
                # 兼容 Dataset 返回格式：可能返回 (img, label) 或是 (img, ft_sample, label)
                if len(data) == 3:
                    sample, _, target = data
                else:
                    sample, target = data

                sample = sample.to(self.conf.device)
                target = target.to(self.conf.device)
                batch_size = target.size(0)
                
                # 验证集只评估分类损失 (ft_sample可能不存在)
                # embeddings = self.model.forward(sample)
                embeddings = self.model(sample)
                loss_cls = self.cls_criterion(embeddings, target)
                
                acc = self._get_accuracy(embeddings, target)[0].item()
                
                _, preds = torch.max(embeddings, 1) # 获取概率最大的类别索引
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(target.cpu().tolist())

                # 累加 loss 和 acc (乘以 batch_size 后续求全局平均)
                val_loss += loss_cls.item() * batch_size
                val_acc += acc * batch_size
                total_samples += batch_size
                
        # 计算整个测试集的平均 loss 和 accuracy
        avg_val_loss = val_loss / total_samples if total_samples > 0 else 0
        avg_val_acc = val_acc / total_samples if total_samples > 0 else 0
        
        # 记录到 swanlab (使用 step 对齐训练进度，便于在同一个图表时间轴上对比)
        swanlab.log({
            'Validation/Loss_cls': avg_val_loss,
            'Validation/Acc': avg_val_acc
        }, step=self.step)
        
        print(f"\n[Epoch {epoch} 验证结果] Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc*100:.2f}%\n")
        
        print_confusion_matrix(all_preds, all_targets, self.conf.num_classes)
        print("\n")

        # 验证结束后务必切回 train 模式
        self.model.train()

        return avg_val_acc

    def _define_network(self):
        param = {
            'num_classes': self.conf.num_classes,
            'img_channel': self.conf.input_channel,
            'embedding_size': self.conf.embedding_size,
            'conv6_kernel': self.conf.kernel_size,
            'conf': self.conf}

        # 1. 实例化基础模型
        model = MultiFTNet(**param).to(self.conf.device)

        # 2. 判断并加载预训练模型
        pretrain_src = getattr(self.conf, 'pretrain_model_src', None)
        if pretrain_src:
            if os.path.exists(pretrain_src):
                print(f"\n[*] 正在加载预训练模型: {pretrain_src}")
                state_dict = torch.load(pretrain_src, map_location=self.conf.device)
                
                # 处理可能由 DataParallel 保存带来的 'module.' 前缀问题
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k[7:] if k.startswith('module.') else k
                    if not new_key.startswith('model.'):
                        new_key = 'model.' + new_key
                    
                    new_state_dict[new_key] = v
                
                # 使用 strict=False 允许部分权重不匹配 (比如修改了分类类别数 num_classes)
                missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                print("[*] 预训练模型加载成功！")
                
                if missing_keys or unexpected_keys:
                    print(f"[-] 警告: 存在不匹配的键值 (通常发生在最后分类层发生改变时)。")
                    print(f"Missing keys: {missing_keys}")
                    print(f"Unexpected keys: {unexpected_keys}")
            else:
                print(f"\n[!] 错误: 找不到预训练模型文件 {pretrain_src}，将使用随机初始化继续训练。")
        return model

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret

    def _save_state(self, extra=None, best=False):
        # 使用 os.path.join 拼接路径，存入 self.save_dir
        if best:
            save_name = 'best.pth'
            save_path = os.path.join(self.save_dir, save_name)
            torch.save(self.model.state_dict(), save_path)
            return None

        save_name = '{}_{}_model_iter-{}.pth'.format(self.start_time, extra, self.step)
        save_path = os.path.join(self.save_dir, save_name)
        
        torch.save(self.model.state_dict(), save_path)