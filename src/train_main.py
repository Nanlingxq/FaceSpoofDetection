# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:59
# @Author : zhuying
# @Company : Minivision
# @File : train_main.py
# @Software : PyCharm

import os
import logging
import torch
import swanlab
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from src.utility import get_time
from src.model_lib.MultiFTNet import MultiFTNet
from src.model_lib.depth_auxiliary import depth_auxiliary_loss
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
        self.depth_aux_enabled = bool(getattr(self.conf, "depth_aux_enabled", False))
        self._setup_text_logger()
        self._log(f"Training started. depth_aux_enabled={self.depth_aux_enabled}")
        self._save_init_sample_images()
        self.best_val_acc = 0.0
        self.scaler = GradScaler()

    def _setup_text_logger(self):
        log_root = getattr(self.conf, "text_log_root", os.path.join(self.conf.model_path, "..", ".."))
        log_dir = os.path.join(log_root, "train_logs")
        os.makedirs(log_dir, exist_ok=True)
        self.text_log_path = os.path.join(log_dir, f"{self.conf.job_name}_{self.start_time}.txt")
        self.logger = logging.getLogger(f"face_spoof_train_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler = logging.FileHandler(self.text_log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)
        print(f"[Log] Text training log: {self.text_log_path}")

    def _log(self, message):
        print(message)
        if hasattr(self, "logger"):
            self.logger.info(message)

    def _save_init_sample_images(self):
        print("\n[Debug] 正在从 DataLoader 提取初始 Batch 用于保存预览图...")
        
        # --- 保存训练集样本 ---
        try:
            train_batch = next(iter(self.train_loader))
            if self.depth_aux_enabled:
                sample, ft_sample, _, _, target = train_batch
            else:
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
        running_loss = 0.0
        running_acc = 0.0
        running_loss_cls = 0.0
        running_loss_ft = 0.0
        running_loss_depth = 0.0
        is_first = True

        for e in range(self.start_epoch, self.conf.epochs):
            if is_first:
                val_acc = self._validate_stage(epoch=e)
                self.writer = SummaryWriter(self.conf.log_path)
                is_first = False

            self._log(f"epoch {e} started")
            self._log(f"lr: {self.schedule_lr.get_last_lr()[0]:.8f}")
            if self.depth_aux_enabled:
                warmup = max(1, int(getattr(self.conf, "depth_loss_warmup_epochs", 5)))
                depth_weight = float(getattr(self.conf, "depth_loss_weight", 0.1)) * min(1.0, (e + 1) / warmup)
            else:
                depth_weight = 0.0

            for batch in tqdm(iter(self.train_loader)):
                if self.depth_aux_enabled:
                    sample, ft_sample, depth_target, depth_mask, target = batch
                    imgs = [sample, ft_sample, depth_target, depth_mask]
                else:
                    sample, ft_sample, target = batch
                    imgs = [sample, ft_sample]
                loss, acc, loss_cls, loss_ft, loss_depth, _, _ = self._train_batch_data(
                    imgs, target, depth_weight=depth_weight
                )
                running_loss_cls += loss_cls
                running_loss_ft += loss_ft
                running_loss_depth += loss_depth
                running_loss += loss
                running_acc += acc
                self.step += 1

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    acc_board = running_acc / self.board_loss_every
                    lr = self.optimizer.param_groups[0]['lr']
                    loss_cls_board = running_loss_cls / self.board_loss_every
                    loss_ft_board = running_loss_ft / self.board_loss_every
                    loss_depth_board = running_loss_depth / self.board_loss_every
                    swanlab.log({
                        'Training/Loss': loss_board,
                        'Training/Acc': acc_board,
                        'Training/Learning_rate': lr,
                        'Training/Loss_cls': loss_cls_board,
                        'Training/Loss_ft': loss_ft_board,
                        'Training/Loss_depth': loss_depth_board,
                        'Training/Depth_weight': depth_weight,
                    }, step=self.step)
                    self._log(
                        f"step={self.step} loss={loss_board:.6f} acc={acc_board:.6f} "
                        f"cls={loss_cls_board:.6f} ft={loss_ft_board:.6f} "
                        f"depth={loss_depth_board:.6f} depth_weight={depth_weight:.4f}"
                    )
                    running_loss = running_acc = running_loss_cls = running_loss_ft = running_loss_depth = 0.0
                if self.step % self.save_every == 0 and self.step != 0:
                    self._save_state(extra=self.conf.job_name)

            val_acc = self._validate_stage(epoch=e)
            self._log(f"epoch={e} validation_acc={val_acc:.6f}")
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._log(f"New best validation accuracy: {self.best_val_acc * 100:.2f}%")
                self._save_state(extra="best_model", best=True)
            self.schedule_lr.step()

        self._save_state(extra=self.conf.job_name)
        self._log("Training finished")
        self.writer.close()
        for handler in getattr(self, "logger", logging.getLogger()).handlers[:]:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)

    def _train_batch_data(self, imgs, labels, depth_weight=0.0):
        self.optimizer.zero_grad()
        labels = labels.to(self.conf.device, non_blocking=True)
        rgb = imgs[0].to(self.conf.device, non_blocking=True)
        ft_target = imgs[1].to(self.conf.device, non_blocking=True)
        outputs = self.model(rgb)
        if self.depth_aux_enabled:
            embeddings, feature_map, depth_prediction = outputs
            depth_target = imgs[2].to(self.conf.device, non_blocking=True)
            depth_mask = imgs[3].to(self.conf.device, non_blocking=True)
            depth_loss, depth_pixel, depth_gradient = depth_auxiliary_loss(
                depth_prediction,
                depth_target,
                depth_mask,
                gradient_weight=float(getattr(self.conf, "depth_gradient_weight", 0.1)),
            )
        else:
            embeddings, feature_map = outputs
            depth_loss = depth_pixel = depth_gradient = torch.zeros((), device=self.conf.device)

        loss_cls = self.cls_criterion(embeddings, labels)
        loss_fea = self.ft_criterion(feature_map, ft_target)
        loss = 0.5 * loss_cls + 0.5 * loss_fea + depth_weight * depth_loss
        acc = self._get_accuracy(embeddings, labels)[0]
        loss.backward()
        self.optimizer.step()
        return (
            loss.item(), acc.item(), loss_cls.item(), loss_fea.item(),
            depth_loss.item(), depth_pixel.item(), depth_gradient.item()
        )
    
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