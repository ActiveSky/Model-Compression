#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models.vgg import VGG
from utils.merge_bn import fuse_model_vgg
from ptflops import get_model_complexity_info


def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    # 定义命令行参数
    parser = argparse.ArgumentParser(description='PyTorch VGG模型剪枝')
    parser.add_argument('--depth', type=int, default=16,
                        help='VGG网络深度 (默认: 16)')
    parser.add_argument('--percent', type=float, default=0.5,
                        help='剪枝比例 (默认: 0.5)')
    parser.add_argument('--pretrain_model', type=str, default='',
                        help='预训练模型路径')
    parser.add_argument('--save_path', type=str, default='',
                        help='剪枝后模型保存路径')
    parser.add_argument('--data_root', type=str, default='',
                        help='CIFAR数据集路径')
    parser.add_argument('--test_batch_size', type=int, default=256,
                        help='测试批次大小')
    parser.add_argument('--use_gpu', action='store_true',
                        help='是否使用GPU')
    
    return parser.parse_args()


def calc_time_and_flops(input_tensor, model, repeat=50):
    """
    计算模型推理时间和FLOPs
    
    Args:
        input_tensor: 输入张量样例
        model: 要评估的模型
        repeat: 重复测试次数，用于计算平均推理时间
        
    Returns:
        float: 平均推理时间
        str: FLOPs数量的字符串表示
        str: 参数数量的字符串表示
    """
    model.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(input_tensor)
        avg_infer_time = (time.time() - start) / repeat
        
        # 计算FLOPs和参数量
        flops, params = get_model_complexity_info(
            model, 
            (3, 32, 32), 
            as_strings=True,
            print_per_layer_stat=True
        )
    return avg_infer_time, flops, params


def load_model(model, pretrain_path):
    """
    加载预训练模型
    
    Args:
        model: 模型实例
        pretrain_path: 预训练模型路径
        
    Returns:
        tuple: (开始轮次, 最佳精度)
    """
    if not os.path.isfile(pretrain_path):
        print(f"=> 未找到预训练模型 '{pretrain_path}'")
        return 0, 0.0
    
    print(f"=> 加载预训练模型 '{pretrain_path}'")
    checkpoint = torch.load(pretrain_path)
    start_epoch = checkpoint.get('epoch', 0)
    best_prec1 = checkpoint.get('best_prec1', 0.0)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"=> 已加载预训练模型 '{pretrain_path}' (轮次 {start_epoch}) 精度: {best_prec1:.2f}%")
    
    return start_epoch, best_prec1


def test_model(model, data_root, test_batch_size, device):
    """
    在测试集上评估模型精度
    
    Args:
        model: 要评估的模型
        data_root: 数据集根目录
        test_batch_size: 测试批次大小
        device: 计算设备
        
    Returns:
        float: 测试精度
    """
    kwargs = {'num_workers': 1, 'pin_memory': True} if 'cuda' in device else {}
    
    # 加载CIFAR-10测试集
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            data_root, 
            train=False, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), 
                    (0.2023, 0.1994, 0.2010)
                )
            ])
        ),
        batch_size=test_batch_size, 
        shuffle=True, 
        **kwargs
    )
    
    # 评估模式
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 获取最大概率的索引
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'测试集精度: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


def get_pruning_threshold(model, prune_percent):
    """
    根据剪枝比例确定剪枝阈值
    
    Args:
        model: 要剪枝的模型
        prune_percent: 剪枝比例
        
    Returns:
        tuple: (阈值, 总通道数, BN权重排序结果)
    """
    # 收集所有BN层权重
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    
    bn_weights = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn_weights[index:(index+size)] = m.weight.data.abs().clone()
            index += size
    
    # 排序并确定阈值
    sorted_weights, _ = torch.sort(bn_weights)
    thre_index = int(total * prune_percent)
    threshold = sorted_weights[thre_index]
    
    return threshold, total, sorted_weights


def generate_pruning_masks(model, threshold):
    """
    根据阈值生成剪枝掩码
    
    Args:
        model: 要剪枝的模型
        threshold: 剪枝阈值
        
    Returns:
        tuple: (剪枝后通道配置, 掩码列表, 剪枝通道数)
    """
    pruned = 0
    cfg = []
    cfg_mask = []
    
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(threshold).float().to(m.weight.device)
            pruned += mask.shape[0] - torch.sum(mask)
            
            # 应用掩码到BN层参数
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            
            # 保存配置和掩码
            remaining_channels = int(torch.sum(mask))
            cfg.append(remaining_channels)
            cfg_mask.append(mask.clone())
            
            print(f'层索引: {k} \t 总通道数: {mask.shape[0]} \t 保留通道数: {remaining_channels}')
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    
    return cfg, cfg_mask, pruned


def create_pruned_model(model, new_model, cfg_mask):
    """
    创建剪枝后的模型并复制权重
    
    Args:
        model: 原始模型
        new_model: 剪枝后的新模型
        cfg_mask: 剪枝掩码列表
    """
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)  # 输入图像通道掩码
    end_mask = cfg_mask[layer_id_in_cfg]
    
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            # 复制保留的通道的BN参数
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
                
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            
            # 更新掩码
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
                
        elif isinstance(m0, nn.Conv2d):
            # 复制保留的输入和输出通道的卷积权重
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
                
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            
        elif isinstance(m0, nn.Linear):
            # 复制保留的输入通道的全连接层权重
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
                
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载原始模型
    model = VGG(dataset='cifar10', depth=args.depth)
    _, best_prec1 = load_model(model, args.pretrain_model)
    model = model.to(device)
    
    # 记录原始模型参数量
    orig_params = sum([param.nelement() for param in model.parameters()])
    print(f"原始模型参数量: {orig_params}")
    
    # 测量原始模型性能
    random_input = torch.rand((1, 3, 32, 32)).to(device)
    model = model.to(device)
    orig_time, orig_flops, orig_params_info = calc_time_and_flops(random_input, model)
    
    # 确定剪枝阈值
    print("\n--- 开始确定剪枝阈值 ---")
    threshold, total, _ = get_pruning_threshold(model, args.percent)
    
    # 生成剪枝掩码
    print("\n--- 开始生成剪枝掩码 ---")
    cfg, cfg_mask, pruned = generate_pruning_masks(model, threshold)
    pruned_ratio = pruned / total
    print(f'剪枝率: {pruned_ratio:.2f}')
    
    # 测试简单剪枝效果（仅将权重置零）
    print("\n--- 测试简单剪枝效果 ---")
    fake_prune_acc = test_model(model, args.data_root, args.test_batch_size, device)
    
    # 创建真正剪枝后的模型
    print("\n--- 创建真正剪枝后的模型 ---")
    new_model = VGG(dataset='cifar10', cfg=cfg)
    new_model = new_model.to(device)
    create_pruned_model(model, new_model, cfg_mask)
    
    # 合并BN层并保存模型
    print("\n--- 合并BN层并保存模型 ---")
    fuse_model_vgg(new_model)
    pruned_params = sum([param.nelement() for param in new_model.parameters()])
    
    # 保存剪枝后的模型
    torch.save(
        {
            'cfg': cfg, 
            'state_dict': new_model.state_dict(),
            'pruned_ratio': pruned_ratio
        }, 
        args.save_path
    )
    print(f"剪枝后模型已保存至: {args.save_path}")
    
    # 测试真正剪枝后的模型
    print("\n--- 测试真正剪枝后的模型 ---")
    real_prune_acc = test_model(new_model, args.data_root, args.test_batch_size, device)
    
    # 测量剪枝后模型性能
    new_model = new_model.to(device)
    pruned_time, pruned_flops, pruned_params_info = calc_time_and_flops(random_input, new_model)
    
    # 打印性能对比
    print("\n--- 性能对比 ---")
    print(f"原始模型推理时间: {orig_time:.6f}s  vs  剪枝模型推理时间: {pruned_time:.6f}s")
    print(f"原始模型FLOPS: {orig_flops}  vs  剪枝模型FLOPS: {pruned_flops}")
    print(f"原始模型参数: {orig_params_info}  vs  剪枝模型参数: {pruned_params_info}")
    print(f"原始模型参数数量: {orig_params}  vs  剪枝模型参数数量: {pruned_params}")
    print(f"参数减少率: {(1 - pruned_params/orig_params):.2f}")
    print(f"原始模型精度: {best_prec1:.2f}%  vs  剪枝模型精度: {real_prune_acc:.2f}%")
    print(f"简单剪枝精度: {fake_prune_acc:.2f}%  vs  结构剪枝精度: {real_prune_acc:.2f}%")


if __name__ == "__main__":
    main()
