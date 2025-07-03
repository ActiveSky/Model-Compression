import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
from utils.merge_bn import fuse_model_resnet
import time
from ptflops import get_model_complexity_info
from typing import List, Tuple, Optional, Union


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PyTorch ResNet模型剪枝工具')
    parser.add_argument('--depth', type=int, default=164,
                        help='ResNet深度 (default: 164)')
    parser.add_argument('--percent', type=float, default=0.5,
                        help='剪枝比例 (default: 0.5)')
    return parser.parse_args()


def load_dataset(data_root: str, batch_size: int = 256):
    """加载CIFAR-10测试数据集
    
    Args:
        data_root: 数据集根目录
        batch_size: 批次大小
        
    Returns:
        数据加载器
    """
    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            data_root, train=False, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        ),
        batch_size=batch_size, shuffle=False, **kwargs
    )
    return test_loader


def test(model: nn.Module, data_root: str, test_batch_size: int = 256) -> float:
    """评估模型在测试集上的准确率
    
    Args:
        model: 待评估的模型
        data_root: 数据集根目录
        test_batch_size: 测试批次大小
        
    Returns:
        测试集上的准确率
    """
    test_loader = load_dataset(data_root, test_batch_size)
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # 获取最大概率对应的索引
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    accuracy = correct / float(len(test_loader.dataset))
    return accuracy


def calc_time_and_flops(model: nn.Module, input_tensor: torch.Tensor, repeat: int = 20) -> Tuple[float, str, str]:
    """计算模型的推理时间和计算复杂度
    
    Args:
        model: 待评估的模型
        input_tensor: 输入样例张量
        repeat: 重复测试的次数
        
    Returns:
        平均推理时间, FLOPS计算量(字符串), 参数数量(字符串)
    """
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
            _ = model(input_tensor)
        avg_infer_time = (time.time() - start) / repeat
        
        flops, params = get_model_complexity_info(
            model, (3, 32, 32), 
            as_strings=True,
            print_per_layer_stat=True
        )  # 默认batch_size=1

    return avg_infer_time, flops, params


def get_bn_threshold(model: nn.Module, percent: float) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """获取BatchNorm剪枝阈值
    
    Args:
        model: 原始模型
        percent: 剪枝比例
        
    Returns:
        剪枝阈值, BatchNorm权重列表, 总通道数
    """
    # 计算模型中所有BN层的通道数量
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
    
    # 收集所有BN层权重
    bn_weights = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn_weights[index:(index + size)] = m.weight.data.abs().clone()
            index += size
    
    # 按权重大小排序并确定阈值
    y, _ = torch.sort(bn_weights)
    thre_index = int(total * percent)
    threshold = y[thre_index]
    
    return threshold, bn_weights, total


def create_pruning_mask(model: nn.Module, threshold: float) -> Tuple[List, List, int]:
    """创建剪枝掩码
    
    Args:
        model: 原始模型
        threshold: 剪枝阈值
        
    Returns:
        剪枝后的通道配置, 剪枝掩码列表, 被剪枝的通道数量
    """
    pruned = 0
    cfg = []
    cfg_mask = []
    
    # 为每一层创建掩码
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(threshold).float().cuda()
            pruned += mask.shape[0] - torch.sum(mask)
            
            # 应用掩码到BN层权重和偏置
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            
            # 记录剪枝后的配置
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            
            print(f'层索引: {k} \t 总通道数: {mask.shape[0]} \t 保留通道数: {int(torch.sum(mask))}')
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
    
    return cfg, cfg_mask, pruned


def copy_weights_to_new_model(model: nn.Module, new_model: nn.Module, cfg_mask: List) -> None:
    """将权重从原始模型复制到剪枝后的新模型
    
    Args:
        model: 原始模型
        new_model: 剪枝后的新模型
        cfg_mask: 剪枝掩码列表
    """
    old_modules = list(model.modules())
    new_modules = list(new_model.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0
    
    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        
        # 处理批归一化层
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            
            # 处理通道选择层
            if isinstance(old_modules[layer_id + 1], channel_selection):
                # 如果下一层是通道选择层，当前BN层不会被剪枝
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
                
                # 设置通道选择层
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0
                
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                # 普通BN层，直接复制保留通道的参数
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # 最后的全连接层不需要更改
                    end_mask = cfg_mask[layer_id_in_cfg]
                    
        # 处理卷积层
        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                # 第一个卷积层保持不变
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
                
            if isinstance(old_modules[layer_id - 1], channel_selection) or isinstance(old_modules[layer_id - 1], nn.BatchNorm2d):
                # 处理残差块中的卷积层
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                    
                # 应用输入通道掩码
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                
                # 如果不是残差块中的第一个卷积，还需应用输出通道掩码
                if conv_count % 3 != 1:
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                    
                m1.weight.data = w1.clone()
                continue
            
            # 处理下采样卷积层，直接复制权重
            m1.weight.data = m0.weight.data.clone()
            
        # 处理全连接层
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            
            # 只需要调整输入特征
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()


def main():
    """主函数"""
    # 配置参数
    args = parse_arguments()
    model_path = "YOUR_SR_MODEL_PATH"
    data_root = "YOUR_CIFAR_PATH"
    model_save_path = "YOUR_MODEL_SAVE_PATH"
    test_batch_size = 256
    
    # 创建并加载原始模型
    model = ResNet(depth=args.depth, dataset='cifar10')
    
    # 加载预训练权重
    if os.path.isfile(model_path):
        print(f"=> 加载检查点 '{model_path}'")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        best_prec1 = checkpoint['best_prec1']
        print(f"=> 已加载检查点 '{model_path}' (epoch {checkpoint['epoch']}) Prec1: {best_prec1:.4f}")
    else:
        print("模型路径错误...")
        return
    
    # 记录原始模型性能指标
    ori_model_acc = best_prec1
    ori_model_parameters = sum([param.nelement() for param in model.parameters()])
    
    # 计算原始模型推理时间和计算复杂度
    cpu_device = torch.device("cpu")
    device = torch.device("cuda")
    random_input = torch.rand((1, 3, 32, 32)).to(cpu_device)
    model.to(cpu_device)
    origin_forward_time, origin_flops, origin_params = calc_time_and_flops(random_input, model)
    
    # 将模型转移到GPU上进行剪枝操作
    model.to(device)
    
    # 计算剪枝阈值
    threshold, bn_weights, total = get_bn_threshold(model, args.percent)
    
    # 创建剪枝掩码
    cfg, cfg_mask, pruned = create_pruning_mask(model, threshold)
    pruned_ratio = pruned / total
    
    # 剪枝后的模型准确率初步测试
    fake_prune_acc = test(model, data_root, test_batch_size)
    print(f"应用掩码后的模型准确率: {fake_prune_acc:.4f}")
    
    # 创建新的剪枝模型
    new_model = ResNet(depth=args.depth, dataset='cifar10', cfg=cfg)
    new_model.to(device)
    
    # 统计剪枝后模型参数量
    prune_model_parameters = sum([param.nelement() for param in new_model.parameters()])
    
    # 将原模型权重复
    # 将原模型权重复制到新模型中
    copy_weights_to_new_model(model, new_model, cfg_mask)
    
    # 测试剪枝后模型的准确率
    real_prune_model_acc = test(new_model, data_root, test_batch_size)
    print(f"剪枝后模型准确率: {real_prune_model_acc:.4f}")
    
    # 融合BatchNorm层到卷积层中
    fuse_model_resnet(new_model)
    real_prune_fuse_model_acc = test(new_model, data_root, test_batch_size)
    print(f"剪枝并融合BN后模型准确率: {real_prune_fuse_model_acc:.4f}")
    
    # 保存剪枝后的模型
    torch.save({'cfg': cfg, 'state_dict': new_model.state_dict()}, model_save_path)
    print(f"剪枝后的模型已保存到 {model_save_path}")
    
    # 评估剪枝后模型的效率指标
    new_model.to(cpu_device)
    pruned_forward_time, pruned_flops, pruned_params = calc_time_and_flops(random_input, new_model)
    
    # 输出性能对比结果
    print("\n========== 性能对比 ==========")
    print(f"推理时间: 原始模型 {origin_forward_time:.6f}秒 vs 剪枝模型 {pruned_forward_time:.6f}秒")
    print(f"计算量: 原始模型 {origin_flops} vs 剪枝模型 {pruned_flops}")
    print(f"参数量: 原始模型 {origin_params} vs 剪枝模型 {pruned_params}")
    print(f"参数数量: 原始模型 {ori_model_parameters:,} vs 剪枝模型 {prune_model_parameters:,}")
    print(f"准确率: 原始模型 {ori_model_acc:.4f} vs 剪枝模型 {real_prune_model_acc:.4f}")
    print(f"剪枝率: {pruned_ratio:.4f}")
    print("===============================")


if __name__ == "__main__":
    main()

