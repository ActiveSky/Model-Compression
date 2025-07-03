#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于知识蒸馏(Knowledge Distillation)的VGG网络训练脚本
该脚本实现了从大型VGG教师模型向小型VGG学生模型的知识迁移
"""

import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from models.vgg import VGG


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 知识蒸馏训练')
    parser.add_argument('--lr', default=0.1, type=float, help='学习率')
    parser.add_argument('--temperature', default=10, type=float, help='蒸馏温度参数')
    parser.add_argument('--alpha', default=0.5, type=float, help='软硬损失平衡系数')
    parser.add_argument('--epochs', default=200, type=int, help='训练轮数')
    parser.add_argument('--batch-size', default=64, type=int, help='训练批次大小')
    parser.add_argument('--test-batch-size', default=256, type=int, help='测试批次大小')
    parser.add_argument('--student-path', default="STUDENT_PATH", type=str, help='学生模型路径')
    parser.add_argument('--teacher-path', default="TEACHER_PATH", type=str, help='教师模型路径')
    parser.add_argument('--model-save-path', default="YOUR_KD_MODEL_SAVE_PATH", type=str, help='模型保存路径')
    parser.add_argument('--cifar-path', default="YOUR_CIFAR_PATH", type=str, help='CIFAR10数据集路径')
    
    return parser.parse_args()


def get_data_loaders(cifar_path, batch_size, test_batch_size):
    """
    准备CIFAR10数据加载器
    
    Args:
        cifar_path: CIFAR10数据集路径
        batch_size: 训练批次大小
        test_batch_size: 测试批次大小
        
    Returns:
        train_loader, test_loader: 训练和测试数据加载器
    """
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    # 定义训练数据变换：数据增强 + 标准化
    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 定义测试数据变换：仅标准化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 创建训练和测试数据集
    train_dataset = datasets.CIFAR10(cifar_path, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(cifar_path, train=False, transform=test_transform)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                              shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                             shuffle=False, **kwargs)
    
    return train_loader, test_loader


def load_models(student_path, teacher_path, device):
    """
    加载学生和教师模型
    
    Args:
        student_path: 学生模型路径
        teacher_path: 教师模型路径
        device: 设备(CPU/GPU)
        
    Returns:
        student, teacher: 加载好的学生和教师模型
    """
    # 加载学生模型（较小的VGG16）
    student_checkpoint = torch.load(student_path)
    student = VGG(depth=16, dataset='cifar10', slim_channel="quarter") #通道数为教师的1/4
    student.load_state_dict(student_checkpoint["state_dict"])
    student.to(device)
    
    # 加载教师模型（标准VGG16）
    teacher_checkpoint = torch.load(teacher_path)
    teacher = VGG(depth=16, dataset='cifar10', slim_channel="normal")
    teacher.load_state_dict(teacher_checkpoint["state_dict"])
    teacher.to(device)
    
    return student, teacher


def train_epoch(student, teacher, train_loader, optimizer, device, temperature, alpha):
    """
    训练一个轮次
    
    Args:
        student: 学生模型
        teacher: 教师模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备(CPU/GPU)
        temperature: 温度参数
        alpha: 软硬损失平衡系数
        
    Returns:
        train_loss: 平均训练损失
        train_acc: 训练准确率
    """
    student.train()  # 设置学生模型为训练模式
    teacher.eval()   # 设置教师模型为评估模式
    
    # 定义损失函数
    hard_criterion = torch.nn.CrossEntropyLoss()
    soft_criterion = torch.nn.KLDivLoss(reduction='batchmean')
    
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_data, batch_label in train_loader:
        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)
        
        # 前向传播
        student_output = student(batch_data)
        with torch.no_grad():  # 教师模型不需要梯度
            teacher_output = teacher(batch_data)
        
        # 计算硬损失（学生模型与真实标签）
        hard_loss = hard_criterion(student_output, batch_label)
        
        # 计算软损失（学生模型与教师模型的概率分布）
        soft_loss = soft_criterion(
            F.log_softmax(student_output/temperature, dim=1),
            F.softmax(teacher_output/temperature, dim=1)
        ) * (temperature ** 2)
        
        # 组合损失
        loss = hard_loss * (1. - alpha) + soft_loss * alpha
        
        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计数据
        train_loss += loss.item()
        _, student_predict = student_output.max(1)
        total += batch_label.size(0)
        correct += student_predict.eq(batch_label).sum().item()
    
    # 计算平均损失和准确率
    train_loss = train_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc


def test_model(student, test_loader, device):
    """
    测试模型性能
    
    Args:
        student: 学生模型
        test_loader: 测试数据加载器
        device: 设备(CPU/GPU)
        
    Returns:
        test_acc: 测试准确率
    """
    student.eval()  # 设置为评估模式
    
    correct = 0
    total = 0
    
    with torch.no_grad():  # 测试阶段不需要计算梯度
        for batch_data, batch_label in test_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)
            
            # 获取学生模型预测
            student_output = student(batch_data)
            _, student_predict = student_output.max(1)
            
            # 统计准确率
            total += batch_label.size(0)
            correct += student_predict.eq(batch_label).sum().item()
    
    test_acc = 100. * correct / total
    return test_acc


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 打印训练参数信息
    print("==================== 训练参数 ====================")
    print(f"温度(Temperature): {args.temperature}")
    print(f"软硬损失平衡系数(Alpha): {args.alpha}")
    print(f"学习率(Learning Rate): {args.lr}")
    print(f"设备(Device): {device}")
    print("=================================================")
    
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(
        args.cifar_path, args.batch_size, args.test_batch_size
    )
    
    # 加载模型
    student, teacher = load_models(args.student_path, args.teacher_path, device)
    
    # 配置优化器
    optimizer = torch.optim.SGD(student.parameters(),
                               lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # 训练过程
    best_acc = 0
    for epoch in range(args.epochs):
        # 学习率调度
        if epoch in [int(args.epochs * 0.5), int(args.epochs * 0.75)]:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.1
            print(f"Epoch {epoch}: 学习率调整为 {optimizer.param_groups[0]['lr']}")
        
        # 训练一个轮次
        train_loss, train_acc = train_epoch(
            student, teacher, train_loader, optimizer, 
            device, args.temperature, args.alpha
        )
        print(f"Epoch {epoch}: 训练损失 {train_loss:.4f}, 训练准确率 {train_acc:.2f}%")
        
        # 测试模型
        test_acc = test_model(student, test_loader, device)
        print(f"Epoch {epoch}: 测试准确率 {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            print(f"保存最佳模型，准确率: {test_acc:.2f}%")
            best_acc = test_acc
            state = {
                "state_dict": student.state_dict(),
                "acc": test_acc
            }
            torch.save(state, args.model_save_path)
    
    print(f"训练完成，最佳测试准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
