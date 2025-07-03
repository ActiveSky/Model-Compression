#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ResNet知识蒸馏训练脚本
通过知识蒸馏(Knowledge Distillation)技术，将大模型(教师模型)的知识迁移到小模型(学生模型)
"""

import argparse
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from models.preresnet import ResNet
from utils.merge_bn import fuse_model_resnet


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 知识蒸馏训练')
    parser.add_argument('--lr', default=0.1, type=float, help='学习率')
    parser.add_argument('--epochs', default=200, type=int, help='训练轮数')
    parser.add_argument('--batch_size', default=64, type=int, help='训练批次大小')
    parser.add_argument('--test_batch_size', default=256, type=int, help='测试批次大小')
    parser.add_argument('--temperature', default=50, type=float, help='蒸馏温度参数')
    parser.add_argument('--alpha', default=0.9, type=float, help='蒸馏软标签权重')
    parser.add_argument('--student_model', default="STUDENT_MODEL", type=str, help='学生模型路径')
    parser.add_argument('--teacher_model', default="TEACHER_MODEL", type=str, help='教师模型路径')
    parser.add_argument('--save_path', default="YOUR_KD_MODEL_SAVE_PATH", type=str, help='模型保存路径')
    parser.add_argument('--cifar_path', default="YOUR_CIFAR_PATH", type=str, help='CIFAR数据集路径')
    return parser.parse_args()


def load_data(cifar_path, batch_size, test_batch_size):
    """
    加载并预处理CIFAR10数据集
    
    参数:
        cifar_path: CIFAR10数据集路径
        batch_size: 训练批次大小
        test_batch_size: 测试批次大小
    
    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    
    # 定义数据预处理和增强
    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载训练集和测试集
    train_dataset = datasets.CIFAR10(cifar_path, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(cifar_path, train=False, transform=test_transform)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True, **kwargs)
    
    return train_loader, test_loader


def load_models(student_weight, teacher_weight, device):
    """
    加载学生模型和教师模型
    
    参数:
        student_weight: 学生模型权重文件路径
        teacher_weight: 教师模型权重文件路径
        device: 计算设备(CPU/GPU)
    
    返回:
        student: 学生模型
        teacher: 教师模型
    """
    # 加载学生模型（较小的ResNet-38）
    student_checkpoint = torch.load(student_weight)
    student = ResNet(depth=38, dataset='cifar10', cfg=student_checkpoint["cfg"])
    fuse_model_resnet(student)  # 融合BN层
    student.load_state_dict(student_checkpoint["state_dict"])
    student.to(device)
    
    # 加载教师模型（较大的ResNet-92）
    teacher_checkpoint = torch.load(teacher_weight)
    teacher = ResNet(depth=92, dataset='cifar10', cfg=None)
    teacher.load_state_dict(teacher_checkpoint["state_dict"])
    teacher.to(device)
    
    return student, teacher


def train(epoch, student, teacher, train_loader, optimizer, 
          hard_criterion, soft_criterion, temperature, alpha, device):
    """
    单个epoch的训练函数
    
    参数:
        epoch: 当前训练轮次
        student: 学生模型
        teacher: 教师模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        hard_criterion: 硬目标损失函数
        soft_criterion: 软目标损失函数
        temperature: 蒸馏温度参数
        alpha: 软目标权重
        device: 计算设备(CPU/GPU)
    
    返回:
        train_loss: 训练损失
        train_acc: 训练准确率
    """
    student.train()
    teacher.eval()

    train_loss = 0
    total = 0
    correct = 0
    
    for batch_data, batch_label in train_loader:
        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)
        
        # 前向传播
        student_output = student(batch_data)
        with torch.no_grad():
            teacher_output = teacher(batch_data)
        
        # 计算硬目标损失（学生预测与真实标签之间的交叉熵）
        hard_loss = hard_criterion(student_output, batch_label)
        
        # 计算软目标损失（学生和教师输出概率分布的KL散度）
        soft_loss = soft_criterion(
            F.log_softmax(student_output/temperature, dim=1),
            F.softmax(teacher_output/temperature, dim=1)
        ) * temperature * temperature
        
        # 结合硬目标和软目标的损失
        loss = hard_loss * (1. - alpha) + soft_loss * alpha
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计训练准确率
        _, student_predict = student_output.max(1)
        total += batch_label.size(0)
        correct += student_predict.eq(batch_label).sum().item()
        train_loss += loss.item()
    
    # 计算平均训练损失和准确率
    train_loss = train_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    print(f"Epoch {epoch}, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}% ({correct}/{total})")
    return train_loss, train_acc


def test(epoch, student, test_loader, model_save_path, device):
    """
    测试函数
    
    参数:
        epoch: 当前训练轮次
        student: 学生模型
        test_loader: 测试数据加载器
        model_save_path: 模型保存路径
        device: 计算设备(CPU/GPU)
    
    返回:
        test_acc: 测试准确率
        best_acc: 如果达到更好的准确率，返回新的最佳准确率
    """
    global best_acc
    student.eval()

    total = 0
    correct = 0
    
    with torch.no_grad():
        for batch_data, batch_label in test_loader:
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            student_output = student(batch_data)
            _, student_predict = student_output.max(1)

            total += batch_label.size(0)
            correct += student_predict.eq(batch_label).sum().item()

    test_acc = 100. * correct / total
    print(f"Epoch {epoch}, 测试准确率: {test_acc:.2f}% ({correct}/{total})")

    # 如果达到更好的准确率，保存模型
    if test_acc > best_acc:
        print(f"保存最佳模型，准确率从 {best_acc:.2f}% 提高到 {test_acc:.2f}%")
        state = {
            "state_dict": student.state_dict(),
            "acc": test_acc
        }
        torch.save(state, model_save_path)
        best_acc = test_acc
    
    return test_acc, best_acc


def adjust_learning_rate(optimizer, epoch, epochs):
    """
    根据训练进度调整学习率
    
    参数:
        optimizer: 优化器
        epoch: 当前轮次
        epochs: 总训练轮数
    """
    # 在训练的50%和75%处降低学习率
    if epoch in [int(epochs * 0.5), int(epochs * 0.75)]:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.1
        print(f"学习率调整为: {param_group['lr']}")


def main():
    """主函数，执行知识蒸馏训练"""
    global best_acc
    
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 打印训练参数
    print(f"{'='*20} 知识蒸馏训练参数 {'='*20}")
    print(f"温度参数: {args.temperature}, 软标签权重: {args.alpha}, 学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}, 批次大小: {args.batch_size}, 测试批次大小: {args.test_batch_size}")
    print(f"{'='*60}")
    
    # 加载数据集
    train_loader, test_loader = load_data(args.cifar_path, args.batch_size, args.test_batch_size)
    
    # 加载模型
    student, teacher = load_models(args.student_model, args.teacher_model, device)
    
    # 定义损失函数
    hard_criterion = torch.nn.CrossEntropyLoss()
    soft_criterion = torch.nn.KLDivLoss(reduction='batchmean')
    
    # 定义优化器
    optimizer = torch.optim.SGD(student.parameters(),
                               lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # 初始化最佳准确率
    best_acc = 0.0
    
    # 开始训练
    for epoch in range(args.epochs):
        # 调整学习率
        adjust_learning_rate(optimizer, epoch, args.epochs)
        
        # 训练和测试
        train(epoch, student, teacher, train_loader, optimizer, 
              hard_criterion, soft_criterion, args.temperature, args.alpha, device)
        test_acc, best_acc = test(epoch, student, test_loader, args.save_path, device)
    
    print(f"训练结束，最佳测试准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
