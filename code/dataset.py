# code/dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_clip_preprocess(image_size=224):
    """
    获取 CLIP 模型官方推荐的图像预处理流程。
    """
    return transforms.Compose([
        # 将图片尺寸调整为 image_size x image_size
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        # 从中心裁剪出 image_size x image_size 的区域
        transforms.CenterCrop(image_size),
        # 将可能存在的 RGBA (4通道) 图片转为 RGB (3通道)
        lambda image: image.convert("RGB"),
        # 将 PIL Image 对象转为 PyTorch 张量 (Tensor)，并将像素值从 [0, 255] 缩放到 [0, 1]
        transforms.ToTensor(),
        # 使用 CLIP 预训练时用的均值和标准差对图像进行归一化
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), 
            (0.26862954, 0.26130258, 0.27577711)
        ),
    ])

def create_dataloaders(dataset_path, batch_size=32, num_workers=4):
    """
    根据给定的数据集路径创建训练、验证和测试的 DataLoader。
    
    Args:
        dataset_path (str): 数据集根目录路径 (例如 "../datasets/dataset-A")
        batch_size (int): 每个批次的样本数
        num_workers (int): 用于加载数据的子进程数
        
    Returns:
        tuple: 包含 (train_loader, val_loader, test_loader) 的元组
    """
    preprocess = get_clip_preprocess()
    
    # 使用 ImageFolder 加载数据，它会自动从文件夹名推断标签
    train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, "train"), transform=preprocess)
    val_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, "val"), transform=preprocess)
    test_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, "test"), transform=preprocess)
    
    print(f"训练集信息: 共 {len(train_dataset)} 个样本。 类别: {train_dataset.classes} (0: real, 1: fake)")
    print(f"验证集信息: 共 {len(val_dataset)} 个样本。 类别: {val_dataset.classes}")
    print(f"测试集信息: 共 {len(test_dataset)} 个样本。 类别: {test_dataset.classes}")

    # 创建 DataLoader
    # 训练集需要打乱 (shuffle=True)，验证和测试集则不需要
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

# --- 使用 __name__ == "__main__" 进行模块化测试 ---
# 这样你可以直接运行 `python dataset.py` 来检查数据加载是否正常
if __name__ == '__main__':
    # 假设你的 dataset-A 文件夹在 datasets/ 目录下，并且 code/ 和 datasets/ 是同级目录
    DATA_PATH = "../datasets/dataset-A"
    
    if not os.path.exists(DATA_PATH):
        print(f"错误：数据集路径 '{DATA_PATH}' 不存在。请检查路径是否正确。")
    else:
        print("--- 开始测试数据加载模块 ---")
        train_loader, _, _ = create_dataloaders(dataset_path=DATA_PATH, batch_size=4, num_workers=0)
        
        # 从 train_loader 中取一个 batch 的数据看看
        try:
            images, labels = next(iter(train_loader))
            
            print("\n成功取出一个批次的数据！")
            print(f"图像张量的形状 (Shape): {images.shape}") # 应该类似 [4, 3, 224, 224]
            print(f"标签张量的形状 (Shape): {labels.shape}") # 应该类似 [4]
            print(f"标签示例: {labels}")
            print("\n--- 数据加载模块测试通过 ---")
        except Exception as e:
            print(f"\n在取出数据时发生错误: {e}")
