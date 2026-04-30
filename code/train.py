# code/train.py

import torch
import time
import datetime
import logging
import csv
import shutil
import sys
import os
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

# 从我们自己的模块中导入
from dataset import create_dataloaders
from model import PGMoE
from loss import OrthogonalLoss

def setup_logging(log_file):
    """配置日志，同时输出到控制台和文件"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def calculate_metrics(labels, preds_prob):
    """计算二分类的各种指标"""
    preds_binary = np.round(preds_prob)
    
    auc = roc_auc_score(labels, preds_prob)
    acc = accuracy_score(labels, preds_binary)
    f1 = f1_score(labels, preds_binary, zero_division=0)
    precision = precision_score(labels, preds_binary, zero_division=0)
    recall = recall_score(labels, preds_binary, zero_division=0)
    
    return {
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def get_args():
    parser = argparse.ArgumentParser(description="PG-MoE Training Script")
    
    # 数据集配置
    parser.add_argument("--dataset", type=str, default="dataset-A", help="Name of the dataset (folder name in datasets/)")
    parser.add_argument("--data_root", type=str, default="./datasets", help="Root directory for datasets")
    
    # 训练超参数
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lambda_orth", type=float, default=0.05, help="Weight for orthogonal loss")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # 结果保存路径
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save results")
    
    return parser.parse_args()

def run():
    args = get_args()
    
    # --- 0. 基础配置与目录初始化 ---
    start_time = time.time()
    
    # 超参数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_NAME = args.dataset
    DATASET_PATH = os.path.join(args.data_root, DATASET_NAME)
    
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    LAMBDA_ORTH = args.lambda_orth
    NUM_WORKERS = args.num_workers
    RESULTS_ROOT = args.results_dir
    
    # 获取当前日期和时间，生成唯一ID
    now = datetime.datetime.now()
    date_str = now.strftime("%m-%d")
    time_str = now.strftime("%H%M%S")
    run_id = f"{date_str}-{time_str}"
    
    # 初始结果目录 (临时名称，最后会重命名)
    if not os.path.exists(RESULTS_ROOT):
        os.makedirs(RESULTS_ROOT)
        
    temp_folder_name = f"Temp_{DATASET_NAME}_{run_id}"
    OUTPUT_DIR = os.path.join(RESULTS_ROOT, temp_folder_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(OUTPUT_DIR, "train.log")
    setup_logging(log_file)
    logging.info(f"--- 开始训练任务: {run_id} ---")
    logging.info(f"使用设备: {DEVICE}")
    logging.info(f"结果输出目录: {OUTPUT_DIR}")
    logging.info(f"配置参数: {args}")

    # --- 1. 数据准备 ---
    if not os.path.exists(DATASET_PATH):
        logging.error(f"数据集路径不存在: {DATASET_PATH}")
        logging.error("请确保数据集已解压并在正确位置。")
        return

    logging.info(f"加载数据集: {DATASET_PATH}")
    train_loader, val_loader, test_loader = create_dataloaders(DATASET_PATH, BATCH_SIZE, NUM_WORKERS)
    
    # --- 2. 模型与优化器 ---
    model = PGMoE().to(DEVICE)
    criterion = OrthogonalLoss(lambda_orth=LAMBDA_ORTH)

    params_to_train = list(model.spatial_expert.parameters()) + \
                      list(model.frequency_expert.parameters()) + \
                      list(model.router.parameters()) + \
                      list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(params_to_train, lr=LEARNING_RATE)
    
    # --- 新增: 学习率调度器 (Cosine Annealing) ---
    # T_max 设置为 EPOCHS，让学习率在整个训练过程中从 LR 降到 0 (或 eta_min)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # --- 3. 训练循环 ---
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0 # 用于 Early Stopping 的计数器
    # patience = 5 # Early Stopping 的耐心值
    patience = 20
    # patience = 40




    
    # CSV 记录文件
    csv_file = os.path.join(OUTPUT_DIR, "training_metrics.csv")
    csv_headers = ["epoch", "lr", "train_loss", "train_bce", "train_orth", 
                   "val_loss", "val_acc", "val_auc", "val_f1", "val_precision", "val_recall", "inference_fps"]
    
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    logging.info("开始训练...")
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"\n--- Epoch {epoch+1}/{EPOCHS} (LR: {current_lr:.6f}) ---")
        
        # --- 训练阶段 ---
        model.train()
        train_loss_total, train_loss_bce, train_loss_orth = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss, l_bce, l_orth = criterion(outputs, labels)
            loss.backward()
            
            # --- 新增: 梯度裁剪 (Gradient Clipping) ---
            # 阈值通常设为 1.0 或 5.0，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
            
            optimizer.step()
            
            train_loss_total += loss.item()
            train_loss_bce += l_bce.item()
            train_loss_orth += l_orth.item()
        
        # 更新学习率
        scheduler.step()
        
        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_focal = train_loss_bce / len(train_loader) # 变量名虽然叫 train_loss_bce, 但实际存的是 focal loss
        avg_train_orth = train_loss_orth / len(train_loader)
        logging.info(f"Train Loss -> Total: {avg_train_loss:.4f}, Focal: {avg_train_focal:.4f}, Orth: {avg_train_orth:.4f}")

        # --- 验证阶段 ---
        model.eval()
        val_loss_total = 0
        all_preds, all_labels = [], []
        
        val_start_time = time.time()
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss, _, _ = criterion(outputs, labels)
                val_loss_total += loss.item()
                
                preds = torch.sigmoid(outputs['logits']).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        val_end_time = time.time()
        
        # 计算验证指标
        avg_val_loss = val_loss_total / len(val_loader)
        val_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        
        # 计算推理速度 (FPS) - 基于验证集
        num_val_images = len(val_loader.dataset)
        inference_time = val_end_time - val_start_time
        fps = num_val_images / inference_time if inference_time > 0 else 0
        
        logging.info(f"Val Loss: {avg_val_loss:.4f} | FPS: {fps:.2f}")
        logging.info(f"Val Metrics -> AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # 打印真实显存峰值
        if DEVICE == "cuda":
            max_memory = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
            logging.info(f"当前 Epoch 真实显存峰值占用: {max_memory:.2f} GB")

        # 写入 CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{current_lr:.6f}",
                f"{avg_train_loss:.4f}", f"{avg_train_focal:.4f}", f"{avg_train_orth:.4f}",
                f"{avg_val_loss:.4f}", 
                f"{val_metrics['acc']:.4f}", f"{val_metrics['auc']:.4f}", f"{val_metrics['f1']:.4f}", 
                f"{val_metrics['precision']:.4f}", f"{val_metrics['recall']:.4f}",
                f"{fps:.2f}"
            ])

        # --- 保存最佳模型 ---
        # 只要 AUC 更好，就覆盖保存
        if val_metrics['auc'] > best_val_auc:
            # 只有当 AUC 提升超过阈值 (1e-4) 时，才认为是有效提升，重置 patience
            if val_metrics['auc'] - best_val_auc > 1e-4:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            best_val_auc = val_metrics['auc']
            best_val_acc = val_metrics['acc']
            best_epoch = epoch + 1
            
            # 确定当前数据集的简写前缀
            if DATASET_NAME == "dataset-A":
                tem = "A"
            else:
                tem = "B"
                
            # 删除旧的最佳模型（如果存在）
            # 统一删除当前数据集下所有以 tem_epoch 开头的 .pth 文件
            # 同时也兼容清理旧格式的 best_model_ 和 dataset-X__epoch
            for f in os.listdir(OUTPUT_DIR):
                if f.endswith(".pth"):
                    if f.startswith(f"{tem}_epoch") or \
                       f.startswith(f"{DATASET_NAME}__epoch") or \
                       f.startswith("best_model_"):
                        try:
                            os.remove(os.path.join(OUTPUT_DIR, f))
                        except OSError as e:
                            logging.warning(f"删除旧模型失败: {f}, 错误: {e}")
            
            # 保存新的最佳模型
            # 格式: X_epoch_ACC_Month-Date-序号
            model_save_name = f"{tem}_epoch{epoch+1}_ACC{best_val_acc:.4f}_{run_id}.pth"
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, model_save_name))
            logging.info(f"*** 发现新最佳模型 (AUC: {best_val_auc:.4f})，已保存 ***")
        else:
            epochs_no_improve += 1
            logging.info(f"验证集 AUC 未提升 (当前: {val_metrics['auc']:.4f}, 最佳: {best_val_auc:.4f}). 已连续 {epochs_no_improve} 轮未提升。")

        # --- Early Stopping 检查 ---
        if epochs_no_improve >= patience:
            logging.info(f"早停触发 (Early Stopping)! 连续 {patience} 轮验证集 AUC 未显著提升。停止训练。")
            break
            
        # --- 后期冻结 Backbone (Freeze Backbone) ---
        # 在 Epoch 10 之后冻结 CLIP 的视觉编码器，只训练 Head
        if epoch + 1 == 10:
            logging.info(">>> 触发后期冻结策略: 冻结 CLIP Visual Encoder，仅训练 Experts 和 Heads <<<")
            # 冻结 CLIP 内部参数 (虽然初始化时已经冻结了 CLIP，但这里再次确认并冻结可能被解冻的部分，或者作为显式逻辑)
            # 注意：我们的 PGMoE 设计中，spatial_expert 和 frequency_expert 是独立的模块，不是 CLIP 的一部分
            # 这里的 visual_encoder 指的是 model.clip.visual
            # 但我们需要小心，不要冻结了 expert。
            # 根据代码，CLIP 本身就是冻结的 (requires_grad=False)
            # 用户意图可能是指：如果之前解冻了 CLIP (微调)，现在冻结它。
            # 但现有代码 CLIP 一直是冻结的。
            # 用户的意图可能是：只训练 classifier 和 router，冻结 expert？
            # "只训练 head，可减少震荡" -> 通常指冻结特征提取部分。
            # 我们的特征提取部分是 CLIP (已冻结) + Experts (未冻结)。
            # 所以这里应该是冻结 Experts，只训练 Router 和 Classifier。
            
            # 修正理解：用户代码示例 `for p in visual_encoder.parameters(): p.requires_grad = False`
            # 在 PGMoE 中，Experts 充当了 Adapter 的角色。
            # 让我们冻结 Experts，只让 Router 和 Classifier 继续微调。
            
            logging.info("冻结 Spatial Expert 和 Frequency Expert...")
            for param in model.spatial_expert.parameters():
                param.requires_grad = False
            for param in model.frequency_expert.parameters():
                param.requires_grad = False
                
            # 重新构建优化器，只包含剩余的需更新参数
            params_to_train = list(model.router.parameters()) + \
                              list(model.classifier.parameters())
            
            # 注意：重建优化器会丢失之前的动量信息，但这在 Fine-tuning 后期通常是可以接受的，或者我们可以手动将 param_group 的 lr 设为 0
            # 这里选择重建优化器，并适当调小 LR
            NEW_LR = current_lr * 0.1 # 降低学习率
            optimizer = torch.optim.AdamW(params_to_train, lr=NEW_LR)
            # 重置 Scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - epoch - 1, eta_min=1e-6)
            logging.info(f"优化器已重建，仅训练 Head。学习率调整为: {NEW_LR}")


    # --- 4. 训练结束与最终评估 ---
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"\n--- 训练结束。总耗时: {total_time_str} ---")
    
    # 在测试集上评估最佳模型
    logging.info("开始在测试集上评估最佳模型...")
    # 查找符合新命名格式的最佳模型
    if DATASET_NAME == "dataset-A":
        tem = "A"
    else:
        tem = "B"
    best_model_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"{tem}_epoch") and f.endswith(".pth")]
    if best_model_files:
        best_model_path = os.path.join(OUTPUT_DIR, best_model_files[0])
        model.load_state_dict(torch.load(best_model_path))
        
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(DEVICE)
                outputs = model(images)
                preds = torch.sigmoid(outputs['logits']).cpu().numpy()
                test_preds.extend(preds)
                test_labels.extend(labels.numpy())
        
        test_metrics = calculate_metrics(np.array(test_labels), np.array(test_preds))
        logging.info(f"测试集最终结果 -> AUC: {test_metrics['auc']:.4f}, Acc: {test_metrics['acc']:.4f}, F1: {test_metrics['f1']:.4f}")
        
        # 将测试集结果追加到 CSV 文件
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            # 在 epoch 列填入 "Test_Set"，其他训练相关列留空，最后填入测试集指标
            # 列顺序: epoch,lr,train_loss,train_bce,train_orth,val_loss,val_acc,val_auc,val_f1,val_precision,val_recall,inference_fps
            writer.writerow([
                "Test_Set", 
                "-", "-", "-", "-", 
                "-", # val_loss (test loss not calculated here, placeholder)
                f"{test_metrics['acc']:.4f}", 
                f"{test_metrics['auc']:.4f}", 
                f"{test_metrics['f1']:.4f}", 
                f"{test_metrics['precision']:.4f}", 
                f"{test_metrics['recall']:.4f}",
                "-" # inference_fps
            ])
            
    else:
        logging.warning("未找到最佳模型文件，跳过测试集评估。")

        test_metrics = {"auc": 0, "acc": 0}

    # --- 5. 生成报告与重命名文件夹 ---
    
    # 写入 train_info.txt
    info_file = os.path.join(OUTPUT_DIR, "train_info.txt")
    with open(info_file, "w") as f:
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Date: {date_str}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Total Time: {total_time_str}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  Epochs: {EPOCHS}\n")
        f.write(f"  Batch Size: {BATCH_SIZE}\n")
        f.write(f"  Learning Rate: {LEARNING_RATE}\n")
        f.write(f"  Lambda Orth: {LAMBDA_ORTH}\n")
        f.write(f"  LR Scheduler: CosineAnnealingLR (eta_min=1e-6)\n")
        f.write(f"  Gradient Clipping: Max Norm = 1.0\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best Validation Results (Epoch {best_epoch}):\n")
        f.write(f"  AUC: {best_val_auc:.4f}\n")
        f.write(f"  Accuracy: {best_val_acc:.4f}\n")
        if best_model_files:
             f.write(f"Test Set Results:\n")
             f.write(f"  AUC: {test_metrics['auc']:.4f}\n")
             f.write(f"  Accuracy: {test_metrics['acc']:.4f}\n")
             f.write(f"  F1: {test_metrics['f1']:.4f}\n")
             f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
             f.write(f"  Recall: {test_metrics['recall']:.4f}\n")

    # 重命名结果文件夹
    # 格式: 数据集名_AUC{XX}_ACC{XX}_{日期}_{序号}
    # 序号使用 run_id 的后半部分 (时间)
    final_folder_name = f"{DATASET_NAME}_AUC{best_val_auc:.4f}_ACC{best_val_acc:.4f}_{run_id}"
    FINAL_OUTPUT_DIR = os.path.join(RESULTS_ROOT, final_folder_name)
    
    try:
        os.rename(OUTPUT_DIR, FINAL_OUTPUT_DIR)
        logging.info(f"结果文件夹已重命名为: {FINAL_OUTPUT_DIR}")
    except Exception as e:
        logging.error(f"重命名文件夹失败: {e}")

    logging.info("--- 任务全部完成 ---")

if __name__ == '__main__':
    run()
