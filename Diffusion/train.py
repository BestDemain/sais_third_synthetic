import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.utils.tensorboard import SummaryWriter

from data_utils import load_data, create_dataloaders, convert_logits_to_labels
from model import DiffusionModel, DiffusionProcess

def train(config, device):
    """训练扩散模型并评估"""
    # 获取保存模型的配置
    save_interval = config.train.get('save_interval', 1)  # 默认为0表示不定期保存
    save_last = config.train.get('save_last', True)  # 默认保存最后一个模型
    
    # 创建TensorBoard日志目录和写入器
    log_dir = os.path.join('runs', 'diffusion_model')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 加载数据
    train_dataset, valid_dataset, test_dataset = load_data(
        config.data.data_path,
        train_rate=0.7,
        valid_rate=0.2,
        use_all_for_train=config.data.get('use_all_for_train', False)
    )
    
    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
        train_dataset, valid_dataset, test_dataset,
        batch_size=config.train.dataloader.batch_size,
        num_workers=config.train.dataloader.num_workers
    )
    
    # 创建模型
    model = DiffusionModel(config.model)
    model = model.to(device)
    
    # 创建扩散过程
    diffusion = DiffusionProcess(
        time_steps=config.model.get('time_steps', 1000),
        beta_start=config.model.get('beta_start', 1e-4),
        beta_end=config.model.get('beta_end', 0.02)
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.optimizer.lr,
        weight_decay=config.train.optimizer.weight_decay
    )
    
    # 初始评估
    print("初始模型评估:")
    initial_metrics = evaluate(model, valid_dataloader, device, diffusion, writer=writer, step=0, prefix='Initial')
    
    # 训练循环
    best_f1 = 0.0
    for epoch in range(config.train.epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(
            train_dataloader,
            desc=f"训练周期:{epoch+1}/{config.train.epochs}"
        )
        
        for sequence, label in progress_bar:
            sequence = sequence.to(device)
            label = label.to(device)
            
            # 随机选择时间步
            batch_size = sequence.shape[0]
            t = torch.randint(0, diffusion.time_steps, (batch_size,), device=device).long()
            
            # 计算损失
            loss = diffusion.p_losses(model, label, t)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新进度条
            progress_bar.set_postfix(loss=loss.item())
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"训练损失: {avg_loss:.4f}")
        
        # 记录训练损失
        writer.add_scalar('Training/Loss', avg_loss, epoch)
        writer.add_scalar('Training/LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # 验证阶段
        print("验证集评估:")
        metrics = evaluate(model, valid_dataloader, device, diffusion, writer=writer, step=epoch+1, prefix='Validation')
        
        # 保存最佳模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            save_model(model, config)
            print(f"保存最佳模型，F1分数: {best_f1:.4f}")
        
        # 按照指定间隔保存当前模型
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            save_model(model, config, suffix=f'epoch_{epoch+1}')
            print(f"已保存第 {epoch+1} 轮模型")
    
    # 保存最后一个模型
    if save_last:
        save_model(model, config, suffix='last')
        print(f"已保存最后一轮模型 (epoch {config.train.epochs})")
    
    # 加载最佳模型进行测试
    model = load_best_model(config, device)
    print("\n测试集最终评估:")
    test_metrics = evaluate(model, test_dataloader, device, diffusion, writer=writer, step=0, prefix='Test')
    
    # 关闭TensorBoard写入器
    writer.close()

def evaluate(model, dataloader, device, diffusion, writer=None, step=None, prefix='Eval'):
    """评估模型性能，计算Precision、Recall和F1分数
    
    Args:
        model: 要评估的模型
        dataloader: 数据加载器
        device: 计算设备
        diffusion: 扩散过程对象
        writer: TensorBoard SummaryWriter对象，如果不为None则记录指标
        step: 当前步数，用于TensorBoard记录
        prefix: 指标前缀，用于区分不同阶段的评估
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for sequence, label in dataloader:
            sequence = sequence.to(device)
            label = label.to(device)
            batch_size = sequence.shape[0]
            
            # 使用扩散模型进行预测
            # 对于评估，我们使用t=0（无噪声）直接预测
            t = torch.zeros(batch_size, device=device).long()
            pred = model(sequence, t)
            
            # 转换为标签
            pred_labels = convert_logits_to_labels(pred)
            true_labels = convert_logits_to_labels(label)
            
            # 计算预测概率（对于扩散模型，我们可以使用sigmoid来获取概率）
            probs = torch.sigmoid(pred)
            
            # 收集预测和真实标签
            all_preds.extend(pred_labels.view(-1).cpu().numpy())
            all_labels.extend(true_labels.view(-1).cpu().numpy())
            all_probs.extend(probs.view(-1).cpu().numpy())
    
    # 计算指标
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds))
    
    # 如果提供了TensorBoard写入器，记录更多详细信息
    if writer is not None and step is not None:
        # 添加评估指标到TensorBoard
        writer.add_scalar(f'{prefix}/Precision', precision, step)
        writer.add_scalar(f'{prefix}/Recall', recall, step)
        writer.add_scalar(f'{prefix}/F1_Score', f1, step)
        
        # 创建混淆矩阵图
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import io
            from PIL import Image
            
            cm = confusion_matrix(all_labels, all_preds)
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=[0, 1], yticklabels=[0, 1],
                   title=f'{prefix} 混淆矩阵',
                   ylabel='真实标签',
                   xlabel='预测标签')
            
            # 在每个单元格中添加文本
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            
            # 将图形转换为图像
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf)
            image_tensor = torch.tensor(np.array(image).transpose((2, 0, 1)))
            
            # 添加到TensorBoard
            writer.add_image(f'{prefix}/ConfusionMatrix', image_tensor, step)
            plt.close(fig)
            
            # 记录预测概率分布
            writer.add_histogram(f'{prefix}/Probabilities', np.array(all_probs), step)
        except Exception as e:
            print(f"创建混淆矩阵时出错: {e}")
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def save_model(model, config, suffix=None):
    """保存模型
    
    Args:
        model: 要保存的模型
        config: 配置信息
        suffix: 文件名后缀，如果为None则保存为默认名称
    """
    os.makedirs('models', exist_ok=True)
    if suffix:
        save_path = f'models/diffusion_model_{suffix}.pt'
    else:
        save_path = 'models/diffusion_model.pt'
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")

def load_best_model(config, device):
    """加载最佳模型"""
    model = DiffusionModel(config.model)
    model.load_state_dict(torch.load('models/diffusion_model.pt', map_location=device))
    model = model.to(device)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser('蛋白质序列预测 - 扩散模型训练')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 训练模型
    train(config, device)