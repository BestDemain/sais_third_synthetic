import argparse
import pickle
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from model import DiffusionModel, DiffusionProcess
from data_utils import restypes

def predict_test(model_path, config_path, test_data_path, output_path):
    """使用训练好的扩散模型对测试数据进行预测"""
    # 加载配置
    config = OmegaConf.load(config_path)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    model = DiffusionModel(config.model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 创建扩散过程
    diffusion = DiffusionProcess(
        time_steps=config.model.get('time_steps', 1000),
        beta_start=config.model.get('beta_start', 1e-4),
        beta_end=config.model.get('beta_end', 0.02)
    )
    
    # 加载测试数据
    with open(test_data_path, 'rb') as f:
        test_datas = pickle.load(f)
    
    # 创建残基映射
    residue_mapping = {'X': 20}
    residue_mapping.update(dict(zip(restypes, range(len(restypes)))))
    
    # 预测结果
    submit_data = []
    
    for data in test_datas:
        # 准备序列数据
        sequence = list(data["sequence"])
        sequence_tensor = torch.zeros(len(sequence), len(residue_mapping))
        
        for i, c in enumerate(sequence):
            if c not in restypes:
                c = 'X'
            sequence_tensor[i][residue_mapping[c]] = 1
        
        # 添加批次维度
        sequence_tensor = sequence_tensor.unsqueeze(0).to(device)
        
        # 预测 - 使用t=0（无噪声）直接预测
        with torch.no_grad():
            t = torch.zeros(1, device=device).long()
            pred = model(sequence_tensor, t)
            pred_labels = torch.argmax(pred, dim=-1).squeeze(0)
        
        # 转换为字符串
        pred_str = "".join([str(label.item()) for label in pred_labels])
        
        # 添加到提交数据
        submit_data.append([data['id'], data['sequence'], pred_str])
    
    # 创建提交DataFrame
    submit_df = pd.DataFrame(submit_data)
    submit_df.columns = ["proteinID", "sequence", "IDRs"]
    
    # 保存提交文件
    submit_df.to_csv(output_path, index=None)
    print(f"预测完成，结果已保存到: {output_path}")
    
    # 计算并打印评估指标
    if 'label' in test_datas[0]:
        calculate_metrics(submit_data, test_datas)

def calculate_metrics(predictions, test_data):
    """计算Precision、Recall和F1分数"""
    all_true_labels = []
    all_pred_labels = []
    
    for i, data in enumerate(test_data):
        true_labels = [int(c) for c in data['label']]
        pred_labels = [int(c) for c in predictions[i][2]]
        
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)
    
    # 计算TP, FP, FN
    tp = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == 1 and p == 0)
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n评估指标:")
    print(f"Precision = TP/(TP+FP) = {tp}/{tp+fp} = {precision:.4f}")
    print(f"Recall = TP/(TP+FN) = {tp}/{tp+fn} = {recall:.4f}")
    print(f"F1 Score = (2×Precision×Recall)/(Precision+Recall) = {f1:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('蛋白质序列预测 - 扩散模型预测')
    parser.add_argument('--model_path', default='models/diffusion_model.pt', help='模型路径')
    parser.add_argument('--config_path', default='config.yaml', help='配置文件路径')
    parser.add_argument('--test_data', default='../Data/WSAA_data_test.pkl', help='测试数据路径')
    parser.add_argument('--output', default='/saisresult/submit.csv', help='输出文件路径')
    
    args = parser.parse_args()
    
    predict_test(args.model_path, args.config_path, args.test_data, args.output)