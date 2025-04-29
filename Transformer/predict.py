import pickle
import torch
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

from model import TransformerModel
from data_utils import restypes, unsure_restype


def predict_test(path):
    # 加载测试数据
    print(f"加载测试数据: {path}")
    test_datas = pickle.load(open(path, "rb"))
    print(f"成功加载测试数据，共 {len(test_datas)} 条记录")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model_path = "d:/Vscode/Project/ML/sais_third_synthetic/Transformer/model.pt"
    print(f"加载模型: {model_path}")
    try:
        # 尝试直接加载模型状态字典
        model_state = torch.load(model_path, map_location=device)
        print("成功加载模型状态字典")
        
        # 打印模型状态字典的键，帮助调试
        print("模型状态字典的键:")
        for key in model_state.keys():
            print(f"  - {key}")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("尝试其他加载方式...")
        # 如果加载失败，尝试作为OrderedDict加载
        model_state = torch.load(model_path, map_location=device)
    
    # 创建模型配置
    config = {
        'd_model': 64,  # 根据模型参数设置
        'n_head': 8,
        'n_layer': 4,
        'dropout': 0.1,
        'i_dim': 21,  # 输入维度（氨基酸种类数）
        'o_dim': 2    # 输出维度（二分类）
    }
    
    # 注意：如果加载模型失败，可能需要调整上述配置参数以匹配模型结构
    
    # 初始化模型
    model = TransformerModel(config)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    # 创建氨基酸映射字典
    residue_mapping = {'X': 20}
    residue_mapping.update(dict(zip(restypes, range(len(restypes)))))
    
    submit_data = []
    
    with torch.no_grad():
        for data in test_datas:
            sequence = data['sequence']
            seq_len = len(sequence)
            
            # 创建输入张量
            input_tensor = torch.zeros(seq_len, len(residue_mapping))
            
            # 填充序列
            for i, c in enumerate(sequence):
                if c not in restypes:
                    c = 'X'
                input_tensor[i][residue_mapping[c]] = 1
            
            # 添加批次维度并移至设备
            input_tensor = input_tensor.unsqueeze(0).to(device)
            
            # 预测
            outputs = model(input_tensor)
            
            # 打印原始输出，查看模型输出的分布情况
            raw_outputs = outputs.squeeze(0)
            print(f"\n序列ID: {data['id']}")
            print(f"原始输出形状: {raw_outputs.shape}")
            print(f"前5个位置的输出: {raw_outputs[:5]}")
            
            # 计算每个类别的平均概率
            softmax_outputs = torch.softmax(raw_outputs, dim=1)
            avg_probs = softmax_outputs.mean(dim=0)
            print(f"平均概率分布: {avg_probs}")
            
            # 检查是否所有预测都是同一个类别
            _, predicted_raw = torch.max(raw_outputs, 1)
            unique_preds, counts = torch.unique(predicted_raw, return_counts=True)
            print(f"预测类别统计: {list(zip(unique_preds.tolist(), counts.tolist()))}")
            
            # 尝试多种预测方法
            print("\n尝试不同的预测方法:")
            
            # 方法1: 使用argmax直接预测
            _, predicted_argmax = torch.max(raw_outputs, 1)
            
            # 方法2: 使用softmax后的argmax
            _, predicted_softmax = torch.max(softmax_outputs, 1)
            
            # 方法3: 使用阈值
            threshold = 0.1  # 非常低的阈值
            predicted_threshold = (softmax_outputs[:, 1] > threshold).long()
            
            # 方法4: 使用相对阈值 - 如果类别1的概率大于类别0的概率的一定比例
            relative_threshold = 0.8  # 如果类别1的概率至少是类别0的80%，就预测为1
            predicted_relative = ((softmax_outputs[:, 1] / (softmax_outputs[:, 0] + 1e-6)) > relative_threshold).long()
            
            # 方法5: 随机分配一些1，用于测试
            predicted_random = torch.zeros_like(predicted_argmax)
            random_indices = torch.randperm(len(predicted_random))[:int(len(predicted_random)*0.2)]  # 随机20%位置为1
            predicted_random[random_indices] = 1
            
            # 打印各种方法的预测结果统计
            print(f"方法1 (argmax): 1的数量 = {predicted_argmax.sum().item()}, 比例 = {predicted_argmax.sum().item()/len(predicted_argmax):.4f}")
            print(f"方法2 (softmax+argmax): 1的数量 = {predicted_softmax.sum().item()}, 比例 = {predicted_softmax.sum().item()/len(predicted_softmax):.4f}")
            print(f"方法3 (阈值={threshold}): 1的数量 = {predicted_threshold.sum().item()}, 比例 = {predicted_threshold.sum().item()/len(predicted_threshold):.4f}")
            print(f"方法4 (相对阈值={relative_threshold}): 1的数量 = {predicted_relative.sum().item()}, 比例 = {predicted_relative.sum().item()/len(predicted_relative):.4f}")
            print(f"方法5 (随机20%): 1的数量 = {predicted_random.sum().item()}, 比例 = {predicted_random.sum().item()/len(predicted_random):.4f}")
            
            # 使用方法4作为最终预测
            predicted = predicted_relative
            
            # 转换为字符串
            pred_str = "".join([str(p.item()) for p in predicted[:seq_len]])
            
            # 添加到结果
            submit_data.append([data['id'], data['sequence'], pred_str])
    
    # 创建提交数据框
    submit_df = pd.DataFrame(submit_data)
    submit_df.columns = ["proteinID", "sequence", "IDRs"]
    
    # 保存结果
    output_path = "submit.csv"
    submit_df.to_csv(output_path, index=None)
    print(f"预测完成，结果已保存至 {output_path}")
    
    # 分析预测结果
    label_counts = submit_df['IDRs'].apply(lambda x: [x.count('0'), x.count('1')])
    total_0 = sum([counts[0] for counts in label_counts])
    total_1 = sum([counts[1] for counts in label_counts])
    print(f"\n预测结果统计:")
    print(f"预测为0的位置总数: {total_0}")
    print(f"预测为1的位置总数: {total_1}")
    print(f"1的比例: {total_1/(total_0+total_1):.4f}")
    
    # 如果所有预测都是0，提示可能的问题
    if total_1 == 0:
        print("\n警告: 所有位置都被预测为0，可能存在以下问题:")
        print("1. 模型训练不足，只学会了预测多数类别")
        print("2. 预测阈值设置不当")
        print("3. 模型输出层可能需要调整")
        print("4. 数据分布不平衡，导致模型偏向预测多数类别")


if __name__ == "__main__":
    predict_test("../Data/WSAA_data_public.pkl")