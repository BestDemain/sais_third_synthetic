import pickle
import torch
import numpy as np
import pandas as pd
import sys
import os

from model import TransformerModel
from data_utils import restypes, unsure_restype


def predict_test(path):
    # 加载测试数据
    test_datas = pickle.load(open(path, "rb"))
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model_path = "model.pt"
    try:
        # 尝试直接加载模型状态字典
        model_state = torch.load(model_path, map_location=device)
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
            
            # 获取预测类别
            _, predicted = torch.max(outputs.squeeze(0), 1)
            
            # 转换为字符串
            pred_str = "".join([str(p.item()) for p in predicted[:seq_len]])
            
            # 添加到结果
            submit_data.append([data['id'], data['sequence'], pred_str])
    
    # 创建提交数据框
    submit_df = pd.DataFrame(submit_data)
    submit_df.columns = ["proteinID", "sequence", "IDRs"]
    
    # 保存结果
    submit_df.to_csv("/saisresult/submit.csv", index=None)
    print(f"预测完成，结果已保存至 /saisresult/submit.csv")


if __name__ == "__main__":
    predict_test("../Data/WSAA_data_public.pkl")