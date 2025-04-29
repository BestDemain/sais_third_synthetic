# -*- coding: utf-8 -*-
"""
蛋白质序列标签预测 - ProtVec 预测脚本

本脚本使用训练好的ProtVec模型和分类器对测试数据进行预测，并生成提交结果。
"""

import pickle
import gensim
import gensim.models
import os
import sys
import numpy as np
import pandas as pd
from joblib import load


def generate_ngrams(sequence, n=3):
    """生成蛋白质序列的n-gram"""
    return [sequence[i:i+n] for i in range(len(sequence)-n+1)]


def extract_features_for_position(sequence, position, model, n=3):
    """为序列中的特定位置提取特征"""
    # 获取包含该位置的所有n-gram
    relevant_ngrams = []
    for i in range(max(0, position-n+1), min(len(sequence)-n+1, position+1)):
        ngram = sequence[i:i+n]
        if ngram in model.wv:
            relevant_ngrams.append(ngram)
    
    # 如果没有相关n-gram，使用随机向量
    if not relevant_ngrams:
        return np.random.randn(model.vector_size)
    
    # 计算所有相关n-gram向量的平均值
    return np.mean([model.wv[ngram] for ngram in relevant_ngrams], axis=0)


def predict_test(path):
    """使用训练好的ProtVec模型和分类器对测试数据进行预测"""
    # 加载模型
    print("加载模型...")
    model_protvec = gensim.models.Word2Vec.load("protvec_model.w2v")
    classifier = load("classifier_model.joblib")
    
    # 加载测试数据
    print("加载测试数据...")
    test_datas = pickle.load(open(path, "rb"))
    
    # 预测结果
    print("开始预测...")
    submit_data = []
    
    for i, data in enumerate(test_datas):
        if i % 10 == 0:
            print(f"处理第 {i+1}/{len(test_datas)} 个序列...")
        
        sequence = data["sequence"]
        
        # 为序列中的每个位置提取特征并预测
        data_x = []
        for idx in range(len(sequence)):
            features = extract_features_for_position(sequence, idx, model_protvec)
            data_x.append(features)
        
        # 转换为numpy数组并预测
        data_x = np.array(data_x)
        pred = classifier.predict(data_x)
        
        # 转换为字符串
        pred_str = "".join([str(int(p)) for p in pred])
        
        # 添加到提交数据
        submit_data.append([data['id'], data['sequence'], pred_str])
    
    # 创建提交DataFrame
    submit_df = pd.DataFrame(submit_data)
    submit_df.columns = ["proteinID", "sequence", "IDRs"]
    
    # 保存提交文件
    submit_df.to_csv("/saisresult/submit.csv", index=None)
    print("预测完成，结果已保存至 /saisresult/submit.csv")


if __name__ == "__main__":
    predict_test("saisdata/WSAA_data_test.pkl")