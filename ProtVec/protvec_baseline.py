# -*- coding: utf-8 -*-
"""
蛋白质序列标签预测 - ProtVec 实现

本脚本实现了使用 ProtVec（蛋白质特定嵌入）进行蛋白质序列标签预测的功能。
ProtVec 是一种专门为蛋白质序列设计的嵌入方法，它通过将蛋白质序列分解为重叠的 n-gram（通常是 3-gram），
然后使用 Word2Vec 类似的方法学习这些 n-gram 的向量表示。
"""

import pickle
import gensim
import gensim.models
import os
import sys
import random
import numpy as np
import pandas as pd
from joblib import load, dump

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report


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


def main():
    # 加载数据
    print("加载数据...")
    datas = pickle.load(open("../Data/WSAA_data_public.pkl", "rb"))
    
    # 为每个蛋白质序列生成n-gram
    print("生成n-gram...")
    ngram_sentences = []
    for data in datas:
        sequence = data["sequence"]
        ngrams = generate_ngrams(sequence, n=3)  # 使用3-gram
        ngram_sentences.append(ngrams)
    
    # 设置随机种子以确保可重复性
    random_seed = random.randint(0, 10000)
    print(f"随机种子: {random_seed}")
    
    # 训练ProtVec模型
    print("训练ProtVec模型...")
    vector_size = random.choice([10, 20, 40, 50, 100])
    model_protvec = gensim.models.Word2Vec(
        sentences=ngram_sentences,
        vector_size=vector_size,  # 随机选择向量维度
        window=5,  # 上下文窗口大小
        min_count=1,  # 最小词频
        workers=4,  # 并行训练的线程数
        seed=random_seed
    )
    
    print(f"ProtVec向量维度: {model_protvec.vector_size}")
    
    # 提取特征
    print("提取特征...")
    data_x = []
    data_y = []
    
    for i, data in enumerate(datas):
        if i % 10 == 0:
            print(f"处理第 {i+1}/{len(datas)} 个序列...")
        
        sequence = data["sequence"]
        labels = data["label"]
        
        for idx, label in enumerate(labels):
            # 提取该位置的特征
            features = extract_features_for_position(sequence, idx, model_protvec)
            data_x.append(features)
            data_y.append(label)
    
    # 转换为numpy数组
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    
    print(f"特征形状: {data_x.shape}")
    print(f"标签形状: {data_y.shape}")
    
    # 使用高斯朴素贝叶斯分类器
    print("\n使用高斯朴素贝叶斯分类器训练和评估...")
    model = GaussianNB()
    pred = cross_val_predict(model, data_x, data_y)
    print(classification_report(data_y, pred))
    
    # 使用逻辑回归分类器
    print("\n使用逻辑回归分类器训练和评估...")
    model_lr = LogisticRegression(max_iter=1000, random_state=random_seed)
    pred_lr = cross_val_predict(model_lr, data_x, data_y)
    print("逻辑回归分类器性能：")
    print(classification_report(data_y, pred_lr))
    
    # 训练完整数据集上的模型
    print("\n在完整数据集上训练最终模型...")
    final_model = GaussianNB()
    final_model.fit(data_x, data_y)
    
    # 保存ProtVec模型
    model_protvec.save("protvec_model.w2v")
    
    # 保存分类器
    dump(final_model, "classifier_model.joblib")
    
    print("模型已保存！")
    print("\n完成！")


if __name__ == "__main__":
    main()