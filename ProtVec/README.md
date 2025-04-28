# ProtVec 蛋白质序列标签预测

## 项目简介

本项目实现了使用 ProtVec（蛋白质特定嵌入）进行蛋白质序列标签预测的功能。ProtVec 是一种专门为蛋白质序列设计的嵌入方法，它通过将蛋白质序列分解为重叠的 n-gram（通常是 3-gram），然后使用 Word2Vec 类似的方法学习这些 n-gram 的向量表示。

与基线模型中的 Word2Vec 相比，ProtVec 更适合蛋白质序列分析，因为它考虑了蛋白质序列的特定结构和模式。通过使用 n-gram 捕获局部氨基酸模式，ProtVec 能够更好地表示蛋白质序列中的功能和结构信息。

## 文件说明

- `protvec_baseline.ipynb`: Jupyter Notebook 实现，包含详细的步骤说明和可视化
- `protvec_baseline.py`: Python 脚本实现，可直接运行
- `protvec_model.w2v`: 训练好的 ProtVec 模型（运行后生成）
- `classifier_model.joblib`: 训练好的分类器模型（运行后生成）

## 实现原理

1. **数据加载**：加载蛋白质序列数据
2. **n-gram 生成**：将每个蛋白质序列分解为重叠的 3-gram
3. **ProtVec 训练**：使用 Word2Vec 模型训练 n-gram 的向量表示
4. **特征提取**：为每个氨基酸位置提取特征，考虑其所在的 n-gram 上下文
5. **模型训练**：使用提取的特征训练分类模型（高斯朴素贝叶斯和逻辑回归）
6. **模型评估**：使用交叉验证评估模型性能
7. **模型保存**：保存训练好的 ProtVec 模型和分类器

## 使用方法

### 运行 Python 脚本

```bash
python protvec_baseline.py
```

### 使用 Jupyter Notebook

```bash
jupyter notebook protvec_baseline.ipynb
```

## 与基线模型的比较

基线模型（`ml_baseline.ipynb`）使用 Word2Vec 直接对氨基酸字符进行嵌入，而本实现使用 ProtVec 对蛋白质序列的 n-gram 进行嵌入，能够更好地捕获蛋白质序列的局部结构和功能信息。

## 依赖库

- gensim
- numpy
- pandas
- scikit-learn
- joblib

## 参考文献

- Asgari, E., & Mofrad, M. R. (2015). Continuous distributed representation of biological sequences for deep proteomics and genomics. PloS one, 10(11), e0141287.