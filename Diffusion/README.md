# 蛋白质序列标签预测 - 扩散模型

本项目实现了一个基于扩散模型的蛋白质序列标签预测系统，用于预测蛋白质序列中的无序区域（IDRs）。

## 模型架构

该模型基于扩散模型（Diffusion Model）的思想，结合了Transformer架构，通过逐步去噪过程学习从氨基酸序列到标签的映射关系。主要组件包括：

1. **扩散过程**：实现了前向扩散（添加噪声）和反向扩散（去噪）过程
2. **Transformer编码器**：捕获序列中的长距离依赖关系
3. **时间步嵌入**：将扩散过程的时间步信息融入模型

## 文件结构

- `model.py`: 定义扩散模型和扩散过程
- `data_utils.py`: 数据处理工具
- `train.py`: 模型训练脚本
- `predict.py`: 模型预测脚本
- `main.py`: 主程序入口
- `config.yaml`: 模型配置文件
- `Submit/`: 提交相关文件
  - `Dockerfile`: Docker配置文件
  - `requirements.txt`: 依赖包列表
  - `run.sh`: 运行脚本
  - `submit.py`: 提交脚本

## 使用方法

### 训练模型

```bash
python main.py --mode train --config config.yaml
```

### 预测

```bash
python main.py --mode predict --model_path models/diffusion_model.pt --config config.yaml --test_data ../Data/WSAA_data_test.pkl --output /saisresult/submit.csv
```

或直接使用预测脚本：

```bash
python predict.py --model_path models/diffusion_model.pt --config_path config.yaml --test_data ../Data/WSAA_data_test.pkl --output /saisresult/submit.csv
```

## 配置说明

在`config.yaml`中可以调整以下参数：

- 模型参数：维度、层数、头数、Dropout率等
- 扩散过程参数：时间步数、噪声系数等
- 训练参数：批量大小、学习率、训练轮数等
- 数据参数：数据路径、是否使用全部数据训练等

## 依赖包

- torch
- numpy
- pandas
- scikit-learn
- tqdm
- omegaconf
- pickle5