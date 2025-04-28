#!/bin/bash

# 检查是否存在预训练模型，如果不存在则训练
if [ ! -f "models/diffusion_model.pt" ]; then
    echo "预训练模型不存在，开始训练..."
    mkdir -p models
    python main.py --mode train --config config.yaml
fi

# 使用模型进行预测
python predict.py --model_path models/diffusion_model.pt --config_path config.yaml --test_data WSAA_data_test.pkl --output /saisresult/submit.csv