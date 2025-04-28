import argparse
import os
from omegaconf import OmegaConf

from train import train
from predict import predict_test

def main():
    parser = argparse.ArgumentParser('蛋白质序列预测 - 扩散模型')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train', help='运行模式：训练或预测')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--model_path', default='models/diffusion_model.pt', help='模型路径')
    parser.add_argument('--test_data', default='../Data/WSAA_data_test.pkl', help='测试数据路径')
    parser.add_argument('--output', default='/saisresult/submit.csv', help='输出文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 根据模式执行相应操作
    if args.mode == 'train':
        # 训练模型
        train(config, device)
    else:
        # 预测
        predict_test(args.model_path, args.config, args.test_data, args.output)

if __name__ == '__main__':
    import torch
    main()