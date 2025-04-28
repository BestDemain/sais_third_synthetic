import argparse
import os
import torch
from omegaconf import OmegaConf

from train import train
from predict import predict_test

def main():
    """主函数，用于运行Transformer模型的训练和预测"""
    parser = argparse.ArgumentParser('蛋白质序列预测 - Transformer模型')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train',
                        help='运行模式：train-训练模型，predict-使用模型预测')
    parser.add_argument('--config_path', default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--model_path', default='models/transformer_model.pt',
                        help='模型保存/加载路径')
    parser.add_argument('--test_data', default='../Data/WSAA_data_test.pkl',
                        help='测试数据路径')
    parser.add_argument('--output', default='/saisresult/submit.csv',
                        help='预测结果输出路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config_path)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 根据模式执行相应操作
    if args.mode == 'train':
        print("开始训练模型...")
        train(config, device)
        print("训练完成！")
    elif args.mode == 'predict':
        print("开始预测...")
        predict_test(args.model_path, args.config_path, args.test_data, args.output)
        print("预测完成！")

def print_usage():
    """打印使用说明"""
    print("\n使用说明：")
    print("1. 训练模型：")
    print("   python main.py --mode train --config_path config.yaml")
    print("\n2. 预测：")
    print("   python main.py --mode predict --model_path models/transformer_model.pt --test_data ../Data/WSAA_data_test.pkl --output /saisresult/submit.csv")
    print("\n注意：训练模式会自动保存最佳模型到models目录下。")

if __name__ == '__main__':
    # 创建模型保存目录
    os.makedirs('models', exist_ok=True)
    
    # 打印使用说明
    print_usage()
    
    # 运行主函数
    main()