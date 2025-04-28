import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import predict_test

def main():
    # 设置路径
    model_path = 'models/diffusion_model.pt'
    config_path = 'config.yaml'
    test_data_path = 'WSAA_data_test.pkl'
    output_path = '/saisresult/submit.csv'
    
    # 执行预测
    predict_test(model_path, config_path, test_data_path, output_path)

if __name__ == '__main__':
    main()