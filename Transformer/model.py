import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class PositionalEncoding(nn.Module):
    """位置编码层，为序列中的每个位置添加位置信息"""
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # 不再预先计算固定长度的位置编码
        # 而是在forward中根据输入序列长度动态计算

    def _get_positional_encoding(self, seq_len):
        """根据序列长度动态生成位置编码"""
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)
        )
        
        pe = torch.zeros(1, seq_len, self.d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:self.d_model//2])
        
        return pe

    def forward(self, x):
        """添加位置编码到输入张量"""
        # 获取当前序列长度
        seq_len = x.size(1)
        
        # 动态生成位置编码
        pe = self._get_positional_encoding(seq_len).to(x.device)
        
        if len(x.shape) == 3:
            x = x + pe
        elif len(x.shape) == 4:
            x = x + pe[:, :, None, :]
            
        return self.dropout(x)

class TransformerModel(nn.Module):
    """基于Transformer的蛋白质序列预测模型"""
    def __init__(self, config):
        super().__init__()
        
        # 模型配置参数
        self.d_model = config.get('d_model', 128)
        self.n_head = config.get('n_head', 8)
        self.n_layer = config.get('n_layer', 4)
        self.dropout = config.get('dropout', 0.1)
        self.i_dim = config.get('i_dim', 21)  # 输入维度（氨基酸种类数）
        self.o_dim = config.get('o_dim', 2)   # 输出维度（二分类）
        
        # 输入层：将one-hot编码转换为模型维度
        self.input_layer = nn.Linear(self.i_dim, self.d_model)
        
        # 位置编码
        self.position_embed = PositionalEncoding(self.d_model, dropout=self.dropout)
        
        # 输入归一化
        self.input_norm = nn.LayerNorm(self.d_model)
        
        # Transformer编码器层
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=4*self.d_model,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        
        # Transformer编码器
        self.transformer = TransformerEncoder(encoder_layer, num_layers=self.n_layer)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.d_model, self.o_dim)
        )

    def forward(self, x):
        """前向传播"""
        # 输入编码
        x = self.input_layer(x)
        
        # 添加位置编码
        x = self.position_embed(x)
        
        # 输入归一化
        x = self.input_norm(x)
        
        # Transformer编码器
        x = self.transformer(x)
        
        # 输出层
        x = self.output_layer(x)
        
        return x