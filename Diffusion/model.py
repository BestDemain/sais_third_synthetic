import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 定义氨基酸残基类型
restypes = [
    'A', 'R', 'N', 'D', 'C',
    'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), 0].unsqueeze(0)
        return self.dropout(x)

class DiffusionModel(nn.Module):
    """基于扩散模型的蛋白质序列标签预测模型"""
    def __init__(self, config):
        super().__init__()
        
        # 模型配置参数
        self.d_model = config.get('d_model', 128)
        self.n_head = config.get('n_head', 8)
        self.n_layer = config.get('n_layer', 4)
        self.dropout = config.get('dropout', 0.1)
        self.i_dim = config.get('i_dim', 21)  # 输入维度（氨基酸种类数）
        self.o_dim = config.get('o_dim', 2)   # 输出维度（二分类）
        self.time_steps = config.get('time_steps', 1000)  # 扩散步数
        
        # 输入层：将one-hot编码转换为模型维度
        self.input_layer = nn.Linear(self.i_dim, self.d_model)
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
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

    def forward(self, x, t):
        """前向传播
        Args:
            x: 输入序列 [batch_size, seq_len, i_dim]
            t: 时间步 [batch_size]
        """
        # 输入编码
        x = self.input_layer(x)
        
        # 时间步嵌入
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        
        # 添加时间步嵌入到每个位置
        x = x + t_emb.unsqueeze(1)
        
        # 添加位置编码
        x = self.position_embed(x)
        
        # 输入归一化
        x = self.input_norm(x)
        
        # Transformer编码器
        x = self.transformer(x)
        
        # 输出层
        x = self.output_layer(x)
        
        return x

class DiffusionProcess:
    """扩散过程管理类"""
    def __init__(self, time_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # 线性噪声调度
        self.betas = torch.linspace(beta_start, beta_end, time_steps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 计算扩散过程中的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：给定x_0和时间步t，采样x_t"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].to(x_start.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(x_start.device)
        
        # 重塑为适合广播的形状
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, denoise_model, x_start, t, noise=None):
        """计算扩散模型的损失"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # 获取带噪声的样本
        x_noisy = self.q_sample(x_start, t, noise=noise)
        
        # 预测噪声
        predicted = denoise_model(x_noisy, t)
        
        # 计算损失（简单MSE损失）
        loss = F.mse_loss(predicted, x_start)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """从x_t采样x_{t-1}"""
        betas_t = self.betas[t_index].to(x.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index].to(x.device)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_index].to(x.device)
        
        # 重塑为适合广播的形状
        betas_t = betas_t.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1)
        sqrt_recip_alphas_t = sqrt_recip_alphas_t.view(-1, 1, 1)
        
        # 预测x_0
        model_output = model(x, t)
        
        # 计算均值
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)
        
        # 只有t>0时添加噪声
        if t_index > 0:
            noise = torch.randn_like(x)
            posterior_variance_t = self.posterior_variance[t_index].to(x.device).view(-1, 1, 1)
            model_mean = model_mean + torch.sqrt(posterior_variance_t) * noise
            
        return model_mean
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, device):
        """从纯噪声采样完整的去噪过程"""
        batch_size = shape[0]
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        # 逐步去噪
        for i in reversed(range(0, self.time_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, i)
            
        return x
    
    @torch.no_grad()
    def sample(self, model, batch_size, seq_len, i_dim, device):
        """生成样本"""
        return self.p_sample_loop(model, (batch_size, seq_len, i_dim), device)