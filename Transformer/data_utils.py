import pickle
import torch
from torch.utils.data import Dataset, DataLoader

# 定义氨基酸残基类型
restypes = [
    'A', 'R', 'N', 'D', 'C',
    'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
unsure_restype = 'X'
unknown_restype = 'U'

class ProteinDataset(Dataset):
    """蛋白质序列数据集类"""
    def __init__(self, dict_data, max_seq_len=None):
        sequences = [d['sequence'] for d in dict_data]
        labels = [d['label'] for d in dict_data]
        assert len(sequences) == len(labels)

        self.sequences = sequences
        self.labels = labels
        self.residue_mapping = {'X': 20}
        self.residue_mapping.update(dict(zip(restypes, range(len(restypes)))))
        
        # 计算最大序列长度（如果未指定）
        if max_seq_len is None:
            self.max_seq_len = max([len(seq) for seq in sequences])
        else:
            self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 创建填充后的序列张量
        sequence = torch.zeros(self.max_seq_len, len(self.residue_mapping))
        seq_len = min(len(self.sequences[idx]), self.max_seq_len)
        
        # 填充序列
        for i, c in enumerate(self.sequences[idx][:seq_len]):
            if c not in restypes:
                c = 'X'
            sequence[i][self.residue_mapping[c]] = 1

        # 创建填充后的标签张量
        label_seq = self.labels[idx][:seq_len]
        label = torch.zeros(self.max_seq_len, dtype=torch.long)
        for i, l in enumerate(label_seq):
            label[i] = int(l)
            
        return sequence, label

def load_data(data_path, train_rate=0.7, valid_rate=0.2, use_all_for_train=False):
    """加载并分割数据集
    
    Args:
        data_path: 数据路径
        train_rate: 训练集比例
        valid_rate: 验证集比例
        use_all_for_train: 是否将所有数据用于训练
    
    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    if use_all_for_train:
        # 使用所有数据作为训练集
        train_data_dicts = data
        # 创建一个小的验证集和测试集（仅用于评估，不影响训练）
        # 从训练数据中随机选择少量样本作为验证集和测试集
        import random
        random.seed(42)  # 设置随机种子以确保可重复性
        
        # 随机选择少量样本（例如5%）作为验证集
        valid_indices = random.sample(range(len(data)), max(1, int(len(data) * 0.05)))
        valid_data_dicts = [data[i] for i in valid_indices]
        
        # 随机选择少量样本（例如5%）作为测试集
        test_indices = random.sample(range(len(data)), max(1, int(len(data) * 0.05)))
        test_data_dicts = [data[i] for i in test_indices]
    else:
        # 原始的数据集划分逻辑
        total_number = len(data)
        train_sep = int(total_number * train_rate)
        valid_sep = int(total_number * (train_rate + valid_rate))
        
        train_data_dicts = data[:train_sep]
        valid_data_dicts = data[train_sep:valid_sep]
        test_data_dicts = data[valid_sep:]
    
    # 计算所有序列的最大长度
    all_sequences = [d['sequence'] for d in data]
    max_seq_len = max([len(seq) for seq in all_sequences])
    print(f"最大序列长度: {max_seq_len}")
    
    # 使用相同的最大长度创建数据集
    train_dataset = ProteinDataset(train_data_dicts, max_seq_len=max_seq_len)
    valid_dataset = ProteinDataset(valid_data_dicts, max_seq_len=max_seq_len)
    test_dataset = ProteinDataset(test_data_dicts, max_seq_len=max_seq_len)

    return train_dataset, valid_dataset, test_dataset

def create_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size=1, num_workers=4):
    """创建数据加载器"""
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,  # 验证时使用批量大小为1
        shuffle=False,
        num_workers=num_workers
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,  # 测试时使用批量大小为1
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_dataloader, valid_dataloader, test_dataloader