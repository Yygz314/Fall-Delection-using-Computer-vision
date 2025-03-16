import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import os

# 改进的LSTM模型
class FallDetectionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.4):
        super(FallDetectionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # 使用双向LSTM
        )
        
        # 更复杂的分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),  # *2是因为双向LSTM
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        output = self.classifier(last_output)
        return output

def create_sequences(features, labels, sequence_length=20):  # 增加序列长度
    """创建序列数据，添加数据增强"""
    X, y = [], []
    
    # 正常创建序列
    for i in range(len(features) - sequence_length + 1):
        X.append(features[i:i + sequence_length])
        y.append(labels[i + sequence_length - 1])
    
    # 数据增强：添加噪声
    X = np.array(X)
    y = np.array(y)
    noise_factor = 0.05
    noisy_X = X + np.random.normal(0, noise_factor, X.shape)
    
    # 合并原始数据和增强数据
    X = np.concatenate([X, noisy_X])
    y = np.concatenate([y, y])
    
    return X, y

def train_model(adls_csv_path, falls_csv_path, num_epochs=50, batch_size=32, learning_rate=0.001, sequence_length=10):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 读取数据
    print("正在读取数据...")
    adls_data = pd.read_csv(adls_csv_path, header=None)
    falls_data = pd.read_csv(falls_csv_path, header=None)
    
    # 提取特征和标签，跳过序列名称列和帧号列，只使用从第3列开始的数值特征
    adls_features = adls_data.iloc[:, 2:].values
    falls_features = falls_data.iloc[:, 2:].values
    
    # 创建标签
    adls_labels = np.zeros(len(adls_features))
    falls_labels = np.ones(len(falls_features))
    
    # 合并数据
    X = np.vstack((adls_features, falls_features))
    y = np.hstack((adls_labels, falls_labels))
    
    print(f"数据集大小: {X.shape}, 标签分布: {np.bincount(y.astype(int))}")
    
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 创建序列数据
    sequences = []
    sequence_labels = []
    
    for i in range(len(X) - sequence_length + 1):
        sequences.append(X[i:i + sequence_length])
        sequence_labels.append(y[i + sequence_length - 1])
    
    sequences = np.array(sequences)
    sequence_labels = np.array(sequence_labels)
    
    print(f"序列数据大小: {sequences.shape}, 序列标签分布: {np.bincount(sequence_labels.astype(int))}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(sequences, sequence_labels, test_size=0.2, random_state=42)
    
    # 数据增强
    print("正在进行数据增强...")
    X_train_augmented = list(X_train)
    y_train_augmented = list(y_train)
    
    # 对跌倒样本进行数据增强
    fall_indices = np.where(y_train == 1)[0]
    for idx in fall_indices:
        # 添加噪声的增强
        for _ in range(2):  # 每个跌倒样本生成2个增强样本
            augmented_sequence = X_train[idx] + np.random.normal(0, 0.05, X_train[idx].shape)
            X_train_augmented.append(augmented_sequence)
            y_train_augmented.append(1)
    
    # 转换为numpy数组
    X_train_augmented = np.array(X_train_augmented)
    y_train_augmented = np.array(y_train_augmented)
    
    print(f"数据增强后的训练集大小: {X_train_augmented.shape}")
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train_augmented)
    y_train = torch.LongTensor(y_train_augmented)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 初始化模型
    model = FallDetectionLSTM(input_size=X_train.shape[2])
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 用于存储训练历史
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    print("\n开始训练...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_sequences, batch_labels in train_loader:
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_labels.size(0)
            train_correct += predicted.eq(batch_labels).sum().item()
        
        train_accuracy = train_correct / train_total
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss / len(train_loader))
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_sequences, batch_labels in test_loader:
                batch_sequences = batch_sequences.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_sequences)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()
        
        val_accuracy = val_correct / val_total
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss / len(test_loader))
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy:.4f}')
        print(f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracy:.4f}')
    
    print("\n训练完成!")
    
    # 保存模型和训练历史
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler,
        'sequence_length': sequence_length,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    return model, checkpoint

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train fall detection LSTM model')
    parser.add_argument('--adls', type=str, 
                       default=r'D:\xiaoD_keshe\kes\dataset\urfall-cam0-adls.csv', 
                       help='Path to ADLS CSV file')
    parser.add_argument('--falls', type=str, 
                       default=r'D:\xiaoD_keshe\kes\dataset\urfall-cam0-falls.csv', 
                       help='Path to Falls CSV file')
    parser.add_argument('--output', type=str, 
                       default=r'D:\xiaoD_keshe\kes\dataset\fall_detection_lstm_model.pth', 
                       help='Path to save model')
    
    args = parser.parse_args()
    model, checkpoint = train_model(args.adls, args.falls)
    torch.save(checkpoint, args.output)
    print(f"模型已保存至: {args.output}") 