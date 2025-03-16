import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os

# 数据集类
class FallDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ResNet模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

# 跌倒检测模型
class FallDetectionModel(nn.Module):
    def __init__(self, input_size):
        super(FallDetectionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.classifier = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def add_noise(features, noise_factor=0.05):
    noise = np.random.normal(0, noise_factor, features.shape)
    return features + noise

def train_model(adls_csv_path, falls_csv_path, num_epochs=50, batch_size=32, learning_rate=0.001):
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
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 数据增强
    print("正在进行数据增强...")
    X_train_augmented = list(X_train)
    y_train_augmented = list(y_train)
    
    # 对跌倒样本进行数据增强
    fall_indices = np.where(y_train == 1)[0]
    for idx in fall_indices:
        # 添加噪声的增强
        for _ in range(2):  # 每个跌倒样本生成2个增强样本
            augmented_features = add_noise(X_train[idx])
            X_train_augmented.append(augmented_features)
            y_train_augmented.append(1)
    
    # 转换为numpy数组
    X_train_augmented = np.array(X_train_augmented)
    y_train_augmented = np.array(y_train_augmented)
    
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
    model = FallDetectionModel(X_train.shape[1])
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
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
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
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features)
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
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    return model, checkpoint

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train fall detection model')
    parser.add_argument('--adls', type=str, 
                       default=r'D:\xiaoD_keshe\kes\dataset\urfall-cam0-adls.csv', 
                       help='Path to ADLS CSV file')
    parser.add_argument('--falls', type=str, 
                       default=r'D:\xiaoD_keshe\kes\dataset\urfall-cam0-falls.csv', 
                       help='Path to Falls CSV file')
    parser.add_argument('--output', type=str, 
                       default=r'D:\xiaoD_keshe\kes\dataset\fall_detection_model.pth', 
                       help='Path to save model')
    
    args = parser.parse_args()
    model, checkpoint = train_model(args.adls, args.falls)
    torch.save(checkpoint, args.output)
    print(f"模型已保存至: {args.output}")
