"""
深度学习模型集合
包含：ResNet-1D、DenseNet-1D、Transformer、BiLSTM、TCN等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResNet1D(nn.Module):
    """
    ResNet-1D for ECG Classification
    残差网络，解决梯度消失问题
    """
    
    class ResidualBlock(nn.Module):
        """残差块"""
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, 
                                   stride=stride, padding=3, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7,
                                   stride=1, padding=3, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)
            
            # 短路连接
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels)
                )
        
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out
    
    def __init__(self, num_classes=12, input_channels=1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=15, 
                               stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 4个残差层
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(self.ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(self.ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, channels, length)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    通道注意力机制
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEResNet1D(nn.Module):
    """
    SE-ResNet-1D
    ResNet + SE注意力机制
    """
    
    class SEResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, reduction=16):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7,
                                   stride=stride, padding=3, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7,
                                   stride=1, padding=3, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)
            
            self.se = SEBlock(out_channels, reduction)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels)
                )
        
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.se(out)
            out += self.shortcut(x)
            out = F.relu(out)
            return out
    
    def __init__(self, num_classes=12):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(self.SEResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(self.SEResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


class TransformerECG(nn.Module):
    """
    Transformer for ECG Classification
    全局依赖建模
    """
    def __init__(self, num_classes=12, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # 输入嵌入
        self.embedding = nn.Conv1d(1, d_model, kernel_size=7, padding=3)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 分类头
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x: (batch, 1, length)
        x = self.embedding(x)  # (batch, d_model, length)
        x = x.permute(0, 2, 1)  # (batch, length, d_model)
        
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # 全局平均池化
        x = x.mean(dim=1)  # (batch, d_model)
        x = self.fc(x)
        
        return x


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BiLSTMECG(nn.Module):
    """
    Bidirectional LSTM for ECG Classification
    双向长短期记忆网络
    """
    def __init__(self, num_classes=12, input_size=1, hidden_size=128, 
                 num_layers=2, dropout=0.5):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # x: (batch, 1, length)
        x = x.permute(0, 2, 1)  # (batch, length, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, length, hidden_size*2)
        
        # 注意力机制
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch, length, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_size*2)
        
        # 分类
        out = self.fc(context)
        
        return out


class TCN(nn.Module):
    """
    Temporal Convolutional Network
    时间卷积网络，因果卷积
    """
    
    class TemporalBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, 
                     dilation, padding, dropout=0.2):
            super().__init__()
            
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.dropout1 = nn.Dropout(dropout)
            
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.dropout2 = nn.Dropout(dropout)
            
            self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
                if in_channels != out_channels else None
            
            self.relu = nn.ReLU()
        
        def forward(self, x):
            out = self.conv1(x)
            # 裁剪到原始长度（因果卷积）
            if out.size(2) != x.size(2):
                out = out[:, :, :x.size(2)]
            
            out = self.bn1(out)
            out = self.relu(out)
            out = self.dropout1(out)
            
            out = self.conv2(out)
            # 裁剪到原始长度
            if out.size(2) != x.size(2):
                out = out[:, :, :x.size(2)]
            
            out = self.bn2(out)
            out = self.relu(out)
            out = self.dropout2(out)
            
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)
    
    def __init__(self, num_classes=12, num_channels=[64, 128, 256], 
                 kernel_size=7, dropout=0.2):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = 1 if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(self.TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size, padding=padding, dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
    
    def forward(self, x):
        # x: (batch, 1, length)
        out = self.network(x)
        out = out.mean(dim=2)  # 全局平均池化
        out = self.fc(out)
        return out


class InceptionBlock1D(nn.Module):
    """
    Inception Block for 1D signals
    多尺度卷积核
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 1x1 卷积
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU()
        )
        
        # 1x3 卷积
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU()
        )
        
        # 1x5 卷积
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU()
        )
        
        # 1x7 卷积
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU()
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class InceptionECG(nn.Module):
    """
    Inception Network for ECG
    多尺度特征提取
    """
    def __init__(self, num_classes=12):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.inception1 = InceptionBlock1D(64, 128)
        self.inception2 = InceptionBlock1D(128, 256)
        self.inception3 = InceptionBlock1D(256, 512)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.inception1(x)
        x = F.max_pool1d(x, 2)
        
        x = self.inception2(x)
        x = F.max_pool1d(x, 2)
        
        x = self.inception3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
