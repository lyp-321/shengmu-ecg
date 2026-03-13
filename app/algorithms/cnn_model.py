import torch
import torch.nn as nn
import torch.nn.functional as F

class ECG1DCNN(nn.Module):
    """
    1D-CNN 架构用于 ECG 分类
    参考经典研究架构：Conv1D -> ReLU -> MaxPool -> Conv1D -> ReLU -> MaxPool -> FC
    """
    def __init__(self, num_classes=5):
        super(ECG1DCNN, self).__init__()
        
        # 第一层卷积：提取局部波形特征
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 第二层卷积：提取更抽象的特征
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 第三层卷积
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 全连接层
        # 假设输入长度为 1000，经过三次池化（/2, /2, /2）变为 125
        self.fc1 = nn.Linear(128 * 125, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, length)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def load_cnn_model(model_path, num_classes=3, device='cpu'):
    """加载模型"""
    model = ECG1DCNN(num_classes=num_classes)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
