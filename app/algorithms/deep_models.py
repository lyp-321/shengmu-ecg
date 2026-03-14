"""
深度学习模型集合
包含：ResNet-1D、SE-ResNet-1D、Transformer、BiLSTM、TCN、Inception
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ResNet-1D
# ============================================================
class ResNet1D(nn.Module):
    """ResNet-1D for ECG Classification"""

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7,
                                   stride=stride, padding=3, bias=False)
            self.bn1   = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7,
                                   stride=1, padding=3, bias=False)
            self.bn2   = nn.BatchNorm1d(out_channels)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out)

    def __init__(self, num_classes=3, input_channels=1):
        super().__init__()
        self.conv1   = nn.Conv1d(input_channels, 64, kernel_size=15,
                                 stride=2, padding=7, bias=False)
        self.bn1     = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1  = self._make_layer(64,  64,  2, stride=1)
        self.layer2  = self._make_layer(64,  128, 2, stride=2)
        self.layer3  = self._make_layer(128, 256, 2, stride=2)
        self.layer4  = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch, out_ch, n, stride):
        layers = [self.ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, n):
            layers.append(self.ResidualBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out).view(out.size(0), -1)
        return self.fc(self.dropout(out))


# ============================================================
# SE Block
# ============================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze    = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


# ============================================================
# SE-ResNet-1D
# ============================================================
class SEResNet1D(nn.Module):
    """SE-ResNet-1D：ResNet + 通道注意力"""

    class SEResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, reduction=16):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7,
                                   stride=stride, padding=3, bias=False)
            self.bn1   = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7,
                                   stride=1, padding=3, bias=False)
            self.bn2   = nn.BatchNorm1d(out_channels)
            self.se    = SEBlock(out_channels, reduction)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.se(out)
            out += self.shortcut(x)
            return F.relu(out)

    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1   = nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1     = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1  = self._make_layer(64,  64,  2, stride=1)
        self.layer2  = self._make_layer(64,  128, 2, stride=2)
        self.layer3  = self._make_layer(128, 256, 2, stride=2)
        self.layer4  = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch, out_ch, n, stride):
        layers = [self.SEResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, n):
            layers.append(self.SEResidualBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out).view(out.size(0), -1)
        return self.fc(self.dropout(out))


# ============================================================
# Positional Encoding
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ============================================================
# Transformer ECG
# ============================================================
class TransformerECG(nn.Module):
    """
    Transformer for ECG Classification
    先用步长卷积把 1000 → 125，再做 self-attention，大幅降低显存
    """
    def __init__(self, num_classes=3, d_model=128, nhead=8,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        # 步长卷积下采样：1000 → 125（stride=8）
        self.embedding = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=15, stride=8, padding=7, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)           # (B, d_model, 125)
        x = x.permute(0, 2, 1)         # (B, 125, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)              # global avg pool
        return self.fc(self.dropout(x))


# ============================================================
# BiLSTM ECG
# ============================================================
class BiLSTMECG(nn.Module):
    """BiLSTM + 注意力机制"""
    def __init__(self, num_classes=3, input_size=1, hidden_size=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # attention：tanh 激活后再 softmax，避免梯度消失
        self.attn_w  = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_v  = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)                          # (B, L, 1)
        lstm_out, _ = self.lstm(x)                       # (B, L, H*2)
        # Bahdanau-style attention
        score   = self.attn_v(torch.tanh(self.attn_w(lstm_out)))  # (B, L, 1)
        weights = F.softmax(score, dim=1)                # (B, L, 1)
        context = (weights * lstm_out).sum(dim=1)        # (B, H*2)
        return self.fc(self.dropout(context))


# ============================================================
# TCN（真正的因果卷积）
# ============================================================
class TCN(nn.Module):
    """
    Temporal Convolutional Network — 因果卷积（只 pad 左边）
    修复原版双边 padding 导致的非因果问题
    """

    class CausalConv1d(nn.Module):
        """只在左边 pad，保证因果性"""
        def __init__(self, in_ch, out_ch, kernel_size, dilation):
            super().__init__()
            self.pad  = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                                  dilation=dilation, bias=False)
            self.bn   = nn.BatchNorm1d(out_ch)

        def forward(self, x):
            x = F.pad(x, (self.pad, 0))   # 只 pad 左边
            return self.bn(self.conv(x))

    class TemporalBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
            super().__init__()
            self.conv1    = TCN.CausalConv1d(in_ch,  out_ch, kernel_size, dilation)
            self.conv2    = TCN.CausalConv1d(out_ch, out_ch, kernel_size, dilation)
            self.dropout  = nn.Dropout(dropout)
            self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        def forward(self, x):
            out = F.relu(self.conv1(x))
            out = self.dropout(out)
            out = F.relu(self.conv2(out))
            out = self.dropout(out)
            res = x if self.downsample is None else self.downsample(x)
            return F.relu(out + res)

    def __init__(self, num_classes=3, num_channels=None, kernel_size=7, dropout=0.2):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 128, 256]
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch      = 1 if i == 0 else num_channels[i - 1]
            dilation   = 2 ** i
            layers.append(self.TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        out = self.network(x)
        out = out.mean(dim=2)          # global avg pool
        return self.fc(self.dropout(out))


# ============================================================
# Inception ECG
# ============================================================
class InceptionBlock1D(nn.Module):
    """多尺度卷积核 Inception Block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        c = out_channels // 4
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, c, 1, bias=False), nn.BatchNorm1d(c), nn.ReLU())
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, c, 3, padding=1, bias=False), nn.BatchNorm1d(c), nn.ReLU())
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, c, 5, padding=2, bias=False), nn.BatchNorm1d(c), nn.ReLU())
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, c, 7, padding=3, bias=False), nn.BatchNorm1d(c), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], dim=1)


class InceptionECG(nn.Module):
    """Inception Network for ECG — 多尺度特征提取"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1     = nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1       = nn.BatchNorm1d(64)
        self.maxpool   = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.inception1 = InceptionBlock1D(64,  128)
        self.inception2 = InceptionBlock1D(128, 256)
        self.inception3 = InceptionBlock1D(256, 512)
        self.avgpool   = nn.AdaptiveAvgPool1d(1)
        self.dropout   = nn.Dropout(0.5)
        self.fc        = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.inception1(x)
        x = F.max_pool1d(x, 2)
        x = self.inception2(x)
        x = F.max_pool1d(x, 2)
        x = self.inception3(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return self.fc(self.dropout(x))
