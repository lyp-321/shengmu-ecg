"""
图神经网络模型
用于处理多导联ECG的空间关系
包含：GCN、GAT、ST-GCN等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphConvolution(nn.Module):
    """
    图卷积层 (GCN Layer)
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_nodes, in_features)
            adj: (batch, num_nodes, num_nodes) 邻接矩阵
        """
        support = torch.matmul(x, self.weight)  # (batch, num_nodes, out_features)
        output = torch.matmul(adj, support)  # (batch, num_nodes, out_features)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GCN(nn.Module):
    """
    Graph Convolutional Network for Multi-lead ECG
    将12导联ECG建模为图结构
    """
    def __init__(self, num_nodes=12, node_features=128, hidden_dim=256, 
                 num_classes=12, dropout=0.5):
        super().__init__()
        
        # 节点特征提取（每个导联独立提取特征）
        self.feature_extractor = nn.Conv1d(1, node_features, kernel_size=7, padding=3)
        
        # 图卷积层
        self.gc1 = GraphConvolution(node_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gc3 = GraphConvolution(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # 分类层
        self.fc = nn.Linear(num_nodes * hidden_dim, num_classes)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_nodes, 1, length) 多导联ECG
            adj: (batch, num_nodes, num_nodes) 邻接矩阵
        """
        batch_size, num_nodes, _, length = x.size()
        
        # 提取每个导联的特征
        x = x.view(batch_size * num_nodes, 1, length)
        x = self.feature_extractor(x)  # (batch*num_nodes, node_features, length)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (batch*num_nodes, node_features)
        x = x.view(batch_size, num_nodes, -1)  # (batch, num_nodes, node_features)
        
        # 图卷积
        x = F.relu(self.gc1(x, adj))
        x = self.dropout(x)
        
        x = F.relu(self.gc2(x, adj))
        x = self.dropout(x)
        
        x = F.relu(self.gc3(x, adj))
        
        # 展平并分类
        x = x.view(batch_size, -1)
        x = self.fc(x)
        
        return x


class GraphAttentionLayer(nn.Module):
    """
    图注意力层 (GAT Layer)
    自适应学习导联间的重要性
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        """
        Args:
            h: (batch, num_nodes, in_features)
            adj: (batch, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = h.size()
        
        # 线性变换
        Wh = torch.matmul(h, self.W)  # (batch, num_nodes, out_features)
        
        # 计算注意力系数
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        # 掩码（只考虑邻接节点）
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 加权聚合
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        """准备注意力机制输入"""
        batch_size, num_nodes, out_features = Wh.size()
        
        # 广播拼接
        Wh_repeated_in_chunks = Wh.repeat_interleave(num_nodes, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, num_nodes, 1)
        
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        
        return all_combinations_matrix.view(batch_size, num_nodes, num_nodes, 2 * out_features)


class GAT(nn.Module):
    """
    Graph Attention Network for Multi-lead ECG
    自适应学习导联间的重要性
    """
    def __init__(self, num_nodes=12, node_features=128, hidden_dim=256,
                 num_classes=12, dropout=0.6, alpha=0.2, nheads=8):
        super().__init__()
        
        # 节点特征提取
        self.feature_extractor = nn.Conv1d(1, node_features, kernel_size=7, padding=3)
        
        # 多头注意力
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(node_features, hidden_dim, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        
        # 输出注意力层
        self.out_att = GraphAttentionLayer(hidden_dim * nheads, hidden_dim, 
                                           dropout, alpha, concat=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # 分类层
        self.fc = nn.Linear(num_nodes * hidden_dim, num_classes)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_nodes, 1, length)
            adj: (batch, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _, length = x.size()
        
        # 提取特征
        x = x.view(batch_size * num_nodes, 1, length)
        x = self.feature_extractor(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = x.view(batch_size, num_nodes, -1)
        
        # 多头注意力
        x = self.dropout(x)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        
        # 输出层
        x = self.dropout(x)
        x = F.elu(self.out_att(x, adj))
        
        # 分类
        x = x.view(batch_size, -1)
        x = self.fc(x)
        
        return x


class SpatialTemporalGraphConv(nn.Module):
    """
    时空图卷积层
    同时建模空间（导联间）和时间依赖
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        
        # 空间图卷积
        self.gcn = GraphConvolution(in_channels, out_channels)
        
        # 时间卷积
        self.tcn = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, 
                     padding=(kernel_size-1)//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
        # 残差连接
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.residual = lambda x: x
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_nodes, in_channels, length)
            adj: (batch, num_nodes, num_nodes)
        """
        batch_size, num_nodes, in_channels, length = x.size()
        
        # 空间图卷积（对每个时间步）
        x_spatial = []
        for t in range(length):
            x_t = x[:, :, :, t]  # (batch, num_nodes, in_channels)
            x_t = self.gcn(x_t, adj)  # (batch, num_nodes, out_channels)
            x_spatial.append(x_t)
        
        x_spatial = torch.stack(x_spatial, dim=-1)  # (batch, num_nodes, out_channels, length)
        
        # 时间卷积（对每个节点）
        x_temporal = []
        for n in range(num_nodes):
            x_n = x_spatial[:, n, :, :]  # (batch, out_channels, length)
            x_n = self.tcn(x_n)  # (batch, out_channels, length)
            x_temporal.append(x_n)
        
        x_temporal = torch.stack(x_temporal, dim=1)  # (batch, num_nodes, out_channels, length)
        
        # 残差连接
        res = self.residual(x.view(batch_size * num_nodes, in_channels, length))
        res = res.view(batch_size, num_nodes, -1, length)
        
        return F.relu(x_temporal + res)


class STGCN(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network
    时空图卷积网络，用于多导联ECG分析
    """
    def __init__(self, num_nodes=12, in_channels=1, hidden_channels=64,
                 num_classes=12, kernel_size=9, num_layers=3):
        super().__init__()
        
        # ST-GCN层
        self.st_gcn_layers = nn.ModuleList()
        
        # 第一层
        self.st_gcn_layers.append(
            SpatialTemporalGraphConv(in_channels, hidden_channels, kernel_size)
        )
        
        # 中间层
        for _ in range(num_layers - 2):
            self.st_gcn_layers.append(
                SpatialTemporalGraphConv(hidden_channels, hidden_channels, kernel_size)
            )
        
        # 最后一层
        self.st_gcn_layers.append(
            SpatialTemporalGraphConv(hidden_channels, hidden_channels, kernel_size)
        )
        
        # 全局池化和分类
        self.fc = nn.Linear(num_nodes * hidden_channels, num_classes)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch, num_nodes, 1, length)
            adj: (batch, num_nodes, num_nodes)
        """
        batch_size = x.size(0)
        
        # ST-GCN层
        for layer in self.st_gcn_layers:
            x = layer(x, adj)
        
        # 全局平均池化
        x = x.mean(dim=-1)  # (batch, num_nodes, hidden_channels)
        
        # 展平并分类
        x = x.view(batch_size, -1)
        x = self.fc(x)
        
        return x


def build_ecg_adjacency_matrix(num_nodes=12, device='cpu'):
    """
    构建12导联ECG的邻接矩阵
    基于医学知识定义导联间的连接关系
    
    12导联：I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    """
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    
    # 肢体导联连接 (I, II, III, aVR, aVL, aVF)
    limb_leads = [0, 1, 2, 3, 4, 5]
    for i in limb_leads:
        for j in limb_leads:
            if i != j:
                adj[i, j] = 1
    
    # 胸导联连接 (V1-V6)
    chest_leads = [6, 7, 8, 9, 10, 11]
    for i in range(len(chest_leads) - 1):
        adj[chest_leads[i], chest_leads[i+1]] = 1
        adj[chest_leads[i+1], chest_leads[i]] = 1
    
    # 肢体导联与胸导联的连接（较弱）
    for i in limb_leads:
        for j in chest_leads:
            adj[i, j] = 0.5
            adj[j, i] = 0.5
    
    # 添加自环
    adj = adj + torch.eye(num_nodes, device=device)
    
    # 归一化（对称归一化）
    degree = adj.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    adj_normalized = torch.matmul(torch.matmul(D_inv_sqrt, adj), D_inv_sqrt)
    
    return adj_normalized
