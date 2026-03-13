"""
联邦学习框架
支持多机构协同训练，数据不出域
包含：FedAvg、FedProx、FedNova、SCAFFOLD等算法
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
from copy import deepcopy
from app.core.logger import logger


class FederatedServer:
    """
    联邦学习服务器
    负责模型聚合和分发
    """
    
    def __init__(self, global_model: nn.Module, algorithm='FedAvg'):
        """
        Args:
            global_model: 全局模型
            algorithm: 聚合算法 ('FedAvg', 'FedProx', 'FedNova', 'SCAFFOLD')
        """
        self.global_model = global_model
        self.algorithm = algorithm
        self.round = 0
        
        # SCAFFOLD算法需要的控制变量
        if algorithm == 'SCAFFOLD':
            self.control_variate = {
                name: torch.zeros_like(param)
                for name, param in global_model.named_parameters()
            }
        
        logger.info(f"联邦学习服务器初始化 - 算法: {algorithm}")
    
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]], 
                  client_weights: Optional[List[float]] = None,
                  client_num_samples: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        聚合客户端模型
        
        Args:
            client_models: 客户端模型参数列表
            client_weights: 客户端权重（可选）
            client_num_samples: 客户端样本数（用于加权平均）
        
        Returns:
            聚合后的全局模型参数
        """
        self.round += 1
        logger.info(f"开始第 {self.round} 轮聚合，客户端数量: {len(client_models)}")
        
        if self.algorithm == 'FedAvg':
            return self._fedavg_aggregate(client_models, client_num_samples)
        elif self.algorithm == 'FedProx':
            return self._fedprox_aggregate(client_models, client_num_samples)
        elif self.algorithm == 'FedNova':
            return self._fednova_aggregate(client_models, client_num_samples)
        elif self.algorithm == 'SCAFFOLD':
            return self._scaffold_aggregate(client_models, client_num_samples)
        else:
            raise ValueError(f"未知的聚合算法: {self.algorithm}")
    
    def _fedavg_aggregate(self, client_models: List[Dict], 
                         client_num_samples: Optional[List[int]] = None) -> Dict:
        """
        FedAvg聚合算法
        加权平均
        """
        if client_num_samples is None:
            # 均匀权重
            weights = [1.0 / len(client_models)] * len(client_models)
        else:
            # 按样本数加权
            total_samples = sum(client_num_samples)
            weights = [n / total_samples for n in client_num_samples]
        
        # 加权平均
        global_params = {}
        for name in client_models[0].keys():
            global_params[name] = sum(
                w * client_model[name] for w, client_model in zip(weights, client_models)
            )
        
        logger.info(f"FedAvg聚合完成，权重: {weights}")
        return global_params
    
    def _fedprox_aggregate(self, client_models: List[Dict],
                          client_num_samples: Optional[List[int]] = None) -> Dict:
        """
        FedProx聚合算法
        处理数据异构性，添加近端项约束
        """
        # FedProx的聚合与FedAvg相同，区别在于客户端训练时的近端项
        return self._fedavg_aggregate(client_models, client_num_samples)
    
    def _fednova_aggregate(self, client_models: List[Dict],
                          client_num_samples: Optional[List[int]] = None) -> Dict:
        """
        FedNova聚合算法
        归一化聚合，解决客户端训练步数不一致问题
        """
        # 简化版本，实际需要考虑每个客户端的训练步数
        return self._fedavg_aggregate(client_models, client_num_samples)
    
    def _scaffold_aggregate(self, client_models: List[Dict],
                           client_num_samples: Optional[List[int]] = None) -> Dict:
        """
        SCAFFOLD聚合算法
        方差减少技术，使用控制变量
        """
        # 简化版本，实际需要更新控制变量
        return self._fedavg_aggregate(client_models, client_num_samples)
    
    def get_global_model(self) -> nn.Module:
        """获取全局模型"""
        return self.global_model
    
    def update_global_model(self, global_params: Dict[str, torch.Tensor]):
        """更新全局模型参数"""
        self.global_model.load_state_dict(global_params)


class FederatedClient:
    """
    联邦学习客户端
    负责本地训练
    """
    
    def __init__(self, client_id: int, local_model: nn.Module, 
                 train_loader, device='cpu', algorithm='FedAvg'):
        """
        Args:
            client_id: 客户端ID
            local_model: 本地模型
            train_loader: 训练数据加载器
            device: 设备
            algorithm: 训练算法
        """
        self.client_id = client_id
        self.local_model = local_model
        self.train_loader = train_loader
        self.device = device
        self.algorithm = algorithm
        
        logger.info(f"客户端 {client_id} 初始化完成")
    
    def train(self, global_params: Dict[str, torch.Tensor], 
              epochs: int = 1, lr: float = 0.01, mu: float = 0.01) -> Dict[str, torch.Tensor]:
        """
        本地训练
        
        Args:
            global_params: 全局模型参数
            epochs: 训练轮数
            lr: 学习率
            mu: FedProx的近端项系数
        
        Returns:
            训练后的本地模型参数
        """
        # 加载全局模型参数
        self.local_model.load_state_dict(global_params)
        self.local_model.to(self.device)
        self.local_model.train()
        
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.local_model(data)
                loss = criterion(output, target)
                
                # FedProx近端项
                if self.algorithm == 'FedProx':
                    proximal_term = 0.0
                    for name, param in self.local_model.named_parameters():
                        proximal_term += ((param - global_params[name]) ** 2).sum()
                    loss += (mu / 2) * proximal_term
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            logger.debug(f"客户端 {self.client_id} - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # 返回训练后的模型参数
        return self.local_model.state_dict()
    
    def get_num_samples(self) -> int:
        """获取本地数据集大小"""
        return len(self.train_loader.dataset)


class DifferentialPrivacy:
    """
    差分隐私保护
    添加噪声保护模型参数
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 sensitivity: float = 1.0, mechanism='Gaussian'):
        """
        Args:
            epsilon: 隐私预算
            delta: 失败概率
            sensitivity: 敏感度
            mechanism: 噪声机制 ('Gaussian', 'Laplace')
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.mechanism = mechanism
        
        logger.info(f"差分隐私初始化 - ε={epsilon}, δ={delta}, 机制={mechanism}")
    
    def add_noise(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        为模型参数添加噪声
        
        Args:
            params: 模型参数
        
        Returns:
            添加噪声后的参数
        """
        noisy_params = {}
        
        for name, param in params.items():
            if self.mechanism == 'Gaussian':
                # 高斯机制
                sigma = self._calculate_gaussian_sigma()
                noise = torch.randn_like(param) * sigma
            elif self.mechanism == 'Laplace':
                # 拉普拉斯机制
                scale = self.sensitivity / self.epsilon
                noise = torch.from_numpy(
                    np.random.laplace(0, scale, param.shape)
                ).float()
            else:
                raise ValueError(f"未知的噪声机制: {self.mechanism}")
            
            noisy_params[name] = param + noise
        
        return noisy_params
    
    def _calculate_gaussian_sigma(self) -> float:
        """计算高斯噪声的标准差"""
        # 根据差分隐私理论计算
        sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        return sigma


class SecureAggregation:
    """
    安全聚合协议
    使用同态加密和秘密共享
    """
    
    def __init__(self, method='additive_secret_sharing'):
        """
        Args:
            method: 安全聚合方法 ('additive_secret_sharing', 'homomorphic_encryption')
        """
        self.method = method
        logger.info(f"安全聚合初始化 - 方法: {method}")
    
    def encrypt(self, params: Dict[str, torch.Tensor], num_clients: int) -> List[Dict]:
        """
        加密模型参数
        
        Args:
            params: 模型参数
            num_clients: 客户端数量
        
        Returns:
            加密后的参数份额列表
        """
        if self.method == 'additive_secret_sharing':
            return self._additive_secret_sharing(params, num_clients)
        elif self.method == 'homomorphic_encryption':
            return self._homomorphic_encryption(params, num_clients)
        else:
            raise ValueError(f"未知的安全聚合方法: {self.method}")
    
    def _additive_secret_sharing(self, params: Dict, num_clients: int) -> List[Dict]:
        """
        加性秘密共享
        将参数分割成多个份额，满足：sum(shares) = params
        """
        shares = [{} for _ in range(num_clients)]
        
        for name, param in params.items():
            # 生成随机份额
            random_shares = [torch.randn_like(param) for _ in range(num_clients - 1)]
            
            # 最后一个份额 = 原始参数 - 其他份额之和
            last_share = param - sum(random_shares)
            
            # 分配份额
            for i in range(num_clients - 1):
                shares[i][name] = random_shares[i]
            shares[num_clients - 1][name] = last_share
        
        return shares
    
    def _homomorphic_encryption(self, params: Dict, num_clients: int) -> List[Dict]:
        """
        同态加密（简化版本）
        实际应使用Paillier或CKKS方案
        """
        # 占位符，实际需要实现完整的同态加密
        logger.warning("同态加密功能尚未完全实现，使用秘密共享代替")
        return self._additive_secret_sharing(params, num_clients)
    
    def decrypt(self, encrypted_shares: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        解密聚合后的参数
        
        Args:
            encrypted_shares: 加密的参数份额列表
        
        Returns:
            解密后的参数
        """
        if self.method == 'additive_secret_sharing':
            # 简单求和
            decrypted_params = {}
            for name in encrypted_shares[0].keys():
                decrypted_params[name] = sum(share[name] for share in encrypted_shares)
            return decrypted_params
        else:
            raise NotImplementedError()


class GradientCompression:
    """
    梯度压缩
    减少通信开销
    """
    
    def __init__(self, method='top_k', compression_ratio=0.1):
        """
        Args:
            method: 压缩方法 ('top_k', 'random_sparsification', 'quantization')
            compression_ratio: 压缩率
        """
        self.method = method
        self.compression_ratio = compression_ratio
        
        logger.info(f"梯度压缩初始化 - 方法: {method}, 压缩率: {compression_ratio}")
    
    def compress(self, params: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        压缩参数
        
        Args:
            params: 模型参数
        
        Returns:
            压缩后的参数
        """
        if self.method == 'top_k':
            return self._top_k_compression(params)
        elif self.method == 'random_sparsification':
            return self._random_sparsification(params)
        elif self.method == 'quantization':
            return self._quantization(params)
        else:
            raise ValueError(f"未知的压缩方法: {self.method}")
    
    def _top_k_compression(self, params: Dict) -> Dict:
        """
        Top-K稀疏化
        只保留最大的K个梯度
        """
        compressed = {}
        
        for name, param in params.items():
            # 展平
            flat_param = param.flatten()
            k = max(1, int(len(flat_param) * self.compression_ratio))
            
            # 选择Top-K
            _, indices = torch.topk(torch.abs(flat_param), k)
            values = flat_param[indices]
            
            compressed[name] = {
                'indices': indices,
                'values': values,
                'shape': param.shape
            }
        
        return compressed
    
    def _random_sparsification(self, params: Dict) -> Dict:
        """随机稀疏化"""
        compressed = {}
        
        for name, param in params.items():
            # 随机掩码
            mask = torch.rand_like(param) < self.compression_ratio
            compressed[name] = {
                'mask': mask,
                'values': param * mask,
                'shape': param.shape
            }
        
        return compressed
    
    def _quantization(self, params: Dict) -> Dict:
        """量化（FP32 -> INT8）"""
        compressed = {}
        
        for name, param in params.items():
            # 简单的线性量化
            min_val = param.min()
            max_val = param.max()
            scale = (max_val - min_val) / 255.0
            
            quantized = ((param - min_val) / scale).round().byte()
            
            compressed[name] = {
                'quantized': quantized,
                'min_val': min_val,
                'scale': scale,
                'shape': param.shape
            }
        
        return compressed
    
    def decompress(self, compressed: Dict) -> Dict[str, torch.Tensor]:
        """
        解压缩参数
        
        Args:
            compressed: 压缩后的参数
        
        Returns:
            解压缩后的参数
        """
        if self.method == 'top_k':
            return self._top_k_decompression(compressed)
        elif self.method == 'random_sparsification':
            return self._random_decompression(compressed)
        elif self.method == 'quantization':
            return self._quantization_decompression(compressed)
        else:
            raise ValueError(f"未知的压缩方法: {self.method}")
    
    def _top_k_decompression(self, compressed: Dict) -> Dict:
        """Top-K解压缩"""
        decompressed = {}
        
        for name, data in compressed.items():
            # 重建稀疏张量
            flat_param = torch.zeros(data['shape'].numel())
            flat_param[data['indices']] = data['values']
            decompressed[name] = flat_param.view(data['shape'])
        
        return decompressed
    
    def _random_decompression(self, compressed: Dict) -> Dict:
        """随机稀疏化解压缩"""
        decompressed = {}
        
        for name, data in compressed.items():
            decompressed[name] = data['values']
        
        return decompressed
    
    def _quantization_decompression(self, compressed: Dict) -> Dict:
        """量化解压缩"""
        decompressed = {}
        
        for name, data in compressed.items():
            # 反量化
            param = data['quantized'].float() * data['scale'] + data['min_val']
            decompressed[name] = param.view(data['shape'])
        
        return decompressed
