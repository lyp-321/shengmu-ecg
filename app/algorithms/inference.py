"""
ECG 推理/结果计算模块 - 多模态融合版本
集成：传统ML + 深度学习 + 图神经网络 + 规则引擎
"""
import os
import pickle
import torch
import numpy as np
from typing import Dict, Any, Optional
from app.algorithms.cnn_model import ECG1DCNN
from app.algorithms.multimodal_fusion import MultiModalFusionEngine
from app.core.logger import logger
from app.core.exceptions import ModelLoadError, InferenceError


class ECGInference:
    """
    ECG 推理引擎 (多模态融合)
    使用单例模式，避免重复加载模型
    """
    
    _instance: Optional['ECGInference'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """单例模式：确保只创建一个实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化模型（只执行一次）"""
        if self._initialized:
            return
        
        logger.info("初始化ECG多模态推理引擎...")
        
        self.model_dir = "app/algorithms/models"
        
        # 懒加载：只在需要时加载模型
        self._rf_model = None
        self._svm_model = None
        self._scaler = None
        self._cnn_model = None
        self._device = None
        
        # 多模态融合引擎
        self._fusion_engine = None
        
        self._initialized = True
        logger.info("ECG多模态推理引擎初始化完成（懒加载模式）")
    
    @property
    def device(self):
        """延迟初始化设备"""
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"使用设备: {self._device}")
        return self._device
    
    @property
    def fusion_engine(self):
        """延迟加载多模态融合引擎"""
        if self._fusion_engine is None:
            self._fusion_engine = MultiModalFusionEngine()
            logger.info("多模态融合引擎加载完成")
        return self._fusion_engine
    
    @property
    def rf_model(self):
        """延迟加载随机森林模型"""
        if self._rf_model is None:
            self._rf_model = self._load_pickle("rf_model.pkl")
        return self._rf_model
    
    @property
    def svm_model(self):
        """延迟加载SVM模型"""
        if self._svm_model is None:
            self._svm_model = self._load_pickle("svm_model.pkl")
        return self._svm_model
    
    @property
    def scaler(self):
        """延迟加载标准化器"""
        if self._scaler is None:
            self._scaler = self._load_pickle("scaler.pkl")
        return self._scaler
    
    @property
    def cnn_model(self):
        """延迟加载CNN模型"""
        if self._cnn_model is None:
            self._cnn_model = self._load_cnn("cnn_model.pth")
        return self._cnn_model

    def _load_pickle(self, filename: str):
        """
        加载pickle模型文件
        
        Args:
            filename: 模型文件名
            
        Returns:
            加载的模型对象，失败返回None
        """
        import joblib
        path = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(path):
            logger.warning(f"模型文件不存在: {path}")
            return None
        
        try:
            model = joblib.load(path)
            logger.info(f"成功加载模型: {filename}")
            return model
        except Exception as e:
            logger.warning(f"使用joblib加载 {filename} 失败: {e}，尝试使用pickle")
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"使用pickle成功加载模型: {filename}")
                return model
            except Exception as e2:
                logger.error(f"加载模型 {filename} 完全失败: {e2}")
                return None

    def _load_cnn(self, filename: str):
        """
        加载CNN深度学习模型
        
        Args:
            filename: 模型文件名
            
        Returns:
            加载的CNN模型
            
        Raises:
            ModelLoadError: 模型加载失败
        """
        path = os.path.join(self.model_dir, filename)
        model = ECG1DCNN(num_classes=3)
        
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location=self.device))
                logger.info(f"成功加载CNN模型: {filename}")
            except Exception as e:
                logger.error(f"加载CNN模型失败: {e}")
                raise ModelLoadError(f"无法加载CNN模型: {str(e)}")
        else:
            logger.warning(f"CNN模型文件不存在: {path}，使用未训练模型")
        
        model.to(self.device)
        model.eval()
        return model

    def predict(self, features: Dict[str, Any], use_fusion: bool = True) -> Dict[str, Any]:
        """
        基于特征和原始信号进行推理
        
        Args:
            features: 包含心率、HRV、原始信号等特征的字典
            use_fusion: 是否使用多模态融合（默认True）
            
        Returns:
            推理结果字典
            
        Raises:
            InferenceError: 推理失败
        """
        try:
            heart_rate = features.get('heart_rate', 0)
            hrv = features.get('hrv', {})
            signal = features.get('signal', np.zeros(1000))
            
            logger.info(f"开始推理 - 心率: {heart_rate:.1f}, HRV SDNN: {hrv.get('sdnn', 0):.1f}, "
                       f"融合模式: {'开启' if use_fusion else '关闭'}")
            
            if use_fusion:
                # 使用多模态融合引擎
                result = self.fusion_engine.predict(features)
            else:
                # 使用传统双驱动推理（向后兼容）
                # 1. 机器学习推理 (基于 HRV 特征)
                ml_diagnosis = self._ml_inference(heart_rate, hrv)
                
                # 2. 深度学习推理 (基于原始波形)
                afib_prob = self._dl_inference(signal)
                
                # 3. 综合判定 (双驱动逻辑)
                diagnosis, risk_level = self._综合判定(heart_rate, hrv, afib_prob, ml_diagnosis)
                
                result = {
                    'heart_rate': heart_rate,
                    'hrv_sdnn': hrv.get('sdnn', 0),
                    'hrv_rmssd': hrv.get('rmssd', 0),
                    'afib_prob': afib_prob,
                    'diagnosis': diagnosis,
                    'risk_level': risk_level,
                    'confidence': 0.85  # 默认置信度
                }
            
            logger.info(f"推理完成 - 诊断: {result['diagnosis']}, "
                       f"置信度: {result.get('confidence', 0):.2%}")
            return result
            
        except Exception as e:
            logger.error(f"推理失败: {str(e)}", exc_info=True)
            raise InferenceError(f"推理过程出错: {str(e)}")
    
    def _ml_inference(self, heart_rate: float, hrv: Dict) -> str:
        """机器学习推理"""
        ml_diagnosis = "正常窦性心律"
        
        if self.rf_model and self.scaler:
            try:
                X = np.array([[heart_rate, hrv.get('sdnn', 0), hrv.get('rmssd', 0)]])
                
                if self.scaler.n_features_in_ == 3:
                    X_scaled = self.scaler.transform(X)
                    ml_pred = self.rf_model.predict(X_scaled)[0]
                    ml_diagnosis = str(ml_pred)
                    logger.debug(f"ML推理结果: {ml_diagnosis}")
                else:
                    # 特征不匹配，使用规则引擎
                    if hrv.get('sdnn', 0) < 30 or heart_rate > 110:
                        ml_diagnosis = "心律不齐 (规则判定)"
            except Exception as e:
                logger.warning(f"ML推理失败，使用规则引擎: {e}")
                if hrv.get('sdnn', 0) < 30:
                    ml_diagnosis = "心律不齐"
        
        return ml_diagnosis
    
    def _dl_inference(self, signal: np.ndarray) -> float:
        """深度学习推理"""
        afib_prob = 0.0
        
        if self.cnn_model:
            try:
                # 预处理信号为 CNN 输入格式
                seg = signal[:1000]
                if len(seg) < 1000:
                    seg = np.pad(seg, (0, 1000 - len(seg)))
                
                input_tensor = torch.FloatTensor(seg).view(1, 1, -1).to(self.device)
                
                with torch.no_grad():
                    output = self.cnn_model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    afib_prob = float(probs[0][1])  # 假设索引1是房颤
                
                logger.debug(f"DL推理 - 房颤概率: {afib_prob:.3f}")
            except Exception as e:
                logger.warning(f"DL推理失败: {e}")
        
        return afib_prob
    
    def _综合判定(self, heart_rate: float, hrv: Dict, afib_prob: float, ml_diagnosis: str) -> tuple:
        """综合判定诊断结果和风险等级"""
        diagnosis = ml_diagnosis
        
        # 诊断逻辑
        if afib_prob > 0.5:
            diagnosis = "疑似房颤 (Atrial Fibrillation)"
        elif heart_rate > 100:
            diagnosis = "心动过速"
        elif heart_rate < 60:
            diagnosis = "心动过缓"
        
        # 风险评估
        risk_level = "低风险"
        if afib_prob > 0.7 or heart_rate > 120 or heart_rate < 40:
            risk_level = "高风险"
        elif afib_prob > 0.3 or hrv.get('sdnn', 0) < 50:
            risk_level = "中风险"
        
        return diagnosis, risk_level

