"""
多模态深度融合引擎 - 五维度融合架构
创新点：时域+频域+深度+图网络+规则引擎
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from app.core.logger import logger
from app.core.exceptions import InferenceError


class MultiModalFusionEngine:
    """
    多模态融合引擎
    五维度融合：时域 + 频域 + 深度 + 图网络 + 规则
    """
    
    def __init__(self):
        """初始化融合引擎"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"多模态融合引擎初始化 - 设备: {self.device}")
        
        # 各模态模型（懒加载）
        self._time_domain_models = None
        self._freq_domain_models = None
        self._deep_models = None
        self._graph_model = None
        self._rule_engine = None
        
        # 融合权重（自适应学习）
        self.fusion_weights = {
            'time': 0.25,
            'freq': 0.20,
            'deep': 0.35,
            'graph': 0.15,
            'rule': 0.05
        }
        
        # 置信度阈值
        self.confidence_thresholds = {
            'high': 0.90,
            'medium': 0.70,
            'low': 0.50
        }
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        多模态融合推理
        
        Args:
            features: 包含各种特征的字典
            
        Returns:
            融合后的推理结果
        """
        try:
            logger.info("开始多模态融合推理...")
            
            # 1. 时域模态推理
            time_result = self._time_domain_inference(features)
            
            # 2. 频域模态推理
            freq_result = self._freq_domain_inference(features)
            
            # 3. 深度学习模态推理
            deep_result = self._deep_learning_inference(features)
            
            # 4. 图神经网络推理（多导联）
            graph_result = self._graph_inference(features)
            
            # 5. 规则引擎推理
            rule_result = self._rule_based_inference(features)
            
            # 6. 动态权重融合
            final_result = self._adaptive_fusion(
                time_result, freq_result, deep_result, 
                graph_result, rule_result, features
            )
            
            # 7. 置信度评估与校准
            final_result = self._confidence_calibration(final_result)
            
            logger.info(f"融合推理完成 - 诊断: {final_result['diagnosis']}, "
                       f"置信度: {final_result['confidence']:.2%}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"多模态融合推理失败: {str(e)}", exc_info=True)
            raise InferenceError(f"融合推理出错: {str(e)}")
    
    def _time_domain_inference(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        时域特征推理（传统机器学习集成）
        使用 XGBoost + LightGBM + CatBoost + RandomForest
        """
        logger.debug("时域模态推理...")
        
        # 懒加载模型
        if self._time_domain_models is None:
            logger.info("首次调用，开始加载时域模型...")
            self._load_time_domain_models()
        
        # 提取时域特征
        time_features = self._extract_time_features(features)
        logger.debug(f"提取的特征维度: {len(time_features)}")
        
        # 如果模型加载成功，使用真实模型预测
        if self._time_domain_models and len(time_features) > 0:
            logger.info(f"使用 {len(self._time_domain_models)} 个模型进行预测...")
            try:
                # 标准化（与训练时一致）
                feat_input = time_features.reshape(1, -1)
                if self._scaler is not None:
                    try:
                        feat_input = self._scaler.transform(feat_input)
                    except Exception as e:
                        logger.warning(f"scaler 标准化失败: {e}")

                # ── Stacking：XGBoost 粗筛 + CatBoost 复核 ──
                # XGBoost 召回率高，负责不漏掉任何疑似异常
                # CatBoost 精确率高，负责剔除假阳性，减少误报骚扰
                xgb_model = self._time_domain_models.get('XGBoost')
                cat_model = self._time_domain_models.get('CatBoost')
                lgb_model = self._time_domain_models.get('LightGBM')

                pred_class, prob = 0, 0.5
                stacking_used = False

                if xgb_model is not None and cat_model is not None:
                    try:
                        xgb_probs = xgb_model.predict_proba(feat_input)[0]
                        xgb_pred  = int(np.argmax(xgb_probs))

                        if xgb_pred != 0:
                            # XGBoost 认为异常 → 交给 CatBoost 复核
                            cat_probs = cat_model.predict_proba(feat_input)[0]
                            cat_pred  = int(np.argmax(cat_probs))
                            pred_class = cat_pred
                            prob       = float(cat_probs[cat_pred])
                            logger.info(f"Stacking: XGB粗筛=异常({xgb_pred}), CatBoost复核={cat_pred}, conf={prob:.4f}")
                        else:
                            # XGBoost 认为正常 → 直接采信
                            pred_class = 0
                            prob       = float(xgb_probs[0])
                            logger.info(f"Stacking: XGB粗筛=正常, conf={prob:.4f}")

                        stacking_used = True
                    except Exception as e:
                        logger.warning(f"Stacking 预测失败，回退到 LightGBM: {e}")

                if not stacking_used:
                    # 回退：用 LightGBM 或任意可用模型
                    fallback = lgb_model or next(iter(self._time_domain_models.values()), None)
                    if fallback is not None:
                        try:
                            fb_probs   = fallback.predict_proba(feat_input)[0]
                            pred_class = int(np.argmax(fb_probs))
                            prob       = float(fb_probs[pred_class])
                        except Exception as e:
                            logger.warning(f"回退模型预测失败: {e}")

                predictions = [pred_class]
                confidences = [prob]

                if predictions:
                    logger.info(f"ML集成结果: 类别={pred_class}, 置信度={prob:.4f}")
                    
                    # 3类概率分布映射到12类
                    probs12 = np.zeros(12)
                    class_map = {0: 0, 1: 5, 2: 3}  # 正常→0, 室早→5, 其他→3
                    probs12[class_map.get(pred_class, 0)] = prob
                    probs12[probs12 == 0] = (1 - prob) / 11
                    
                    return {
                        'probabilities': probs12,
                        'prediction': pred_class,
                        'confidence': prob,
                        'features': time_features
                    }
                else:
                    logger.warning("所有模型预测都失败，使用规则推理")
            except Exception as e:
                logger.warning(f"模型预测失败，使用规则推理: {e}")
        else:
            logger.warning(f"模型未加载或特征为空，使用规则推理。模型数={len(self._time_domain_models) if self._time_domain_models else 0}, 特征数={len(time_features)}")
        
        # 回退到规则推理
        logger.info("使用规则推理...")
        heart_rate = features.get('heart_rate', 75)
        hrv_sdnn = features.get('hrv', {}).get('sdnn', 50)
        
        # 简单的分类逻辑
        if heart_rate > 100:
            pred_class = 2  # 心动过速
            prob = 0.85
        elif heart_rate < 60:
            pred_class = 1  # 心动过缓
            prob = 0.80
        else:
            pred_class = 0  # 正常
            prob = 0.90
        
        logger.info(f"规则推理结果: 类别={pred_class}, 置信度={prob}")
        
        # 构造概率分布（12类）
        probs = np.zeros(12)
        probs[pred_class] = prob
        probs[probs == 0] = (1 - prob) / 11
        
        return {
            'probabilities': probs,
            'prediction': pred_class,
            'confidence': prob,
            'features': time_features
        }
    
    def _load_time_domain_models(self):
        """加载时域模型（传统ML模型）和 scaler"""
        import os
        import joblib

        model_dir = "app/algorithms/models"
        model_files = {
            'XGBoost':  'xgboost_model.pkl',
            'LightGBM': 'lightgbm_model.pkl',
            'CatBoost': 'catboost_model.pkl',
        }

        # 加载 scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        self._scaler = None
        if os.path.exists(scaler_path):
            try:
                self._scaler = joblib.load(scaler_path)
                logger.info("加载 scaler 成功")
            except Exception as e:
                logger.warning(f"加载 scaler 失败: {e}")

        self._time_domain_models = {}
        for model_name, filename in model_files.items():
            model_path = os.path.join(model_dir, filename)
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    self._time_domain_models[model_name] = model
                    logger.info(f"加载模型成功: {model_name}")
                except Exception as e:
                    logger.warning(f"加载模型失败 {model_name}: {e}")
            else:
                logger.warning(f"模型文件不存在: {model_path}")

        if not self._time_domain_models:
            logger.warning("未加载任何时域模型，将使用规则推理")
        else:
            logger.info(f"成功加载 {len(self._time_domain_models)} 个模型: {list(self._time_domain_models.keys())}")
    
    def _freq_domain_inference(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        频域特征推理（小波变换 + SVM）
        使用多尺度小波分析
        """
        logger.debug("频域模态推理...")
        
        # 提取频域特征
        freq_features = self._extract_freq_features(features)
        
        # 模拟SVM预测
        signal = features.get('signal', np.zeros(1000))
        
        # 简单的频域分析
        fft = np.fft.fft(signal)
        power = np.abs(fft) ** 2
        dominant_freq = np.argmax(power[:len(power)//2])
        
        # 根据主频判断
        if dominant_freq > 50:
            pred_class = 3  # 房颤
            prob = 0.75
        else:
            pred_class = 0  # 正常
            prob = 0.85
        
        probs = np.zeros(12)
        probs[pred_class] = prob
        probs[probs == 0] = (1 - prob) / 11
        
        return {
            'probabilities': probs,
            'prediction': pred_class,
            'confidence': prob,
            'features': freq_features
        }
    
    def _deep_learning_inference(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度学习推理（ResNet-1D + BiLSTM + TCN + Inception）
        多模型集成
        """
        logger.debug("深度学习模态推理...")
        
        # 懒加载深度学习模型
        if self._deep_models is None:
            logger.info("首次调用，开始加载深度学习模型...")
            self._load_deep_models()
        
        signal = features.get('signal', np.zeros(1000))
        
        # 如果模型加载成功，使用真实模型预测
        if self._deep_models and len(signal) > 0:
            try:
                # 准备输入数据
                signal_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0)  # (1, 1, 1000)
                signal_tensor = signal_tensor.to(self.device)
                
                # 集成多个深度学习模型的预测
                predictions = []
                confidences = []
                
                for model_name, model in self._deep_models.items():
                    try:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(signal_tensor)
                            probs = torch.softmax(outputs, dim=1)[0]
                            pred = torch.argmax(probs).item()
                            conf = probs[pred].item()
                        
                        predictions.append(pred)
                        confidences.append(conf)
                        
                        logger.info(f"{model_name} 预测: 类别={pred}, 置信度={conf:.4f}")
                    except Exception as e:
                        logger.warning(f"{model_name} 预测失败: {e}")
                        continue
                
                # 投票决定最终预测
                if predictions:
                    from collections import Counter
                    pred_class = Counter(predictions).most_common(1)[0][0]
                    prob = np.mean(confidences)
                    
                    logger.info(f"深度学习集成结果: 类别={pred_class}, 平均置信度={prob:.4f}")
                    
                    # 构造概率分布（3类或12类，取决于训练时的类别数）
                    # 训练时使用3类，这里映射到12类
                    probs = np.zeros(12)
                    if pred_class == 0:  # 正常
                        probs[0] = prob
                    elif pred_class == 1:  # 室性早搏
                        probs[5] = prob
                    elif pred_class == 2:  # 其他异常
                        probs[3] = prob  # 映射到房颤
                    
                    probs[probs == 0] = (1 - prob) / 11
                    
                    return {
                        'probabilities': probs,
                        'prediction': pred_class,
                        'confidence': prob,
                        'attention_weights': None
                    }
                else:
                    logger.warning("所有深度学习模型预测都失败，使用规则推理")
            except Exception as e:
                logger.warning(f"深度学习模型预测失败，使用规则推理: {e}")
        else:
            logger.warning(f"深度学习模型未加载或信号为空，使用规则推理。模型数={len(self._deep_models) if self._deep_models else 0}, 信号长度={len(signal)}")
        
        # 回退到规则推理
        logger.info("使用规则推理...")
        signal_std = np.std(signal)
        signal_mean = np.mean(signal)
        
        if signal_std > 0.3:
            pred_class = 3  # 房颤
            prob = 0.88
        elif signal_std < 0.1:
            pred_class = 1  # 心动过缓
            prob = 0.82
        else:
            pred_class = 0  # 正常
            prob = 0.92
        
        probs = np.zeros(12)
        probs[pred_class] = prob
        probs[probs == 0] = (1 - prob) / 11
        
        return {
            'probabilities': probs,
            'prediction': pred_class,
            'confidence': prob,
            'attention_weights': None
        }
    
    def _load_deep_models(self):
        """加载深度学习模型（PyTorch模型）"""
        import os
        from app.algorithms.deep_models import ResNet1D, SEResNet1D, BiLSTMECG, TCN, InceptionECG
        
        model_dir = "app/algorithms/models"
        num_classes = 3  # 训练时使用3类
        
        # 定义模型架构和文件
        model_configs = {
            'ResNet1D': (ResNet1D(num_classes=num_classes), 'resnet1d_best.pth'),
            'SEResNet1D': (SEResNet1D(num_classes=num_classes), 'seresnet1d_best.pth'),
            'BiLSTM': (BiLSTMECG(num_classes=num_classes, hidden_size=128), 'bilstm_best.pth'),
            'TCN': (TCN(num_classes=num_classes), 'tcn_best.pth'),
            'Inception': (InceptionECG(num_classes=num_classes), 'inception_best.pth')
        }
        
        self._deep_models = {}
        
        for model_name, (model, filename) in model_configs.items():
            model_path = os.path.join(model_dir, filename)
            if os.path.exists(model_path):
                try:
                    # 加载模型权重
                    state_dict = torch.load(model_path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    model = model.to(self.device)
                    model.eval()
                    
                    self._deep_models[model_name] = model
                    logger.info(f"加载深度学习模型成功: {model_name}")
                except Exception as e:
                    logger.warning(f"加载深度学习模型失败 {model_name}: {e}")
            else:
                logger.warning(f"深度学习模型文件不存在: {model_path}")
        
        if not self._deep_models:
            logger.warning("未加载任何深度学习模型，将使用规则推理")
        else:
            logger.info(f"成功加载 {len(self._deep_models)} 个深度学习模型: {list(self._deep_models.keys())}")
    
    def _graph_inference(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        图神经网络推理（处理多导联ECG）
        使用 GCN + GAT
        """
        logger.debug("图神经网络推理...")
        
        # 模拟图网络预测
        # 实际应构建导联间的图结构
        
        # 简单占位符
        pred_class = 0
        prob = 0.80
        
        probs = np.zeros(12)
        probs[pred_class] = prob
        probs[probs == 0] = (1 - prob) / 11
        
        return {
            'probabilities': probs,
            'prediction': pred_class,
            'confidence': prob,
            'graph_embedding': None
        }
    
    def _rule_based_inference(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        规则引擎推理（医学先验知识）
        硬规则 + 软规则 + 模糊逻辑
        """
        logger.debug("规则引擎推理...")
        
        heart_rate = features.get('heart_rate', 75)
        hrv = features.get('hrv', {})
        sdnn = hrv.get('sdnn', 50)
        rmssd = hrv.get('rmssd', 30)
        
        # 硬规则判定
        if heart_rate > 100:
            pred_class = 2  # 心动过速
            prob = 1.0
        elif heart_rate < 60:
            pred_class = 1  # 心动过缓
            prob = 1.0
        elif sdnn < 30 and rmssd < 20:
            pred_class = 3  # 疑似房颤
            prob = 0.70
        else:
            pred_class = 0  # 正常
            prob = 0.95
        
        probs = np.zeros(12)
        probs[pred_class] = prob
        probs[probs == 0] = (1 - prob) / 11
        
        return {
            'probabilities': probs,
            'prediction': pred_class,
            'confidence': prob,
            'rules_fired': []  # 触发的规则列表
        }
    
    def _adaptive_fusion(
        self, 
        time_result: Dict, 
        freq_result: Dict, 
        deep_result: Dict,
        graph_result: Dict,
        rule_result: Dict,
        features: Dict
    ) -> Dict[str, Any]:
        """
        自适应权重融合
        根据信号质量动态调整权重
        """
        logger.debug("自适应权重融合...")
        
        # 计算信号质量指标（SQI）
        sqi = self._calculate_signal_quality(features)
        
        # 根据SQI调整权重
        weights = self._adjust_weights_by_sqi(sqi)
        
        # 临时方案：时域模型和深度学习模型都已训练，调整权重
        weights = {
            'time': 0.45,  # 时域模型（传统ML Stacking）
            'freq': 0.05,  # 频域（模拟）
            'deep': 0.45,  # 深度学习模型（已训练）
            'graph': 0.02,  # 图网络（模拟）
            'rule': 0.03   # 规则引擎
        }
        logger.debug(f"使用时域+深度学习双主导权重: {weights}")

        # 加权融合概率
        fused_probs = (
            weights['time'] * time_result['probabilities'] +
            weights['freq'] * freq_result['probabilities'] +
            weights['deep'] * deep_result['probabilities'] +
            weights['graph'] * graph_result['probabilities'] +
            weights['rule'] * rule_result['probabilities']
        )

        # 归一化
        fused_probs = fused_probs / np.sum(fused_probs)

        # 最终预测
        pred_class = int(np.argmax(fused_probs))
        confidence = float(fused_probs[pred_class])

        # ── 双端互证（ML vs DL 一致性检验）──
        # ML 判正常但 DL 坚定认为异常 → 触发强制预警，不能因为 ML 多数票而漏诊
        ml_pred  = time_result.get('prediction', 0)
        dl_pred  = deep_result.get('prediction', 0)
        dl_conf  = deep_result.get('confidence', 0.0)
        cross_alert = False

        if ml_pred == 0 and dl_pred != 0 and dl_conf >= 0.70:
            # ML 说正常，DL 高置信度说异常 → 以 DL 为准，强制上调风险
            cross_alert = True
            pred_class  = dl_pred
            confidence  = dl_conf * 0.9   # 略微折扣，体现不确定性
            logger.warning(
                f"双端互证触发：ML=正常, DL=异常({dl_pred}, conf={dl_conf:.2f}) "
                f"→ 强制采信DL结果，触发3级预警"
            )
        elif ml_pred != 0 and dl_pred == 0 and dl_conf >= 0.80:
            # ML 说异常，DL 高置信度说正常 → DL 起到"假阳性过滤"作用
            cross_alert = False
            pred_class  = 0
            confidence  = (float(time_result.get('confidence', 0.5)) + dl_conf) / 2
            logger.info(
                f"双端互证：ML=异常({ml_pred}), DL=正常(conf={dl_conf:.2f}) "
                f"→ DL过滤假阳性，采信正常"
            )
        
        # 诊断标签映射
        diagnosis_map = {
            0: "正常窦性心律",
            1: "窦性心动过缓",
            2: "窦性心动过速",
            3: "心房颤动",
            4: "心房扑动",
            5: "室性早搏",
            6: "室性心动过速",
            7: "室性颤动",
            8: "一度房室传导阻滞",
            9: "二度房室传导阻滞",
            10: "三度房室传导阻滞",
            11: "束支传导阻滞"
        }

        diagnosis = diagnosis_map.get(pred_class, "未知")

        # 风险等级：双端互证触发时强制3级及以上
        risk_level = self._assess_risk(pred_class, confidence, features)
        if cross_alert and risk_level == "低风险":
            risk_level = "中风险"   # 至少中风险，不允许漏诊

        # 确保所有numpy类型转换为Python原生类型
        return {
            'diagnosis': diagnosis,
            'prediction_class': int(pred_class),
            'confidence': float(confidence),
            'probabilities': fused_probs.tolist() if hasattr(fused_probs, 'tolist') else list(fused_probs),
            'fusion_weights': {k: float(v) for k, v in weights.items()},
            'signal_quality': float(sqi),
            'heart_rate': float(features.get('heart_rate', 0)),
            'hrv_sdnn': float(features.get('hrv', {}).get('sdnn', 0)),
            'hrv_rmssd': float(features.get('hrv', {}).get('rmssd', 0)),
            'risk_level': risk_level,
            'cross_alert': cross_alert,   # 双端互证是否触发
            'afib_prob': float(fused_probs[1]) if len(fused_probs) > 1 else 0.0
        }
    
    def _calculate_signal_quality(self, features: Dict[str, Any]) -> float:
        """
        计算信号质量指标（SQI）
        
        Args:
            features: 特征字典
            
        Returns:
            信号质量分数 (0-1)
        """
        signal = features.get('signal', np.array([]))
        
        if len(signal) == 0:
            return 0.5
        
        # 简单的SQI计算：基于信号的统计特性
        try:
            # 1. 信噪比估计
            signal_power = np.var(signal)
            noise_estimate = np.var(np.diff(signal))
            snr = signal_power / (noise_estimate + 1e-10)
            
            # 2. 归一化到0-1
            sqi = min(1.0, snr / 100.0)
            
            return float(sqi)
        except:
            return 0.5
    
    def _assess_risk(self, pred_class: int, confidence: float, features: Dict[str, Any]) -> str:
        """
        评估风险等级
        
        Args:
            pred_class: 预测类别
            confidence: 置信度
            features: 特征字典
            
        Returns:
            风险等级: "低风险", "中风险", "高风险"
        """
        heart_rate = features.get('heart_rate', 75)
        hrv_sdnn = features.get('hrv', {}).get('sdnn', 50)
        
        # 高风险条件
        high_risk_classes = [1, 2, 3, 4, 9, 10, 11]  # 房颤、室颤、室速等
        if pred_class in high_risk_classes and confidence > 0.7:
            return "高风险"
        
        if heart_rate > 120 or heart_rate < 40:
            return "高风险"
        
        # 中风险条件
        medium_risk_classes = [5, 6, 7, 8]  # 早搏、传导阻滞等
        if pred_class in medium_risk_classes:
            return "中风险"
        
        if hrv_sdnn < 30 or confidence < 0.6:
            return "中风险"
        
        if heart_rate > 100 or heart_rate < 50:
            return "中风险"
        
        # 低风险
        return "低风险"
    
    def _adjust_weights_by_sqi(self, sqi: float) -> Dict[str, float]:
        """
        根据信号质量调整融合权重
        高质量信号：增加深度学习权重
        低质量信号：增加规则引擎权重
        """
        if sqi > 0.9:
            # 高质量：深度学习主导
            weights = {
                'time': 0.20,
                'freq': 0.15,
                'deep': 0.45,
                'graph': 0.15,
                'rule': 0.05
            }
        elif sqi > 0.7:
            # 中等质量：均衡
            weights = {
                'time': 0.25,
                'freq': 0.20,
                'deep': 0.35,
                'graph': 0.15,
                'rule': 0.05
            }
        else:
            # 低质量：规则主导
            weights = {
                'time': 0.25,
                'freq': 0.15,
                'deep': 0.20,
                'graph': 0.10,
                'rule': 0.30
            }
        
        logger.debug(f"SQI={sqi:.2f}, 调整后权重: {weights}")
        return weights
    
    def _confidence_calibration(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        置信度校准与不确定性量化
        使用 Temperature Scaling（已禁用，直接返回原始置信度）
        """
        confidence = result['confidence']
        
        # 暂时禁用Temperature Scaling，直接使用原始置信度
        calibrated_confidence = confidence
        
        # 不修改概率分布
        probs = np.array(result['probabilities'])
        
        result['confidence'] = float(calibrated_confidence)
        result['probabilities'] = probs.tolist() if hasattr(probs, 'tolist') else list(probs)
        result['calibrated'] = False  # 标记为未校准
        
        # 风险评估
        if calibrated_confidence > 0.85:
            result['recommendation'] = "AI自动诊断"
        elif calibrated_confidence > 0.70:
            result['recommendation'] = "AI辅助+医生审核"
        else:
            result['recommendation'] = "转人工诊断"
        
        return result
    
    def _extract_time_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        提取与训练脚本完全一致的特征（41维）
        时域16维 + 频域7维 + 小波18维 = 41维
        与 scripts/train_traditional_ml.py 保持一致
        """
        import pywt
        from scipy.signal import find_peaks

        signal = np.array(features.get('signal', np.zeros(1000)), dtype=np.float64)
        fs = features.get('sampling_rate', 360)

        feat = {}

        # ── 时域特征（16维）──
        feat['mean'] = np.mean(signal)
        feat['std'] = np.std(signal)
        feat['max'] = np.max(signal)
        feat['min'] = np.min(signal)
        feat['median'] = np.median(signal)
        feat['range'] = feat['max'] - feat['min']
        import pandas as pd
        feat['skewness'] = float(pd.Series(signal).skew())
        feat['kurtosis'] = float(pd.Series(signal).kurtosis())

        peaks, _ = find_peaks(signal, distance=int(0.6 * fs), prominence=0.3)
        if len(peaks) > 1:
            rr = np.diff(peaks) / fs * 1000
            feat['hr_mean'] = 60000 / np.mean(rr) if len(rr) > 0 else 0
            feat['hr_std'] = np.std(60000 / rr) if len(rr) > 0 else 0
            feat['sdnn'] = np.std(rr)
            feat['rmssd'] = np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else 0
            feat['pnn50'] = np.sum(np.abs(np.diff(rr)) > 50) / len(rr) * 100 if len(rr) > 1 else 0
            feat['rr_irregularity'] = np.std(rr) / (np.mean(rr) + 1e-8)
        else:
            feat['hr_mean'] = 0; feat['hr_std'] = 0; feat['sdnn'] = 0
            feat['rmssd'] = 0; feat['pnn50'] = 0; feat['rr_irregularity'] = 0

        feat['r_amplitude'] = feat['max'] - feat['min']
        feat['r_peak_count'] = float(len(peaks))

        # ── 频域特征（7维）──
        fft_vals = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal), 1 / fs)
        power = np.abs(fft_vals) ** 2

        p_t_band  = (fft_freq >= 0.5) & (fft_freq < 5)
        qrs_band  = (fft_freq >= 5)   & (fft_freq < 20)
        hf_band   = (fft_freq >= 20)  & (fft_freq < 40)
        total_pow = np.sum(power[(fft_freq >= 0.5) & (fft_freq < 40)]) + 1e-8

        feat['pt_power']        = np.sum(power[p_t_band])
        feat['qrs_power']       = np.sum(power[qrs_band])
        feat['hf_power']        = np.sum(power[hf_band])
        feat['qrs_pt_ratio']    = feat['qrs_power'] / (feat['pt_power'] + 1e-8)
        feat['qrs_power_ratio'] = feat['qrs_power'] / total_pow

        pos_mask = fft_freq > 0
        if pos_mask.sum() > 0:
            dom_idx = np.argmax(power[pos_mask])
            feat['dominant_freq']  = fft_freq[pos_mask][dom_idx]
            feat['dominant_power'] = power[pos_mask][dom_idx]
        else:
            feat['dominant_freq'] = 0; feat['dominant_power'] = 0

        # ── 小波特征（6层×3 = 18维）──
        coeffs = pywt.wavedec(signal, 'db4', level=5)
        for i, coeff in enumerate(coeffs):
            feat[f'wavelet_energy_level_{i}'] = np.sum(coeff ** 2)
            feat[f'wavelet_std_level_{i}']    = np.std(coeff)
            feat[f'wavelet_max_level_{i}']    = np.max(np.abs(coeff))

        # 按固定顺序转为数组（与训练时DataFrame列顺序一致）
        ordered_keys = [
            'mean', 'std', 'max', 'min', 'median', 'range', 'skewness', 'kurtosis',
            'hr_mean', 'hr_std', 'sdnn', 'rmssd', 'pnn50', 'rr_irregularity',
            'r_amplitude', 'r_peak_count',
            'pt_power', 'qrs_power', 'hf_power', 'qrs_pt_ratio', 'qrs_power_ratio',
            'dominant_freq', 'dominant_power',
        ]
        for i in range(6):
            ordered_keys += [f'wavelet_energy_level_{i}', f'wavelet_std_level_{i}', f'wavelet_max_level_{i}']

        arr = np.array([feat.get(k, 0.0) for k in ordered_keys], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _extract_freq_features(self, features: Dict[str, Any]) -> np.ndarray:
        """频域特征（复用 _extract_time_features 中的频域部分，保持接口兼容）"""
        signal = np.array(features.get('signal', np.zeros(1000)))
        fft = np.fft.fft(signal)
        power = np.abs(fft[:28]) ** 2
        return power.astype(np.float32)
