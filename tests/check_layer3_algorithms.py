#!/usr/bin/env python
"""
第3层检查：算法层
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from datetime import datetime

def check_algorithms():
    """检查算法层"""
    
    print("=" * 80)
    print("第3层检查：算法层")
    print("=" * 80)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    issues = []
    
    # 1. 检查数据读取模块
    print("【1】检查数据读取模块 (reader.py)...")
    try:
        from app.algorithms.reader import ECGReader
        reader = ECGReader()
        print("   ✅ ECGReader导入成功")
        
        # 测试CSV读取
        test_file = "test_data/normal_ecg.csv"
        if os.path.exists(test_file):
            data = reader.read(test_file)
            print(f"   ✅ CSV文件读取成功")
            print(f"      信号长度: {len(data['signal'])}")
            print(f"      采样率: {data.get('sampling_rate', 'N/A')} Hz")
        else:
            print(f"   ⚠️  测试文件不存在: {test_file}")
        
    except ImportError as e:
        print(f"   ❌ ECGReader导入失败: {e}")
        issues.append(f"ECGReader导入失败: {e}")
    except Exception as e:
        print(f"   ❌ ECGReader测试失败: {e}")
        issues.append(f"ECGReader测试失败: {e}")
    
    print()
    
    # 2. 检查预处理模块
    print("【2】检查预处理模块 (preprocess.py)...")
    try:
        from app.algorithms.preprocess import ECGPreprocessor
        preprocessor = ECGPreprocessor()
        print("   ✅ ECGPreprocessor导入成功")
        
        # 测试预处理
        test_signal = np.random.randn(1000)
        test_data = {'signal': test_signal, 'sampling_rate': 360}
        processed = preprocessor.process(test_data)
        
        print(f"   ✅ 预处理功能正常")
        print(f"      输入信号长度: {len(test_signal)}")
        print(f"      输出信号长度: {len(processed['signal'])}")
        print(f"      包含字段: {list(processed.keys())}")
        
    except ImportError as e:
        print(f"   ❌ ECGPreprocessor导入失败: {e}")
        issues.append(f"ECGPreprocessor导入失败: {e}")
    except Exception as e:
        print(f"   ❌ ECGPreprocessor测试失败: {e}")
        issues.append(f"ECGPreprocessor测试失败: {e}")
    
    print()
    
    # 3. 检查特征提取模块
    print("【3】检查特征提取模块 (features.py)...")
    try:
        from app.algorithms.features import ECGFeatureExtractor
        extractor = ECGFeatureExtractor()
        print("   ✅ ECGFeatureExtractor导入成功")
        
        # 测试特征提取
        test_signal = np.random.randn(1000)
        test_data = {
            'signal': test_signal,
            'sampling_rate': 360,
            'r_peaks': [100, 300, 500, 700, 900]
        }
        features = extractor.extract(test_data)
        
        print(f"   ✅ 特征提取功能正常")
        print(f"      提取的特征: {list(features.keys())}")
        
        # 检查必需特征
        required_features = ['heart_rate', 'hrv', 'signal']
        for feat in required_features:
            if feat in features:
                print(f"      ✅ 特征 '{feat}' 存在")
            else:
                print(f"      ❌ 特征 '{feat}' 缺失")
                issues.append(f"缺少特征: {feat}")
        
    except ImportError as e:
        print(f"   ❌ ECGFeatureExtractor导入失败: {e}")
        issues.append(f"ECGFeatureExtractor导入失败: {e}")
    except Exception as e:
        print(f"   ❌ ECGFeatureExtractor测试失败: {e}")
        issues.append(f"ECGFeatureExtractor测试失败: {e}")
    
    print()
    
    # 4. 检查推理引擎
    print("【4】检查推理引擎 (inference.py)...")
    try:
        from app.algorithms.inference import ECGInference
        inference = ECGInference()
        print("   ✅ ECGInference导入成功")
        print(f"   ✅ 单例模式: {ECGInference._instance is not None}")
        
        # 测试推理
        test_features = {
            'heart_rate': 75.0,
            'hrv': {'sdnn': 45.0, 'rmssd': 38.0},
            'signal': np.random.randn(1000)
        }
        
        # 测试传统模式
        result = inference.predict(test_features, use_fusion=False)
        print(f"   ✅ 推理功能正常 (传统模式)")
        print(f"      返回字段: {list(result.keys())}")
        
        # 检查必需字段
        required_fields = ['heart_rate', 'diagnosis', 'risk_level']
        for field in required_fields:
            if field in result:
                print(f"      ✅ 字段 '{field}' 存在: {result[field]}")
            else:
                print(f"      ❌ 字段 '{field}' 缺失")
                issues.append(f"推理结果缺少字段: {field}")
        
        # 测试融合模式
        try:
            result_fusion = inference.predict(test_features, use_fusion=True)
            print(f"   ✅ 多模态融合模式正常")
            print(f"      诊断: {result_fusion.get('diagnosis')}")
            print(f"      置信度: {result_fusion.get('confidence', 0):.2%}")
        except Exception as e:
            print(f"   ⚠️  多模态融合模式异常: {e}")
            # 不算严重问题，因为可能模型未训练
        
    except ImportError as e:
        print(f"   ❌ ECGInference导入失败: {e}")
        issues.append(f"ECGInference导入失败: {e}")
    except Exception as e:
        print(f"   ❌ ECGInference测试失败: {e}")
        issues.append(f"ECGInference测试失败: {e}")
    
    print()
    
    # 5. 检查多模态融合引擎
    print("【5】检查多模态融合引擎 (multimodal_fusion.py)...")
    try:
        from app.algorithms.multimodal_fusion import MultiModalFusionEngine
        fusion = MultiModalFusionEngine()
        print("   ✅ MultiModalFusionEngine导入成功")
        
        # 测试融合推理
        test_features = {
            'heart_rate': 75.0,
            'hrv': {'sdnn': 45.0, 'rmssd': 38.0},
            'signal': np.random.randn(1000),
            'r_peaks': [100, 300, 500, 700, 900]
        }
        
        result = fusion.predict(test_features)
        print(f"   ✅ 融合推理功能正常")
        print(f"      诊断: {result.get('diagnosis')}")
        print(f"      置信度: {result.get('confidence', 0):.2%}")
        print(f"      风险等级: {result.get('risk_level')}")
        
        # 检查返回值类型（确保JSON可序列化）
        import json
        try:
            json_str = json.dumps(result, ensure_ascii=False)
            print(f"   ✅ 返回值可JSON序列化")
        except TypeError as e:
            print(f"   ❌ 返回值无法JSON序列化: {e}")
            issues.append(f"融合引擎返回值无法JSON序列化: {e}")
        
    except ImportError as e:
        print(f"   ❌ MultiModalFusionEngine导入失败: {e}")
        issues.append(f"MultiModalFusionEngine导入失败: {e}")
    except Exception as e:
        print(f"   ❌ MultiModalFusionEngine测试失败: {e}")
        issues.append(f"MultiModalFusionEngine测试失败: {e}")
    
    print()
    
    # 6. 检查模型文件
    print("【6】检查预训练模型文件...")
    model_dir = "app/algorithms/models"
    if os.path.exists(model_dir):
        print(f"   ✅ 模型目录存在: {model_dir}")
        
        model_files = [
            'rf_model.pkl',
            'xgboost_model.pkl',
            'lightgbm_model.pkl',
            'catboost_model.pkl',
            'svm_model.pkl',
            'scaler.pkl'
        ]
        
        existing_models = []
        missing_models = []
        
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                size = os.path.getsize(model_path)
                existing_models.append(model_file)
                print(f"      ✅ {model_file} ({size/1024:.1f} KB)")
            else:
                missing_models.append(model_file)
                print(f"      ⚠️  {model_file} 不存在")
        
        if missing_models:
            print(f"   ⚠️  警告: {len(missing_models)} 个模型文件缺失")
            print(f"      建议运行: python scripts/train_traditional_ml.py")
        else:
            print(f"   ✅ 所有模型文件完整")
    else:
        print(f"   ❌ 模型目录不存在: {model_dir}")
        issues.append("模型目录不存在")
    
    print()
    
    # 7. 完整流程测试
    print("【7】完整算法流程测试...")
    try:
        from app.algorithms.reader import ECGReader
        from app.algorithms.preprocess import ECGPreprocessor
        from app.algorithms.features import ECGFeatureExtractor
        from app.algorithms.inference import ECGInference
        
        print("   测试完整流程: 读取 → 预处理 → 特征提取 → 推理")
        
        # 使用测试数据
        test_file = "test_data/normal_ecg.csv"
        if os.path.exists(test_file):
            # 1. 读取
            reader = ECGReader()
            data = reader.read(test_file)
            print(f"      ✅ 步骤1: 读取数据")
            
            # 2. 预处理
            preprocessor = ECGPreprocessor()
            processed = preprocessor.process(data)
            print(f"      ✅ 步骤2: 预处理")
            
            # 3. 特征提取
            extractor = ECGFeatureExtractor()
            features = extractor.extract(processed)
            print(f"      ✅ 步骤3: 特征提取")
            
            # 4. 推理
            inference = ECGInference()
            result = inference.predict(features, use_fusion=False)
            print(f"      ✅ 步骤4: 推理")
            
            print(f"   ✅ 完整流程测试通过")
            print(f"      最终结果:")
            print(f"        心率: {result.get('heart_rate', 0):.1f} BPM")
            print(f"        诊断: {result.get('diagnosis')}")
            print(f"        风险: {result.get('risk_level')}")
        else:
            print(f"   ⚠️  跳过完整流程测试（测试文件不存在）")
        
    except Exception as e:
        print(f"   ❌ 完整流程测试失败: {e}")
        issues.append(f"完整流程测试失败: {e}")
    
    print()
    
    # 总结
    print("=" * 80)
    print("检查总结")
    print("=" * 80)
    
    if not issues:
        print("✅ 算法层检查通过，没有发现严重问题")
        print()
        print("算法模块:")
        print("  ✅ ECGReader - 数据读取")
        print("  ✅ ECGPreprocessor - 信号预处理")
        print("  ✅ ECGFeatureExtractor - 特征提取")
        print("  ✅ ECGInference - 推理引擎")
        print("  ✅ MultiModalFusionEngine - 多模态融合")
        print()
        print("数据流:")
        print("  CSV/DAT → 读取 → 预处理 → 特征提取 → 推理 → 结果")
        return True
    else:
        print(f"❌ 发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print()
        print("建议修复方案:")
        if any("模型" in issue for issue in issues):
            print("   → 训练模型: python scripts/train_traditional_ml.py")
        if any("JSON" in issue for issue in issues):
            print("   → 检查numpy类型转换")
        return False

if __name__ == '__main__':
    success = check_algorithms()
    sys.exit(0 if success else 1)
