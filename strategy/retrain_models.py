#!/usr/bin/env python3
"""
重新训练AI模型以支持80个特征
Author: Alvin
"""

import sys
import os
import json
import numpy as np
from datetime import datetime, timedelta
from ai_model_service import ai_model

def generate_training_data(num_samples=1000):
    """生成模拟训练数据以重新训练模型"""
    print(f"🔄 生成{num_samples}个训练样本...")
    
    training_data = []
    base_price = 100.0
    
    for i in range(num_samples):
        # 模拟价格走势
        trend = 0.001 if i < num_samples // 2 else -0.0005
        noise = np.random.normal(0, 0.02)
        price_change = trend + noise
        base_price *= (1 + price_change)
        
        # 生成OHLCV数据
        open_price = base_price
        high_price = base_price * (1 + abs(np.random.normal(0, 0.01)))
        low_price = base_price * (1 - abs(np.random.normal(0, 0.01)))
        close_price = base_price
        volume = int(np.random.normal(10000, 3000))
        
        sample = {
            'timestamp': (datetime.now() - timedelta(days=num_samples-i)).isoformat(),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': max(volume, 1000)
        }
        training_data.append(sample)
    
    return training_data

def retrain_models():
    """重新训练模型以支持80个特征"""
    print("🚀 开始重新训练AI模型...")
    print("目标: 支持80个特征，提升收益率")
    print("="*60)
    
    # 生成训练数据
    training_data = generate_training_data(2000)  # 增加训练数据
    
    # 备份原有模型
    print("📦 备份原有模型...")
    import shutil
    import glob
    
    backup_dir = f"models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    for file in glob.glob("models/*.pkl") + glob.glob("models/*.json"):
        shutil.copy2(file, backup_dir)
    print(f"✅ 模型已备份到: {backup_dir}")
    
    # 重新训练
    print("🤖 开始训练新模型...")
    success = ai_model.train_models(training_data)
    
    if success:
        print("✅ 模型重新训练成功!")
        print(f"✅ 新特征数量: {len(ai_model.feature_columns)}")
        print(f"✅ 模型性能: {ai_model.model_performance}")
        
        # 测试新模型
        print("\n🧪 测试新模型...")
        test_signal = ai_model.generate_signal(
            {'close': 100, 'volume': 1000000},
            {'RSI': 25, 'MACD': 1.5, 'MA5': 100, 'MA20': 98, 'VOLATILITY': 0.03, 'VOLUME_RATIO': 2.5},
            []
        )
        print(f"测试信号: {test_signal['action']} 置信度:{test_signal['confidence']:.1%}")
        
        return True
    else:
        print("❌ 模型训练失败")
        # 恢复备份
        print("🔄 恢复原有模型...")
        for file in glob.glob(f"{backup_dir}/*"):
            filename = os.path.basename(file)
            shutil.copy2(file, f"models/{filename}")
        print("✅ 原有模型已恢复")
        return False

if __name__ == "__main__":
    print("🚀 AI模型重新训练工具 v0.1")
    print("作者: Alvin")
    print("="*60)
    
    # 检查是否确认重新训练
    response = input("⚠️ 这将重新训练模型以支持80个特征，是否继续? (y/N): ")
    if response.lower() != 'y':
        print("❌ 取消重新训练")
        sys.exit(0)
    
    # 执行重新训练
    success = retrain_models()
    
    if success:
        print("\n🎉 重新训练完成!")
        print("📊 新模型特点:")
        print("- 特征数量: 80个")
        print("- 高收益特征: 突破、机构资金、超强信号")
        print("- 预期提升: 准确率和收益率")
        print("\n🔄 请重启AI服务以使用新模型:")
        print("pkill -f ai_model_service && python3 ai_model_service.py")
    else:
        print("\n❌ 重新训练失败，请检查错误信息")
    
    print("="*60)
