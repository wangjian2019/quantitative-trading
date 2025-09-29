#!/usr/bin/env python3
"""
内存优化训练脚本 - 解决MPS内存不足问题
使用更小的模型和批次大小，CPU验证
Author: Alvin
"""

import torch
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
import pickle
from sklearn.preprocessing import StandardScaler

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_basic_features(df):
    """计算基础技术指标特征"""
    features = pd.DataFrame(index=df.index)

    # 价格特征
    features['open'] = df['Open']
    features['high'] = df['High']
    features['low'] = df['Low']
    features['close'] = df['Close']
    features['volume'] = df['Volume']

    # 收益率
    features['returns'] = df['Close'].pct_change()
    features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # 移动平均
    for window in [5, 10, 20]:  # 减少特征数量
        features[f'ma_{window}'] = df['Close'].rolling(window).mean()
        features[f'price_to_ma_{window}'] = df['Close'] / features[f'ma_{window}']

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))

    # 布林带
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    features['bb_upper'] = ma20 + (std20 * 2)
    features['bb_lower'] = ma20 - (std20 * 2)
    features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    features['macd'] = ema12 - ema26

    # 波动率
    features['volatility'] = features['returns'].rolling(20).std()

    # 成交量指标
    features['volume_ma'] = df['Volume'].rolling(20).mean()
    features['volume_ratio'] = df['Volume'] / features['volume_ma']

    # 缺失值处理
    features = features.fillna(method='ffill').fillna(0)

    return features

def memory_efficient_transformer():
    """创建内存优化的小型Transformer模型"""

    class TinyTransformer(torch.nn.Module):
        def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, num_classes=3):  # 更小的模型
            super().__init__()

            self.input_projection = torch.nn.Linear(input_dim, d_model)
            self.pos_encoding = torch.nn.Parameter(torch.randn(100, d_model) * 0.1)  # 更短的位置编码

            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=128,  # 更小的前馈网络
                dropout=0.1,
                batch_first=True
            )
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)

            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(d_model, 32),  # 更小的分类器
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(32, num_classes)
            )

        def forward(self, x):
            # x: (batch, seq_len, features)
            seq_len = x.size(1)

            # 投影到模型维度
            x = self.input_projection(x)

            # 添加位置编码（截断到实际序列长度）
            pos_len = min(seq_len, self.pos_encoding.size(0))
            x[:, :pos_len, :] = x[:, :pos_len, :] + self.pos_encoding[:pos_len].unsqueeze(0)

            # Transformer编码
            x = self.transformer(x)

            # 使用最后一个时间步进行分类
            x = x[:, -1, :]

            # 分类
            output = self.classifier(x)

            return output

    return TinyTransformer

def memory_efficient_train():
    """内存优化训练函数"""
    logger.info("🚀 开始内存优化训练")

    # 获取数据文件
    data_dir = "data/training_data"
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    # 只使用前20个文件进行内存友好训练
    selected_files = all_files[:20]
    logger.info(f"📊 选择 {len(selected_files)} 只股票进行训练")

    # 收集所有数据
    all_features = []
    all_targets = []

    logger.info("📦 开始数据加载和特征计算...")

    for i, file in enumerate(selected_files):
        try:
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)

            if len(df) < 200:  # 跳过数据太少的股票
                continue

            # 计算特征
            features = calculate_basic_features(df)

            if len(features) > 70:
                # 创建序列数据 - 使用更短的序列
                seq_len = 30  # 减少到30个时间步
                for j in range(seq_len, min(len(features) - 1, seq_len + 500)):  # 限制样本数量
                    # 特征序列 (取数值列)
                    feature_seq = features.iloc[j-seq_len:j].select_dtypes(include=[np.number]).values

                    # 目标（下一天的收益率）
                    next_return = df['Close'].iloc[j+1] / df['Close'].iloc[j] - 1
                    if next_return > 0.02:
                        target = 0  # BUY
                    elif next_return < -0.02:
                        target = 2  # SELL
                    else:
                        target = 1  # HOLD

                    all_features.append(feature_seq)
                    all_targets.append(target)

            logger.info(f"✅ 已处理 {i+1}/{len(selected_files)} 只股票，样本数: {len(all_features)}")

        except Exception as e:
            logger.warning(f"❌ 处理 {file} 失败: {e}")
            continue

    logger.info(f"📊 数据收集完成: {len(all_features)} 个样本")

    if len(all_features) < 100:
        logger.error("❌ 训练样本太少，无法训练")
        return

    # 转换为numpy数组
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_targets, dtype=np.int64)

    logger.info(f"📏 特征形状: {X.shape}")
    logger.info(f"📏 目标形状: {y.shape}")

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # 分割训练集和测试集
    train_size = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    logger.info(f"📊 训练集: {X_train.shape[0]} 样本")
    logger.info(f"📊 测试集: {X_test.shape[0]} 样本")

    # 获取设备
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"💻 使用设备: {device}")

    # 创建模型
    input_dim = X.shape[-1]
    TinyTransformerClass = memory_efficient_transformer()
    model = TinyTransformerClass(input_dim=input_dim).to(device)

    logger.info("🤖 内存优化模型创建成功")

    # 训练设置
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 转换为tensor
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)

    logger.info("🚀 开始训练...")

    # 训练循环
    batch_size = 16  # 更小的批次大小
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        # 批次训练
        for i in range(0, len(X_train_tensor), batch_size):
            batch_x = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # 清理GPU内存
            if device == 'mps':
                torch.mps.empty_cache()

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

        # 每3个epoch验证一次，使用CPU避免内存问题
        if (epoch + 1) % 3 == 0:
            model.eval()
            model_cpu = model.cpu()  # 移到CPU进行验证

            with torch.no_grad():
                X_test_cpu = torch.FloatTensor(X_test)
                y_test_cpu = torch.LongTensor(y_test)

                # 分批验证
                correct = 0
                total = 0
                for i in range(0, len(X_test_cpu), batch_size):
                    batch_x = X_test_cpu[i:i+batch_size]
                    batch_y = y_test_cpu[i:i+batch_size]

                    outputs = model_cpu(batch_x)
                    predictions = torch.argmax(outputs, dim=1)
                    correct += (predictions == batch_y).sum().item()
                    total += batch_y.size(0)

                accuracy = correct / total
                logger.info(f"验证准确率: {accuracy:.4f}")

            # 移回GPU继续训练
            model = model_cpu.to(device)

    logger.info("✅ 训练完成")

    # 最终验证并保存模型
    model.eval()
    model_cpu = model.cpu()

    # 保存模型
    model_save_path = "tiny_transformer_model.pth"
    torch.save(model_cpu.state_dict(), model_save_path)
    logger.info(f"💾 模型已保存: {model_save_path}")

    # 保存缩放器
    scaler_save_path = "tiny_scaler.pkl"
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"💾 缩放器已保存: {scaler_save_path}")

    # 最终验证
    with torch.no_grad():
        X_test_cpu = torch.FloatTensor(X_test)
        y_test_cpu = torch.LongTensor(y_test)

        # 分批验证
        correct = 0
        total = 0
        for i in range(0, len(X_test_cpu), batch_size):
            batch_x = X_test_cpu[i:i+batch_size]
            batch_y = y_test_cpu[i:i+batch_size]

            outputs = model_cpu(batch_x)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

        final_accuracy = correct / total
        logger.info(f"🎯 最终准确率: {final_accuracy:.4f}")

    # 保存模型结构信息
    model_info = {
        'input_dim': input_dim,
        'model_class': 'TinyTransformer',
        'accuracy': final_accuracy,
        'num_samples': len(X),
        'seq_len': 30,
        'd_model': 64,
        'feature_names': list(range(input_dim))
    }

    with open('tiny_model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)

    logger.info("🎉 内存优化训练完成!")
    return model_save_path, scaler_save_path

if __name__ == "__main__":
    memory_efficient_train()