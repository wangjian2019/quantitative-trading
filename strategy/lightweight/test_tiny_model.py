#!/usr/bin/env python3
"""
测试 TinyTransformer 模型
使用正确的模型架构加载训练好的模型

作者: Alvin
"""

import torch
import torch.nn as nn
import pickle
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings("ignore")

# 复制TinyTransformer架构定义
class TinyTransformer(nn.Module):
    def __init__(self, input_dim=21, d_model=64, nhead=8, num_layers=2, num_classes=3, seq_len=30):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        seq_len = x.size(1)

        # 投影到模型维度
        x = self.input_projection(x)

        # 添加位置编码
        pos_len = min(seq_len, self.pos_encoding.size(0))
        x[:, :pos_len, :] = x[:, :pos_len, :] + self.pos_encoding[:pos_len].unsqueeze(0)

        # Transformer编码
        x = self.transformer(x)

        # 使用最后一个时间步
        x = x[:, -1, :]

        # 分类
        output = self.classifier(x)

        return output

def calculate_basic_features(df):
    """计算基础特征 - 与训练时保持一致"""
    features = []

    # 价格相关特征
    close = df['Close'].values
    volume = df['Volume'].values

    for i in range(len(df)):
        feature_row = []

        # 基础价格特征
        current_close = close[i]
        feature_row.extend([
            df['Open'].iloc[i] / current_close,
            df['High'].iloc[i] / current_close,
            df['Low'].iloc[i] / current_close,
            1.0,  # close / close = 1
            df['Volume'].iloc[i] / 1e6  # 标准化成交量
        ])

        # 历史特征（如果有足够历史数据）
        if i > 0:
            prev_close = close[i-1]
            feature_row.extend([
                (current_close - prev_close) / prev_close,  # 日收益率
            ])
        else:
            feature_row.extend([0.0])

        # 移动平均特征
        if i >= 4:
            ma5 = np.mean(close[max(0, i-4):i+1])
            feature_row.extend([current_close / ma5 - 1])
        else:
            feature_row.extend([0.0])

        if i >= 9:
            ma10 = np.mean(close[max(0, i-9):i+1])
            feature_row.extend([current_close / ma10 - 1])
        else:
            feature_row.extend([0.0])

        # 波动率特征
        if i >= 9:
            recent_prices = close[max(0, i-9):i+1]
            volatility = np.std(recent_prices) / np.mean(recent_prices)
            feature_row.extend([volatility])
        else:
            feature_row.extend([0.02])

        # RSI特征（简化版）
        if i >= 13:
            gains = []
            losses = []
            for j in range(max(0, i-13), i):
                change = close[j+1] - close[j]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(-change)

            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            rs = avg_gain / (avg_loss + 1e-6)
            rsi = 100 - (100 / (1 + rs))
            feature_row.extend([rsi / 100.0])
        else:
            feature_row.extend([0.5])

        # 成交量比率
        if i >= 19:
            vol_ma20 = np.mean(volume[max(0, i-19):i+1])
            vol_ratio = volume[i] / (vol_ma20 + 1e-6)
            feature_row.extend([min(vol_ratio, 5.0)])
        else:
            feature_row.extend([1.0])

        # 填充到21个特征
        while len(feature_row) < 21:
            feature_row.append(0.0)

        features.append(feature_row[:21])

    return np.array(features)

class TinyModelTester:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.model_info = None
        self.load_model()

    def load_model(self):
        """加载TinyTransformer模型"""
        try:
            # 加载模型信息
            with open('tiny_model_info.pkl', 'rb') as f:
                self.model_info = pickle.load(f)
            print(f"✅ 模型信息: {self.model_info}")

            # 加载缩放器
            with open('tiny_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ 缩放器加载成功")

            # 创建正确的模型实例
            self.model = TinyTransformer(
                input_dim=21,
                d_model=64,  # 训练时使用的参数
                nhead=8,
                num_layers=2,
                num_classes=3,
                seq_len=30
            ).to(self.device)

            # 加载模型权重
            checkpoint = torch.load('tiny_transformer_model.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            print("✅ TinyTransformer模型加载成功!")
            print(f"📊 模型准确率: {self.model_info.get('accuracy', 'Unknown'):.2%}")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")

    def predict_signal(self, symbol):
        """预测交易信号"""
        if self.model is None:
            return None

        try:
            # 获取历史数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)

            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)

            if len(hist) < 50:
                print(f"❌ {symbol} 历史数据不足")
                return None

            # 计算特征
            features = calculate_basic_features(hist)

            if len(features) < 30:
                print(f"❌ {symbol} 特征数据不足")
                return None

            # 取最后30天的数据
            recent_features = features[-30:]

            # 标准化特征
            feature_shape = recent_features.shape
            recent_features = recent_features.reshape(-1, feature_shape[-1])
            recent_features = self.scaler.transform(recent_features)
            recent_features = recent_features.reshape(1, 30, -1)

            # 转换为张量
            X = torch.FloatTensor(recent_features).to(self.device)

            # 模型预测
            with torch.no_grad():
                logits = self.model(X)
                probabilities = torch.softmax(logits, dim=1)

                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()

            # 转换为交易动作
            actions = ["SELL", "HOLD", "BUY"]
            action = actions[predicted_class]

            current_price = hist['Close'].iloc[-1]

            signal = {
                'symbol': symbol,
                'action': action,
                'confidence': confidence,
                'current_price': float(current_price),
                'predicted_class': predicted_class,
                'timestamp': datetime.now().isoformat(),
                'model_type': 'TinyTransformer',
                'model_accuracy': self.model_info.get('accuracy', 0.746)
            }

            print(f"🎯 {symbol}: {action} (置信度: {confidence:.2%}, 类别: {predicted_class})")
            return signal

        except Exception as e:
            print(f"❌ {symbol} 预测失败: {e}")
            return None

    def test_multiple_stocks(self, symbols):
        """测试多个股票"""
        print("🚀 开始TinyTransformer模型测试...")
        print("="*60)

        results = {}

        for symbol in symbols:
            signal = self.predict_signal(symbol)
            if signal:
                results[symbol] = signal

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tiny_model_results_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 结果已保存到: {filename}")

        # 统计结果
        if results:
            print("\n📊 预测结果汇总:")
            print("="*60)
            print(f"{'股票':<8s} | {'动作':<4s} | {'置信度':<8s} | {'当前价格':<10s}")
            print("-" * 40)

            action_counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
            total_confidence = 0

            for symbol, result in results.items():
                print(f"{symbol:<8s} | {result['action']:<4s} | {result['confidence']:<8.2%} | ${result['current_price']:<9.2f}")
                action_counts[result['action']] += 1
                total_confidence += result['confidence']

            print("-" * 40)
            print(f"📈 BUY: {action_counts['BUY']} | 📊 HOLD: {action_counts['HOLD']} | 📉 SELL: {action_counts['SELL']}")
            print(f"🎯 平均置信度: {total_confidence/len(results):.2%}")
            print(f"✅ 成功测试 {len(results)} 只股票")

        return results

def main():
    tester = TinyModelTester()

    if tester.model is None:
        print("❌ 模型加载失败")
        return

    # 测试训练集中的股票
    train_symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'QQQ', 'NVDA', 'AMZN', 'META']

    print("🧪 测试训练集股票...")
    train_results = tester.test_multiple_stocks(train_symbols)

    print("\n" + "="*60)
    print("🆕 测试训练集外股票...")

    # 测试一些训练集外的股票
    test_symbols = ['UBER', 'SHOP', 'NFLX', 'AMD']
    test_results = tester.test_multiple_stocks(test_symbols)

    print(f"\n🎉 测试完成! 共测试了 {len(train_results) + len(test_results)} 只股票")

if __name__ == "__main__":
    main()