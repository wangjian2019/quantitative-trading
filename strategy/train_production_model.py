#!/usr/bin/env python3
"""
生产级AI量化交易模型训练
使用167只股票的5年历史数据训练MultiStockTransformerModel
Author: Alvin
"""

import torch
import numpy as np
import pandas as pd
import os
import glob
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

# 导入我们的模型
from models.transformer_model import (
    MultiStockTransformerModel,
    IndustryLeadingFeatureExtractor,
    IndustryLeadingTransformerTrainer
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionModelTrainer:
    """生产级模型训练器"""

    def __init__(self, data_dir: str = "data/training_data"):
        self.data_dir = data_dir
        self.device = self._get_device()

        # 模型配置（基于实际数据优化）
        self.config = {
            'input_dim': 100,     # 保持100维特征
            'time_dim': 8,
            'd_model': 256,       # 增加模型容量
            'nhead': 16,
            'num_layers': 6,
            'seq_len': 60,        # 60天序列长度
            'dropout': 0.1,
            'enable_cross_stock': True
        }

        # 初始化组件
        self.feature_extractor = IndustryLeadingFeatureExtractor()
        self.model = None
        self.trainer = None
        self.scaler = StandardScaler()

    def _get_device(self):
        """获取计算设备"""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def load_training_data(self) -> Dict[str, pd.DataFrame]:
        """加载训练数据"""
        print("📊 加载167只股票的训练数据...")

        # 读取所有CSV文件
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        data_dict = {}

        for file_path in csv_files:
            try:
                symbol = os.path.basename(file_path).replace('.csv', '').replace('_', '.')
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)

                if len(df) >= 500:  # 至少2年数据
                    data_dict[symbol] = df

            except Exception as e:
                logger.warning(f"加载 {file_path} 失败: {e}")

        print(f"✅ 成功加载 {len(data_dict)} 只股票数据")

        # 选择流动性最好的前20只股票作为核心股票
        sorted_symbols = sorted(data_dict.items(), key=lambda x: len(x[1]), reverse=True)
        self.core_symbols = [symbol for symbol, _ in sorted_symbols[:20]]

        # 更新股票数量
        self.config['num_stocks'] = len(self.core_symbols)

        print(f"📈 核心股票池: {self.core_symbols}")

        return data_dict

    def prepare_production_samples(self, data_dict: Dict[str, pd.DataFrame]) -> Tuple:
        """准备生产级训练样本"""
        print("🔧 准备生产级训练样本...")

        all_features = []
        all_time_features = []
        all_targets = []
        all_stock_ids = []

        # 创建股票ID映射
        stock_id_map = {symbol: i for i, symbol in enumerate(self.core_symbols)}

        total_samples = 0

        for symbol in self.core_symbols:
            if symbol not in data_dict:
                continue

            print(f"  处理 {symbol}...")
            data = data_dict[symbol]
            stock_id = stock_id_map[symbol]

            try:
                # 重置索引，获取时间戳
                data_reset = data.reset_index()
                # 处理时区问题
                timestamps_dt = pd.to_datetime(data_reset.iloc[:, 0])
                if timestamps_dt.dt.tz is not None:
                    timestamps_dt = timestamps_dt.dt.tz_localize(None)
                timestamps = timestamps_dt.astype(np.int64) // 10**9

                ohlcv_data = np.column_stack([
                    timestamps,
                    data['Open'].values,
                    data['High'].values,
                    data['Low'].values,
                    data['Close'].values,
                    data['Volume'].values
                ])

                # 提取特征
                features = self.feature_extractor.extract_features(ohlcv_data)
                time_features = self._extract_time_features(timestamps)

                if features is None or len(features) < self.config['seq_len'] + 10:
                    continue

                # 创建序列样本
                for i in range(self.config['seq_len'], len(features) - 10):
                    # 特征序列
                    feature_seq = features[i-self.config['seq_len']:i]
                    time_seq = time_features[i-self.config['seq_len']:i]

                    # 创建多种时间跨度的标签
                    current_price = ohlcv_data[i, 4]

                    # 1天、3天、5天、10天后的价格
                    future_prices = []
                    valid_sample = True

                    for days in [1, 3, 5, 10]:
                        if i + days < len(features):
                            future_price = ohlcv_data[i + days, 4]
                            future_prices.append(future_price)
                        else:
                            valid_sample = False
                            break

                    if not valid_sample:
                        continue

                    # 计算综合标签（加权平均多个时间跨度的收益率）
                    returns = [(fp - current_price) / current_price for fp in future_prices]
                    weights = [0.4, 0.3, 0.2, 0.1]  # 偏重短期预测
                    weighted_return = sum(r * w for r, w in zip(returns, weights))

                    # 方向标签
                    if weighted_return > 0.02:
                        direction = 2  # BUY
                    elif weighted_return < -0.02:
                        direction = 0  # SELL
                    else:
                        direction = 1  # HOLD

                    # 计算其他标签
                    volatility = np.std(returns) if len(returns) > 1 else abs(weighted_return)
                    confidence = min(0.95, abs(weighted_return) * 20 + 0.5)
                    expected_return = weighted_return
                    sharpe_ratio = weighted_return / (volatility + 1e-6)

                    all_features.append(feature_seq)
                    all_time_features.append(time_seq)
                    all_targets.append([direction, volatility, confidence, expected_return, sharpe_ratio])
                    all_stock_ids.append(stock_id)

                    total_samples += 1

            except Exception as e:
                print(f"    ❌ 处理失败: {e}")
                continue

        print(f"🔧 生成了 {total_samples} 个高质量训练样本")

        # 转换为tensor
        features_tensor = torch.FloatTensor(np.array(all_features))
        time_tensor = torch.FloatTensor(np.array(all_time_features))
        stock_ids_tensor = torch.LongTensor(all_stock_ids)
        targets_array = np.array(all_targets)

        # 标准化特征
        original_shape = features_tensor.shape
        features_flat = features_tensor.reshape(-1, original_shape[-1])
        features_scaled = self.scaler.fit_transform(features_flat.numpy())
        features_tensor = torch.FloatTensor(features_scaled.reshape(original_shape))

        targets_dict = {
            'direction': torch.LongTensor(targets_array[:, 0].astype(int)),
            'volatility': torch.FloatTensor(targets_array[:, 1].reshape(-1, 1)),
            'confidence': torch.FloatTensor(targets_array[:, 2].reshape(-1, 1)),
            'expected_return': torch.FloatTensor(targets_array[:, 3].reshape(-1, 1)),
            'sharpe_ratio': torch.FloatTensor(targets_array[:, 4].reshape(-1, 1))
        }

        return features_tensor, time_tensor, stock_ids_tensor, targets_dict

    def _extract_time_features(self, timestamps: np.ndarray) -> np.ndarray:
        """提取时间特征"""
        dt_array = pd.to_datetime(timestamps, unit='s')

        features = np.column_stack([
            dt_array.hour.values,
            dt_array.dayofweek.values,
            dt_array.day.values,
            dt_array.month.values,
            np.sin(2 * np.pi * dt_array.hour / 24),
            np.cos(2 * np.pi * dt_array.hour / 24),
            np.sin(2 * np.pi * dt_array.dayofweek / 7),
            np.cos(2 * np.pi * dt_array.dayofweek / 7)
        ])

        return features.astype(np.float32)

    def train_production_model(self, features: torch.Tensor, time_features: torch.Tensor,
                              stock_ids: torch.Tensor, targets: Dict):
        """训练生产级模型"""
        print("🚀 开始训练生产级模型...")

        # 创建模型
        self.model = MultiStockTransformerModel(
            input_dim=self.config['input_dim'],
            time_dim=self.config['time_dim'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            seq_len=self.config['seq_len'],
            num_stocks=self.config['num_stocks'],
            dropout=self.config['dropout'],
            enable_cross_stock=self.config['enable_cross_stock']
        )

        # 创建训练器
        self.trainer = IndustryLeadingTransformerTrainer(self.model, self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 模型参数量: {total_params / 1e6:.2f}M")
        print(f"💻 训练设备: {self.device}")

        # 数据集分割
        total_samples = len(features)
        train_size = int(0.8 * total_samples)
        val_size = int(0.1 * total_samples)

        # 按时间顺序分割数据（更贴近实际交易情况）
        train_features = features[:train_size]
        train_time = time_features[:train_size]
        train_stock_ids = stock_ids[:train_size]
        train_targets = {k: v[:train_size] for k, v in targets.items()}

        val_features = features[train_size:train_size+val_size]
        val_time = time_features[train_size:train_size+val_size]
        val_stock_ids = stock_ids[train_size:train_size+val_size]
        val_targets = {k: v[train_size:train_size+val_size] for k, v in targets.items()}

        print(f"📊 训练样本: {len(train_features):,}, 验证样本: {len(val_features):,}")

        # 训练参数
        batch_size = 64  # 增加批次大小
        num_epochs = 30  # 更多训练轮次

        train_losses = []
        val_accuracies = []
        best_accuracy = 0

        print("🔄 开始训练循环...")

        for epoch in range(num_epochs):
            print(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")

            # 训练阶段
            self.model.train()
            epoch_loss = 0
            num_batches = 0

            # 随机打乱训练数据
            indices = torch.randperm(len(train_features))

            for i in range(0, len(train_features), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_features = train_features[batch_indices]
                batch_time = train_time[batch_indices]
                batch_stock_ids = train_stock_ids[batch_indices]
                batch_targets = {k: v[batch_indices] for k, v in train_targets.items()}

                loss_dict = self.trainer.train_step(
                    batch_features, batch_time, batch_targets, batch_stock_ids
                )

                epoch_loss += loss_dict['total_loss']
                num_batches += 1

                if num_batches % 50 == 0:
                    print(f"  Batch {num_batches}: Loss = {loss_dict['total_loss']:.4f}")

            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)

            # 验证阶段
            metrics = self.trainer.evaluate(
                val_features, val_time, val_targets, val_stock_ids
            )
            val_accuracies.append(metrics['direction_accuracy'])

            print(f"  训练损失: {avg_loss:.4f}")
            print(f"  验证准确率: {metrics['direction_accuracy']:.4f}")
            print(f"  验证指标: {metrics}")

            # 保存最佳模型
            if metrics['direction_accuracy'] > best_accuracy:
                best_accuracy = metrics['direction_accuracy']
                model_path = 'models/best_production_model.pth'
                self.trainer.save_model(model_path)
                print(f"  🎯 保存最佳模型 (准确率: {best_accuracy:.4f})")

            # 学习率衰减
            if epoch > 0 and epoch % 10 == 0:
                for param_group in self.trainer.optimizer.param_groups:
                    param_group['lr'] *= 0.8
                print(f"  📉 学习率衰减至: {param_group['lr']:.6f}")

        # 绘制训练曲线
        self._plot_training_curves(train_losses, val_accuracies)

        # 保存最终模型和配置
        final_model_path = 'models/final_production_model.pth'
        self.trainer.save_model(final_model_path)

        # 保存scaler
        with open('models/feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        # 保存配置
        config_save = {
            'model_config': self.config,
            'core_symbols': self.core_symbols,
            'stock_id_map': {symbol: i for i, symbol in enumerate(self.core_symbols)},
            'best_accuracy': best_accuracy,
            'training_samples': len(train_features),
            'training_date': datetime.now().isoformat()
        }

        with open('models/production_config.pkl', 'wb') as f:
            pickle.dump(config_save, f)

        print(f"✅ 生产模型训练完成！")
        print(f"🎯 最佳验证准确率: {best_accuracy:.4f}")
        print(f"💾 模型保存到: {final_model_path}")

    def _plot_training_curves(self, losses: List[float], accuracies: List[float]):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.plot(accuracies)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        if len(losses) > 10:
            smooth_losses = np.convolve(losses, np.ones(5)/5, mode='valid')
            plt.plot(smooth_losses, label='Smoothed Loss')
        plt.plot(accuracies, label='Accuracy', color='orange')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('production_training_curves.png', dpi=300, bbox_inches='tight')
        print("📊 训练曲线已保存到: production_training_curves.png")

    def run_production_training(self):
        """运行完整的生产级训练"""
        print("=" * 80)
        print("🚀 生产级AI量化交易模型训练")
        print("=" * 80)

        try:
            # 1. 加载数据
            data_dict = self.load_training_data()
            if len(data_dict) < 10:
                print("❌ 数据不足，无法训练")
                return

            # 2. 准备训练样本
            features, time_features, stock_ids, targets = self.prepare_production_samples(data_dict)

            if len(features) < 10000:
                print("❌ 训练样本不足")
                return

            # 3. 训练模型
            self.train_production_model(features, time_features, stock_ids, targets)

            print("\n✅ 生产级模型训练完成！")
            print("💡 模型已可用于实际交易信号生成")

        except Exception as e:
            print(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    trainer = ProductionModelTrainer()
    trainer.run_production_training()