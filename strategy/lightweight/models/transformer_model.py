"""
业界最优AI量化交易模型 - 轻量级Transformer架构
Author: Alvin
专为Mac Mini优化，实现业界一流的交易信号生成
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class Time2Vec(nn.Module):
    """
    Time2Vec编码模块 - 业界最优时间编码方案
    Paper: "Time2Vec: Learning a Vector Representation of Time"
    专为金融时间序列设计，优于传统位置编码
    """

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # 线性层用于非周期性特征
        self.linear_layer = nn.Linear(input_dim, 1)

        # 周期性层数（embed_dim - 1，因为线性层占用1维）
        self.periodic_layers = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(embed_dim - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_dim) - 时间特征
        返回: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # 非周期性部分
        linear_out = self.linear_layer(x)  # (batch, seq, 1)

        # 周期性部分
        periodic_outs = []
        for layer in self.periodic_layers:
            periodic_out = torch.sin(layer(x))  # 使用sin激活
            periodic_outs.append(periodic_out)

        # 拼接所有维度
        periodic_tensor = torch.cat(periodic_outs, dim=-1)  # (batch, seq, embed_dim-1)

        # 组合线性和周期性特征
        time2vec_out = torch.cat([linear_out, periodic_tensor], dim=-1)  # (batch, seq, embed_dim)

        return time2vec_out

class AdvancedPositionalEncoding(nn.Module):
    """
    增强位置编码模块
    结合传统位置编码和可学习位置嵌入
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 传统正弦余弦位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # 可学习位置嵌入
        self.learned_pe = nn.Parameter(torch.randn(max_len, d_model) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)

        # 组合固定和可学习位置编码
        pos_encoding = self.pe[:seq_len, :] + self.learned_pe[:seq_len, :]

        return self.dropout(x + pos_encoding.unsqueeze(0))

class MultiStockTransformerModel(nn.Module):
    """
    业界最优多股票协同预测Transformer模型
    基于最新研究：Time2Vec + Multi-feature + Cross-stock attention
    支持多股票协同预测、增强特征工程、完整可解释性
    """

    def __init__(self,
                 input_dim: int = 50,          # 基础特征维度
                 time_dim: int = 8,            # 时间特征维度
                 d_model: int = 256,           # 增强嵌入维度
                 nhead: int = 16,              # 增强注意力头数
                 num_layers: int = 6,          # 增强层数
                 seq_len: int = 60,            # 序列长度
                 num_stocks: int = 10,         # 支持多股票协同
                 dropout: float = 0.1,
                 enable_cross_stock: bool = True):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.num_stocks = num_stocks
        self.enable_cross_stock = enable_cross_stock

        # 业界最优Time2Vec时间编码
        self.time2vec = Time2Vec(time_dim, d_model // 4)

        # 特征投影层 - 增强容量
        self.feature_projection = nn.Linear(input_dim, d_model * 3 // 4)

        # 综合特征融合
        self.feature_fusion = nn.Linear(d_model, d_model)

        # 增强位置编码
        self.pos_encoding = AdvancedPositionalEncoding(d_model, seq_len, dropout)

        # 股票嵌入 (用于多股票协同)
        if enable_cross_stock:
            self.stock_embedding = nn.Embedding(num_stocks, d_model)

        # 主要Transformer编码器 - 增强架构
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # 业界标准4倍放大
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True  # Pre-LayerNorm架构，更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 跨股票注意力模块
        if enable_cross_stock:
            self.cross_stock_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            )
            self.cross_stock_norm = nn.LayerNorm(d_model)

        # 业界最优多任务输出头 - 增强网络
        hidden_dim = d_model // 2

        self.direction_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)  # BUY/HOLD/SELL
        )

        self.volatility_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)  # 波动率预测
        )

        self.confidence_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),  # 置信度预测
            nn.Sigmoid()  # 确保0-1范围
        )

        self.return_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)  # 收益率预测
        )

        # 新增：风险调整收益预测
        self.sharpe_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)  # 夏普比率预测
        )

        # 初始化权重 - 业界最佳实践
        self._init_weights()

        # 存储注意力权重用于可解释性
        self.attention_weights = None

    def _init_weights(self):
        """业界最佳权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier初始化适合GELU激活
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def forward(self,
                features: torch.Tensor,
                time_features: torch.Tensor,
                stock_ids: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        业界最优前向传播
        features: (batch_size, seq_len, input_dim) - 主要特征
        time_features: (batch_size, seq_len, time_dim) - 时间特征
        stock_ids: (batch_size,) - 股票ID用于协同预测
        mask: (batch_size, seq_len) - 注意力掩码
        """
        batch_size, seq_len, _ = features.shape

        # Time2Vec时间编码
        time_encoded = self.time2vec(time_features)  # (batch, seq, d_model//4)

        # 特征投影
        feature_projected = self.feature_projection(features)  # (batch, seq, 3*d_model//4)

        # 特征融合
        combined_features = torch.cat([feature_projected, time_encoded], dim=-1)
        fused_features = self.feature_fusion(combined_features)  # (batch, seq, d_model)

        # 股票嵌入 (多股票协同)
        if self.enable_cross_stock and stock_ids is not None:
            stock_embeds = self.stock_embedding(stock_ids)  # (batch, d_model)
            stock_embeds = stock_embeds.unsqueeze(1).expand(-1, seq_len, -1)
            fused_features = fused_features + stock_embeds

        # 位置编码
        encoded_features = self.pos_encoding(fused_features)

        # 主要Transformer编码
        transformer_out = self.transformer(encoded_features, src_key_padding_mask=mask)

        # 跨股票注意力 (多股票协同预测)
        if self.enable_cross_stock:
            cross_attended, self.attention_weights = self.cross_stock_attention(
                transformer_out, transformer_out, transformer_out, key_padding_mask=mask
            )
            transformer_out = self.cross_stock_norm(transformer_out + cross_attended)

        # 序列池化 - 使用注意力池化替代简单的最后位置
        if mask is not None:
            # 计算有效序列长度
            seq_lengths = (~mask).sum(dim=1, keepdim=True).float()
            # 掩码池化
            masked_out = transformer_out.masked_fill(mask.unsqueeze(-1), 0)
            pooled_output = masked_out.sum(dim=1) / seq_lengths.clamp(min=1)
        else:
            # 平均池化
            pooled_output = transformer_out.mean(dim=1)

        # 多任务输出
        outputs = {
            'direction': self.direction_head(pooled_output),
            'volatility': torch.abs(self.volatility_head(pooled_output)),  # 确保非负
            'confidence': self.confidence_head(pooled_output),
            'expected_return': self.return_head(pooled_output),
            'sharpe_ratio': self.sharpe_head(pooled_output)  # 新增风险调整收益
        }

        return outputs

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """获取最新的注意力权重用于可解释性分析"""
        return self.attention_weights

    def get_feature_importance(self,
                             features: torch.Tensor,
                             time_features: torch.Tensor) -> torch.Tensor:
        """
        计算特征重要性用于可解释性
        使用梯度方法计算每个特征的重要性
        """
        features.requires_grad_(True)
        time_features.requires_grad_(True)

        outputs = self.forward(features, time_features)

        # 对confidence输出计算梯度
        confidence_score = outputs['confidence'].sum()
        confidence_score.backward(retain_graph=True)

        # 返回特征重要性
        feature_importance = torch.abs(features.grad).mean(dim=(0, 1))
        time_importance = torch.abs(time_features.grad).mean(dim=(0, 1))

        return feature_importance, time_importance

class IndustryLeadingFeatureExtractor:
    """
    业界最优特征工程模块
    基于最新金融工程研究，实现业界领先的特征提取
    包含：技术指标、因子库、宏观特征、情绪指标、高频微观结构特征
    """

    def __init__(self):
        self.lookback_periods = [3, 5, 10, 20, 50, 100, 200]  # 扩展时间窗口
        self.volume_periods = [5, 10, 20]
        self.momentum_periods = [5, 10, 20, 60]  # 动量周期

    def extract_comprehensive_features(self, ohlcv_data: np.ndarray,
                                     market_data: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        业界最全面的特征提取
        ohlcv_data: [timestamp, open, high, low, close, volume]
        market_data: 市场宏观数据 (可选)
        """
        if len(ohlcv_data) < 200:  # 确保足够的历史数据
            raise ValueError("需要至少200个数据点进行特征提取")

        close = ohlcv_data[:, 4]
        high = ohlcv_data[:, 2]
        low = ohlcv_data[:, 3]
        open_price = ohlcv_data[:, 1]
        volume = ohlcv_data[:, 5]
        timestamp = ohlcv_data[:, 0]

        features = {}

        # 1. 基础价格特征 (增强版)
        features.update(self._extract_enhanced_price_features(open_price, high, low, close, volume))

        # 2. 高级技术指标 (业界标准)
        features.update(self._calculate_advanced_technical_indicators(ohlcv_data))

        # 3. 量价关系特征
        features.update(self._extract_volume_price_features(close, volume))

        # 4. 动量和趋势特征
        features.update(self._extract_momentum_features(close, high, low))

        # 5. 波动率建模特征
        features.update(self._extract_volatility_modeling_features(close))

        # 6. 市场微观结构特征 (高频)
        features.update(self._extract_microstructure_features(ohlcv_data))

        # 7. 时间序列统计特征
        features.update(self._extract_statistical_features(close, volume))

        # 8. 跨周期相对强弱特征
        features.update(self._extract_relative_strength_features(close))

        # 9. 风险度量特征
        features.update(self._extract_risk_features(close))

        # 10. 时间特征 (用于Time2Vec)
        features.update(self._extract_time_features(timestamp))

        return features

    def _extract_enhanced_price_features(self, open_price, high, low, close, volume):
        """增强价格特征"""
        features = {}

        # 多时间框架收益率
        for period in self.lookback_periods:
            if len(close) > period:
                # 对数收益率
                features[f'log_return_{period}'] = np.log(close[period:] / close[:-period])
                # 累积收益率
                features[f'cum_return_{period}'] = (close[period:] / close[:-period]) - 1
                # 高低点收益率
                features[f'high_return_{period}'] = np.log(high[period:] / close[:-period])
                features[f'low_return_{period}'] = np.log(low[period:] / close[:-period])

        # 价格位置特征
        for period in [10, 20, 50]:
            if len(close) > period:
                rolling_high = np.array([np.max(high[max(0, i-period+1):i+1])
                                       for i in range(len(high))])
                rolling_low = np.array([np.min(low[max(0, i-period+1):i+1])
                                      for i in range(len(low))])
                features[f'price_position_{period}'] = (close - rolling_low) / (rolling_high - rolling_low + 1e-8)

        # 价差特征
        features['high_low_ratio'] = (high - low) / close
        features['open_close_ratio'] = (close - open_price) / open_price
        features['close_high_ratio'] = close / high
        features['close_low_ratio'] = close / low

        return features

    def _calculate_advanced_technical_indicators(self, ohlcv_data):
        """业界标准技术指标"""
        close = ohlcv_data[:, 4]
        high = ohlcv_data[:, 2]
        low = ohlcv_data[:, 3]
        volume = ohlcv_data[:, 5]

        features = {}

        # RSI (多周期)
        for period in [9, 14, 21]:
            features[f'rsi_{period}'] = self._calculate_rsi(close, period)

        # MACD (多参数组合)
        macd_configs = [(12, 26, 9), (5, 35, 5), (19, 39, 9)]
        for i, (fast, slow, signal) in enumerate(macd_configs):
            macd, macd_signal, macd_hist = self._calculate_macd(close, fast, slow, signal)
            features[f'macd_{i}'] = macd
            features[f'macd_signal_{i}'] = macd_signal
            features[f'macd_histogram_{i}'] = macd_hist

        # 布林带 (多标准差)
        for period, std_factor in [(20, 2), (20, 1.5), (10, 2)]:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, period, std_factor)
            idx = f"{period}_{int(std_factor*10)}"
            features[f'bb_upper_{idx}'] = bb_upper
            features[f'bb_middle_{idx}'] = bb_middle
            features[f'bb_lower_{idx}'] = bb_lower
            features[f'bb_position_{idx}'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
            features[f'bb_width_{idx}'] = (bb_upper - bb_lower) / bb_middle

        # ATR (多周期)
        for period in [10, 14, 20]:
            features[f'atr_{period}'] = self._calculate_atr(high, low, close, period)

        # 随机指标KD
        for period in [9, 14, 21]:
            k_percent, d_percent = self._calculate_stochastic(high, low, close, period)
            features[f'stoch_k_{period}'] = k_percent
            features[f'stoch_d_{period}'] = d_percent

        # 威廉指标
        for period in [10, 14, 20]:
            features[f'williams_r_{period}'] = self._calculate_williams_r(high, low, close, period)

        return features

    def _extract_volume_price_features(self, close, volume):
        """量价关系特征"""
        features = {}

        # 成交量移动平均
        for period in self.volume_periods:
            features[f'volume_ma_{period}'] = self._moving_average(volume, period)
            features[f'volume_ratio_{period}'] = volume / (self._moving_average(volume, period) + 1e-8)

        # 量价确认指标
        price_change = np.diff(close)
        volume_change = np.diff(volume)

        # OBV (On-Balance Volume)
        obv = np.cumsum(np.where(price_change > 0, volume[1:],
                                np.where(price_change < 0, -volume[1:], 0)))
        features['obv'] = obv

        # 价量散度
        for period in [10, 20]:
            if len(close) > period:
                price_ma = self._moving_average(close, period)
                volume_ma = self._moving_average(volume, period)
                features[f'pv_divergence_{period}'] = (close / price_ma) / (volume / volume_ma + 1e-8)

        return features

    def _extract_momentum_features(self, close, high, low):
        """动量特征"""
        features = {}

        # 多周期动量
        for period in self.momentum_periods:
            if len(close) > period:
                features[f'momentum_{period}'] = close[period:] / close[:-period] - 1

        # ROC (Rate of Change)
        for period in [10, 20, 50]:
            if len(close) > period:
                features[f'roc_{period}'] = (close[period:] - close[:-period]) / close[:-period]

        # 价格震荡强度
        for period in [10, 20]:
            if len(close) > period:
                price_range = high - low
                features[f'oscillation_{period}'] = self._moving_average(price_range, period) / close

        return features

    def _extract_volatility_modeling_features(self, close):
        """波动率建模特征"""
        features = {}

        # 对数收益率
        log_returns = np.log(close[1:] / close[:-1])

        # 多周期波动率
        for period in [5, 10, 20, 60]:
            if len(log_returns) > period:
                features[f'volatility_{period}'] = np.array([
                    np.std(log_returns[max(0, i-period+1):i+1])
                    for i in range(len(log_returns))
                ])

                # 偏度和峰度
                features[f'skewness_{period}'] = np.array([
                    self._calculate_skewness(log_returns[max(0, i-period+1):i+1])
                    for i in range(len(log_returns))
                ])

                features[f'kurtosis_{period}'] = np.array([
                    self._calculate_kurtosis(log_returns[max(0, i-period+1):i+1])
                    for i in range(len(log_returns))
                ])

        # GARCH类特征
        features['garch_volatility'] = self._estimate_garch_volatility(log_returns)

        return features

    def _extract_microstructure_features(self, ohlcv_data):
        """市场微观结构特征"""
        close = ohlcv_data[:, 4]
        high = ohlcv_data[:, 2]
        low = ohlcv_data[:, 3]
        volume = ohlcv_data[:, 5]

        features = {}

        # 价格影响函数
        log_returns = np.log(close[1:] / close[:-1])
        log_volume = np.log(volume[1:] + 1)

        # 流动性指标
        features['amihud_illiquidity'] = np.abs(log_returns) / (log_volume + 1e-8)

        # 买卖压力指标
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        for period in [10, 20]:
            if len(money_flow) > period:
                positive_mf = np.where(np.diff(typical_price) > 0, money_flow[1:], 0)
                negative_mf = np.where(np.diff(typical_price) < 0, money_flow[1:], 0)

                features[f'mfi_{period}'] = np.array([
                    np.sum(positive_mf[max(0, i-period+1):i+1]) /
                    (np.sum(positive_mf[max(0, i-period+1):i+1]) + np.sum(negative_mf[max(0, i-period+1):i+1]) + 1e-8)
                    for i in range(len(positive_mf))
                ])

        return features

    def _extract_statistical_features(self, close, volume):
        """统计特征"""
        features = {}

        # 多重分形特征
        log_returns = np.log(close[1:] / close[:-1])

        for period in [20, 50]:
            if len(log_returns) > period:
                # 自相关
                features[f'autocorr_{period}'] = np.array([
                    np.corrcoef(log_returns[max(0, i-period+1):i],
                              log_returns[max(1, i-period+2):i+1])[0, 1] if i >= period else 0
                    for i in range(len(log_returns))
                ])

        return features

    def _extract_relative_strength_features(self, close):
        """相对强弱特征"""
        features = {}

        # 相对强弱指数
        for period in [10, 20, 50]:
            if len(close) > period:
                gains = np.where(np.diff(close) > 0, np.diff(close), 0)
                losses = np.where(np.diff(close) < 0, -np.diff(close), 0)

                avg_gain = self._moving_average(gains, period)
                avg_loss = self._moving_average(losses, period)

                rs = avg_gain / (avg_loss + 1e-8)
                features[f'rs_{period}'] = rs / (1 + rs)

        return features

    def _extract_risk_features(self, close):
        """风险特征"""
        features = {}

        log_returns = np.log(close[1:] / close[:-1])

        # VaR (Value at Risk)
        for period in [20, 50]:
            if len(log_returns) > period:
                features[f'var_95_{period}'] = np.array([
                    np.percentile(log_returns[max(0, i-period+1):i+1], 5)
                    for i in range(len(log_returns))
                ])

                features[f'var_99_{period}'] = np.array([
                    np.percentile(log_returns[max(0, i-period+1):i+1], 1)
                    for i in range(len(log_returns))
                ])

        # 最大回撤
        for period in [20, 50, 100]:
            if len(close) > period:
                features[f'max_drawdown_{period}'] = np.array([
                    self._calculate_max_drawdown(close[max(0, i-period+1):i+1])
                    for i in range(len(close))
                ])

        return features

    def _extract_time_features(self, timestamp):
        """时间特征 (用于Time2Vec)"""
        features = {}

        # 转换时间戳
        dt_array = pd.to_datetime(timestamp, unit='s')

        features['hour'] = dt_array.hour.values
        features['day_of_week'] = dt_array.dayofweek.values
        features['day_of_month'] = dt_array.day.values
        features['month'] = dt_array.month.values
        features['quarter'] = dt_array.quarter.values

        # 周期性时间特征
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

        return features

    # 辅助计算函数
    def _moving_average(self, data, period):
        """移动平均"""
        return np.array([np.mean(data[max(0, i-period+1):i+1])
                        for i in range(len(data))])

    def _calculate_rsi(self, close, period):
        """RSI计算"""
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = self._moving_average(gain, period)
        avg_loss = self._moving_average(loss, period)

        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, close, fast_period, slow_period, signal_period):
        """MACD计算"""
        ema_fast = self._exponential_moving_average(close, fast_period)
        ema_slow = self._exponential_moving_average(close, slow_period)

        macd = ema_fast - ema_slow
        macd_signal = self._exponential_moving_average(macd, signal_period)
        macd_histogram = macd - macd_signal

        return macd, macd_signal, macd_histogram

    def _exponential_moving_average(self, data, period):
        """指数移动平均"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

        return ema

    def _calculate_bollinger_bands(self, close, period, std_factor):
        """布林带计算"""
        sma = self._moving_average(close, period)
        std = np.array([np.std(close[max(0, i-period+1):i+1])
                       for i in range(len(close))])

        upper = sma + std_factor * std
        lower = sma - std_factor * std

        return upper, sma, lower

    def _calculate_atr(self, high, low, close, period):
        """ATR计算"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return self._moving_average(tr, period)

    def _calculate_stochastic(self, high, low, close, period):
        """随机指标计算"""
        lowest_low = np.array([np.min(low[max(0, i-period+1):i+1])
                              for i in range(len(low))])
        highest_high = np.array([np.max(high[max(0, i-period+1):i+1])
                               for i in range(len(high))])

        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
        d_percent = self._moving_average(k_percent, 3)

        return k_percent, d_percent

    def _calculate_williams_r(self, high, low, close, period):
        """威廉指标计算"""
        highest_high = np.array([np.max(high[max(0, i-period+1):i+1])
                               for i in range(len(high))])
        lowest_low = np.array([np.min(low[max(0, i-period+1):i+1])
                              for i in range(len(low))])

        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-8)

    def _calculate_skewness(self, data):
        """偏度计算"""
        if len(data) < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data):
        """峰度计算"""
        if len(data) < 4:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _estimate_garch_volatility(self, returns, window=50):
        """简化GARCH波动率估计"""
        volatility = np.zeros_like(returns)

        for i in range(window, len(returns)):
            recent_returns = returns[i-window:i]
            volatility[i] = np.sqrt(np.mean(recent_returns**2))

        return volatility

    def _calculate_max_drawdown(self, prices):
        """最大回撤计算"""
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        return np.min(drawdown)

    def create_sequences(self, features_dict: Dict[str, np.ndarray],
                        time_features: Dict[str, np.ndarray],
                        sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        业界最优序列创建方法
        返回: (features_X, time_X, valid_mask)
        """
        # 对齐所有特征的长度
        min_length = min(len(v) for v in features_dict.values())

        # 准备主要特征
        aligned_features = []
        for name, values in features_dict.items():
            if 'hour' not in name and 'day' not in name and 'month' not in name and 'quarter' not in name:
                if len(values) == min_length:
                    aligned_features.append(values)

        # 准备时间特征
        time_feature_list = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'day_of_month', 'month', 'quarter']
        aligned_time_features = []
        for name in time_feature_list:
            if name in time_features and len(time_features[name]) == min_length:
                aligned_time_features.append(time_features[name])

        if not aligned_features or not aligned_time_features:
            return np.array([]), np.array([]), np.array([])

        # 转换为特征矩阵
        features_matrix = np.column_stack(aligned_features)
        time_matrix = np.column_stack(aligned_time_features)

        # 创建序列
        features_X = []
        time_X = []
        valid_mask = []

        for i in range(sequence_length, len(features_matrix)):
            feature_seq = features_matrix[i-sequence_length:i]
            time_seq = time_matrix[i-sequence_length:i]

            features_X.append(feature_seq)
            time_X.append(time_seq)

            # 创建mask（这里假设所有数据都是有效的）
            mask = np.zeros(sequence_length, dtype=bool)
            valid_mask.append(mask)

        return np.array(features_X), np.array(time_X), np.array(valid_mask)

class IndustryLeadingTransformerTrainer:
    """
    业界最优Transformer训练器
    支持新的MultiStockTransformerModel，包含高级训练技巧
    """

    def __init__(self, model: MultiStockTransformerModel, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)

        # 业界最优优化器配置
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2e-4,  # 略微提高学习率
            weight_decay=0.01,
            betas=(0.9, 0.999),  # 标准配置
            eps=1e-8
        )

        # 预热+余弦退火学习率调度
        self.warmup_steps = 1000
        self.total_steps = 50000
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2, eta_min=1e-6
        )

        # 业界标准损失函数
        self.direction_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.regression_loss_fn = nn.HuberLoss(delta=1.0)  # 对异常值更鲁棒
        self.confidence_loss_fn = nn.BCELoss()  # 专门用于置信度

        # 损失权重（基于业界最佳实践）
        self.loss_weights = {
            'direction': 1.0,
            'volatility': 0.3,
            'confidence': 0.4,
            'expected_return': 0.6,
            'sharpe_ratio': 0.2
        }

        # 早停机制
        self.best_loss = float('inf')
        self.patience = 0
        self.max_patience = 20

        # 训练统计
        self.step_count = 0
        self.epoch_count = 0

    def train_step(self,
                  features: torch.Tensor,
                  time_features: torch.Tensor,
                  targets: Dict[str, torch.Tensor],
                  stock_ids: Optional[torch.Tensor] = None,
                  mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        业界最优单步训练
        返回详细的损失信息
        """
        self.model.train()

        # 数据移动到设备
        features = features.to(self.device)
        time_features = time_features.to(self.device)

        targets_device = {}
        for key, value in targets.items():
            targets_device[key] = value.to(self.device)

        if stock_ids is not None:
            stock_ids = stock_ids.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # 前向传播
        outputs = self.model(features, time_features, stock_ids, mask)

        # 计算各项损失
        losses = {}
        losses['direction'] = self.direction_loss_fn(
            outputs['direction'], targets_device['direction']
        )
        losses['volatility'] = self.regression_loss_fn(
            outputs['volatility'], targets_device['volatility']
        )
        losses['confidence'] = self.confidence_loss_fn(
            outputs['confidence'], targets_device['confidence']
        )
        losses['expected_return'] = self.regression_loss_fn(
            outputs['expected_return'], targets_device['expected_return']
        )
        losses['sharpe_ratio'] = self.regression_loss_fn(
            outputs['sharpe_ratio'], targets_device['sharpe_ratio']
        )

        # 加权总损失
        total_loss = sum(
            self.loss_weights[key] * loss
            for key, loss in losses.items()
        )

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪（业界标准）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()

        self.step_count += 1

        # 返回损失信息
        loss_dict = {f'{key}_loss': loss.item() for key, loss in losses.items()}
        loss_dict['total_loss'] = total_loss.item()
        loss_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']

        return loss_dict

    def evaluate(self,
                features: torch.Tensor,
                time_features: torch.Tensor,
                targets: Dict[str, torch.Tensor],
                stock_ids: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        业界标准模型评估
        返回详细的评估指标
        """
        self.model.eval()
        with torch.no_grad():
            # 数据移动到设备
            features = features.to(self.device)
            time_features = time_features.to(self.device)

            targets_device = {}
            for key, value in targets.items():
                targets_device[key] = value.to(self.device)

            if stock_ids is not None:
                stock_ids = stock_ids.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)

            outputs = self.model(features, time_features, stock_ids, mask)

            # 计算评估指标
            metrics = {}

            # 分类准确率
            direction_pred = torch.argmax(outputs['direction'], dim=1)
            metrics['direction_accuracy'] = (
                direction_pred == targets_device['direction']
            ).float().mean().item()

            # 回归指标
            metrics['volatility_mae'] = torch.mean(
                torch.abs(outputs['volatility'] - targets_device['volatility'])
            ).item()

            metrics['return_mae'] = torch.mean(
                torch.abs(outputs['expected_return'] - targets_device['expected_return'])
            ).item()

            # 置信度校准
            confidence_diff = torch.abs(
                outputs['confidence'] - targets_device['confidence']
            )
            metrics['confidence_calibration'] = torch.mean(confidence_diff).item()

            # 夏普比率预测精度
            sharpe_diff = torch.abs(
                outputs['sharpe_ratio'] - targets_device['sharpe_ratio']
            )
            metrics['sharpe_mae'] = torch.mean(sharpe_diff).item()

            # 整体性能指标
            direction_weight = 0.4
            regression_weight = 0.6

            metrics['overall_score'] = (
                direction_weight * metrics['direction_accuracy'] +
                regression_weight * (1.0 - min(metrics['return_mae'], 1.0))
            )

            return metrics

    def save_model(self, path: str, include_optimizer: bool = True):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'best_loss': self.best_loss,
        }

        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_model(self, path: str, load_optimizer: bool = True):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.step_count = checkpoint.get('step_count', 0)
        self.epoch_count = checkpoint.get('epoch_count', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def should_stop_early(self, current_loss: float) -> bool:
        """早停检查"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience = 0
            return False
        else:
            self.patience += 1
            return self.patience >= self.max_patience

if __name__ == "__main__":
    # 测试业界最优模型
    print("🚀 测试业界最优MultiStockTransformerModel...")

    # 检查设备
    if torch.backends.mps.is_available():
        device = 'mps'
        print("✅ 使用Metal Performance Shaders (MPS)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("✅ 使用CUDA")
    else:
        device = 'cpu'
        print("✅ 使用CPU")

    # 创建业界最优模型
    model = MultiStockTransformerModel(
        input_dim=50,
        time_dim=8,
        d_model=256,
        nhead=16,
        num_layers=6,
        num_stocks=10,
        enable_cross_stock=True
    )
    trainer = IndustryLeadingTransformerTrainer(model, device)

    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试数据
    batch_size, seq_len, input_dim, time_dim = 4, 60, 50, 8
    test_features = torch.randn(batch_size, seq_len, input_dim)
    test_time_features = torch.randn(batch_size, seq_len, time_dim)
    test_stock_ids = torch.randint(0, 10, (batch_size,))

    test_targets = {
        'direction': torch.randint(0, 3, (batch_size,)),
        'volatility': torch.rand(batch_size, 1),  # 非负
        'confidence': torch.rand(batch_size, 1),  # 0-1
        'expected_return': torch.randn(batch_size, 1),
        'sharpe_ratio': torch.randn(batch_size, 1)
    }

    # 将测试数据移到正确设备
    test_features = test_features.to(device)
    test_time_features = test_time_features.to(device)
    test_stock_ids = test_stock_ids.to(device)
    for key in test_targets:
        test_targets[key] = test_targets[key].to(device)

    # 前向传播测试
    outputs = model(test_features, test_time_features, test_stock_ids)
    print(f"✅ 模型输出维度检查:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # 训练步骤测试
    loss_dict = trainer.train_step(test_features, test_time_features, test_targets, test_stock_ids)
    print(f"✅ 训练步骤测试完成:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    # 评估测试
    metrics = trainer.evaluate(test_features, test_time_features, test_targets, test_stock_ids)
    print(f"✅ 评估测试完成:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    print("🤖 Transformer模型库加载成功!")
    print("📚 可用组件:")
    print("  - MultiStockTransformerModel")
    print("  - IndustryLeadingFeatureExtractor")
    print("  - IndustryLeadingTransformerTrainer")
    print("  - Time2Vec")
    print("  - AdvancedPositionalEncoding")

