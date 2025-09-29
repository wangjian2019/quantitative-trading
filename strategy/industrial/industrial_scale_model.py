#!/usr/bin/env python3
"""
工业级大规模量化交易AI模型
专为GPU集群设计的高性能架构

作者: Alvin
目标: 1000只热门美股港股 + 5年历史数据 + 最先进的模型架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import pandas as pd
import math
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings("ignore")

class PositionalEncoding(nn.Module):
    """增强位置编码"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Time2Vec(nn.Module):
    """Time2Vec时间编码层"""
    def __init__(self, input_dim: int, time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim

        # 线性变换层
        self.linear_layer = nn.Linear(input_dim, 1)

        # 周期性变换层
        self.periodic_layers = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(time_dim - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 线性部分
        linear_output = self.linear_layer(x)

        # 周期性部分
        periodic_outputs = []
        for layer in self.periodic_layers:
            periodic_outputs.append(torch.sin(layer(x)))

        # 拼接
        time_encoding = torch.cat([linear_output] + periodic_outputs, dim=-1)
        return time_encoding

class MultiHeadCrossStockAttention(nn.Module):
    """多头跨股票注意力机制"""
    def __init__(self, d_model: int, num_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)

        # 线性变换
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 拼接多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)

        # 最终线性变换
        output = self.w_o(attention_output)

        # 残差连接和层归一化
        output = self.layer_norm(output + query)

        return output, attention_weights

class IndustryLeadingFeatureExtractor(nn.Module):
    """工业级特征提取器"""
    def __init__(self, raw_features: int = 200):
        super().__init__()

        # 技术指标特征提取
        self.technical_extractor = nn.Sequential(
            nn.Linear(raw_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 价格动量特征
        self.momentum_extractor = nn.Sequential(
            nn.Conv1d(in_channels=raw_features, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 波动率特征
        self.volatility_extractor = nn.Sequential(
            nn.Linear(raw_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(256 + 64 + 64, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = x.shape

        # 技术指标特征
        tech_features = self.technical_extractor(x)  # [B, T, 256]

        # 动量特征 (需要转换维度)
        x_transposed = x.transpose(1, 2)  # [B, F, T]
        momentum_features = self.momentum_extractor(x_transposed)  # [B, 64, 1]
        momentum_features = momentum_features.squeeze(-1).unsqueeze(1)  # [B, 1, 64]
        momentum_features = momentum_features.expand(-1, seq_len, -1)  # [B, T, 64]

        # 波动率特征
        volatility_features = self.volatility_extractor(x)  # [B, T, 64]

        # 特征融合
        combined_features = torch.cat([tech_features, momentum_features, volatility_features], dim=-1)
        final_features = self.feature_fusion(combined_features)

        return final_features

class AdvancedTransformerBlock(nn.Module):
    """高级Transformer模块"""
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()

        # 多头注意力
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # 跨股票注意力
        self.cross_stock_attention = MultiHeadCrossStockAttention(d_model, num_heads, dropout)

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力
        attn_output, _ = self.self_attention(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 跨股票注意力
        cross_attn_output, _ = self.cross_stock_attention(x, x, x)
        x = self.norm2(x + cross_attn_output)

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x

class IndustryLeadingTransformer(nn.Module):
    """工业级大规模Transformer模型"""
    def __init__(
        self,
        num_stocks: int = 1000,
        raw_features: int = 200,
        d_model: int = 1024,
        num_heads: int = 16,
        num_layers: int = 12,
        dim_feedforward: int = 4096,
        seq_len: int = 252,  # 一年交易日
        dropout: float = 0.1,
        num_tasks: int = 5,  # 多任务学习
    ):
        super().__init__()

        self.num_stocks = num_stocks
        self.d_model = d_model
        self.seq_len = seq_len

        # 股票嵌入
        self.stock_embedding = nn.Embedding(num_stocks, d_model // 4)

        # 特征提取器
        self.feature_extractor = IndustryLeadingFeatureExtractor(raw_features)

        # Time2Vec时间编码
        self.time2vec = Time2Vec(256, d_model // 4)

        # 特征投影层
        self.feature_projection = nn.Linear(256, d_model // 2)

        # 特征融合层
        self.feature_fusion = nn.Linear(d_model, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout, seq_len)

        # Transformer层
        self.transformer_layers = nn.ModuleList([
            AdvancedTransformerBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 多任务预测头
        self.task_heads = nn.ModuleDict({
            'direction': self._create_prediction_head(d_model, 3),      # 涨/跌/横盘
            'volatility': self._create_prediction_head(d_model, 1),     # 波动率预测
            'return': self._create_prediction_head(d_model, 1),         # 收益率预测
            'confidence': self._create_prediction_head(d_model, 1),     # 置信度
            'risk_level': self._create_prediction_head(d_model, 5),     # 风险等级
        })

        # 初始化权重
        self._init_weights()

    def _create_prediction_head(self, d_model: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, output_dim)
        )

    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, stock_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # 特征提取
        features = self.feature_extractor(x)  # [B, T, 256]

        # Time2Vec编码
        time_features = self.time2vec(features)  # [B, T, d_model//4]

        # 股票嵌入
        stock_emb = self.stock_embedding(stock_ids)  # [B, d_model//4]
        stock_emb = stock_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, d_model//4]

        # 特征投影
        projected_features = self.feature_projection(features)  # [B, T, d_model//2]

        # 特征拼接
        combined_features = torch.cat([
            projected_features,
            time_features,
            stock_emb
        ], dim=-1)  # [B, T, d_model]

        # 特征融合
        x = self.feature_fusion(combined_features)

        # 位置编码
        x = self.pos_encoding(x)

        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x)

        # 全局表示 - 使用多种聚合方式
        # 1. 最后时间步
        last_hidden = x[:, -1, :]

        # 2. 平均池化
        avg_hidden = torch.mean(x, dim=1)

        # 3. 最大池化
        max_hidden, _ = torch.max(x, dim=1)

        # 4. 注意力加权池化
        attention_weights = F.softmax(
            torch.sum(x * x[:, -1:, :], dim=-1), dim=1
        )  # [B, T]
        weighted_hidden = torch.sum(
            x * attention_weights.unsqueeze(-1), dim=1
        )  # [B, d_model]

        # 组合所有表示
        final_representation = (
            last_hidden + avg_hidden + max_hidden + weighted_hidden
        ) / 4

        # 多任务预测
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(final_representation)

        return outputs

class IndustrialTrainingEngine:
    """工业级训练引擎"""
    def __init__(
        self,
        model: nn.Module,
        device_ids: List[int],
        mixed_precision: bool = True,
        gradient_checkpointing: bool = True
    ):
        self.model = model
        self.device_ids = device_ids
        self.mixed_precision = mixed_precision

        # 分布式训练设置
        if len(device_ids) > 1:
            self.model = DDP(model, device_ids=device_ids)

        # 混合精度训练
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        # 梯度检查点
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def train_epoch(self, dataloader, optimizer, scheduler, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch_idx, (x, y, stock_ids) in enumerate(dataloader):
            x = x.cuda()
            stock_ids = stock_ids.cuda()

            # 多任务标签
            y_direction = y['direction'].cuda()
            y_volatility = y['volatility'].cuda()
            y_return = y['return'].cuda()
            y_confidence = y['confidence'].cuda()
            y_risk = y['risk_level'].cuda()

            optimizer.zero_grad()

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(x, stock_ids)

                    # 多任务损失
                    loss_direction = F.cross_entropy(outputs['direction'], y_direction)
                    loss_volatility = F.mse_loss(outputs['volatility'].squeeze(), y_volatility)
                    loss_return = F.mse_loss(outputs['return'].squeeze(), y_return)
                    loss_confidence = F.mse_loss(outputs['confidence'].squeeze(), y_confidence)
                    loss_risk = F.cross_entropy(outputs['risk_level'], y_risk)

                    # 总损失
                    total_batch_loss = (
                        2.0 * loss_direction +      # 方向预测权重最高
                        1.5 * loss_return +         # 收益率预测
                        1.0 * loss_volatility +     # 波动率预测
                        1.0 * loss_confidence +     # 置信度预测
                        0.5 * loss_risk             # 风险等级预测
                    )

                self.scaler.scale(total_batch_loss).backward()

                # 梯度裁剪
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(x, stock_ids)

                # 计算损失（同上）
                loss_direction = F.cross_entropy(outputs['direction'], y_direction)
                loss_volatility = F.mse_loss(outputs['volatility'].squeeze(), y_volatility)
                loss_return = F.mse_loss(outputs['return'].squeeze(), y_return)
                loss_confidence = F.mse_loss(outputs['confidence'].squeeze(), y_confidence)
                loss_risk = F.cross_entropy(outputs['risk_level'], y_risk)

                total_batch_loss = (
                    2.0 * loss_direction + 1.5 * loss_return +
                    1.0 * loss_volatility + 1.0 * loss_confidence + 0.5 * loss_risk
                )

                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            total_loss += total_batch_loss.item()

            # 打印进度
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                      f"Loss: {total_batch_loss.item():.6f}")

        return total_loss / num_batches

def create_industrial_model(config: Dict) -> IndustryLeadingTransformer:
    """创建工业级模型"""
    return IndustryLeadingTransformer(
        num_stocks=config.get('num_stocks', 1000),
        raw_features=config.get('raw_features', 200),
        d_model=config.get('d_model', 1024),
        num_heads=config.get('num_heads', 16),
        num_layers=config.get('num_layers', 12),
        dim_feedforward=config.get('dim_feedforward', 4096),
        seq_len=config.get('seq_len', 252),
        dropout=config.get('dropout', 0.1),
        num_tasks=config.get('num_tasks', 5)
    )

def get_model_config() -> Dict:
    """获取模型配置"""
    return {
        'num_stocks': 1000,      # 1000只股票
        'raw_features': 200,     # 200个原始特征
        'd_model': 1024,         # 模型维度
        'num_heads': 16,         # 注意力头数
        'num_layers': 12,        # Transformer层数
        'dim_feedforward': 4096, # 前馈网络维度
        'seq_len': 252,          # 序列长度（一年）
        'dropout': 0.1,          # Dropout率
        'num_tasks': 5           # 多任务数量
    }

if __name__ == "__main__":
    print("🚀 工业级量化交易AI模型架构")
    print("=" * 60)

    config = get_model_config()
    model = create_industrial_model(config)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📊 模型参数统计:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print("=" * 60)

    # 模型架构信息
    print(f"🏗️  模型架构:")
    print(f"   股票数量: {config['num_stocks']}")
    print(f"   特征维度: {config['raw_features']}")
    print(f"   模型维度: {config['d_model']}")
    print(f"   注意力头: {config['num_heads']}")
    print(f"   层数: {config['num_layers']}")
    print(f"   序列长度: {config['seq_len']}")
    print(f"   多任务数: {config['num_tasks']}")
    print("=" * 60)
    print("✅ 模型架构设计完成!")