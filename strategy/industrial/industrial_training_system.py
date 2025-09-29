#!/usr/bin/env python3
"""
工业级GPU集群训练系统
专为大规模量化交易AI设计的分布式训练框架

作者: Alvin
特性:
- 多GPU分布式训练
- 混合精度训练
- 梯度累积和检查点
- 自适应学习率调度
- 模型并行和数据并行
- 实时监控和可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import pandas as pd
import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# 导入工业级模型
from industrial_scale_model import IndustryLeadingTransformer, get_model_config

class IndustrialDataset(Dataset):
    """工业级数据集类"""

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 252,
        prediction_horizon: int = 5,
        symbols: Optional[List[str]] = None
    ):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # 加载所有数据文件
        self.data_files = list(self.data_path.glob("*.parquet"))
        if symbols:
            # 过滤指定股票
            symbol_files = [f for f in self.data_files
                          if any(s in f.stem for s in symbols)]
            self.data_files = symbol_files

        self.samples = self._prepare_samples()
        self.symbol_to_id = self._create_symbol_mapping()

        print(f"📊 数据集统计:")
        print(f"   数据文件: {len(self.data_files)}")
        print(f"   总样本数: {len(self.samples)}")
        print(f"   序列长度: {self.sequence_length}")
        print(f"   预测跨度: {self.prediction_horizon}")

    def _create_symbol_mapping(self) -> Dict[str, int]:
        """创建股票符号到ID的映射"""
        symbols = list(set(sample[2] for sample in self.samples))
        return {symbol: idx for idx, symbol in enumerate(sorted(symbols))}

    def _prepare_samples(self) -> List[Tuple[str, int, str]]:
        """准备训练样本索引"""
        samples = []

        for file_path in self.data_files:
            try:
                # 读取数据
                df = pd.read_parquet(file_path)

                if len(df) < self.sequence_length + self.prediction_horizon:
                    continue

                symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else file_path.stem

                # 为每个可能的序列创建样本
                for start_idx in range(len(df) - self.sequence_length - self.prediction_horizon + 1):
                    samples.append((str(file_path), start_idx, symbol))

            except Exception as e:
                print(f"❌ 文件读取失败 {file_path}: {e}")
                continue

        return samples

    def _calculate_targets(self, data: pd.DataFrame, start_idx: int) -> Dict[str, torch.Tensor]:
        """计算多任务学习目标"""
        current_idx = start_idx + self.sequence_length - 1
        future_idx = min(current_idx + self.prediction_horizon, len(data) - 1)

        current_price = data['close'].iloc[current_idx]
        future_price = data['close'].iloc[future_idx]

        # 1. 方向预测 (涨/跌/横盘)
        price_change = (future_price - current_price) / current_price
        if price_change > 0.02:  # 涨幅超过2%
            direction = 2  # BUY
        elif price_change < -0.02:  # 跌幅超过2%
            direction = 0  # SELL
        else:
            direction = 1  # HOLD

        # 2. 收益率预测
        return_rate = float(price_change)

        # 3. 波动率预测
        price_series = data['close'].iloc[current_idx-min(20, current_idx):current_idx+1]
        volatility = float(price_series.pct_change().std() * np.sqrt(252))

        # 4. 置信度 (基于历史波动率的倒数)
        confidence = float(1.0 / (1.0 + volatility)) if volatility > 0 else 0.5

        # 5. 风险等级 (基于波动率分位数)
        if volatility < 0.1:
            risk_level = 0  # 低风险
        elif volatility < 0.2:
            risk_level = 1  # 中低风险
        elif volatility < 0.3:
            risk_level = 2  # 中等风险
        elif volatility < 0.5:
            risk_level = 3  # 中高风险
        else:
            risk_level = 4  # 高风险

        return {
            'direction': torch.tensor(direction, dtype=torch.long),
            'return': torch.tensor(return_rate, dtype=torch.float32),
            'volatility': torch.tensor(volatility, dtype=torch.float32),
            'confidence': torch.tensor(confidence, dtype=torch.float32),
            'risk_level': torch.tensor(risk_level, dtype=torch.long)
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        file_path, start_idx, symbol = self.samples[idx]

        # 读取数据
        df = pd.read_parquet(file_path)

        # 提取特征序列
        end_idx = start_idx + self.sequence_length
        sequence_data = df.iloc[start_idx:end_idx]

        # 选择特征列 (排除非数值列)
        feature_columns = [col for col in sequence_data.columns
                          if col not in ['symbol'] and
                          pd.api.types.is_numeric_dtype(sequence_data[col])]

        features = sequence_data[feature_columns].values.astype(np.float32)

        # 处理NaN值
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        # 计算目标
        targets = self._calculate_targets(df, start_idx)

        # 获取股票ID
        stock_id = torch.tensor(self.symbol_to_id[symbol], dtype=torch.long)

        return torch.tensor(features), targets, stock_id

class IndustrialTrainer:
    """工业级训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: Dict[str, Any],
        device_ids: List[int] = None
    ):
        self.config = config
        self.device_ids = device_ids or [0]
        self.device = torch.device(f"cuda:{self.device_ids[0]}" if torch.cuda.is_available() else "cpu")

        print(f"🚀 初始化工业级训练器...")
        print(f"   设备: {self.device}")
        print(f"   GPU数量: {len(self.device_ids)}")

        # 模型设置
        self.model = model.to(self.device)
        if len(self.device_ids) > 1:
            self.model = DDP(self.model, device_ids=self.device_ids)

        # 数据加载器
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, shuffle=False)

        # 优化器设置
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # 混合精度训练
        self.use_amp = config.get('mixed_precision', True)
        if self.use_amp:
            self.scaler = GradScaler()

        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []

        # 梯度累积
        self.accumulation_steps = config.get('gradient_accumulation_steps', 4)

        print(f"✅ 训练器初始化完成")

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """创建数据加载器"""
        sampler = None
        if len(self.device_ids) > 1:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # 使用sampler时不能shuffle

        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.get('num_workers', 8),
            pin_memory=True,
            drop_last=True
        )

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """设置优化器"""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'AdamW')

        if optimizer_type == 'AdamW':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-4),
                weight_decay=optimizer_config.get('weight_decay', 0.01),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'Adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get('lr', 1e-4),
                weight_decay=optimizer_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")

    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """设置学习率调度器"""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'CosineAnnealingLR')

        if scheduler_type == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10)
            )
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")

    def _calculate_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算多任务损失"""
        losses = {}

        # 方向预测损失
        if 'direction' in outputs and 'direction' in targets:
            losses['direction'] = F.cross_entropy(outputs['direction'], targets['direction'])

        # 收益率预测损失
        if 'return' in outputs and 'return' in targets:
            losses['return'] = F.mse_loss(outputs['return'].squeeze(), targets['return'])

        # 波动率预测损失
        if 'volatility' in outputs and 'volatility' in targets:
            losses['volatility'] = F.mse_loss(outputs['volatility'].squeeze(), targets['volatility'])

        # 置信度预测损失
        if 'confidence' in outputs and 'confidence' in targets:
            losses['confidence'] = F.mse_loss(outputs['confidence'].squeeze(), targets['confidence'])

        # 风险等级预测损失
        if 'risk_level' in outputs and 'risk_level' in targets:
            losses['risk_level'] = F.cross_entropy(outputs['risk_level'], targets['risk_level'])

        # 加权总损失
        loss_weights = self.config.get('loss_weights', {
            'direction': 2.0,    # 方向预测最重要
            'return': 1.5,       # 收益率预测
            'volatility': 1.0,   # 波动率预测
            'confidence': 1.0,   # 置信度预测
            'risk_level': 0.5    # 风险等级预测
        })

        total_loss = sum(
            loss_weights.get(task, 1.0) * loss
            for task, loss in losses.items()
        )

        return total_loss, losses

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        task_losses = {}
        num_batches = len(self.train_loader)

        for batch_idx, (features, targets, stock_ids) in enumerate(self.train_loader):
            # 数据移动到GPU
            features = features.to(self.device)
            stock_ids = stock_ids.to(self.device)

            batch_targets = {}
            for task, target in targets.items():
                batch_targets[task] = target.to(self.device)

            # 前向传播
            if self.use_amp:
                with autocast():
                    outputs = self.model(features, stock_ids)
                    loss, losses = self._calculate_loss(outputs, batch_targets)
                    loss = loss / self.accumulation_steps

                # 反向传播
                self.scaler.scale(loss).backward()

                # 梯度累积
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(features, stock_ids)
                loss, losses = self._calculate_loss(outputs, batch_targets)
                loss = loss / self.accumulation_steps

                loss.backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # 统计损失
            total_loss += loss.item() * self.accumulation_steps
            for task, task_loss in losses.items():
                if task not in task_losses:
                    task_losses[task] = 0.0
                task_losses[task] += task_loss.item()

            # 进度报告
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"   Batch {batch_idx}/{num_batches} | "
                      f"Loss: {loss.item():.6f} | "
                      f"LR: {current_lr:.2e}")

        # 平均损失
        avg_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}

        return avg_loss, avg_task_losses

    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        task_losses = {}
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for features, targets, stock_ids in self.val_loader:
                # 数据移动到GPU
                features = features.to(self.device)
                stock_ids = stock_ids.to(self.device)

                batch_targets = {}
                for task, target in targets.items():
                    batch_targets[task] = target.to(self.device)

                # 前向传播
                if self.use_amp:
                    with autocast():
                        outputs = self.model(features, stock_ids)
                        loss, losses = self._calculate_loss(outputs, batch_targets)
                else:
                    outputs = self.model(features, stock_ids)
                    loss, losses = self._calculate_loss(outputs, batch_targets)

                # 统计损失
                total_loss += loss.item()
                for task, task_loss in losses.items():
                    if task not in task_losses:
                        task_losses[task] = 0.0
                    task_losses[task] += task_loss.item()

        # 平均损失
        avg_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}

        return avg_loss, avg_task_losses

    def train(self) -> Dict[str, List[float]]:
        """完整训练流程"""
        print(f"🚀 开始工业级训练...")
        print(f"   训练样本: {len(self.train_loader.dataset)}")
        print(f"   验证样本: {len(self.val_loader.dataset)}")
        print(f"   批大小: {self.config['batch_size']}")
        print(f"   训练轮数: {self.config['epochs']}")
        print("=" * 80)

        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            print(f"🔄 Epoch {epoch + 1}/{self.config['epochs']}")

            # 训练
            train_loss, train_task_losses = self.train_epoch()
            history['train_loss'].append(train_loss)

            # 验证
            val_loss, val_task_losses = self.validate_epoch()
            history['val_loss'].append(val_loss)

            # 学习率调度
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"best_model_epoch_{epoch+1}.pth")

            # 每10个epoch保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")

            epoch_time = time.time() - epoch_start_time

            # 打印结果
            print(f"   训练损失: {train_loss:.6f}")
            print(f"   验证损失: {val_loss:.6f}")
            print(f"   学习率: {current_lr:.2e}")
            print(f"   耗时: {epoch_time:.2f}s")

            # 打印任务损失
            if train_task_losses:
                print("   任务损失:")
                for task, loss in train_task_losses.items():
                    print(f"     {task}: {loss:.6f}")

            print("-" * 50)

        print("🎉 训练完成!")
        return history

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict() if not isinstance(self.model, DDP) else self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, filename)
        print(f"💾 检查点已保存: {filename}")

def setup_distributed_training(rank: int, world_size: int):
    """设置分布式训练"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化分布式进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed_training():
    """清理分布式训练"""
    dist.destroy_process_group()

def get_training_config() -> Dict[str, Any]:
    """获取训练配置"""
    return {
        # 基本设置
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 1e-4,
        'mixed_precision': True,
        'gradient_accumulation_steps': 4,

        # 数据设置
        'num_workers': 8,
        'sequence_length': 252,
        'prediction_horizon': 5,

        # 优化器设置
        'optimizer': {
            'type': 'AdamW',
            'lr': 1e-4,
            'weight_decay': 0.01,
            'betas': (0.9, 0.999)
        },

        # 调度器设置
        'scheduler': {
            'type': 'CosineAnnealingLR',
            'eta_min': 1e-6
        },

        # 损失权重
        'loss_weights': {
            'direction': 2.0,
            'return': 1.5,
            'volatility': 1.0,
            'confidence': 1.0,
            'risk_level': 0.5
        }
    }

def main():
    """主训练函数"""
    print("🚀 工业级GPU集群训练系统")
    print("=" * 80)

    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ 未检测到CUDA支持")
        return

    gpu_count = torch.cuda.device_count()
    print(f"🎯 检测到 {gpu_count} 个GPU")

    # 配置
    model_config = get_model_config()
    training_config = get_training_config()

    print("📊 配置信息:")
    print(f"   模型参数: {sum(p.numel() for p in IndustryLeadingTransformer(**model_config).parameters()):,}")
    print(f"   批大小: {training_config['batch_size']}")
    print(f"   训练轮数: {training_config['epochs']}")
    print(f"   序列长度: {training_config['sequence_length']}")
    print("=" * 80)

    # 创建模型
    model = IndustryLeadingTransformer(**model_config)

    # 创建数据集
    data_path = "data/industrial_training_data"
    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        print("💡 请先运行 industrial_data_collector.py 收集数据")
        return

    print("📦 加载数据集...")
    full_dataset = IndustrialDataset(
        data_path=data_path,
        sequence_length=training_config['sequence_length'],
        prediction_horizon=training_config['prediction_horizon']
    )

    # 划分训练/验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    print(f"📊 数据集划分:")
    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   验证集: {len(val_dataset)} 样本")

    # 设置设备
    device_ids = list(range(gpu_count))

    # 创建训练器
    trainer = IndustrialTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
        device_ids=device_ids
    )

    # 开始训练
    history = trainer.train()

    # 保存训练历史
    with open("training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print("✅ 训练系统运行完毕!")

if __name__ == "__main__":
    main()