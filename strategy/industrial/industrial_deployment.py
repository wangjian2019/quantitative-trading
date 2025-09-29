#!/usr/bin/env python3
"""
工业级模型部署系统
高性能量化交易AI推理服务

作者: Alvin
特性:
- 多GPU推理加速
- 模型并行部署
- 实时批处理推理
- 自动模型热更新
- 高可用服务架构
- 性能监控和负载均衡
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
import pandas as pd
import asyncio
import aiohttp
from aiohttp import web
import uvloop
import logging
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from datetime import datetime, timedelta
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import threading
from pathlib import Path
import psutil
import warnings
warnings.filterwarnings("ignore")

# 导入模型
from industrial_scale_model import IndustryLeadingTransformer, get_model_config

class ModelManager:
    """模型管理器"""

    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.models = {}  # GPU_ID -> model
        self.device_count = torch.cuda.device_count()
        self.current_model_version = None

        print(f"🚀 初始化模型管理器")
        print(f"   GPU数量: {self.device_count}")
        print(f"   模型路径: {model_path}")

        self.load_models()

    def load_models(self):
        """在所有GPU上加载模型"""
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，使用CPU模式")
            device = torch.device('cpu')
            model = IndustryLeadingTransformer(**self.config)
            model.load_state_dict(torch.load(self.model_path, map_location=device)['model_state_dict'])
            model.eval()
            self.models[0] = model
            return

        # 在每个GPU上加载模型副本
        for gpu_id in range(self.device_count):
            device = torch.device(f'cuda:{gpu_id}')
            model = IndustryLeadingTransformer(**self.config)

            # 加载模型权重
            checkpoint = torch.load(self.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            self.models[gpu_id] = model
            print(f"✅ GPU {gpu_id} 模型加载完成")

        self.current_model_version = datetime.now().isoformat()

    def get_model(self, gpu_id: Optional[int] = None) -> Tuple[nn.Module, torch.device]:
        """获取模型实例"""
        if gpu_id is None:
            # 自动选择负载最低的GPU
            gpu_id = self._select_optimal_gpu()

        if gpu_id not in self.models:
            gpu_id = 0

        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        return self.models[gpu_id], device

    def _select_optimal_gpu(self) -> int:
        """选择最优GPU"""
        if not torch.cuda.is_available():
            return 0

        # 简单轮询策略
        return int(time.time()) % self.device_count

    def update_model(self, new_model_path: str):
        """热更新模型"""
        print(f"🔄 开始模型热更新: {new_model_path}")
        try:
            # 创建新模型实例
            new_models = {}

            for gpu_id in range(self.device_count if torch.cuda.is_available() else 1):
                device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
                model = IndustryLeadingTransformer(**self.config)

                checkpoint = torch.load(new_model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval()

                new_models[gpu_id] = model

            # 原子替换
            old_models = self.models
            self.models = new_models
            self.model_path = new_model_path
            self.current_model_version = datetime.now().isoformat()

            print(f"✅ 模型热更新完成")

            # 清理旧模型
            del old_models

        except Exception as e:
            print(f"❌ 模型更新失败: {e}")

class PredictionBatch:
    """预测批次"""

    def __init__(self, batch_id: str, requests: List[Dict]):
        self.batch_id = batch_id
        self.requests = requests
        self.created_at = time.time()
        self.results = {}
        self.completed = False

class IndustrialInferenceEngine:
    """工业级推理引擎"""

    def __init__(self, model_manager: ModelManager, config: Dict[str, Any]):
        self.model_manager = model_manager
        self.config = config

        # 批处理设置
        self.batch_size = config.get('batch_size', 64)
        self.batch_timeout = config.get('batch_timeout', 0.1)  # 100ms
        self.max_sequence_length = config.get('max_sequence_length', 252)

        # 预处理队列
        self.request_queue = queue.Queue()
        self.batch_queue = queue.Queue()
        self.result_cache = {}

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=config.get('inference_workers', 4))

        # 启动批处理线程
        self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.batch_thread.start()

        # 启动推理线程
        for i in range(config.get('inference_workers', 4)):
            worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
            worker_thread.start()

        print(f"🚀 推理引擎初始化完成")
        print(f"   批大小: {self.batch_size}")
        print(f"   批超时: {self.batch_timeout}s")
        print(f"   推理工作线程: {config.get('inference_workers', 4)}")

    def _batch_processor(self):
        """批处理器"""
        current_batch = []
        last_batch_time = time.time()

        while True:
            try:
                # 尝试获取请求
                try:
                    request = self.request_queue.get(timeout=0.01)
                    current_batch.append(request)
                except queue.Empty:
                    request = None

                # 检查是否需要发送批次
                should_send_batch = (
                    len(current_batch) >= self.batch_size or
                    (current_batch and time.time() - last_batch_time > self.batch_timeout)
                )

                if should_send_batch and current_batch:
                    # 创建批次
                    batch_id = f"batch_{int(time.time() * 1000)}"
                    batch = PredictionBatch(batch_id, current_batch.copy())

                    # 发送到推理队列
                    self.batch_queue.put(batch)

                    # 重置
                    current_batch.clear()
                    last_batch_time = time.time()

            except Exception as e:
                print(f"❌ 批处理器错误: {e}")

    def _inference_worker(self):
        """推理工作线程"""
        while True:
            try:
                # 获取批次
                batch = self.batch_queue.get()

                # 执行推理
                self._process_batch(batch)

                # 标记完成
                self.batch_queue.task_done()

            except Exception as e:
                print(f"❌ 推理工作线程错误: {e}")

    def _process_batch(self, batch: PredictionBatch):
        """处理单个批次"""
        try:
            start_time = time.time()

            # 获取模型
            model, device = self.model_manager.get_model()

            # 准备批次数据
            batch_features = []
            batch_stock_ids = []
            valid_indices = []

            for i, request in enumerate(batch.requests):
                try:
                    features, stock_id = self._prepare_request_data(request)
                    batch_features.append(features)
                    batch_stock_ids.append(stock_id)
                    valid_indices.append(i)
                except Exception as e:
                    # 标记失败请求
                    batch.results[request['request_id']] = {
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }

            if not batch_features:
                return

            # 转换为张量
            batch_tensor = torch.stack(batch_features).to(device)
            stock_ids_tensor = torch.tensor(batch_stock_ids, dtype=torch.long).to(device)

            # 推理
            with torch.no_grad():
                if self.config.get('use_amp', True):
                    with autocast():
                        outputs = model(batch_tensor, stock_ids_tensor)
                else:
                    outputs = model(batch_tensor, stock_ids_tensor)

            # 处理结果
            for i, request_idx in enumerate(valid_indices):
                request = batch.requests[request_idx]
                result = self._process_model_output(outputs, i, request)
                batch.results[request['request_id']] = result

            inference_time = time.time() - start_time
            print(f"⚡ 批次 {batch.batch_id} 完成: {len(valid_indices)} 个请求, "
                  f"耗时 {inference_time:.3f}s")

            # 缓存结果
            for request_id, result in batch.results.items():
                self.result_cache[request_id] = result

            batch.completed = True

        except Exception as e:
            print(f"❌ 批次处理失败 {batch.batch_id}: {e}")

            # 标记所有请求失败
            for request in batch.requests:
                batch.results[request['request_id']] = {
                    'error': f"批次处理失败: {str(e)}",
                    'timestamp': datetime.now().isoformat()
                }

    def _prepare_request_data(self, request: Dict) -> Tuple[torch.Tensor, int]:
        """准备请求数据"""
        # 提取特征
        features = np.array(request['features'], dtype=np.float32)

        # 检查形状
        if features.shape[0] > self.max_sequence_length:
            features = features[-self.max_sequence_length:]
        elif features.shape[0] < self.max_sequence_length:
            # 填充
            padding = np.zeros((self.max_sequence_length - features.shape[0], features.shape[1]), dtype=np.float32)
            features = np.vstack([padding, features])

        # 处理NaN
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        # 股票ID
        stock_id = request.get('stock_id', 0)

        return torch.tensor(features), stock_id

    def _process_model_output(self, outputs: Dict[str, torch.Tensor], index: int, request: Dict) -> Dict:
        """处理模型输出"""
        result = {
            'timestamp': datetime.now().isoformat(),
            'model_version': self.model_manager.current_model_version,
            'symbol': request.get('symbol', 'UNKNOWN')
        }

        # 方向预测
        if 'direction' in outputs:
            direction_logits = outputs['direction'][index]
            direction_probs = torch.softmax(direction_logits, dim=0)
            predicted_direction = torch.argmax(direction_probs).item()

            direction_names = ['SELL', 'HOLD', 'BUY']
            result['action'] = direction_names[predicted_direction]
            result['confidence'] = float(torch.max(direction_probs))

        # 收益率预测
        if 'return' in outputs:
            result['expected_return'] = float(outputs['return'][index])

        # 波动率预测
        if 'volatility' in outputs:
            result['volatility'] = float(outputs['volatility'][index])

        # 置信度（如果模型输出）
        if 'confidence' in outputs:
            result['model_confidence'] = float(outputs['confidence'][index])

        # 风险等级
        if 'risk_level' in outputs:
            risk_logits = outputs['risk_level'][index]
            risk_probs = torch.softmax(risk_logits, dim=0)
            predicted_risk = torch.argmax(risk_probs).item()

            risk_names = ['LOW', 'MEDIUM_LOW', 'MEDIUM', 'MEDIUM_HIGH', 'HIGH']
            result['risk_level'] = risk_names[predicted_risk]
            result['risk_confidence'] = float(torch.max(risk_probs))

        return result

    async def predict(self, request: Dict) -> Dict:
        """异步预测接口"""
        request_id = f"req_{int(time.time() * 1000000)}"
        request['request_id'] = request_id

        # 加入队列
        self.request_queue.put(request)

        # 轮询结果
        max_wait_time = 10.0  # 最大等待时间
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            if request_id in self.result_cache:
                result = self.result_cache.pop(request_id)
                return result

            await asyncio.sleep(0.01)  # 10ms

        # 超时
        return {
            'error': 'Request timeout',
            'timestamp': datetime.now().isoformat()
        }

class IndustrialAPIServer:
    """工业级API服务器"""

    def __init__(self, inference_engine: IndustrialInferenceEngine):
        self.inference_engine = inference_engine
        self.app = web.Application()
        self.setup_routes()

        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'start_time': time.time()
        }

    def setup_routes(self):
        """设置路由"""
        self.app.router.add_post('/predict', self.predict_handler)
        self.app.router.add_post('/batch_predict', self.batch_predict_handler)
        self.app.router.add_get('/health', self.health_handler)
        self.app.router.add_get('/stats', self.stats_handler)
        self.app.router.add_get('/model_info', self.model_info_handler)

    async def predict_handler(self, request):
        """单个预测请求处理"""
        start_time = time.time()
        self.stats['total_requests'] += 1

        try:
            data = await request.json()

            # 验证输入
            if 'features' not in data:
                raise ValueError("缺少 'features' 字段")

            # 执行预测
            result = await self.inference_engine.predict(data)

            # 更新统计
            response_time = time.time() - start_time
            self.stats['successful_requests'] += 1
            self._update_average_response_time(response_time)

            return web.json_response(result)

        except Exception as e:
            self.stats['failed_requests'] += 1
            return web.json_response(
                {'error': str(e), 'timestamp': datetime.now().isoformat()},
                status=400
            )

    async def batch_predict_handler(self, request):
        """批量预测请求处理"""
        start_time = time.time()

        try:
            data = await request.json()
            requests = data.get('requests', [])

            if not requests:
                raise ValueError("请求列表不能为空")

            # 并行处理所有请求
            tasks = [
                self.inference_engine.predict(req)
                for req in requests
            ]

            results = await asyncio.gather(*tasks)

            # 更新统计
            response_time = time.time() - start_time
            self.stats['total_requests'] += len(requests)
            self.stats['successful_requests'] += len([r for r in results if 'error' not in r])
            self.stats['failed_requests'] += len([r for r in results if 'error' in r])
            self._update_average_response_time(response_time)

            return web.json_response({
                'results': results,
                'count': len(results),
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            return web.json_response(
                {'error': str(e), 'timestamp': datetime.now().isoformat()},
                status=400
            )

    async def health_handler(self, request):
        """健康检查"""
        # 系统资源检查
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        gpu_info = []

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                gpu_allocated = torch.cuda.memory_allocated(i)
                gpu_info.append({
                    'gpu_id': i,
                    'name': torch.cuda.get_device_name(i),
                    'total_memory': gpu_memory,
                    'allocated_memory': gpu_allocated,
                    'utilization': gpu_allocated / gpu_memory * 100
                })

        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time() - self.stats['start_time'],
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'gpu_info': gpu_info
            },
            'model': {
                'version': self.inference_engine.model_manager.current_model_version,
                'device_count': self.inference_engine.model_manager.device_count
            }
        }

        return web.json_response(health_status)

    async def stats_handler(self, request):
        """统计信息"""
        uptime = time.time() - self.stats['start_time']
        qps = self.stats['total_requests'] / uptime if uptime > 0 else 0

        stats = {
            **self.stats,
            'uptime': uptime,
            'qps': qps,
            'success_rate': (self.stats['successful_requests'] / max(self.stats['total_requests'], 1)) * 100
        }

        return web.json_response(stats)

    async def model_info_handler(self, request):
        """模型信息"""
        info = {
            'model_architecture': 'IndustryLeadingTransformer',
            'model_version': self.inference_engine.model_manager.current_model_version,
            'device_count': self.inference_engine.model_manager.device_count,
            'batch_size': self.inference_engine.batch_size,
            'max_sequence_length': self.inference_engine.max_sequence_length,
            'capabilities': [
                'Multi-task prediction',
                'Direction forecasting',
                'Return estimation',
                'Volatility prediction',
                'Risk assessment',
                'Real-time inference'
            ]
        }

        return web.json_response(info)

    def _update_average_response_time(self, response_time: float):
        """更新平均响应时间"""
        if self.stats['successful_requests'] == 1:
            self.stats['average_response_time'] = response_time
        else:
            self.stats['average_response_time'] = (
                (self.stats['average_response_time'] * (self.stats['successful_requests'] - 1) + response_time) /
                self.stats['successful_requests']
            )

    def run(self, host: str = '0.0.0.0', port: int = 8001):
        """启动服务器"""
        print(f"🚀 启动工业级API服务器")
        print(f"   地址: http://{host}:{port}")
        print(f"   批大小: {self.inference_engine.batch_size}")

        # 使用uvloop优化
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

        web.run_app(self.app, host=host, port=port, access_log=None)

def main():
    """主部署函数"""
    print("🚀 工业级模型部署系统")
    print("=" * 80)

    # 配置
    model_config = get_model_config()
    deployment_config = {
        'model_path': 'models/best_industrial_model.pth',
        'batch_size': 32,
        'batch_timeout': 0.1,
        'max_sequence_length': 252,
        'inference_workers': 4,
        'use_amp': True
    }

    print("📊 配置信息:")
    print(f"   模型路径: {deployment_config['model_path']}")
    print(f"   批大小: {deployment_config['batch_size']}")
    print(f"   推理工作线程: {deployment_config['inference_workers']}")

    # 检查模型文件
    if not Path(deployment_config['model_path']).exists():
        print(f"❌ 模型文件不存在: {deployment_config['model_path']}")
        print("💡 请先训练模型或提供正确的模型路径")
        return

    try:
        # 创建模型管理器
        model_manager = ModelManager(
            model_path=deployment_config['model_path'],
            config=model_config
        )

        # 创建推理引擎
        inference_engine = IndustrialInferenceEngine(
            model_manager=model_manager,
            config=deployment_config
        )

        # 创建API服务器
        api_server = IndustrialAPIServer(inference_engine)

        # 启动服务
        api_server.run(host='0.0.0.0', port=8001)

    except KeyboardInterrupt:
        print("\n🛑 服务器关闭")
    except Exception as e:
        print(f"❌ 部署失败: {e}")

if __name__ == "__main__":
    main()