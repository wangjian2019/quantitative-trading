# MacMini轻量级量化交易系统

适合MacMini等轻量级设备的AI量化交易系统。

## 📁 文件说明

- `memory_efficient_train.py` - TinyTransformer模型训练
- `test_tiny_model.py` - 训练好的模型测试
- `ai_service.py` - 生产环境AI服务API
- `config.py` - 系统配置文件

## 🎯 模型文件

- `tiny_transformer_model.pth` (312KB) - 训练好的TinyTransformer模型
- `tiny_model_info.pkl` - 模型元信息
- `tiny_scaler.pkl` - 特征标准化器

## 🚀 使用方法

### 训练模型
```bash
python3 memory_efficient_train.py
```

### 测试模型
```bash
python3 test_tiny_model.py
```

### 启动AI服务
```bash
python3 ai_service.py
```

## 📊 模型性能

- **准确率**: 74.6%
- **模型大小**: 312KB
- **参数量**: ~20K
- **架构**: TinyTransformer (d_model=64, nhead=8, num_layers=2)
- **特征维度**: 21个技术指标
- **序列长度**: 30天

## 💻 系统要求

- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM
- MPS/CPU支持