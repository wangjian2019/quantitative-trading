# 策略模块文件结构说明

## 📁 核心文件 (清理后版本)

### 🤖 AI训练与数据处理
- **`memory_efficient_train.py`** - 生产级模型训练脚本
  - TinyTransformer架构，内存优化
  - 74.6%准确率，支持MPS加速
  - 生成: `tiny_transformer_model.pth`, `tiny_scaler.pkl`, `tiny_model_info.pkl`

- **`enhanced_data_collector.py`** - 增强型数据收集器
  - 支持断点续传
  - 521只热门美股港股数据
  - Yahoo Finance API集成

### ⚙️ 自动化运维
- **`daily_data_update.py`** - 每日数据更新
  - 工作日自动增量更新
  - 数据质量检查
  - 详细日志记录

- **`weekly_model_retrain.py`** - 周度模型重训练
  - 自动模型备份
  - 质量验证机制
  - 失败回滚保护

- **`system_health_check.py`** - 系统健康监控
  - 全方位健康检查
  - 自动异常告警
  - 状态报告生成

### 🔧 工具脚本
- **`setup_scheduler.sh`** - 定时任务配置
- **`run_manual_update.sh`** - 手动更新工具

### 📱 服务文件
- **`ai_service.py`** - AI预测服务 (端口: 5001)
- **`config.py`** - 配置管理

## 🗑️ 已删除的冗余文件

### 训练脚本清理
- ❌ `train_production_model.py` - 原始生产版本 (内存问题)
- ❌ `simple_train.py` - 简化测试版本 (已被替代)
- ❌ `quick_train_model.py` - 早期测试版本 (不完整)

### API和服务模块清理
- ❌ `app.py` - 冗余Flask应用 (依赖不存在的模块)
- ❌ `api/model_api.py` - 未使用的模型API
- ❌ `api/backtest_api.py` - 未使用的回测API
- ❌ `api/signal_api.py` - 未使用的信号API
- ❌ `services/backtest_service.py` - 未使用的回测服务
- ❌ `api/` 和 `services/` 目录 - 完全移除

### 功能重复模块清理
- ❌ `utils/feature_engineering.py` - 与transformer_model.py功能重复
- ❌ `utils/technical_indicators.py` - 完全未使用，功能已集成
- ❌ `utils/` 目录 - 完全移除

### 清理原因
1. **内存优化**: 旧版本存在MPS内存溢出问题
2. **功能重复**: 多个训练脚本和特征工程模块功能重叠
3. **版本迭代**: 新版本已解决所有问题
4. **代码简洁**: 避免混淆，明确单一入口
5. **模块整合**: 特征工程功能已集成到transformer_model.py中

## 📊 当前模型文件
- `tiny_transformer_model.pth` (312KB) - 训练好的模型权重
- `tiny_scaler.pkl` (953B) - 数据标准化器
- `tiny_model_info.pkl` (185B) - 模型配置信息

## 🔄 推荐工作流程

### 日常使用
```bash
# 手动数据更新
python3 daily_data_update.py

# 手动模型重训练
python3 weekly_model_retrain.py

# 系统健康检查
python3 system_health_check.py
```

### 自动化部署
```bash
# 配置定时任务
./setup_scheduler.sh

# 手动更新工具
./run_manual_update.sh
```

## ✅ 清理效果

- **Python文件**: 从19个减少到11个核心文件
- **功能整合**: 特征工程统一到transformer_model.py
- **代码维护**: 消除重复，单一职责原则
- **功能完整**: 保留所有必要功能，无功能缺失
- **性能优化**: 解决内存问题，提高稳定性
- **架构清晰**: memory_efficient_train.py专注训练，transformer_model.py提供完整模型库

现在系统结构清晰，只保留经过验证的可用版本，便于维护和使用。