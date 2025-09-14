# AI量化交易平台 v0.1 - 系统设计文档

**作者**: Alvin  
**版本**: v0.1 (首个版本)  
**编码**: UTF-8  
**更新时间**: 2025年9月13日  

---

## 🏗️ 系统架构概述

### 整体架构
```
┌─────────────────┐    HTTP请求    ┌─────────────────┐    AI计算    ┌─────────────┐
│ SpringBoot平台  │ ────────────► │ Python AI服务   │ ──────────► │ ML模型      │
│ SmartTradingEngine              │ ai_model_service│             │ RF/GB/LR    │
│ (端口8080)      │               │ (端口5001)      │             │ 集成学习    │
└─────────────────┘               └─────────────────┘             └─────────────┘
       │                              │                              │
       │                              │                              │
       ▼                              ▼                              ▼
┌─────────────────┐    JSON响应    ┌─────────────────┐    预测结果   ┌─────────────┐
│ 用户通知        │ ◄──────────── │ 返回信号        │ ◄──────────── │ 信号生成    │
│ 邮件+微信       │               │ {action,conf}   │               │ {BUY/SELL}  │
└─────────────────┘               └─────────────────┘               └─────────────┘
```

### 核心组件

#### **Java平台 (端口8080)**
```
platform/src/main/java/com/alvin/quantitative/trading/platform/
├── engine/
│   ├── SmartTradingEngine.java        # 主交易引擎
│   ├── TradingEngineInterface.java    # 引擎接口
│   └── BacktestEngine.java           # 回测引擎
├── data/
│   ├── MarketDataManager.java        # 实时技术指标管理
│   ├── DataSourceFactory.java       # 数据源工厂
│   └── impl/
│       └── YahooFinanceDataSource.java # Yahoo Finance API
├── controller/
│   ├── SpringBootApiController.java # SpringBoot REST API控制器
│   └── SpringBootWebController.java # SpringBoot Web控制器
├── portfolio/
│   └── PortfolioManager.java        # 投资组合管理
├── risk/
│   └── RiskManager.java             # 风险管理
├── notification/
│   └── NotificationService.java     # 通知服务
├── strategy/
│   └── AIStrategyClient.java        # AI策略客户端
└── TradingPlatformApplication.java  # 主程序入口
```

#### **Python AI服务 (端口5001)**
```
strategy/
├── ai_model_service.py              # AI模型服务主程序
├── app.py                          # Flask应用
├── models/
│   ├── ai_strategy.py              # AI策略实现
│   ├── ultra_high_return_model.py  # 高收益模型
│   ├── rf_model.pkl                # RandomForest模型
│   ├── gb_model.pkl                # GradientBoosting模型
│   └── lr_model.pkl                # LogisticRegression模型
├── api/
│   ├── signal_api.py               # 信号API
│   ├── backtest_api.py             # 回测API
│   └── model_api.py                # 模型管理API
├── services/
│   └── backtest_service.py         # 回测服务
└── utils/
    ├── feature_engineering.py      # 特征工程
    ├── technical_indicators.py     # 技术指标
    └── advanced_features.py        # 高级特征
```

---

## 🔄 数据流程设计

### 1. 实时数据收集流程
```
每30秒执行:
SmartTradingEngine.collectMarketData()
├── 调用 YahooFinanceDataSource.getRealTimeData()
├── HTTP请求到 https://query1.finance.yahoo.com/v8/finance/chart/{symbol}
├── 解析JSON响应提取OHLCV数据
├── 存储到 MarketDataManager.addKlineData()
└── 自动计算9个技术指标 (RSI, MACD, MA5, MA10, MA20, ATR, 波动率等)
```

### 2. AI信号生成流程
```
每180秒执行:
SmartTradingEngine.executeStrategy()
├── 获取历史数据和技术指标
├── 调用 AIStrategyClient.getSignal()
├── HTTP请求到 http://localhost:5001/get_signal
├── Python AI模型推理 (RF+GB+LR集成)
├── 风险管理评估 (RiskManager)
├── 投资组合权重计算 (PortfolioManager)
└── 发送通知 (NotificationService)
```

### 3. 通知发送流程
```
高置信度信号 (≥70%):
SmartTradingEngine.sendTradingNotificationToUser()
├── 计算建议仓位大小
├── 计算止损止盈价格
├── 格式化通知消息
├── 控制台输出详细信息
├── 发送邮件通知
└── 发送微信通知
```

---

## 🤖 AI模型设计

### 模型架构
```python
# 集成学习模型
models = {
    'rf': RandomForestClassifier(n_estimators=500, max_depth=15),
    'gb': GradientBoostingClassifier(n_estimators=300, max_depth=8),
    'lr': LogisticRegression(max_iter=2000, C=0.1)
}

# 动态权重分配
ensemble_weights = {'rf': 0.3, 'gb': 0.5, 'lr': 0.2}
```

### 特征工程
```python
# 当前特征 (39个) - 实际使用
- MA相对位置: ma5_ratio, ma10_ratio, ma20_ratio, ma_slope, ma_convergence
- RSI状态: rsi, rsi_oversold, rsi_overbought, rsi_neutral, rsi_extreme
- MACD信号: macd, macd_bullish, macd_strength
- 价格位置: price_position
- 波动率: volatility, high_volatility, low_volatility
- 成交量: volume_ratio, high_volume, low_volume, volume_surge
- ATR: atr_ratio, high_atr
- 价格趋势: price_trend_5, price_trend_10, consecutive_up, consecutive_down
- 成交量趋势: volume_trend
- 动量: momentum_3, momentum_5
- 市场时间: morning, afternoon, near_close, market_open
- 信号强度: bullish_strength, bearish_strength, signal_divergence
- 风险: risk_level, trend_strength
```

### 信号生成逻辑
```python
def generate_signal(current_data, indicators, history):
    # 1. 特征工程 (40个基础特征)
    features = prepare_features(current_data, indicators, history)
    
    # 2. 模型预测
    ensemble_prob = np.zeros(3)  # [SELL, HOLD, BUY]
    for model_name, model in models.items():
        prob = model.predict_proba(features)[0]
        weight = ensemble_weights[model_name]
        ensemble_prob += weight * prob
    
    # 3. 信号增强
    ultra_bullish = features.get('ultra_bullish_strength', 0)
    if ultra_bullish > 0.7:
        ensemble_prob[2] *= 1.3  # 增强BUY概率
    
    # 4. 最终信号
    action = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}[np.argmax(ensemble_prob)]
    confidence = np.max(ensemble_prob)
    
    return {'action': action, 'confidence': confidence}
```

---

## 📊 API接口设计

### RESTful API端点
```
Java Web API (localhost:8080):
GET  /api/health           - 系统健康检查
GET  /api/status           - 系统运行状态  
GET  /api/portfolio        - 投资组合 (真实价格)
GET  /api/indicators       - 实时技术指标
GET  /api/trading-signals  - 手动交易信号
POST /api/backtest         - 回测分析
POST /api/test-notification - 测试通知

Python AI API (localhost:5001):
GET  /health               - AI服务健康检查
POST /get_signal           - 获取AI交易信号
GET  /model_info           - 模型信息
POST /train_model          - 训练ML模型
```

### 数据格式规范

#### 交易信号响应格式
```json
{
  "action": "BUY",
  "confidence": 0.875,
  "expected_return": 0.068,
  "reason": "BUY信号 置信度87.5%: 🚀 超强买入信号组合 | 💪 强势上涨动量",
  "metadata": {
    "model_predictions": {"rf": 2, "gb": 2, "lr": 2},
    "ensemble_probabilities": [0.05, 0.098, 0.852],
    "ultra_bullish_strength": 0.83,
    "key_features": {
      "momentum_5": 0.045,
      "rsi_oversold": 1.0,
      "volume_surge": 1.0
    }
  }
}
```

#### 手动交易信号格式
```json
{
  "symbol": "TSLA",
  "current_price": 395.94,
  "suggested_position_million": 150.0,
  "suggested_position_percent": 15.0,
  "risk_assessment": {
    "risk_level": "中等风险",
    "suggested_stop_loss": 376.14,
    "suggested_take_profit": 455.33
  },
  "technical_analysis": "RSI超卖，可能反弹; MACD金叉，上升趋势"
}
```

---

## 🛡️ 风险管理设计

### 风险控制层次
```java
// 1. 配置级风险控制
risk.max.position.ratio=0.15           // 单股票最大15%
risk.stop.loss.ratio=0.04              // 4%止损
risk.max.daily.loss=500000.0           // 日最大亏损50万
risk.emergency.stop.loss=0.03          // 紧急止损3%

// 2. 算法级风险控制
private boolean passRiskCheck(String symbol, AISignal signal, double price) {
    // 置信度检查
    if (signal.getConfidence() < config.getMinConfidence()) return false;
    
    // 仓位检查
    if ("BUY".equals(signal.getAction())) {
        return riskManager.canBuy(symbol, price, totalCapital);
    }
    
    return true;
}

// 3. 动态仓位计算
private double calculatePositionSizeForManualTrading(String symbol, AISignal signal, double price) {
    double basePosition = 0.05;  // 基础5%
    double confidenceMultiplier = Math.min(3.0, signal.getConfidence() / 0.6);
    double volatilityAdjustment = Math.min(1.0, 0.02 / volatility);
    
    // 最终仓位：2%-20% (20万-200万)
    return Math.max(0.02, Math.min(0.20, finalPosition)) * 100;
}
```

---

## 📧 通知系统设计

### 通知触发条件
```java
// 信号通知条件
if (!passRiskCheck(symbol, signal, currentData.getClose())) {
    // 信号被风险控制拒绝，不发送通知
    return;
}

// 发送交易通知
sendTradingNotificationToUser(symbol, signal, currentData.getClose());
```

### 通知消息格式
```
🚨 AI交易信号 - 买入信号
📊 股票: TSLA
🎯 操作: BUY
💰 价格: $395.94
📈 置信度: 87.5%
🚀 预期收益: 6.8%
💼 建议仓位: 150万 (15%)
🛡️ 建议止损: $376.14
🎯 建议止盈: $455.33
📝 分析理由: 🚀 超强买入信号组合 | 💪 强势上涨动量
⏰ 时间: 2025-09-13 23:57:00
```

---

## 🔧 配置管理设计

### 核心配置文件
```properties
# application.properties

# AI服务配置
ai.service.url=http://localhost:5001
ai.service.timeout.connect=10000
ai.service.retry.max=3

trading.initial.capital=10000000.0
trading.symbols=AAPL,TSLA,MSFT,GOOGL,AMZN,QQQ,VOO,ASML,NVDA,META
trading.data.collection.interval=30    # 30秒数据收集
trading.strategy.execution.interval=180 # 3分钟策略执行

# 风险管理配置
risk.max.position.ratio=0.15           # 单股票最大15%
risk.stop.loss.ratio=0.04              # 4%止损
risk.take.profit.ratio=0.12            # 12%止盈
risk.max.daily.loss=500000.0           # 日最大亏损50万
risk.min.confidence=0.70               # 最小置信度70%

# 通知配置
email.enabled=true
email.username=your_qq_email@qq.com
wechat.enabled=true
notification.min.confidence=0.70
```

### 投资组合配置
```json
// portfolio.json
{
  "portfolio": {
    "name": "Alvin的AI量化投资组合",
    "max_position_size": 0.2,
    "stop_loss_threshold": 0.04,
    "take_profit_threshold": 0.12
  },
  "symbols": [
    {
      "symbol": "TSLA",
      "weight": 0.10,
      "min_confidence": 0.75,
      "priority": "high"
    }
  ]
}
```

---

## 🧵 线程和调度设计

### 多线程架构
```java
// SmartTradingEngine.java
private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(6);

public void start() {
    // 实时数据收集线程 - 每30秒
    scheduler.scheduleAtFixedRate(this::collectMarketData, 0, 30, TimeUnit.SECONDS);
    
    // 策略执行线程 - 每180秒
    scheduler.scheduleAtFixedRate(this::executeStrategy, 30, 180, TimeUnit.SECONDS);
    
    // 风险检查线程 - 每15秒
    scheduler.scheduleAtFixedRate(this::checkRisk, 60, 15, TimeUnit.SECONDS);
    
    // 健康监控线程 - 每60秒
    scheduler.scheduleAtFixedRate(this::performHealthCheck, 60, 60, TimeUnit.SECONDS);
    
    // 日度重置线程 - 每24小时
    scheduler.scheduleAtFixedRate(this::dailyReset, 0, 24, TimeUnit.HOURS);
    
    // 周度回测线程 - 每7天
    scheduler.scheduleAtFixedRate(this::runWeeklyBacktest, 0, 7, TimeUnit.DAYS);
}
```

---

## 🔒 安全设计

### 异常处理机制
```java
// AI服务调用安全
try {
    signal = aiClient.getSignal(symbol, currentData, indicators, history);
} catch (Exception e) {
    logger.severe("🚨 CRITICAL: AI service call failed for " + symbol);
    healthMonitor.recordFailedSignal();
    return; // 跳过此股票
}

// 空指针检查
if (signal == null) {
    logger.warning("AI service returned null signal for " + symbol);
    return;
}

// 数据获取失败处理
if (history.isEmpty()) {
    healthMonitor.setDataManagerHealth(false);
    return;
}
```

### 回退策略
```java
// 增强回退策略
private AISignal createEnhancedFallbackSignal(String symbol, KlineData currentData, 
                                             Map<String, Double> indicators) {
    double rsi = indicators.getOrDefault("RSI", 50.0);
    double macd = indicators.getOrDefault("MACD", 0.0);
    double currentPrice = currentData.getClose();
    
    // 强买入条件
    if (rsi < 20 && macd > 0 && volumeRatio > 2.0) {
        return new AISignal("BUY", 0.75, "AI故障回退策略: RSI极度超卖+MACD金叉");
    }
    
    // 默认保守策略
    return new AISignal("HOLD", 0.5, "AI故障回退策略: 保守持有");
}
```

---

## 📈 性能优化设计

### 内存管理
```java
// MarketDataManager.java
private final Map<String, Queue<KlineData>> dataBuffers = new ConcurrentHashMap<>();
private final int maxBufferSize = 500;  // 限制内存使用

public void addKlineData(String symbol, KlineData data) {
    Queue<KlineData> buffer = dataBuffers.get(symbol);
    while (buffer.size() > maxBufferSize) {
        buffer.poll();  // 删除旧数据
    }
}
```

### 并发安全
```java
// 线程安全的数据结构
private final Map<String, Position> positions = new ConcurrentHashMap<>();
private final AtomicLong totalSignalRequests = new AtomicLong(0);
private volatile boolean isRunning = false;
```

---

## 🔍 监控和日志设计

### 健康监控
```java
// HealthMonitor.java
public class HealthMonitor {
    private final AtomicLong totalSignalRequests = new AtomicLong(0);
    private final AtomicLong successfulSignals = new AtomicLong(0);
    private volatile boolean aiServiceHealthy = true;
    private volatile boolean dataManagerHealthy = true;
    
    public Map<String, Object> getHealthReport() {
        Map<String, Object> report = new HashMap<>();
        report.put("system_healthy", isSystemHealthy());
        report.put("signal_success_rate", getSignalSuccessRate());
        report.put("components", getComponentStatus());
        return report;
    }
}
```

### 日志级别
```java
// 关键操作: SEVERE
logger.severe("🚨 CRITICAL: AI service call failed");

// 警告信息: WARNING  
logger.warning("⚠️ Signal rejected by risk control");

// 正常信息: INFO
logger.info("📊 Real-time data collected for " + symbol);

// 调试信息: FINE
logger.fine("Starting data collection for " + symbols.size() + " symbols");
```

---

## 🎯 业务逻辑设计

### 交易决策流程
```java
private void executeStrategyForSymbol(String symbol) {
    // 1. 数据验证
    List<KlineData> history = dataManager.getRecentData(symbol, 100);
    if (history.isEmpty()) return;
    
    // 2. AI信号生成
    AISignal signal = aiClient.getSignal(symbol, currentData, indicators, history);
    if (signal == null) return;
    
    // 3. 风险检查
    if (!passRiskCheck(symbol, signal, currentData.getClose())) return;
    
    // 4. 发送通知 (不执行交易)
    sendTradingNotificationToUser(symbol, signal, currentData.getClose());
}
```

### 仓位计算逻辑
```java
// 动态仓位计算
private double calculatePositionSizeForManualTrading(String symbol, AISignal signal, double price) {
    double basePosition = 0.05;  // 基础5% = 50万
    
    // 置信度调整: 置信度越高，仓位越大
    double confidenceMultiplier = Math.min(3.0, signal.getConfidence() / 0.6);
    
    // 波动率调整: 波动率越高，仓位越小
    double volatilityAdjustment = Math.min(1.0, 0.02 / volatility);
    
    // RSI调整: 超卖增仓，超买减仓
    double rsiAdjustment = 1.0;
    if (rsi < 30 && "BUY".equals(signal.getAction())) {
        rsiAdjustment = 1.5;  // 增加50%仓位
    }
    
    // 最终仓位: 2%-20% (20万-200万)
    double finalPosition = basePosition * confidenceMultiplier * volatilityAdjustment * rsiAdjustment;
    return Math.max(0.02, Math.min(0.20, finalPosition)) * 100;
}
```

---

## 🚀 部署架构设计

### 生产环境部署
```bash
# 部署结构
quantitative-trading/
├── platform/           # Java服务
├── strategy/           # Python AI服务
├── web/               # 静态资源
├── portfolio.json     # 投资组合配置
├── start_production.sh # 生产环境启动脚本
└── logs/              # 日志目录
```

### 启动顺序
```bash
1. 环境检查 (Java, Python, Maven)
2. 端口检查 (8080, 5001)
3. 配置验证 (资金配置)
4. Python依赖安装
5. Java项目编译
6. Python AI服务启动
7. Java平台启动
8. 系统健康验证
```

---

## 📝 代码调用关系

### 主要调用链
```
TradingPlatformApplication.main()
└── SmartTradingEngine.start()
    ├── collectMarketData() [每30秒]
    │   ├── YahooFinanceDataSource.getRealTimeData()
    │   └── MarketDataManager.addKlineData()
    │       └── updateTechnicalIndicators()
    │
    ├── executeStrategy() [每180秒]
    │   ├── AIStrategyClient.getSignal()
    │   │   └── HTTP POST to localhost:5001/get_signal
    │   ├── passRiskCheck()
    │   └── sendTradingNotificationToUser()
    │       └── NotificationService.sendTradingSignalNotification()
    │
    └── checkRisk() [每15秒]
        └── 风险监控和提醒
```

### 数据流向
```
Yahoo Finance API → YahooFinanceDataSource → MarketDataManager → 技术指标计算
                                                ↓
用户通知 ← NotificationService ← SmartTradingEngine ← AI信号 ← Python AI服务
```

---

**🎯 本设计文档详细描述了AI量化交易平台v0.1的完整架构，所有组件都已在代码中实现并经过测试验证。**
