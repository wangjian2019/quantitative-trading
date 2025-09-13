# 🔍 AI量化交易平台工作原理详解

## 🏗️ 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           🌐 Web浏览器界面 (端口8080)                                │
│                     http://localhost:8080                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ HTTP请求
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          ☕ Java平台服务 (端口8080)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │TradingEngine│  │DataManager  │  │UIController │  │Notification │  │Portfolio    │ │
│  │交易引擎     │  │数据管理     │  │界面控制     │  │通知服务     │  │投资组合     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ HTTP调用AI模型
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         🐍 Python AI模型服务 (端口5001)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │特征工程     │  │RandomForest │  │GradientBoost│  │LogisticReg  │  │信号生成     │ │
│  │39个特征     │  │随机森林     │  │梯度提升     │  │逻辑回归     │  │BUY/SELL/HOLD│ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ 获取市场数据
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              📊 外部数据源                                          │
│     Yahoo Finance    │    Alpha Vantage    │    IEX Cloud    │    实时行情         │
│     免费数据源       │    需要API Key      │    需要API Key   │    WebSocket        │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Java调用Python的详细机制

### 1. 📡 HTTP通信协议

#### Java端发起调用
```java
// AIStrategyClient.java - 负责与Python服务通信
public class AIStrategyClient {
    private final String apiUrl = "http://localhost:5001";  // Python服务地址
    
    public AISignal getSignal(String symbol, KlineData currentData, 
                             Map<String, Double> indicators, List<KlineData> history) {
        
        // 🔧 步骤1: 准备请求数据
        Map<String, Object> request = new HashMap<>();
        request.put("symbol", symbol);                    // 股票代码
        request.put("current_data", currentData);         // 当前K线数据
        request.put("indicators", indicators);            // 技术指标
        request.put("history", history);                  // 历史数据（最近100条）
        
        // 🔧 步骤2: 序列化为JSON
        String jsonRequest = objectMapper.writeValueAsString(request);
        
        // 🔧 步骤3: 构建HTTP POST请求
        HttpPost httpPost = new HttpPost(apiUrl + "/get_signal");
        httpPost.setHeader("Content-Type", "application/json");
        httpPost.setEntity(new StringEntity(jsonRequest, "UTF-8"));
        
        // 🔧 步骤4: 发送请求并处理响应
        try (CloseableHttpResponse response = httpClient.execute(httpPost)) {
            String responseBody = EntityUtils.toString(response.getEntity());
            
            // 🔧 步骤5: 反序列化Python返回的结果
            return objectMapper.readValue(responseBody, AISignal.class);
        }
    }
}
```

#### 请求数据结构
```json
{
  "symbol": "TSLA",
  "current_data": {
    "timestamp": "2024-09-13T15:30:00",
    "open": 248.50,
    "high": 252.30,
    "low": 247.80,
    "close": 250.15,
    "volume": 1500000
  },
  "indicators": {
    "MA5": 249.20,      // 5日移动平均
    "MA10": 248.80,     // 10日移动平均
    "MA20": 245.80,     // 20日移动平均
    "RSI": 45.5,        // 相对强弱指数
    "MACD": 0.3,        // MACD指标
    "VOLATILITY": 0.025, // 波动率
    "VOLUME_RATIO": 1.2, // 成交量比率
    "PRICE_POSITION": 0.6, // 价格位置
    "ATR": 2.1          // 平均真实波动范围
  },
  "history": [
    {"timestamp": "2024-09-12", "open": 245.0, "high": 247.0, "low": 244.0, "close": 246.5, "volume": 1200000},
    {"timestamp": "2024-09-11", "open": 246.5, "high": 249.0, "low": 245.5, "close": 248.2, "volume": 1350000},
    // ... 最近100条历史K线数据
  ]
}
```

### 2. 🤖 Python AI处理流程

#### ai_model_service.py - AI模型处理
```python
@app.route('/get_signal', methods=['POST'])
def get_signal():
    # 📥 步骤1: 接收Java发送的数据
    data = request.json
    symbol = data.get('symbol')
    current_data = data.get('current_data', {})
    indicators = data.get('indicators', {})
    history = data.get('history', [])
    
    # 🔧 步骤2: 特征工程（生成39个特征）
    features = ai_model.prepare_features(current_data, indicators, history)
    
    # 特征包括：
    # - MA相对位置: ma5_ratio, ma10_ratio, ma20_ratio
    # - RSI状态: rsi_oversold, rsi_overbought, rsi_neutral
    # - MACD信号: macd_bullish, macd_strength
    # - 价格动量: momentum_3, momentum_5
    # - 成交量: volume_ratio, high_volume, volume_surge
    # - 市场时间: morning, afternoon, near_close
    # - 风险指标: volatility, atr_ratio, risk_level
    # - 趋势强度: bullish_strength, bearish_strength
    
    # 🤖 步骤3: 多模型集成预测
    feature_array = np.array([list(features.values())]).reshape(1, -1)
    feature_scaled = ai_model.scalers['main'].transform(feature_array)
    
    # 三个模型分别预测
    ensemble_prob = np.zeros(3)  # [SELL概率, HOLD概率, BUY概率]
    
    for model_name, model in ai_model.models.items():
        prob = model.predict_proba(feature_scaled)[0]
        weight = ai_model.model_weights[model_name]  # rf:0.4, gb:0.4, lr:0.2
        ensemble_prob += weight * prob
    
    # 🎯 步骤4: 生成最终信号
    final_prediction = np.argmax(ensemble_prob)  # 选择概率最高的动作
    confidence = np.max(ensemble_prob)           # 最高概率作为置信度
    action = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}[final_prediction]
    
    # 📝 步骤5: 生成解释
    reason = generate_explanation(features, action, confidence)
    
    # 📤 步骤6: 返回结果给Java
    return jsonify({
        'action': action,           # 建议动作: BUY/SELL/HOLD
        'confidence': confidence,   # 置信度: 0.0-1.0
        'reason': reason,          # 决策理由
        'metadata': {              # 额外信息
            'model_predictions': predictions,
            'ensemble_probabilities': ensemble_prob,
            'key_features': top_5_features,
            'market_regime': market_state
        }
    })
```

#### 响应数据结构
```json
{
  "action": "BUY",
  "confidence": 0.852,
  "reason": "BUY信号 置信度85.2%: 强上升趋势, RSI超卖, 高成交量, 强正动量",
  "metadata": {
    "model_predictions": {"rf": 2, "gb": 2, "lr": 1},
    "ensemble_probabilities": [0.05, 0.098, 0.852],
    "key_features": {
      "bullish_strength": 0.83,
      "momentum_5": 0.045,
      "rsi_oversold": 1.0,
      "high_volume": 1.0,
      "ma5_ratio": 0.018
    },
    "market_regime": "趋势市场"
  }
}
```

## 📊 数据获取机制详解

### 1. 🌐 实时数据获取

#### Yahoo Finance API调用
```java
// YahooFinanceDataSource.java
public KlineData getRealTimeData(String symbol) throws DataSourceException {
    try {
        // 🔗 步骤1: 构建API URL
        String url = String.format(
            "https://query1.finance.yahoo.com/v8/finance/chart/%s?interval=1d&range=1d", 
            symbol
        );
        
        // 📡 步骤2: 发送HTTP GET请求
        HttpGet request = new HttpGet(url);
        request.setHeader("User-Agent", "Mozilla/5.0 (AI Trading Platform)");
        
        try (CloseableHttpResponse response = httpClient.execute(request)) {
            String responseBody = EntityUtils.toString(response.getEntity());
            
            // 📊 步骤3: 解析Yahoo Finance JSON响应
            JsonNode rootNode = objectMapper.readTree(responseBody);
            JsonNode chartNode = rootNode.path("chart").path("result").get(0);
            JsonNode meta = chartNode.path("meta");
            
            // 📈 步骤4: 提取OHLCV数据
            double open = meta.path("previousClose").asDouble();
            double high = meta.path("regularMarketDayHigh").asDouble();
            double low = meta.path("regularMarketDayLow").asDouble();
            double close = meta.path("regularMarketPrice").asDouble();
            long volume = meta.path("regularMarketVolume").asLong();
            
            // 🕐 步骤5: 创建时间戳
            long timestamp = meta.path("regularMarketTime").asLong();
            LocalDateTime dateTime = LocalDateTime.ofEpochSecond(timestamp, 0, ZoneOffset.UTC);
            
            // 📦 步骤6: 返回标准化数据
            return new KlineData(dateTime, open, high, low, close, volume);
        }
    } catch (Exception e) {
        throw new DataSourceException("获取实时数据失败: " + e.getMessage());
    }
}
```

#### Yahoo Finance API响应示例
```json
{
  "chart": {
    "result": [{
      "meta": {
        "currency": "USD",
        "symbol": "TSLA",
        "regularMarketPrice": 250.15,
        "previousClose": 248.50,
        "regularMarketDayHigh": 252.30,
        "regularMarketDayLow": 247.80,
        "regularMarketVolume": 1500000,
        "regularMarketTime": 1694678400
      },
      "timestamp": [1694678400],
      "indicators": {
        "quote": [{
          "open": [248.50],
          "high": [252.30],
          "low": [247.80],
          "close": [250.15],
          "volume": [1500000]
        }]
      }
    }]
  }
}
```

### 2. 📈 历史数据获取

#### 批量历史数据下载
```java
public List<KlineData> getHistoricalData(String symbol, int days) throws DataSourceException {
    try {
        // 🕐 步骤1: 计算时间范围
        long endTime = System.currentTimeMillis() / 1000;
        long startTime = endTime - (days * 24 * 60 * 60);
        
        // 🔗 步骤2: 构建历史数据API URL
        String url = String.format(
            "https://query1.finance.yahoo.com/v8/finance/chart/%s?period1=%d&period2=%d&interval=1d",
            symbol, startTime, endTime
        );
        
        // 📡 步骤3: 发送请求
        HttpGet request = new HttpGet(url);
        try (CloseableHttpResponse response = httpClient.execute(request)) {
            String responseBody = EntityUtils.toString(response.getEntity());
            JsonNode rootNode = objectMapper.readTree(responseBody);
            
            // 📊 步骤4: 解析时间序列数据
            JsonNode resultNode = rootNode.path("chart").path("result").get(0);
            JsonNode timestamps = resultNode.path("timestamp");
            JsonNode indicators = resultNode.path("indicators").path("quote").get(0);
            
            // 📈 步骤5: 提取OHLCV数组
            JsonNode opens = indicators.path("open");
            JsonNode highs = indicators.path("high");
            JsonNode lows = indicators.path("low");
            JsonNode closes = indicators.path("close");
            JsonNode volumes = indicators.path("volume");
            
            // 🔄 步骤6: 构建历史数据列表
            List<KlineData> historicalData = new ArrayList<>();
            for (int i = 0; i < timestamps.size(); i++) {
                long timestamp = timestamps.get(i).asLong();
                LocalDateTime dateTime = LocalDateTime.ofEpochSecond(timestamp, 0, ZoneOffset.UTC);
                
                double open = opens.get(i).asDouble();
                double high = highs.get(i).asDouble();
                double low = lows.get(i).asDouble();
                double close = closes.get(i).asDouble();
                long volume = volumes.get(i).asLong();
                
                historicalData.add(new KlineData(dateTime, open, high, low, close, volume));
            }
            
            return historicalData;
        }
    } catch (Exception e) {
        throw new DataSourceException("获取历史数据失败: " + e.getMessage());
    }
}
```

### 3. 🔄 数据更新循环

#### 定时任务调度
```java
// TradingEngine.java
public void start() {
    // 🕐 任务1: 实时数据收集（每1分钟）
    scheduler.scheduleAtFixedRate(() -> {
        for (String symbol : config.getTradingSymbols()) {
            try {
                // 1. 获取最新K线数据
                KlineData latestData = dataSource.getRealTimeData(symbol);
                
                // 2. 更新数据缓存
                dataBuffer.addData(symbol, latestData);
                
                // 3. 检查数据完整性
                if (dataBuffer.hasEnoughData(symbol, 50)) {  // 至少50条数据用于指标计算
                    // 触发AI分析
                    scheduleAIAnalysis(symbol);
                }
                
                logger.info(String.format("✅ 更新%s数据: $%.2f (成交量: %,d)", 
                    symbol, latestData.getClose(), latestData.getVolume()));
                    
            } catch (DataSourceException e) {
                logger.warning("❌ 获取" + symbol + "数据失败: " + e.getMessage());
            }
        }
    }, 0, 1, TimeUnit.MINUTES);
    
    // 🕐 任务2: AI策略分析（每5分钟）
    scheduler.scheduleAtFixedRate(() -> {
        executeAIStrategy();
    }, 0, 5, TimeUnit.MINUTES);
    
    // 🕐 任务3: 投资组合评估（每15分钟）
    scheduler.scheduleAtFixedRate(() -> {
        evaluatePortfolio();
    }, 0, 15, TimeUnit.MINUTES);
}

private void executeAIStrategy() {
    for (String symbol : config.getTradingSymbols()) {
        try {
            // 1. 获取历史数据
            List<KlineData> history = dataBuffer.getHistory(symbol, 100);
            KlineData currentData = history.get(history.size() - 1);
            
            // 2. 计算技术指标
            Map<String, Double> indicators = calculateTechnicalIndicators(history);
            
            // 3. 🔥 关键调用: 向Python AI模型请求预测
            AISignal signal = aiStrategyClient.getSignal(symbol, currentData, indicators, history);
            
            // 4. 处理AI返回的信号
            processAISignal(symbol, signal, currentData);
            
        } catch (Exception e) {
            logger.severe("AI策略执行失败: " + e.getMessage());
        }
    }
}
```

## 📊 支持的股票和数据源

### 1. 🇺🇸 美股数据获取
```java
// 支持的美股格式
String[] usStocks = {
    "ASML",    // 阿斯麦控股
    "TSLA",    // 特斯拉
    "NBIS",    // Nebius Group
    "SE",      // Sea Limited
    "GRAB",    // Grab Holdings
    "MRVL",    // Marvell Technology
    "HIMS",    // Hims & Hers Health
    "NKE"      // 耐克
};

// Yahoo Finance URL格式
String url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol;
```

### 2. 📈 ETF数据获取
```java
// 支持的ETF
String[] etfs = {
    "QQQ",     // 纳斯达克100 ETF
    "VOO",     // 标普500 ETF
    "FPE"      // 优先股ETF
};

// 与股票使用相同的API
```

### 3. 🇭🇰 港股数据获取
```java
// 港股格式（需要添加.HK后缀）
String[] hkStocks = {
    "3690.HK"  // 美团-W
};

// Yahoo Finance支持港股
String url = "https://query1.finance.yahoo.com/v8/finance/chart/3690.HK";
```

## 🔧 技术指标计算详解

### 1. 📊 移动平均线计算
```java
private double calculateMA(double[] prices, int period) {
    if (prices.length < period) return 0;
    
    double sum = 0;
    for (int i = prices.length - period; i < prices.length; i++) {
        sum += prices[i];
    }
    return sum / period;
}
```

### 2. 📈 RSI计算
```java
private double calculateRSI(double[] prices, int period) {
    if (prices.length < period + 1) return 50;
    
    double[] gains = new double[prices.length - 1];
    double[] losses = new double[prices.length - 1];
    
    // 计算每日涨跌
    for (int i = 0; i < prices.length - 1; i++) {
        double change = prices[i + 1] - prices[i];
        gains[i] = Math.max(change, 0);
        losses[i] = Math.max(-change, 0);
    }
    
    // 计算平均收益和损失
    double avgGain = Arrays.stream(gains, gains.length - period, gains.length).average().orElse(0);
    double avgLoss = Arrays.stream(losses, losses.length - period, losses.length).average().orElse(0);
    
    if (avgLoss == 0) return 100;
    
    double rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
}
```

### 3. 📊 MACD计算
```java
private double[] calculateMACD(double[] prices) {
    // 计算EMA12和EMA26
    double ema12 = calculateEMA(prices, 12);
    double ema26 = calculateEMA(prices, 26);
    
    // MACD = EMA12 - EMA26
    double macd = ema12 - ema26;
    
    // Signal Line = EMA9 of MACD
    double signal = macd; // 简化版本
    
    // Histogram = MACD - Signal
    double histogram = macd - signal;
    
    return new double[]{macd, signal, histogram};
}
```

## 🔄 完整的数据处理流程

### 1. 🚀 系统启动时
```java
// 1. 初始化数据源
DataSource dataSource = DataSourceFactory.createDataSource(config);

// 2. 加载历史数据（用于指标计算）
for (String symbol : config.getTradingSymbols()) {
    List<KlineData> history = dataSource.getHistoricalData(symbol, 100);
    dataBuffer.initializeHistory(symbol, history);
}

// 3. 启动定时任务
startDataCollection();
startAIAnalysis();
```

### 2. 🔄 运行时循环
```java
// 每分钟执行
void collectRealTimeData() {
    for (String symbol : monitoredSymbols) {
        // 1. 从Yahoo Finance获取最新数据
        KlineData newData = yahooApi.getLatestData(symbol);
        
        // 2. 更新数据缓存
        dataBuffer.append(symbol, newData);
        
        // 3. 保持数据窗口大小（最近100条）
        dataBuffer.trimToSize(symbol, 100);
    }
}

// 每5分钟执行
void executeAIAnalysis() {
    for (String symbol : monitoredSymbols) {
        // 1. 准备数据
        List<KlineData> history = dataBuffer.getHistory(symbol);
        Map<String, Double> indicators = calculateIndicators(history);
        
        // 2. 🔥 调用Python AI模型
        AISignal signal = callPythonAI(symbol, history.get(-1), indicators, history);
        
        // 3. 处理信号
        if (signal.getConfidence() >= 0.75) {
            // 发送通知
            sendNotification(symbol, signal);
            
            // 记录交易建议
            logTradingSignal(symbol, signal);
        }
    }
}
```

### 3. 📊 回测数据处理
```java
// BacktestEngine.java
public BacktestResult runBacktest(String symbol, int days) {
    // 1. 获取足够的历史数据
    List<KlineData> allData = dataSource.getHistoricalData(symbol, days + 50);
    
    // 2. 分割数据
    List<KlineData> warmupData = allData.subList(0, 50);      // 用于指标计算
    List<KlineData> backtestData = allData.subList(50, allData.size());  // 用于回测
    
    // 3. 模拟交易
    double capital = 100000;  // 初始资金
    int position = 0;         // 持仓数量
    List<Trade> trades = new ArrayList<>();
    
    for (int i = 20; i < backtestData.size() - 1; i++) {
        KlineData currentData = backtestData.get(i);
        List<KlineData> history = backtestData.subList(0, i + 1);
        
        // 计算技术指标
        Map<String, Double> indicators = calculateIndicators(history);
        
        // 🔥 调用AI模型获取信号
        AISignal signal = aiStrategyClient.getSignal(symbol, currentData, indicators, history);
        
        // 执行交易逻辑
        if (signal.getConfidence() >= 0.7) {
            if ("BUY".equals(signal.getAction()) && position == 0) {
                // 买入
                position = (int) (capital / currentData.getClose());
                capital = 0;
                trades.add(new Trade("BUY", currentData));
                
            } else if ("SELL".equals(signal.getAction()) && position > 0) {
                // 卖出
                capital = position * currentData.getClose();
                trades.add(new Trade("SELL", currentData));
                position = 0;
            }
        }
    }
    
    // 4. 计算回测指标
    return calculatePerformanceMetrics(trades, 100000);
}
```

## 🎯 关键文件位置

### Java端核心文件
- **AIStrategyClient.java**: 调用Python AI模型
- **TradingEngine.java**: 主交易引擎
- **YahooFinanceDataSource.java**: Yahoo Finance数据获取
- **EnhancedNotificationService.java**: 通知服务
- **application.properties**: 系统配置

### Python端核心文件
- **ai_model_service.py**: 纯AI模型服务
- **config.py**: Python服务配置

### 前端文件
- **index.html**: Web界面
- **script.js**: 前端逻辑
- **style.css**: 界面样式

## 🧪 测试验证

刚才的测试显示系统正常工作：
- ✅ Python AI服务正常运行 (端口5001)
- ✅ Java项目编译成功
- ✅ AI模型预测正常 (置信度77.8%，HOLD信号)
- ✅ HTTP通信正常
- ✅ JSON序列化/反序列化正常

现在你的AI量化交易平台已经具备完整的数据获取和AI预测功能！🚀
