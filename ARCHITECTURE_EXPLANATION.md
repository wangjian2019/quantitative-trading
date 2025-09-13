# 🏗️ AI量化交易平台架构详解

## 📊 Java调用Python模型预测机制

### 1. 🔄 调用流程

```
┌─────────────┐    HTTP请求    ┌─────────────────┐    AI计算    ┌─────────────┐
│ Java服务    │ ────────────► │ Python AI服务   │ ──────────► │ ML模型      │
│ TradingEngine│               │ ai_model_service│             │ RF/GB/LR    │
└─────────────┘               └─────────────────┘             └─────────────┘
       │                              │                              │
       │                              │                              │
       ▼                              ▼                              ▼
┌─────────────┐    JSON响应    ┌─────────────────┐    预测结果   ┌─────────────┐
│ 处理信号    │ ◄──────────── │ 返回信号        │ ◄──────────── │ 信号生成    │
│ 发送通知    │               │ {action,conf}   │               │ {BUY/SELL}  │
└─────────────┘               └─────────────────┘               └─────────────┘
```

### 2. 🔧 Java端调用代码

#### AIStrategyClient.java - 核心调用类
```java
public class AIStrategyClient {
    private final String apiUrl = "http://localhost:5000";  // Python服务地址
    
    public AISignal getSignal(String symbol, KlineData currentData, 
                             Map<String, Double> indicators, List<KlineData> history) {
        
        // 1. 构建请求数据
        Map<String, Object> request = new HashMap<>();
        request.put("symbol", symbol);
        request.put("current_data", currentData);
        request.put("indicators", indicators);
        request.put("history", history.subList(Math.max(0, history.size() - 100), history.size()));
        
        // 2. 转换为JSON
        String jsonRequest = objectMapper.writeValueAsString(request);
        
        // 3. 发送HTTP POST请求
        HttpPost httpPost = new HttpPost(apiUrl + "/get_signal");
        httpPost.setHeader("Content-Type", "application/json");
        httpPost.setEntity(new StringEntity(jsonRequest, "UTF-8"));
        
        // 4. 执行请求并获取响应
        try (CloseableHttpResponse response = httpClient.execute(httpPost)) {
            String responseBody = EntityUtils.toString(response.getEntity());
            
            // 5. 解析Python返回的JSON结果
            return objectMapper.readValue(responseBody, AISignal.class);
        }
    }
}
```

#### 请求数据格式示例
```json
{
  "symbol": "TSLA",
  "current_data": {
    "open": 248.50,
    "high": 252.30,
    "low": 247.80,
    "close": 250.15,
    "volume": 1500000,
    "timestamp": "2024-09-13T15:30:00"
  },
  "indicators": {
    "RSI": 45.5,
    "MACD": 0.3,
    "MA5": 249.20,
    "MA10": 248.80,
    "MA20": 245.80,
    "VOLATILITY": 0.025,
    "VOLUME_RATIO": 1.2
  },
  "history": [
    {"open": 245.0, "high": 247.0, "low": 244.0, "close": 246.5, "volume": 1200000},
    {"open": 246.5, "high": 249.0, "low": 245.5, "close": 248.2, "volume": 1350000},
    // ... 最近100条历史数据
  ]
}
```

### 3. 🐍 Python端处理机制

#### ai_model_service.py - AI模型服务
```python
@app.route('/get_signal', methods=['POST'])
def get_signal():
    # 1. 接收Java发送的数据
    data = request.json
    current_data = data.get('current_data', {})
    indicators = data.get('indicators', {})
    history = data.get('history', [])
    
    # 2. 特征工程
    features = ai_model.prepare_features(current_data, indicators, history)
    # 生成39个特征：ma5_ratio, rsi, macd_bullish, volume_ratio等
    
    # 3. 模型预测
    feature_array = np.array([list(features.values())]).reshape(1, -1)
    feature_scaled = ai_model.scalers['main'].transform(feature_array)
    
    # 4. 集成3个模型的预测
    ensemble_prob = np.zeros(3)  # [SELL, HOLD, BUY]
    for model_name, model in ai_model.models.items():
        prob = model.predict_proba(feature_scaled)[0]
        weight = ai_model.model_weights[model_name]  # rf:0.4, gb:0.4, lr:0.2
        ensemble_prob += weight * prob
    
    # 5. 生成最终信号
    final_prediction = np.argmax(ensemble_prob)
    confidence = np.max(ensemble_prob)
    action = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}[final_prediction]
    
    # 6. 返回结果给Java
    return jsonify({
        'action': action,
        'confidence': float(confidence),
        'reason': generate_explanation(features, action, confidence)
    })
```

#### 响应数据格式示例
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

## 📈 数据获取机制详解

### 1. 🌐 实时数据获取

#### 数据源工厂模式
```java
// DataSourceFactory.java
public static DataSource createDataSource(ApplicationConfig config) {
    String dataSourceType = config.getDataSourceType();
    
    switch (dataSourceType) {
        case "YAHOO_FINANCE":
            return new YahooFinanceDataSource();
        case "ALPHA_VANTAGE":
            return new AlphaVantageDataSource();
        case "IEX_CLOUD":
            return new IEXCloudDataSource();
        default:
            return new YahooFinanceDataSource(); // 默认使用Yahoo
    }
}
```

#### Yahoo Finance实时数据获取
```java
public KlineData getRealTimeData(String symbol) throws DataSourceException {
    try {
        // 1. 构建Yahoo Finance API URL
        String url = String.format(
            "https://query1.finance.yahoo.com/v8/finance/chart/%s?interval=1d&range=1d", 
            symbol
        );
        
        // 2. 发送HTTP请求
        HttpGet request = new HttpGet(url);
        try (CloseableHttpResponse response = httpClient.execute(request)) {
            String responseBody = EntityUtils.toString(response.getEntity());
            
            // 3. 解析JSON响应
            JsonNode rootNode = objectMapper.readTree(responseBody);
            JsonNode chartNode = rootNode.path("chart").path("result").get(0);
            
            // 4. 提取OHLCV数据
            JsonNode meta = chartNode.path("meta");
            double open = meta.path("previousClose").asDouble();
            double high = meta.path("regularMarketDayHigh").asDouble();
            double low = meta.path("regularMarketDayLow").asDouble();
            double close = meta.path("regularMarketPrice").asDouble();
            long volume = meta.path("regularMarketVolume").asLong();
            
            // 5. 返回KlineData对象
            return new KlineData(LocalDateTime.now(), open, high, low, close, volume);
        }
    } catch (Exception e) {
        throw new DataSourceException("获取实时数据失败: " + e.getMessage());
    }
}
```

### 2. 📊 历史数据获取

#### 批量历史数据下载
```java
public List<KlineData> getHistoricalData(String symbol, int days) throws DataSourceException {
    try {
        // 1. 计算时间范围
        long endTime = System.currentTimeMillis() / 1000;
        long startTime = endTime - (days * 24 * 60 * 60);
        
        // 2. 构建API URL
        String url = String.format(
            "https://query1.finance.yahoo.com/v8/finance/chart/%s?period1=%d&period2=%d&interval=1d",
            symbol, startTime, endTime
        );
        
        // 3. 发送请求获取历史数据
        HttpGet request = new HttpGet(url);
        try (CloseableHttpResponse response = httpClient.execute(request)) {
            String responseBody = EntityUtils.toString(response.getEntity());
            
            // 4. 解析JSON数据
            JsonNode rootNode = objectMapper.readTree(responseBody);
            JsonNode resultNode = rootNode.path("chart").path("result").get(0);
            
            // 5. 提取时间序列数据
            JsonNode timestamps = resultNode.path("timestamp");
            JsonNode indicators = resultNode.path("indicators").path("quote").get(0);
            
            JsonNode opens = indicators.path("open");
            JsonNode highs = indicators.path("high");
            JsonNode lows = indicators.path("low");
            JsonNode closes = indicators.path("close");
            JsonNode volumes = indicators.path("volume");
            
            // 6. 构建KlineData列表
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

#### TradingEngine数据管理
```java
public class TradingEngine {
    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);
    
    public void start() {
        // 1. 启动实时数据收集任务（每60秒）
        scheduler.scheduleAtFixedRate(() -> {
            for (String symbol : getTradingSymbols()) {
                try {
                    // 获取实时数据
                    KlineData realTimeData = dataSource.getRealTimeData(symbol);
                    
                    // 更新数据缓存
                    updateDataBuffer(symbol, realTimeData);
                    
                    // 如果数据足够，进行AI分析
                    if (hasEnoughData(symbol)) {
                        analyzeAndGenerateSignal(symbol);
                    }
                    
                } catch (Exception e) {
                    logger.warning("更新" + symbol + "数据失败: " + e.getMessage());
                }
            }
        }, 0, 60, TimeUnit.SECONDS);
        
        // 2. 启动策略执行任务（每5分钟）
        scheduler.scheduleAtFixedRate(() -> {
            executeStrategy();
        }, 0, 5, TimeUnit.MINUTES);
    }
    
    private void analyzeAndGenerateSignal(String symbol) {
        try {
            // 1. 获取历史数据
            List<KlineData> history = getHistoryData(symbol);
            KlineData currentData = history.get(history.size() - 1);
            
            // 2. 计算技术指标
            Map<String, Double> indicators = calculateTechnicalIndicators(history);
            
            // 3. 调用Python AI模型
            AISignal signal = aiStrategyClient.getSignal(symbol, currentData, indicators, history);
            
            // 4. 处理信号结果
            if (signal.getConfidence() >= config.getNotificationMinConfidence()) {
                // 发送通知
                notificationService.sendTradingSignalNotification(symbol, signal, currentData.getClose());
                
                // 记录信号
                logger.info(String.format("生成高置信度信号: %s %s (%.1f%%)", 
                    symbol, signal.getAction(), signal.getConfidence() * 100));
            }
            
        } catch (Exception e) {
            logger.severe("分析信号失败: " + e.getMessage());
        }
    }
}
```

### 4. 📊 技术指标计算

#### 在Java端计算技术指标
```java
private Map<String, Double> calculateTechnicalIndicators(List<KlineData> history) {
    Map<String, Double> indicators = new HashMap<>();
    
    if (history.size() < 20) {
        return getDefaultIndicators();
    }
    
    // 提取收盘价序列
    double[] closes = history.stream()
        .mapToDouble(KlineData::getClose)
        .toArray();
    
    // 1. 移动平均线
    indicators.put("MA5", calculateMA(closes, 5));
    indicators.put("MA10", calculateMA(closes, 10));
    indicators.put("MA20", calculateMA(closes, 20));
    
    // 2. RSI指标
    indicators.put("RSI", calculateRSI(closes, 14));
    
    // 3. MACD指标
    double[] macd = calculateMACD(closes);
    indicators.put("MACD", macd[0]);
    indicators.put("MACD_SIGNAL", macd[1]);
    indicators.put("MACD_HISTOGRAM", macd[2]);
    
    // 4. 布林带
    double[] bollinger = calculateBollingerBands(closes, 20, 2.0);
    indicators.put("BB_UPPER", bollinger[0]);
    indicators.put("BB_MIDDLE", bollinger[1]);
    indicators.put("BB_LOWER", bollinger[2]);
    
    // 5. 价格位置
    double high20 = Arrays.stream(closes, closes.length - 20, closes.length).max().orElse(0);
    double low20 = Arrays.stream(closes, closes.length - 20, closes.length).min().orElse(0);
    double pricePosition = (closes[closes.length - 1] - low20) / (high20 - low20);
    indicators.put("PRICE_POSITION", pricePosition);
    
    // 6. 波动率
    double volatility = calculateVolatility(closes, 20);
    indicators.put("VOLATILITY", volatility);
    
    // 7. 成交量比率
    long[] volumes = history.stream()
        .mapToLong(KlineData::getVolume)
        .toArray();
    double avgVolume = Arrays.stream(volumes, volumes.length - 20, volumes.length - 1)
        .average().orElse(1);
    double volumeRatio = volumes[volumes.length - 1] / avgVolume;
    indicators.put("VOLUME_RATIO", volumeRatio);
    
    return indicators;
}
```

## 📊 数据获取详细机制

### 1. 🌐 支持的数据源

#### Yahoo Finance (免费，主要使用)
```java
// 实时数据API
https://query1.finance.yahoo.com/v8/finance/chart/TSLA?interval=1d&range=1d

// 历史数据API  
https://query1.finance.yahoo.com/v8/finance/chart/TSLA?period1=1640995200&period2=1672531200&interval=1d

// 支持的股票格式:
// 美股: AAPL, TSLA, MSFT
// 港股: 3690.HK, 0700.HK
// ETF: QQQ, VOO, SPY
```

#### Alpha Vantage (需要API Key)
```java
// 实时报价
https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=TSLA&apikey=YOUR_API_KEY

// 历史数据
https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=TSLA&outputsize=full&apikey=YOUR_API_KEY

// 技术指标
https://www.alphavantage.co/query?function=RSI&symbol=TSLA&interval=daily&time_period=14&apikey=YOUR_API_KEY
```

#### IEX Cloud (需要API Key)
```java
// 实时报价
https://cloud.iexapis.com/stable/stock/TSLA/quote?token=YOUR_TOKEN

// 历史数据
https://cloud.iexapis.com/stable/stock/TSLA/chart/3y?token=YOUR_TOKEN
```

### 2. 📅 数据更新策略

#### 实时数据更新机制
```java
// TradingEngine.java
public void startDataCollection() {
    // 1. 实时数据收集（每1分钟）
    scheduler.scheduleAtFixedRate(() -> {
        for (String symbol : config.getTradingSymbols()) {
            try {
                // 获取最新数据
                KlineData latestData = dataSource.getRealTimeData(symbol);
                
                // 更新内存缓存
                dataBuffer.put(symbol, latestData);
                
                // 检查是否需要AI分析
                if (shouldAnalyze(symbol)) {
                    triggerAIAnalysis(symbol);
                }
                
            } catch (DataSourceException e) {
                logger.warning("获取" + symbol + "实时数据失败: " + e.getMessage());
            }
        }
    }, 0, 1, TimeUnit.MINUTES);
    
    // 2. 历史数据补充（每小时）
    scheduler.scheduleAtFixedRate(() -> {
        supplementHistoricalData();
    }, 0, 1, TimeUnit.HOURS);
}
```

#### 历史数据缓存机制
```java
public class MarketDataManager {
    private final Map<String, List<KlineData>> historicalDataCache = new ConcurrentHashMap<>();
    private final Map<String, LocalDateTime> lastUpdateTime = new ConcurrentHashMap<>();
    
    public List<KlineData> getHistoricalData(String symbol, int days) {
        // 1. 检查缓存
        if (isCacheValid(symbol)) {
            return historicalDataCache.get(symbol);
        }
        
        // 2. 从数据源获取
        try {
            List<KlineData> data = dataSource.getHistoricalData(symbol, days);
            
            // 3. 更新缓存
            historicalDataCache.put(symbol, data);
            lastUpdateTime.put(symbol, LocalDateTime.now());
            
            return data;
        } catch (DataSourceException e) {
            logger.warning("获取历史数据失败: " + e.getMessage());
            return historicalDataCache.getOrDefault(symbol, new ArrayList<>());
        }
    }
    
    private boolean isCacheValid(String symbol) {
        LocalDateTime lastUpdate = lastUpdateTime.get(symbol);
        if (lastUpdate == null) return false;
        
        // 缓存30分钟有效
        return Duration.between(lastUpdate, LocalDateTime.now()).toMinutes() < 30;
    }
}
```

### 3. 🔄 回测数据获取

#### 回测数据准备
```java
public class BacktestEngine {
    public BacktestResult runBacktest(String symbol, int backtestDays) {
        try {
            // 1. 获取足够的历史数据（回测期间 + 指标计算期间）
            int totalDays = backtestDays + 50;  // 额外50天用于指标计算
            List<KlineData> allData = dataSource.getHistoricalData(symbol, totalDays);
            
            // 2. 分割数据：前50天用于指标计算，后面用于回测
            List<KlineData> indicatorData = allData.subList(0, 50);
            List<KlineData> backtestData = allData.subList(50, allData.size());
            
            // 3. 逐日回测
            double capital = config.getInitialCapital();
            int position = 0;
            List<Trade> trades = new ArrayList<>();
            
            for (int i = 20; i < backtestData.size() - 1; i++) {  // 跳过前20天用于指标计算
                KlineData currentData = backtestData.get(i);
                List<KlineData> history = backtestData.subList(0, i + 1);
                
                // 计算指标
                Map<String, Double> indicators = calculateTechnicalIndicators(history);
                
                // 调用AI模型获取信号
                AISignal signal = aiStrategyClient.getSignal(symbol, currentData, indicators, history);
                
                // 执行交易逻辑
                if (signal.getConfidence() >= 0.7) {
                    if ("BUY".equals(signal.getAction()) && position == 0) {
                        // 买入
                        position = (int) (capital / currentData.getClose());
                        capital = 0;
                        trades.add(new Trade("BUY", currentData.getClose(), position, currentData.getTimestamp()));
                        
                    } else if ("SELL".equals(signal.getAction()) && position > 0) {
                        // 卖出
                        capital = position * currentData.getClose();
                        trades.add(new Trade("SELL", currentData.getClose(), position, currentData.getTimestamp()));
                        position = 0;
                    }
                }
            }
            
            // 4. 计算回测结果
            return calculateBacktestMetrics(trades, config.getInitialCapital());
            
        } catch (Exception e) {
            logger.severe("回测失败: " + e.getMessage());
            throw new RuntimeException("回测执行失败", e);
        }
    }
}
```

### 4. 🎯 数据流向图

```
┌─────────────────┐    实时数据     ┌─────────────────┐    处理后数据    ┌─────────────────┐
│ 外部数据源      │ ────────────► │ Java数据管理    │ ─────────────► │ 内存缓存        │
│ Yahoo/Alpha/IEX │               │ MarketDataMgr   │                │ DataBuffer      │
└─────────────────┘               └─────────────────┘                └─────────────────┘
                                           │                                   │
                                           │                                   │
                                           ▼                                   ▼
┌─────────────────┐    历史数据     ┌─────────────────┐    计算指标      ┌─────────────────┐
│ 历史数据存储    │ ◄──────────── │ 数据持久化      │ ◄─────────────── │ 技术指标计算    │
│ 文件/数据库     │               │ DatabaseMgr     │                  │ RSI/MACD/MA     │
└─────────────────┘               └─────────────────┘                  └─────────────────┘
                                                                               │
                                                                               │
                                                                               ▼
┌─────────────────┐    JSON请求     ┌─────────────────┐    AI预测       ┌─────────────────┐
│ Python AI服务   │ ◄──────────── │ AIStrategyClient│ ◄─────────────── │ 特征工程        │
│ ai_model_service│               │ HTTP调用        │                  │ 39个特征        │
└─────────────────┘               └─────────────────┘                  └─────────────────┘
       │                                   │
       │                                   │
       ▼                                   ▼
┌─────────────────┐    信号结果     ┌─────────────────┐    高置信度信号   ┌─────────────────┐
│ 集成模型预测    │ ────────────► │ 信号处理        │ ─────────────► │ 通知发送        │
│ RF+GB+LR       │               │ SignalProcessor │                │ 邮件+微信       │
└─────────────────┘               └─────────────────┘                └─────────────────┘
```

## 🔧 关键配置文件

### 1. 数据源配置
```properties
# platform/src/main/resources/application.properties

# 主数据源选择
data.source.type=YAHOO_FINANCE

# Yahoo Finance配置（免费）
data.source.yahoo.finance.base.url=https://query1.finance.yahoo.com/v8/finance/chart

# Alpha Vantage配置（需要API Key）
data.source.alpha.vantage.api.key=YOUR_API_KEY_HERE
data.source.alpha.vantage.base.url=https://www.alphavantage.co/query

# 数据更新频率
trading.data.collection.interval=60          # 实时数据收集间隔（秒）
trading.strategy.execution.interval=300      # 策略执行间隔（秒）

# 监控的股票列表
trading.symbols=ASML,NBIS,TSLA,SE,GRAB,QQQ,VOO,MRVL,3690.HK,HIMS,FPE,NKE
```

### 2. AI服务配置
```properties
# AI服务连接配置
ai.service.url=http://localhost:5000
ai.service.timeout.connect=10000
ai.service.timeout.socket=30000
ai.service.retry.max=3
```

## 🧪 测试整个数据流

现在让我测试AI模型服务：

<function_calls>
<invoke name="run_terminal_cmd">
<parameter name="command">curl -X POST http://localhost:5000/get_signal -H "Content-Type: application/json" -d '{
  "symbol": "TSLA",
  "current_data": {
    "open": 248.50,
    "high": 252.30,
    "low": 247.80,
    "close": 250.15,
    "volume": 1500000
  },
  "indicators": {
    "RSI": 45.5,
    "MACD": 0.3,
    "MA5": 249.20,
    "MA10": 248.80,
    "MA20": 245.80,
    "VOLATILITY": 0.025,
    "VOLUME_RATIO": 1.2
  },
  "history": []
}'
