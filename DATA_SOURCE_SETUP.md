# 📊 股票数据源配置指南

**Author: Alvin**

本指南帮助你配置真实的股票数据API，获取实时市场数据。

## 🚀 快速开始

### **方式1: 使用Yahoo Finance（免费，推荐）**

```properties
# 在 application.properties 中设置
data.source.type=YAHOO_FINANCE
```

✅ **优点**: 完全免费，无需API密钥，数据质量好  
⚠️ **注意**: 可能有使用限制，仅供个人学习使用

---

### **方式2: 使用Alpha Vantage（免费额度）**

1. **获取API密钥**:
   - 访问: https://www.alphavantage.co/support/#api-key
   - 免费注册获取API密钥
   - 免费额度: 500次/天, 5次/分钟

2. **配置**:
```properties
# 在 application.properties 中设置
data.source.type=ALPHA_VANTAGE
data.source.alpha.vantage.api.key=你的API密钥
```

---

## 📋 所有支持的数据源

| 数据源 | 类型 | 免费额度 | 数据质量 | 配置难度 |
|--------|------|----------|----------|----------|
| **Yahoo Finance** | 免费 | 无限制* | ⭐⭐⭐⭐ | 简单 |
| **Alpha Vantage** | 免费/付费 | 500次/天 | ⭐⭐⭐⭐⭐ | 简单 |
| **IEX Cloud** | 免费/付费 | 50万次/月 | ⭐⭐⭐⭐⭐ | 中等 |
| **Polygon.io** | 付费 | 专业级 | ⭐⭐⭐⭐⭐ | 中等 |
| **模拟数据** | 免费 | 无限制 | ⭐⭐⭐ | 无 |

*合理使用范围内

---

## ⚙️ 详细配置

### **1. Yahoo Finance配置**
```properties
# application.properties
data.source.type=YAHOO_FINANCE
data.source.yahoo.finance.base.url=https://query1.finance.yahoo.com/v8/finance/chart
data.fetch.timeout=10000
data.fetch.retry.max=3
```

### **2. Alpha Vantage配置**
```properties
# application.properties  
data.source.type=ALPHA_VANTAGE
data.source.alpha.vantage.api.key=你的API密钥
data.source.alpha.vantage.base.url=https://www.alphavantage.co/query
data.fetch.timeout=10000
data.fetch.retry.max=3
```

### **3. 环境变量配置（推荐）**
```bash
# 设置环境变量（更安全）
export ALPHA_VANTAGE_API_KEY=你的API密钥
export DATA_SOURCE_TYPE=ALPHA_VANTAGE

# 或在启动时设置
java -Ddata.source.type=ALPHA_VANTAGE -Ddata.source.alpha.vantage.api.key=你的密钥 ...
```

---

## 🔑 获取API密钥指南

### **Alpha Vantage (推荐)**
1. 访问: https://www.alphavantage.co/support/#api-key
2. 填写邮箱地址
3. 点击"GET FREE API KEY"
4. 检查邮箱获取API密钥

**免费限制**: 500次/天, 5次/分钟

### **IEX Cloud**
1. 访问: https://iexcloud.io/
2. 注册账户
3. 在Dashboard获取API密钥

**免费限制**: 50万次/月

### **Polygon.io**
1. 访问: https://polygon.io/
2. 注册账户（需要付费）
3. 获取API密钥

**付费服务**: 专业级实时数据

---

## 🧪 测试数据源

### **方法1: 使用配置测试**
```bash
# 启动系统后查看日志
./start_all.sh

# 看到类似输出表示成功:
# ✅ Data source initialized: Yahoo Finance (Free with reasonable use limits)
# 📊 [AAPL] Real-time data: $175.32 (Vol: 45,123,456) from Yahoo Finance
```

### **方法2: 手动测试API**
```bash
# 测试Alpha Vantage
curl "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=你的密钥"

# 测试Yahoo Finance  
curl "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?period1=1640995200&period2=1641081600&interval=1d"
```

---

## 🔧 故障排除

### **常见问题**

#### **1. API密钥无效**
```
错误: Alpha Vantage error: Invalid API call
解决: 检查API密钥是否正确，是否已激活
```

#### **2. 达到速率限制**
```
错误: Alpha Vantage rate limit: Thank you for using Alpha Vantage!
解决: 等待或切换到Yahoo Finance
```

#### **3. 网络连接问题**
```
错误: Failed to fetch real-time data: Connection timeout
解决: 检查网络连接，增加timeout设置
```

#### **4. 股票代码错误**
```
错误: No data available for symbol: INVALID
解决: 使用正确的股票代码 (如: AAPL, TSLA, MSFT)
```

### **自动降级机制**
系统具有智能降级功能：
1. **主数据源失败** → 自动切换到备用数据源
2. **所有外部API失败** → 自动切换到模拟数据
3. **保证系统持续运行** → 不会因数据问题停止

---

## 📊 数据源对比

### **实时性对比**
- **Yahoo Finance**: 15分钟延迟（免费）
- **Alpha Vantage**: 实时数据（免费有限制）
- **IEX Cloud**: 实时数据
- **Polygon**: 实时数据（毫秒级）

### **数据覆盖**
- **美股**: 所有数据源都支持
- **国际市场**: Alpha Vantage和Polygon支持更多
- **加密货币**: Alpha Vantage和Polygon支持
- **外汇**: Alpha Vantage支持

---

## 🚀 推荐配置

### **学习/测试环境**
```properties
data.source.type=YAHOO_FINANCE
```

### **开发环境**
```properties
data.source.type=ALPHA_VANTAGE
data.source.alpha.vantage.api.key=你的密钥
```

### **生产环境**
```properties
data.source.type=POLYGON
data.source.polygon.api.key=你的付费密钥
```

---

## 🔄 运行示例

配置完成后，启动系统：

```bash
# 启动完整系统
./start_all.sh

# 查看实时数据获取
输入: s (查看状态)

# 期望看到:
# 📊 [AAPL] Real-time data: $175.32 (Vol: 45,123,456) from Yahoo Finance
# 📊 [TSLA] Real-time data: $250.15 (Vol: 32,456,789) from Yahoo Finance
# 📊 [MSFT] Real-time data: $350.67 (Vol: 28,789,123) from Yahoo Finance
```

---

## ⚠️ 重要提醒

1. **API密钥安全**: 不要将API密钥提交到代码仓库
2. **使用限制**: 遵守各API提供商的使用条款
3. **备份方案**: 配置多个数据源以防单点故障
4. **成本控制**: 监控API调用次数，避免超出免费额度
5. **法律合规**: 仅用于个人学习和研究目的

---

**配置完成后，你的AI量化交易平台就可以获取真实的股票市场数据了！** 🎉

需要帮助？请查看系统日志或联系技术支持。
