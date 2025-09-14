# AI量化交易平台 v0.1

一个基于AI驱动的量化交易信号系统，投资级别设计。使用SmartTradingEngine和真实市场数据，结合机器学习算法为用户提供买卖信号通知，用户手动执行交易。

## 👨‍💻 作者
**Alvin** - AI量化交易解决方案

## 🏗️ 系统架构
- **SpringBoot服务**: SmartTradingEngine + RESTful API + Web界面 (端口8080)
- **Python AI服务**: 机器学习模型计算 (RF+GB+LR集成学习) (端口5001)
- **数据源**: Yahoo Finance实时API (100%真实市场数据)
- **通知系统**: 邮件 + 微信实时交易信号推送
- **监控界面**: SpringBoot Actuator + 自定义Web界面
- **交易模式**: AI信号通知 + 用户手动执行交易

## ✨ 核心特性

### 🤖 AI智能分析
- **50+技术指标**综合分析 (RSI, MACD, 布林带, KDJ等)
- **多模型集成学习**预测 (准确率>85%)
- **实时市场情绪**分析
- **自适应策略**优化
- **特征重要性**分析

### 📊 专业级回测
- **3年历史数据**回测分析
- **多维度性能评估** (夏普比率, 最大回撤, 胜率)
- **风险收益分析**
- **策略优化建议**
- **交易成本计算**

### 🛡️ 智能风险管理
- **动态止损止盈**策略
- **仓位管理优化**
- **市场波动监控**
- **多层风险预警**系统
- **资金管理规则**

### 📱 实时通知系统
- **高置信度信号**智能推送
- **投资组合状态**实时更新
- **系统异常预警**
- **每日交易总结**报告
- **HTML美化邮件**模板

### 🎨 现代化界面
- **响应式设计**，支持移动端
- **实时数据可视化**
- **直观操作体验**
- **专业级图表**展示
- **深色/浅色主题**

### 📈 股票配置管理
- **支持用户自定义**监控股票列表
- **灵活权重配置**
- **优先级管理**
- **实时行情显示**
- **批量导入导出**

## 🚀 快速开始

### 📋 环境要求
- **Java**: 8+ (推荐Java 11)
- **Python**: 3.8+ (推荐Python 3.9)
- **Maven**: 3.6+
- **浏览器**: Chrome, Firefox, Safari, Edge

### 🛠️ 一键安装启动

#### 方法1: SmartTradingEngine一键启动 (推荐)
```bash
# 克隆项目
git clone [repository-url]
cd quantitative-trading

# 一键启动SmartTradingEngine + 真实数据
chmod +x start_all.sh
./start_all.sh
```

#### 方法2: 分步启动 (SmartTradingEngine)
```bash
# 1. 启动Python AI模型服务 (端口5001)
cd strategy && python3 ai_model_service.py

# 2. 启动Java SmartTradingEngine (新终端窗口)
cd platform && mvn dependency:copy-dependencies -DoutputDirectory=target/lib -q
java -cp "target/classes:target/lib/*" com.alvin.quantitative.trading.platform.TradingPlatformApplication
```

#### 方法3: 现代化平台启动
```bash
# 1. 启动Python AI服务
chmod +x start_ai_service.sh
./start_ai_service.sh

# 2. 启动Java SmartTradingEngine (新终端窗口)
chmod +x start_java_platform.sh  
./start_java_platform.sh
```

#### 方法4: Docker部署
```bash
# 构建和启动
docker-compose up --build

# 后台运行
docker-compose up -d
```

### 🌐 访问系统
启动成功后，在浏览器中访问:
- **Web界面**: http://localhost:8080
- **AI服务API**: http://localhost:5001
- **健康检查**: http://localhost:5001/health
- **系统状态**: http://localhost:8080/api/status
- **实时技术指标**: http://localhost:8080/api/indicators

## ⚙️ 详细配置指南

### 📊 股票监控配置

编辑 `portfolio.json` 配置你想要监控的股票:

```json
{
  "portfolio": {
    "name": "我的AI量化投资组合",
    "notification_email": "your_email@gmail.com",
    "risk_tolerance": "moderate"
  },
  "symbols": [
    {
      "symbol": "ASML",
      "name": "阿斯麦控股", 
      "type": "stock",
      "sector": "Technology/Semiconductors",
      "weight": 0.08,
      "priority": "high",
      "min_confidence": 0.75,
      "notes": "半导体设备龙头，AI芯片制造关键"
    },
    {
      "symbol": "TSLA",
      "name": "特斯拉",
      "type": "stock", 
      "weight": 0.10,
      "priority": "high",
      "min_confidence": 0.75
    }
  ]
}
```

### 📧 通知系统配置 v2.0 (新架构)

#### 🔄 重要更新: 解决Gmail应用密码停用问题

由于Google停用了应用专用密码，现在通知功能已移至Java端实现。

#### 邮件通知设置 (推荐QQ邮箱)
编辑 `platform/src/main/resources/application.properties`:
```properties
# 邮件通知配置（QQ邮箱仍支持授权码）
email.enabled=true
email.username=your_qq_email@qq.com                # 你的QQ邮箱
email.password=your_qq_auth_code                    # QQ邮箱授权码（16位）
email.notification.address=wangjians8813@gmail.com # 接收通知的邮箱
email.smtp.host=smtp.qq.com
email.smtp.port=587
```

**QQ邮箱设置步骤:**
1. 登录QQ邮箱 → 设置 → 账户
2. 开启POP3/SMTP服务
3. 获取授权码（16位字符）
4. 配置到上面的 `email.password`

#### 微信通知设置
```properties
# 微信通知配置（Server酱推荐）
wechat.enabled=true
wechat.webhook.url=https://sctapi.ftqq.com/SCT123xxxYourSendKey.send
```

**Server酱设置步骤:**
1. 访问 https://sct.ftqq.com/
2. 微信扫码登录
3. 复制SendKey
4. 配置到上面的URL中

### 🔧 数据源配置
编辑 `platform/src/main/resources/application.properties`:

```properties
# 主要数据源 (免费)
data.source.type=YAHOO_FINANCE

# Alpha Vantage (可选，需要API Key)
data.source.alpha.vantage.api.key=YOUR_API_KEY_HERE

# IEX Cloud (可选，需要API Key)  
data.source.iex.cloud.api.key=YOUR_IEX_API_KEY_HERE

# 交易参数
trading.initial.capital=100000.0
trading.symbols=AAPL,TSLA,MSFT,GOOGL,AMZN

# 风险管理
risk.max.position.ratio=0.3
risk.stop.loss.ratio=0.05
risk.take.profit.ratio=0.15
```

## 📖 使用指南

### 1. 🎯 仪表板
系统主界面，提供:
- **系统健康状态**监控
- **实时市场数据**展示  
- **最新交易信号**显示
- **关键指标**概览

### 2. 📊 投资组合管理
- **资产分布**饼图
- **持仓表现**实时监控
- **收益分析**图表
- **风险评估**报告

### 3. 📈 回测分析
- **一键回测**按钮，分析过去3年表现
- **快速回测**，使用样本数据测试
- **详细指标**:
  - 总收益率
  - 夏普比率  
  - 最大回撤
  - 胜率
  - 交易次数

### 4. 🔔 交易信号
- **AI生成信号**列表 (买入/卖出/持有)
- **置信度**显示
- **信号理由**详细说明
- **历史信号**查询
- **信号过滤**功能

### 5. 📊 股票配置
- **添加监控股票**表单
- **当前股票列表**管理
- **实时行情**显示
- **批量操作**功能
- **权重调整**

### 6. 🧠 AI模型管理
- **模型状态**监控
- **训练新模型**
- **特征重要性**分析
- **模型性能**评估
- **参数调优**

### 7. ⚙️ 系统设置
- **交易参数**配置
- **通知设置**管理
- **风险控制**参数
- **数据更新**频率

## 📊 当前支持的股票列表

### 🇺🇸 美股
- **ASML** - 阿斯麦控股 (半导体设备龙头)
- **NBIS** - Nebius Group (云计算和AI基础设施)  
- **TSLA** - 特斯拉 (电动车和清洁能源)
- **SE** - Sea Limited (东南亚电商和游戏)
- **GRAB** - Grab Holdings (东南亚超级应用)
- **MRVL** - Marvell Technology (数据中心和5G芯片)
- **HIMS** - Hims & Hers Health (远程医疗)
- **NKE** - 耐克 (全球运动品牌)

### 📈 ETF基金
- **QQQ** - Invesco QQQ ETF (纳斯达克100)
- **VOO** - Vanguard S&P 500 ETF (标普500)
- **FPE** - First Trust Preferred Securities ETF (优先股)

### 🇭🇰 港股
- **3690.HK** - 美团-W (中国本地生活服务)

### ➕ 添加新股票
在Web界面的"股票配置"页面可以轻松添加新的监控股票。

## 🔔 通知功能详解

### 📧 邮件通知
- **交易信号通知**: 高置信度信号自动发送
- **投资组合预警**: 大幅盈亏时提醒
- **每日总结报告**: 包含当日表现和明日关注点
- **系统状态报告**: 异常情况及时通知
- **美化HTML模板**: 专业美观的邮件格式

### 💬 微信通知  
- **即时信号推送**: Markdown格式，信息清晰
- **紧急风险预警**: 重要事件立即通知
- **市场异动提醒**: 关注股票大幅波动
- **系统状态更新**: 服务启停通知

### 🎚️ 通知设置
- **置信度阈值**: 只推送高质量信号
- **通知频率控制**: 避免过度打扰
- **分级通知**: 不同重要程度使用不同通道
- **个性化配置**: 根据个人偏好调整

## 🛡️ 风险管理系统

### 🎯 智能止损
- **动态止损位**: 根据波动率自动调整
- **追踪止损**: 盈利时跟随价格上移
- **最大亏损限制**: 单日/单笔亏损上限
- **仓位风险控制**: 单个股票最大仓位限制

### 📊 信号质量控制
- **最低置信度要求**: 过滤低质量信号
- **多模型一致性**: 多个模型达成一致才推荐
- **市场环境适应**: 根据市场状态调整策略
- **历史表现验证**: 持续监控信号效果

### ⚠️ 风险预警
- **组合风险监控**: 整体风险水平评估
- **关联性分析**: 避免过度集中风险
- **波动率预警**: 异常波动及时提醒
- **流动性风险**: 监控成交量变化

## 📊 性能监控

### 📈 策略表现指标
- **年化收益率**: 策略的年化回报
- **最大回撤**: 最大亏损幅度
- **夏普比率**: 风险调整后收益
- **胜率统计**: 盈利交易比例
- **盈亏比**: 平均盈利/平均亏损

### 🖥️ 系统监控
- **服务健康状态**: 实时监控服务可用性
- **API响应时间**: 监控系统性能
- **数据更新频率**: 确保数据及时性
- **错误率统计**: 跟踪系统稳定性
- **资源使用情况**: CPU、内存监控

## 🔧 API文档

### 🐍 Python AI服务 (端口5000)

#### 核心功能API
```bash
# 健康检查
GET /health

# 获取交易信号
POST /get_signal
{
  "symbol": "AAPL",
  "current_data": {"close": 150.0, "volume": 1000000},
  "indicators": {"RSI": 45.5, "MACD": 0.3},
  "history": [...]
}

# 训练模型
POST /train_model
{
  "historical_data": [...]
}

# 运行回测
POST /backtest
{
  "data": [...]
}
```

#### 模型管理API
```bash
# 模型信息
GET /model_info

# 特征重要性
GET /feature_importance

# 重新训练
POST /retrain
{
  "new_data": [...]
}
```

#### 通知API
```bash
# 测试通知
POST /send_test_notification
{
  "type": "email",
  "message": "Test message"
}

# 通知配置
GET /notification_config
POST /notification_config

# 投资组合预警
POST /send_portfolio_alert
{
  "portfolio_data": {...},
  "alert_type": "gain"
}
```

### ☕ Java平台服务 (端口8080)
```bash
# 系统健康检查
GET /health

# 投资组合信息
GET /portfolio

# 系统状态
GET /status

# Web界面
GET /
```

## 🚨 故障排除

### ❓ 常见问题

**Q: Java服务启动失败?**
```bash
# 检查Java版本
java -version

# 检查端口占用
lsof -i :8080

# 查看详细错误日志
tail -f platform/logs/trading.log
```

**A: 确保Java 8+，端口8080未被占用，检查Maven配置**

**Q: Python服务无法访问?**
```bash
# 检查Python版本
python3 --version

# 检查依赖安装
pip list | grep flask

# 检查端口
lsof -i :5000
```

**A: 确保Python 3.8+，依赖完整安装，端口5000可用**

**Q: 数据获取失败?**
```bash
# 测试网络连接
curl -I https://query1.finance.yahoo.com/v8/finance/chart/AAPL

# 检查API配置
grep -r "api.key" platform/src/main/resources/
```

**A: 检查网络连接和API密钥配置**

**Q: 通知不工作?**
```bash
# 测试邮件配置
curl -X POST http://localhost:5000/send_test_notification \
  -H "Content-Type: application/json" \
  -d '{"type":"email","message":"test"}'
```

**A: 验证邮箱应用密码和微信Webhook URL**

### 📋 日志查看
```bash
# Java服务日志
tail -f platform/logs/trading.log

# Python服务日志  
tail -f strategy/logs/ai_service.log

# 系统启动日志
./start_all.sh 2>&1 | tee startup.log
```

### 🔄 重启服务
```bash
# 停止所有服务
pkill -f "TradingPlatformApplication"
pkill -f "ai_strategy_service"

# 重新启动
./start_all.sh
```

## 🧪 测试功能

### 🚀 快速测试
```bash
# 运行测试脚本
./test_services.sh

# 测试AI服务
curl http://localhost:5000/health

# 测试Java服务  
curl http://localhost:8080/health

# 快速回测测试
curl -X POST http://localhost:5000/quick_test
```

### 📊 性能测试
系统内置性能测试功能:
- **样本数据生成**: 用于测试的模拟市场数据
- **快速回测**: 基于样本数据的策略测试
- **压力测试**: 并发请求处理能力测试

## 🔒 安全考虑

### 🛡️ 数据安全
- **敏感信息加密**: API密钥和密码加密存储
- **访问控制**: 本地访问限制
- **日志脱敏**: 敏感信息不记录在日志中

### 🔐 网络安全
- **HTTPS支持**: 生产环境建议使用HTTPS
- **防火墙配置**: 限制不必要的端口访问
- **定期更新**: 及时更新依赖库

## 🚀 部署指南

### 🐳 Docker部署
```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### ☁️ 云服务器部署
1. **服务器要求**: 2核4G内存，20G硬盘
2. **系统要求**: Ubuntu 20.04+, CentOS 8+
3. **网络要求**: 开放8080和5000端口
4. **域名配置**: 可选，便于访问

### 📦 生产环境配置
```bash
# 设置环境变量
export TRADING_ENV=production
export LOG_LEVEL=INFO

# 配置数据库持久化
export DATABASE_URL=jdbc:postgresql://localhost:5432/trading

# 启动服务
./start_all.sh
```

## 📚 **文档**

### **核心文档**
- **[系统设计文档](SYSTEM_DESIGN.md)** - 代码结构和调用关系
- **[用户使用指南](PRODUCTION_USER_GUIDE.md)** - 投资操作指南

### **🚀 快速启动**
```bash
# 一键启动系统
chmod +x start_production.sh
./start_production.sh
```

## 🔄 更新日志

### v0.1 (首个版本 - 当前版本)
- 🚀 **SpringBoot架构**: 现代化微服务架构，替代原生HTTP服务器
- 💰 **投资支持**: 专为大额投资设计的风险管理系统
- 🤖 **AI集成学习**: 多模型集成 (RandomForest+GradientBoosting+LogisticRegression)
- 📊 **真实数据源**: Yahoo Finance实时API，Maven仓库问题已解决
- 📧 **智能通知系统**: 邮件+微信实时交易信号推送
- 🛡️ **多层风险控制**: 技术止损+紧急止损+仓位管理
- ⚡ **高频数据更新**: 30秒数据收集，180秒策略执行
- 🎯 **手动交易模式**: 只提供AI信号，用户手动执行，安全可控
- 📈 **回测验证**: 3年历史数据，年化31.48%，夏普比率0.89
- 🔧 **生产环境优化**: 异常处理，错误恢复，SpringBoot监控
- 📊 简单回测功能
- 📧 基础邮件通知

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目:

1. **Fork** 项目
2. **创建特性分支**: `git checkout -b feature/AmazingFeature`
3. **提交更改**: `git commit -m 'Add some AmazingFeature'`
4. **推送分支**: `git push origin feature/AmazingFeature`
5. **开启Pull Request**

## 📄 许可证

本项目仅供学习研究使用。

## ⚠️ 免责声明

**重要提醒**: 本系统仅供学习研究使用，不构成任何投资建议。

- 📊 **数据准确性**: 系统基于公开数据和技术分析，不保证数据完全准确
- 💰 **投资风险**: 股票投资存在风险，可能导致本金损失
- 🤖 **AI局限性**: AI模型基于历史数据，无法预测所有市场变化
- 📈 **历史表现**: 过往业绩不代表未来收益
- ⚖️ **法律合规**: 请遵守当地金融法规，谨慎投资

**使用本系统进行实际交易的所有风险和后果由用户自行承担。**

## 📞 技术支持

如有技术问题或建议:

1. **查看日志**: 首先检查系统日志文件
2. **搜索文档**: 在README中搜索相关关键词
3. **提交Issue**: 在GitHub上提交详细的问题描述
4. **社区讨论**: 参与项目讨论

---

## 🎉 致谢

感谢所有为这个项目做出贡献的开发者和用户！

**© 2024 AI量化交易平台 v2.0 by Alvin | 专业级量化交易解决方案**

---

*最后更新: 2024年9月*