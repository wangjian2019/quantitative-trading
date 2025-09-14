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

**© 2025 AI量化交易平台 v0.1 by Alvin | 专业级量化交易解决方案**

---

*最后更新: 2025年9月*