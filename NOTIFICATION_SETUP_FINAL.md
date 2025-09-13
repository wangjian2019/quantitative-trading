# 🔔 通知配置完整指南 - 最终版本

## 🏗️ 新架构说明

由于Google停用了应用专用密码，我已经重新设计了系统架构：

- **Python服务**: 只负责AI模型和策略计算 (`ai_model_service.py`)
- **Java服务**: 负责数据管理、交易决策、通知发送、Web界面

## 📧 邮件通知配置

### 推荐方案: QQ邮箱（仍支持授权码）

#### 步骤1: 开启QQ邮箱SMTP服务
1. **登录QQ邮箱**: https://mail.qq.com/
2. **进入设置**: 点击"设置" → "账户"
3. **开启SMTP**: 找到"POP3/IMAP/SMTP/Exchange/CardDAV/CalDAV服务"
4. **点击开启**: 按提示发送短信验证
5. **获取授权码**: 开启成功后显示授权码（16位字符）
6. **保存授权码**: 复制保存这个授权码

#### 步骤2: 配置系统
编辑 `platform/src/main/resources/application.properties`:

```properties
# 邮件通知配置（使用QQ邮箱）
email.enabled=true
email.username=your_qq_email@qq.com                # 你的QQ邮箱
email.password=abcdefghijklmnop                     # QQ邮箱授权码（16位）
email.notification.address=wangjians8813@gmail.com # 接收通知的邮箱
email.smtp.host=smtp.qq.com
email.smtp.port=587
```

### 备选方案: 163邮箱

```properties
# 邮件通知配置（使用163邮箱）
email.enabled=true
email.username=your_email@163.com
email.password=your_163_auth_code
email.notification.address=wangjians8813@gmail.com
email.smtp.host=smtp.163.com
email.smtp.port=587
```

## 💬 微信通知配置

### 方案1: Server酱（最简单，推荐）

#### 步骤1: 注册Server酱
1. **访问**: https://sct.ftqq.com/
2. **微信扫码登录**
3. **复制SendKey**: 在首页复制SendKey（格式：SCT123xxx...）

#### 步骤2: 配置系统
```properties
# 微信通知配置（Server酱）
wechat.enabled=true
wechat.webhook.url=https://sctapi.ftqq.com/SCT123xxx你的SendKey.send
```

### 方案2: 企业微信群机器人

#### 步骤1: 创建企业微信群
1. 下载企业微信APP
2. 创建群聊
3. 群设置 → 群机器人 → 添加机器人
4. 复制Webhook URL

#### 步骤2: 配置系统
```properties
# 微信通知配置（企业微信）
wechat.enabled=true
wechat.webhook.url=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=你的key
```

### 方案3: Pushplus

#### 步骤1: 注册Pushplus
1. **访问**: http://www.pushplus.plus/
2. **微信扫码登录**
3. **复制Token**

#### 步骤2: 配置系统
```properties
# 微信通知配置（Pushplus）
wechat.enabled=true
wechat.webhook.url=http://www.pushplus.plus/send?token=你的token
```

## 🔧 完整配置示例

编辑 `platform/src/main/resources/application.properties`:

```properties
# ========== 通知配置 ==========

# 邮件通知配置（QQ邮箱推荐）
email.enabled=true
email.username=your_qq_email@qq.com                # 替换为你的QQ邮箱
email.password=abcdefghijklmnop                     # 替换为QQ邮箱授权码
email.notification.address=wangjians8813@gmail.com # 接收通知的邮箱
email.smtp.host=smtp.qq.com
email.smtp.port=587

# 微信通知配置（Server酱推荐）
wechat.enabled=true
wechat.webhook.url=https://sctapi.ftqq.com/SCT123xxxYourSendKey.send  # 替换为你的SendKey

# 通知规则
notification.min.confidence=0.75          # 只有置信度≥75%的信号才发通知
notification.send.daily.summary=true      # 发送每日总结

# ========== 其他配置 ==========

# AI服务配置
ai.service.url=http://localhost:5000

# 交易配置
trading.initial.capital=100000.0
trading.symbols=ASML,NBIS,TSLA,SE,GRAB,QQQ,VOO,MRVL,3690.HK,HIMS,FPE,NKE

# 风险管理
risk.max.position.ratio=0.3
risk.stop.loss.ratio=0.05
risk.take.profit.ratio=0.15

# UI服务器配置
ui.server.host=localhost
ui.server.port=8080
```

## 🚀 启动和测试

### 1. 启动服务
```bash
# 启动Python AI模型服务
cd strategy && python3 ai_model_service.py

# 启动Java平台服务（新终端）
cd platform && mvn exec:java -Dexec.mainClass="com.alvin.quantitative.trading.platform.TradingPlatformApplication"
```

### 2. 访问Web界面
- **主界面**: http://localhost:8080
- **AI模型服务**: http://localhost:5000

### 3. 测试通知功能
在Java应用的控制台菜单中，可以测试通知配置。

## 📱 通知功能说明

### 🔔 什么时候会收到通知？

#### 📧 邮件通知
- **高置信度交易信号** (≥75%): 买入/卖出建议
- **投资组合预警**: 大幅盈亏时提醒
- **每日总结报告**: 交易日结束后发送
- **系统异常**: 服务故障时通知

#### 💬 微信通知
- **即时交易信号**: 重要交易机会立即推送
- **风险预警**: 市场异动和风险提醒
- **系统状态**: 服务启停和健康状态

### 📊 通知内容示例

#### 交易信号通知
```
📈 交易信号提醒

📊 股票代码: TSLA
📈 建议操作: BUY
💰 当前价格: $250.15
🎯 置信度: 85.2%
📝 分析理由: RSI超卖且突破MA5，成交量放大
⏰ 信号时间: 2024-09-13 15:30:25

请根据您的风险承受能力和投资策略谨慎决策。
```

#### 投资组合预警
```
🎉 投资组合提醒

您的投资组合表现优异！
总资产: $125,000.00
总收益率: 25.00%
今日盈亏: $1,250.00

发送时间: 2024-09-13 16:00:00
```

## 🧪 测试步骤

### 1. 测试QQ邮箱配置
```bash
# 在Java应用控制台中选择测试功能
# 或者查看日志确认配置加载
tail -f platform/logs/trading.log
```

### 2. 测试微信通知
确保配置了正确的Webhook URL后，系统会自动发送测试消息。

### 3. 验证配置
- ✅ 检查QQ邮箱是否收到测试邮件
- ✅ 检查微信是否收到测试消息
- ✅ 查看系统日志确认无错误

## 🔍 故障排除

### QQ邮箱问题
- **授权码错误**: 重新生成QQ邮箱授权码
- **SMTP连接失败**: 检查网络和防火墙
- **认证失败**: 确认用户名和授权码正确

### 微信通知问题
- **Webhook无效**: 检查URL格式和有效性
- **发送失败**: 查看Server酱/企业微信后台状态
- **消息不到达**: 确认微信账号正常

### 系统日志
```bash
# 查看Java服务日志
tail -f platform/logs/trading.log

# 查看Python模型服务日志
tail -f strategy/logs/ai_service.log
```

## 📞 技术支持

### 常用检查命令
```bash
# 检查服务状态
curl http://localhost:8080/health
curl http://localhost:5000/health

# 检查端口占用
lsof -i :8080
lsof -i :5000

# 重启服务
pkill -f TradingPlatformApplication
pkill -f ai_model_service
```

### 配置验证
1. **邮箱配置**: 确认QQ邮箱SMTP已开启
2. **微信配置**: 确认Webhook URL可访问
3. **网络连接**: 确认可以访问外部API
4. **权限设置**: 确认应用有网络访问权限

---

## 🎯 配置完成检查清单

- [ ] QQ邮箱SMTP服务已开启
- [ ] QQ邮箱授权码已获取并配置
- [ ] 微信通知服务已选择并配置
- [ ] `application.properties`文件已更新
- [ ] Python AI模型服务正常运行
- [ ] Java平台服务正常运行
- [ ] Web界面可以正常访问
- [ ] 通知测试功能正常工作

**✅ 完成以上配置后，你的AI量化交易平台将具备完整的智能通知功能！**

---

**🚀 AI量化交易平台 v2.0 by Alvin | 专业级量化交易解决方案**
