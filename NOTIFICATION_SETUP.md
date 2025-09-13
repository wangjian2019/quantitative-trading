# 📧 通知系统配置指南

**Author: Alvin**

本指南帮助你配置邮件和微信通知，实时接收AI量化交易信号。

## 📧 **邮件通知配置**

### **第1步：获取Gmail应用密码**

1. **登录Gmail账户** (建议使用专门的邮箱)
2. **开启两步验证**:
   - 访问: https://myaccount.google.com/security
   - 点击"两步验证"并按提示设置

3. **生成应用密码**:
   - 在"两步验证"页面，点击"应用密码"
   - 选择"邮件"和"其他（自定义名称）"
   - 输入"AI量化交易平台"
   - 复制生成的16位密码

### **第2步：配置邮件设置**

编辑配置文件：
```bash
nano /Users/alvin.wang/alvin-eclipse-workspace-new/quantitative-trading/platform/src/main/resources/application.properties
```

修改邮件配置：
```properties
# 邮件通知配置
email.username=your_email@gmail.com
email.password=your_16_digit_app_password
email.smtp.host=smtp.gmail.com
email.smtp.port=587
```

### **第3步：更新投资组合配置**

编辑投资组合文件：
```bash
nano /Users/alvin.wang/alvin-eclipse-workspace-new/quantitative-trading/portfolio.json
```

确保邮件地址正确：
```json
{
  "portfolio": {
    "notification_email": "wangjians8813@gmail.com"
  },
  "notification_settings": {
    "email": {
      "enabled": true,
      "address": "wangjians8813@gmail.com",
      "send_daily_summary": true,
      "send_trade_signals": true,
      "send_performance_reports": true
    }
  }
}
```

---

## 💬 **微信通知配置**

### **方式1：企业微信机器人（推荐）**

1. **创建企业微信群**:
   - 下载企业微信APP
   - 创建一个群聊
   - 添加机器人

2. **获取Webhook地址**:
   - 在群聊中，点击右上角"..."
   - 选择"群机器人"
   - 添加机器人，获取Webhook URL

3. **配置Webhook**:
   ```json
   {
     "notification_settings": {
       "wechat": {
         "enabled": true,
         "webhook_url": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY_HERE"
       }
     }
   }
   ```

### **方式2：微信测试号（开发者）**

1. **申请测试号**:
   - 访问: https://developers.weixin.qq.com/sandbox
   - 扫码登录获取测试号

2. **配置模板消息**:
   - 设置模板ID和用户OpenID
   - 配置消息模板

---

## 🔧 **配置示例**

### **完整的application.properties配置**
```properties
# 邮件通知配置
email.username=wangjians8813@gmail.com
email.password=abcd efgh ijkl mnop
email.smtp.host=smtp.gmail.com
email.smtp.port=587

# 微信通知配置 
wechat.webhook.url=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=693axxx6-7aoc-4bc4-97a0-0ec2sifa5aaa
```

### **完整的portfolio.json配置**
```json
{
  "portfolio": {
    "name": "Alvin的AI量化投资组合",
    "notification_email": "wangjians8813@gmail.com"
  },
  "notification_settings": {
    "email": {
      "enabled": true,
      "address": "wangjians8813@gmail.com",
      "send_daily_summary": true,
      "send_trade_signals": true,
      "send_performance_reports": true
    },
    "wechat": {
      "enabled": true,
      "webhook_url": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=YOUR_KEY",
      "send_immediate_alerts": true,
      "send_daily_summary": false
    }
  }
}
```

---

## 📨 **通知内容示例**

### **邮件通知示例**
```
主题: 🚀 AI量化交易信号 - AAPL BUY

📊 AAPL (苹果公司)
Technology | stock

💰 当前价格: $175.32
🎯 操作建议: 🚀 BUY
🔥 置信度: 85.0%
📈 权重配置: 15.0%
⭐ 优先级: high

💡 AI分析理由
BUY signal with 85% confidence: RSI oversold condition, Strong positive momentum, Multiple bullish signals

📝 备注信息
科技龙头，长期看好

⏰ 生成时间: 2024-09-13 14:30:25
🤖 AI量化交易系统 by Alvin
```

### **微信通知示例**
```
🚀 **AI量化交易提醒**

📊 **标的**: AAPL (苹果公司)
💰 **当前价格**: $175.32
🎯 **操作建议**: BUY
🔥 **置信度**: 85.0%
💡 **理由**: RSI oversold condition, Strong positive momentum
⏰ **时间**: 09-13 14:30

---
🤖 AI量化交易系统 by Alvin
```

---

## 🔔 **通知触发条件**

### **交易信号通知**
- BUY信号置信度 ≥ 85%
- SELL信号置信度 ≥ 85%
- 符合个股最小置信度要求

### **每日汇总通知**
- 每天18:00发送
- 包含当日所有交易信号
- 包含投资组合表现

### **紧急风险通知**
- 投资组合亏损 ≥ 5%
- 投资组合收益 ≥ 10%
- 系统异常或数据源故障

---

## 🧪 **测试通知**

### **测试邮件通知**
启动系统后，会自动发送测试邮件验证配置：
```bash
./start_all.sh

# 查看日志确认邮件发送
tail -f logs/trading.log | grep "Email"
```

### **测试微信通知**
```bash
# 手动触发测试通知
curl -X POST "YOUR_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{"msgtype":"text","text":{"content":"AI量化交易系统测试通知"}}'
```

---

## 🚨 **故障排除**

### **常见邮件问题**

#### **1. 认证失败**
```
错误: Authentication failed
解决: 检查Gmail应用密码是否正确，确保开启了两步验证
```

#### **2. 连接超时**
```
错误: Connection timeout
解决: 检查网络连接，确认SMTP端口587未被阻塞
```

#### **3. 邮件被拒收**
```
错误: Message rejected
解决: 检查邮件格式，避免触发垃圾邮件过滤器
```

### **常见微信问题**

#### **1. Webhook无效**
```
错误: Invalid webhook URL
解决: 重新获取企业微信机器人Webhook地址
```

#### **2. 消息格式错误**
```
错误: Message format error
解决: 检查JSON格式，确保符合企业微信API规范
```

#### **3. 频率限制**
```
错误: Rate limit exceeded
解决: 降低通知频率，避免短时间内发送过多消息
```

---

## 📋 **配置检查清单**

- [ ] Gmail应用密码已生成并配置
- [ ] application.properties邮件配置正确
- [ ] portfolio.json通知设置已启用
- [ ] 企业微信机器人已创建
- [ ] Webhook URL已配置
- [ ] 测试邮件发送成功
- [ ] 测试微信消息发送成功
- [ ] 通知触发条件已确认

---

## 🎯 **推荐配置**

### **保守型投资者**
```json
{
  "notification_triggers": {
    "strong_buy": 0.9,
    "strong_sell": 0.9,
    "portfolio_loss": 0.03,
    "portfolio_gain": 0.08
  }
}
```

### **积极型投资者**
```json
{
  "notification_triggers": {
    "strong_buy": 0.75,
    "strong_sell": 0.8,
    "portfolio_loss": 0.05,
    "portfolio_gain": 0.12
  }
}
```

---

**配置完成后，你将实时收到AI量化交易信号和投资组合状态通知！** 📧💬

需要帮助？请查看系统日志或联系技术支持。
