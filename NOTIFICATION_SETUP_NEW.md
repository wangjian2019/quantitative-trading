# 🔔 通知配置指南 - 新版本（无需Gmail应用密码）

## 📧 邮件通知配置

由于Google停用了应用专用密码，我们推荐使用以下替代方案：

### 方案1: QQ邮箱（推荐，仍支持授权码）

#### 步骤1: 开启QQ邮箱SMTP服务
1. **登录QQ邮箱**: https://mail.qq.com/
2. **进入设置**: 点击页面上方的"设置" → "账户"
3. **开启服务**: 找到"POP3/IMAP/SMTP/Exchange/CardDAV/CalDAV服务"
4. **开启SMTP**: 点击"开启"，按提示发送短信验证
5. **获取授权码**: 开启成功后，系统会显示一个授权码（16位字符）
6. **保存授权码**: 复制并保存这个授权码

#### 步骤2: 配置系统
编辑 `platform/src/main/resources/application.properties`:

```properties
# 邮件通知配置
email.enabled=true
email.username=your_qq_email@qq.com          # 你的QQ邮箱
email.password=your_qq_auth_code              # QQ邮箱授权码（16位）
email.notification.address=wangjians8813@gmail.com  # 接收通知的邮箱
email.smtp.host=smtp.qq.com
email.smtp.port=587
```

### 方案2: 163邮箱（也支持授权码）

#### 配置163邮箱
```properties
email.enabled=true
email.username=your_email@163.com
email.password=your_163_auth_code
email.notification.address=wangjians8813@gmail.com
email.smtp.host=smtp.163.com
email.smtp.port=587
```

#### 获取163邮箱授权码:
1. 登录163邮箱
2. 设置 → POP3/SMTP/IMAP
3. 开启SMTP服务
4. 获取授权码

### 方案3: 第三方邮件服务（企业级）

可以集成SendGrid、Mailgun等服务（需要API Key）。

## 💬 微信通知配置

### 方案1: Server酱（最简单，推荐）

#### 步骤1: 注册Server酱
1. **访问**: https://sct.ftqq.com/
2. **微信扫码登录**
3. **复制SendKey**: 在首页复制你的SendKey（格式：SCT123xxx...）

#### 步骤2: 配置系统
```properties
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
wechat.enabled=true
wechat.webhook.url=http://www.pushplus.plus/send?token=你的token
```

## 🔧 完整配置示例

编辑 `platform/src/main/resources/application.properties`:

```properties
# 邮件通知配置（使用QQ邮箱）
email.enabled=true
email.username=your_qq_email@qq.com
email.password=abcdefghijklmnop                    # QQ邮箱授权码
email.notification.address=wangjians8813@gmail.com
email.smtp.host=smtp.qq.com
email.smtp.port=587

# 微信通知配置（使用Server酱）
wechat.enabled=true
wechat.webhook.url=https://sctapi.ftqq.com/SCT123xxxYourSendKey.send

# 通知规则
notification.min.confidence=0.75
notification.send.daily.summary=true
```

## 🧪 测试配置

### 1. 启动服务
```bash
# 启动Python AI模型服务
cd strategy && python3 ai_model_service.py

# 启动Java平台服务（新终端）
cd platform && mvn compile exec:java -Dexec.mainClass="com.alvin.quantitative.trading.platform.TradingPlatformApplication"
```

### 2. 测试通知
在Java应用的控制台菜单中选择测试通知功能，或者访问Web界面进行测试。

## 📱 通知效果

配置成功后，系统会在以下情况发送通知：

### 📧 邮件通知
- **交易信号**: 高置信度的买入/卖出建议
- **投资组合预警**: 大幅盈亏提醒
- **每日总结**: 详细的交易报告
- **系统异常**: 服务故障通知

### 💬 微信通知
- **即时信号**: 重要交易机会立即推送
- **风险预警**: 市场异动提醒
- **系统状态**: 服务启停通知

## ⚠️ 故障排除

### QQ邮箱配置问题
- ✅ 确认已开启SMTP服务
- ✅ 授权码是16位字符
- ✅ 检查网络连接
- ✅ 查看Java应用日志

### 微信通知问题
- ✅ 确认Webhook URL格式正确
- ✅ 测试URL是否可访问
- ✅ 检查服务商账户状态

### 常用测试命令
```bash
# 查看Java应用日志
tail -f platform/logs/trading.log

# 检查服务状态
curl http://localhost:8080/health

# 检查AI模型服务
curl http://localhost:5000/health
```

## 📞 技术支持

如遇问题，请：
1. 检查配置文件格式
2. 查看系统日志
3. 测试网络连接
4. 验证授权码/API Key有效性

---

**🎯 配置完成后，你的AI量化交易平台将具备完整的智能通知功能！**
