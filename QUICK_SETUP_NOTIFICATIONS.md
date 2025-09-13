# 🚀 通知配置快速指南

## 📧 Gmail配置（5分钟搞定）

### 步骤1: 获取Gmail应用密码
1. 打开浏览器，访问: **https://myaccount.google.com/security**
2. 登录你的Gmail账户 (wangjians8813@gmail.com)
3. 在"登录Google"部分，点击 **"两步验证"**
4. 如果没有开启，按提示开启两步验证
5. 开启后，在同一页面找到 **"应用专用密码"**
6. 点击 **"应用专用密码"**
7. 选择应用：**"其他（自定义名称）"**
8. 输入名称：**"AI Trading Platform"**
9. 点击 **"生成"**
10. **复制生成的16位密码**（类似：abcd efgh ijkl mnop）

### 步骤2: 配置系统
编辑文件：`strategy/ai_strategy_service.py`

找到这一行：
```python
'password': 'YOUR_16_DIGIT_APP_PASSWORD_HERE',
```

替换为：
```python
'password': 'abcdefghijklmnop',  # 你的16位密码，去掉空格
```

## 💬 微信配置（推荐Server酱）

### 方法1: Server酱（最简单，推荐）

1. **访问Server酱官网**: https://sct.ftqq.com/
2. **微信扫码登录**
3. **复制SendKey**（在首页，格式：SCT123xxx...）
4. **配置系统**:
   ```python
   'webhook_url': 'https://sctapi.ftqq.com/SCT123xxx你的SendKey.send'
   ```

### 方法2: 企业微信群机器人

1. **下载企业微信APP**
2. **创建群聊**
3. **群设置** → **群机器人** → **添加机器人**
4. **复制Webhook URL**
5. **配置系统**:
   ```python
   'webhook_url': 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=你的key'
   ```

## 🧪 测试配置

### 1. 重启Python服务
```bash
# 停止当前服务
pkill -f ai_strategy_service

# 启动服务
cd strategy && python3 ai_strategy_service.py
```

### 2. 测试邮件通知
```bash
curl -X POST http://localhost:5000/send_test_notification \
  -H "Content-Type: application/json" \
  -d '{"type":"email","message":"Gmail配置测试成功！"}'
```

### 3. 测试微信通知
```bash
curl -X POST http://localhost:5000/send_test_notification \
  -H "Content-Type: application/json" \
  -d '{"type":"wechat","message":"微信配置测试成功！"}'
```

## 📝 完整配置示例

```python
notification_config = {
    'email': {
        'enabled': True,
        'username': 'wangjians8813@gmail.com',
        'password': 'abcdefghijklmnop',  # 你的16位Gmail应用密码
        'address': 'wangjians8813@gmail.com',
        'smtp_host': 'smtp.gmail.com',
        'smtp_port': 587,
        'send_daily_summary': True
    },
    'wechat': {
        'enabled': True,
        'webhook_url': 'https://sctapi.ftqq.com/SCT123xxx你的SendKey.send'  # Server酱URL
    },
    'min_notification_confidence': 0.75
}
```

## ⚠️ 常见问题

### Gmail邮件发送失败
- ✅ 确认两步验证已开启
- ✅ 确认应用密码是16位（无空格）
- ✅ 检查网络连接
- ✅ 查看日志：`tail -f strategy/logs/ai_service.log`

### 微信通知失败
- ✅ 确认Webhook URL格式正确
- ✅ 测试URL是否可访问
- ✅ 检查Server酱账户状态

## 🎯 配置完成后的效果

配置成功后，系统会在以下情况自动发送通知：

### 📧 邮件通知
- **交易信号**: 置信度≥75%的买入/卖出信号
- **投资组合预警**: 大幅盈亏时提醒
- **每日总结**: 交易日结束后的详细报告
- **系统异常**: 服务故障时的紧急通知

### 💬 微信通知
- **高置信度信号**: 立即推送重要交易机会
- **风险预警**: 市场异动和风险提醒
- **系统状态**: 服务启停和健康状态

## 🚀 下一步

配置完成后，建议：

1. **测试通知功能**
2. **调整置信度阈值**（根据个人需求）
3. **配置监控股票**（在Web界面的股票配置页面）
4. **运行回测分析**（验证策略效果）

---

**💡 小贴士**: 建议先用较低的资金测试策略效果，确认系统稳定后再增加投资额度。
