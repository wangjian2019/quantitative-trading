# 🔔 通知配置模板

## 📧 Gmail配置步骤

### 1. 获取Gmail应用密码
1. 访问 https://myaccount.google.com/
2. 进入 "安全性" -> "两步验证"
3. 开启两步验证（如未开启）
4. 进入 "应用专用密码"
5. 生成新密码，选择 "其他" -> "AI Trading Platform"
6. 复制生成的16位密码（格式：abcd efgh ijkl mnop）

### 2. 配置代码
编辑 `strategy/ai_strategy_service.py` 文件，找到 notification_config，更新以下内容：

```python
notification_config = {
    'email': {
        'enabled': True,
        'username': 'wangjians8813@gmail.com',        # 你的Gmail地址
        'password': 'abcd efgh ijkl mnop',             # 替换为实际的16位应用密码（去掉空格）
        'address': 'wangjians8813@gmail.com',          # 接收邮件的地址
        'smtp_host': 'smtp.gmail.com',
        'smtp_port': 587,
        'send_daily_summary': True
    }
}
```

## 💬 微信通知配置

### 方法1: 企业微信群机器人（推荐）

1. **创建企业微信群**
   - 下载企业微信
   - 创建群聊并邀请成员

2. **添加群机器人**
   - 群设置 -> 群机器人 -> 添加机器人
   - 命名：AI交易助手

3. **获取Webhook**
   - 复制类似这样的URL：
   ```
   https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
   ```

4. **配置代码**
   ```python
   'wechat': {
       'enabled': True,
       'webhook_url': 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=你的key'
   }
   ```

### 方法2: Server酱（简单推荐）

1. **注册Server酱**
   - 访问 https://sct.ftqq.com/
   - 微信扫码登录

2. **获取SendKey**
   - 登录后复制SendKey（格式：SCTxxxxxxxxxxxxxxxxxxxxxx）

3. **配置代码**
   ```python
   'wechat': {
       'enabled': True,
       'webhook_url': 'https://sctapi.ftqq.com/你的SendKey.send'
   }
   ```

### 方法3: Pushplus

1. **注册Pushplus**
   - 访问 http://www.pushplus.plus/
   - 微信扫码登录

2. **获取Token**
   - 复制你的token

3. **配置代码**
   ```python
   'wechat': {
       'enabled': True,
       'webhook_url': 'http://www.pushplus.plus/send?token=你的token'
   }
   ```

## 🧪 测试配置

配置完成后，重启Python服务并测试：

```bash
# 停止当前服务
pkill -f ai_strategy_service

# 重启服务
cd strategy && python3 ai_strategy_service.py
```

在新终端测试邮件通知：
```bash
curl -X POST http://localhost:5000/send_test_notification \
  -H "Content-Type: application/json" \
  -d '{"type":"email","message":"Gmail配置测试成功！"}'
```

测试微信通知：
```bash
curl -X POST http://localhost:5000/send_test_notification \
  -H "Content-Type: application/json" \
  -d '{"type":"wechat","message":"微信通知配置测试成功！"}'
```

## 📱 完整配置示例

```python
notification_config = {
    'email': {
        'enabled': True,
        'username': 'wangjians8813@gmail.com',
        'password': 'abcdefghijklmnop',  # 实际的16位应用密码
        'address': 'wangjians8813@gmail.com',
        'smtp_host': 'smtp.gmail.com',
        'smtp_port': 587,
        'send_daily_summary': True
    },
    'wechat': {
        'enabled': True,
        'webhook_url': 'https://sct.ftqq.com/SCTxxxxxxxxxxxxxxxxxxxxxx.send'  # 实际的webhook地址
    },
    'min_notification_confidence': 0.75
}
```

## ⚠️ 安全注意事项

1. **不要提交敏感信息到Git**
   - 应用密码和webhook地址不要提交到代码仓库
   - 考虑使用环境变量

2. **定期更换密码**
   - 定期更换Gmail应用密码
   - 监控异常登录活动

3. **限制访问权限**
   - 只给必要的人员提供webhook地址
   - 定期检查群成员

## 🔍 故障排除

### Gmail邮件发送失败
- 检查应用密码是否正确（16位，无空格）
- 确认两步验证已开启
- 检查网络连接
- 查看错误日志：`tail -f strategy/logs/ai_service.log`

### 微信通知失败
- 检查webhook URL是否正确
- 测试webhook是否可访问
- 检查消息格式是否符合要求
- 查看服务日志确认具体错误

## 📞 技术支持

如遇到问题，可以：
1. 查看系统日志文件
2. 使用测试API验证配置
3. 检查网络连接和防火墙设置
