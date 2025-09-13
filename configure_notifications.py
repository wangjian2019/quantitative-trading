#!/usr/bin/env python3
"""
通知配置助手脚本
Author: Alvin
用于配置Gmail和微信通知
"""

import re
import os
import sys

def configure_notifications():
    """配置通知设置的交互式脚本"""
    print("🔔 AI量化交易平台 - 通知配置助手")
    print("=" * 50)
    
    # Gmail配置
    print("\n📧 Gmail邮件通知配置")
    print("-" * 30)
    
    gmail_enabled = input("是否启用Gmail通知？(y/n) [y]: ").strip().lower()
    gmail_enabled = gmail_enabled in ['', 'y', 'yes', '是']
    
    gmail_config = {}
    if gmail_enabled:
        gmail_username = input("请输入Gmail邮箱地址 [wangjians8813@gmail.com]: ").strip()
        if not gmail_username:
            gmail_username = "wangjians8813@gmail.com"
            
        gmail_password = input("请输入Gmail应用专用密码（16位）: ").strip().replace(' ', '')
        if not gmail_password:
            print("⚠️  警告: 未输入Gmail应用密码，邮件通知将无法工作")
            gmail_password = "YOUR_16_DIGIT_APP_PASSWORD_HERE"
        elif len(gmail_password) != 16:
            print("⚠️  警告: Gmail应用密码应该是16位字符")
            
        gmail_address = input(f"接收邮件地址 [{gmail_username}]: ").strip()
        if not gmail_address:
            gmail_address = gmail_username
            
        gmail_config = {
            'enabled': True,
            'username': gmail_username,
            'password': gmail_password,
            'address': gmail_address,
            'smtp_host': 'smtp.gmail.com',
            'smtp_port': 587,
            'send_daily_summary': True
        }
        print(f"✅ Gmail配置完成: {gmail_username}")
    else:
        gmail_config = {'enabled': False}
        print("❌ Gmail通知已禁用")
    
    # 微信配置
    print("\n💬 微信通知配置")
    print("-" * 30)
    
    wechat_enabled = input("是否启用微信通知？(y/n) [y]: ").strip().lower()
    wechat_enabled = wechat_enabled in ['', 'y', 'yes', '是']
    
    wechat_config = {}
    if wechat_enabled:
        print("\n请选择微信通知方式:")
        print("1. 企业微信群机器人（推荐）")
        print("2. Server酱")
        print("3. Pushplus")
        print("4. 自定义Webhook")
        
        choice = input("请选择 (1-4) [1]: ").strip()
        if not choice:
            choice = "1"
            
        webhook_url = ""
        
        if choice == "1":
            print("\n📋 企业微信群机器人配置步骤:")
            print("1. 创建企业微信群")
            print("2. 群设置 -> 群机器人 -> 添加机器人")
            print("3. 复制Webhook URL")
            webhook_url = input("\n请输入企业微信Webhook URL: ").strip()
            
        elif choice == "2":
            print("\n📋 Server酱配置步骤:")
            print("1. 访问 https://sct.ftqq.com/")
            print("2. 微信扫码登录")
            print("3. 复制SendKey")
            sendkey = input("\n请输入Server酱SendKey: ").strip()
            if sendkey:
                webhook_url = f"https://sctapi.ftqq.com/{sendkey}.send"
                
        elif choice == "3":
            print("\n📋 Pushplus配置步骤:")
            print("1. 访问 http://www.pushplus.plus/")
            print("2. 微信扫码登录")
            print("3. 复制Token")
            token = input("\n请输入Pushplus Token: ").strip()
            if token:
                webhook_url = f"http://www.pushplus.plus/send?token={token}"
                
        elif choice == "4":
            webhook_url = input("请输入自定义Webhook URL: ").strip()
        
        if webhook_url and not webhook_url.startswith(('http://', 'https://')):
            print("⚠️  警告: Webhook URL应该以http://或https://开头")
            
        wechat_config = {
            'enabled': True,
            'webhook_url': webhook_url or 'YOUR_WECHAT_BOT_WEBHOOK_HERE'
        }
        
        if webhook_url:
            print(f"✅ 微信配置完成: {webhook_url[:50]}...")
        else:
            print("⚠️  警告: 未输入有效的Webhook URL，微信通知将无法工作")
    else:
        wechat_config = {'enabled': False}
        print("❌ 微信通知已禁用")
    
    # 其他配置
    print("\n⚙️  其他配置")
    print("-" * 30)
    
    min_confidence = input("最小通知置信度 (0.5-0.9) [0.75]: ").strip()
    try:
        min_confidence = float(min_confidence) if min_confidence else 0.75
        if not 0.5 <= min_confidence <= 0.9:
            min_confidence = 0.75
            print("⚠️  置信度超出范围，使用默认值0.75")
    except ValueError:
        min_confidence = 0.75
        print("⚠️  无效的置信度值，使用默认值0.75")
    
    # 生成配置代码
    config_code = f'''# Global notification service
notification_config = {{
    'email': {repr(gmail_config)},
    'wechat': {repr(wechat_config)},
    'min_notification_confidence': {min_confidence}
}}'''
    
    print("\n📝 生成的配置代码:")
    print("=" * 50)
    print(config_code)
    print("=" * 50)
    
    # 询问是否自动更新文件
    update_file = input("\n是否自动更新ai_strategy_service.py文件？(y/n) [y]: ").strip().lower()
    update_file = update_file in ['', 'y', 'yes', '是']
    
    if update_file:
        try:
            update_config_file(gmail_config, wechat_config, min_confidence)
            print("✅ 配置文件已更新!")
        except Exception as e:
            print(f"❌ 更新配置文件失败: {e}")
            print("请手动复制上面的配置代码到ai_strategy_service.py文件中")
    else:
        print("请手动复制上面的配置代码到ai_strategy_service.py文件中")
    
    print("\n🧪 测试配置")
    print("-" * 30)
    print("配置完成后，请执行以下命令测试:")
    print("1. 重启Python服务: cd strategy && python3 ai_strategy_service.py")
    print("2. 测试邮件: curl -X POST http://localhost:5000/send_test_notification -H \"Content-Type: application/json\" -d '{\"type\":\"email\",\"message\":\"测试邮件\"}'")
    print("3. 测试微信: curl -X POST http://localhost:5000/send_test_notification -H \"Content-Type: application/json\" -d '{\"type\":\"wechat\",\"message\":\"测试微信\"}'")

def update_config_file(gmail_config, wechat_config, min_confidence):
    """更新配置文件"""
    file_path = "strategy/ai_strategy_service.py"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 构建新的配置
    new_config = f'''notification_config = {{
    'email': {repr(gmail_config)},
    'wechat': {repr(wechat_config)},
    'min_notification_confidence': {min_confidence}
}}'''
    
    # 使用正则表达式替换配置
    pattern = r'notification_config\s*=\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\}'
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_config, content, flags=re.DOTALL)
    else:
        raise ValueError("找不到notification_config配置块")
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    try:
        configure_notifications()
    except KeyboardInterrupt:
        print("\n\n👋 配置已取消")
    except Exception as e:
        print(f"\n❌ 配置过程中出错: {e}")
        sys.exit(1)
