"""
简化通知服务 - 无需Gmail应用密码
Author: Alvin
使用第三方服务发送邮件和微信通知
"""

import requests
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

class SimpleNotificationService:
    """
    简化通知服务类
    支持多种第三方通知服务
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化通知服务
        
        Args:
            config: 通知配置
        """
        self.config = config
        self.email_config = config.get('email', {})
        self.wechat_config = config.get('wechat', {})
        self.logger = logging.getLogger(__name__)
        
    def send_email_notification(self, subject: str, message: str, to_email: Optional[str] = None) -> bool:
        """
        发送邮件通知 - 使用第三方服务
        
        Args:
            subject: 邮件主题
            message: 邮件内容
            to_email: 收件人邮箱
            
        Returns:
            bool: 发送是否成功
        """
        if not self.email_config.get('enabled', False):
            self.logger.info("邮件通知未启用")
            return False
        
        # 方法1: 使用EmailJS (推荐)
        if self.email_config.get('service') == 'emailjs':
            return self._send_via_emailjs(subject, message, to_email)
        
        # 方法2: 使用SendGrid
        elif self.email_config.get('service') == 'sendgrid':
            return self._send_via_sendgrid(subject, message, to_email)
        
        # 方法3: 使用Mailgun
        elif self.email_config.get('service') == 'mailgun':
            return self._send_via_mailgun(subject, message, to_email)
        
        # 方法4: 使用QQ邮箱SMTP（仍然支持应用密码）
        elif self.email_config.get('service') == 'qq':
            return self._send_via_qq_smtp(subject, message, to_email)
        
        # 默认使用微信通知代替邮件
        else:
            self.logger.warning("未配置邮件服务，尝试使用微信通知")
            return self.send_wechat_notification(f"📧 {subject}\n\n{message}")
    
    def _send_via_emailjs(self, subject: str, message: str, to_email: Optional[str]) -> bool:
        """使用EmailJS发送邮件"""
        try:
            service_id = self.email_config.get('emailjs_service_id')
            template_id = self.email_config.get('emailjs_template_id')
            public_key = self.email_config.get('emailjs_public_key')
            
            if not all([service_id, template_id, public_key]):
                self.logger.error("EmailJS配置不完整")
                return False
            
            data = {
                'service_id': service_id,
                'template_id': template_id,
                'user_id': public_key,
                'template_params': {
                    'to_email': to_email or self.email_config.get('to_email'),
                    'subject': f"🚀 AI量化交易平台 - {subject}",
                    'message': message,
                    'from_name': 'AI Trading Platform'
                }
            }
            
            response = requests.post(
                'https://api.emailjs.com/api/v1.0/email/send',
                data=json.dumps(data),
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                self.logger.info("EmailJS邮件发送成功")
                return True
            else:
                self.logger.error(f"EmailJS发送失败: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"EmailJS发送异常: {e}")
            return False
    
    def _send_via_sendgrid(self, subject: str, message: str, to_email: Optional[str]) -> bool:
        """使用SendGrid发送邮件"""
        try:
            api_key = self.email_config.get('sendgrid_api_key')
            from_email = self.email_config.get('from_email')
            
            if not api_key:
                self.logger.error("SendGrid API Key未配置")
                return False
            
            data = {
                'personalizations': [{
                    'to': [{'email': to_email or self.email_config.get('to_email')}],
                    'subject': f"🚀 AI量化交易平台 - {subject}"
                }],
                'from': {'email': from_email or 'noreply@trading.com'},
                'content': [{
                    'type': 'text/html',
                    'value': self._create_email_html(subject, message)
                }]
            }
            
            response = requests.post(
                'https://api.sendgrid.com/v3/mail/send',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                },
                data=json.dumps(data)
            )
            
            if response.status_code == 202:
                self.logger.info("SendGrid邮件发送成功")
                return True
            else:
                self.logger.error(f"SendGrid发送失败: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"SendGrid发送异常: {e}")
            return False
    
    def _send_via_qq_smtp(self, subject: str, message: str, to_email: Optional[str]) -> bool:
        """使用QQ邮箱SMTP发送（QQ邮箱仍支持应用密码）"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            username = self.email_config.get('qq_username')
            password = self.email_config.get('qq_password')  # QQ邮箱授权码
            
            if not all([username, password]):
                self.logger.error("QQ邮箱配置不完整")
                return False
            
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = to_email or self.email_config.get('to_email', username)
            msg['Subject'] = f"🚀 AI量化交易平台 - {subject}"
            
            html_content = self._create_email_html(subject, message)
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            with smtplib.SMTP('smtp.qq.com', 587) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            self.logger.info("QQ邮箱发送成功")
            return True
            
        except Exception as e:
            self.logger.error(f"QQ邮箱发送失败: {e}")
            return False
    
    def send_wechat_notification(self, message: str, message_type: str = "text") -> bool:
        """
        发送微信通知
        """
        if not self.wechat_config.get('enabled', False):
            self.logger.info("微信通知未启用")
            return False
        
        webhook_url = self.wechat_config.get('webhook_url')
        
        # 支持多种微信通知服务
        if not webhook_url or webhook_url == 'YOUR_WECHAT_BOT_WEBHOOK_HERE':
            self.logger.error("微信Webhook URL未配置")
            return False
        
        try:
            # Server酱
            if 'sctapi.ftqq.com' in webhook_url or 'sc.ftqq.com' in webhook_url:
                return self._send_via_server_chan(message, webhook_url)
            
            # Pushplus
            elif 'pushplus.plus' in webhook_url:
                return self._send_via_pushplus(message, webhook_url)
            
            # 企业微信群机器人
            elif 'qyapi.weixin.qq.com' in webhook_url:
                return self._send_via_wework_bot(message, webhook_url)
            
            # 自定义webhook
            else:
                return self._send_via_custom_webhook(message, webhook_url)
                
        except Exception as e:
            self.logger.error(f"微信通知发送异常: {e}")
            return False
    
    def _send_via_server_chan(self, message: str, webhook_url: str) -> bool:
        """通过Server酱发送"""
        try:
            # 支持新旧版本的Server酱
            if 'sctapi.ftqq.com' in webhook_url:
                # 新版Server酱
                data = {
                    'title': '🚀 AI量化交易平台通知',
                    'desp': message
                }
                response = requests.post(webhook_url, data=data, timeout=10)
            else:
                # 旧版Server酱
                data = {
                    'text': '🚀 AI量化交易平台通知',
                    'desp': message
                }
                response = requests.post(webhook_url, data=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 0 or result.get('errno') == 0:
                    self.logger.info("Server酱通知发送成功")
                    return True
                else:
                    self.logger.error(f"Server酱发送失败: {result}")
                    return False
            else:
                self.logger.error(f"Server酱请求失败: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Server酱发送异常: {e}")
            return False
    
    def _send_via_pushplus(self, message: str, webhook_url: str) -> bool:
        """通过Pushplus发送"""
        try:
            data = {
                'title': '🚀 AI量化交易平台通知',
                'content': message,
                'template': 'html'
            }
            
            response = requests.post(webhook_url, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 200:
                    self.logger.info("Pushplus通知发送成功")
                    return True
                else:
                    self.logger.error(f"Pushplus发送失败: {result}")
                    return False
            else:
                self.logger.error(f"Pushplus请求失败: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Pushplus发送异常: {e}")
            return False
    
    def _send_via_wework_bot(self, message: str, webhook_url: str) -> bool:
        """通过企业微信群机器人发送"""
        try:
            payload = {
                "msgtype": "text",
                "text": {
                    "content": f"🚀 AI量化交易平台通知\n\n{message}"
                }
            }
            
            response = requests.post(
                webhook_url, 
                data=json.dumps(payload), 
                headers={'Content-Type': 'application/json'}, 
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    self.logger.info("企业微信通知发送成功")
                    return True
                else:
                    self.logger.error(f"企业微信发送失败: {result}")
                    return False
            else:
                self.logger.error(f"企业微信请求失败: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"企业微信发送异常: {e}")
            return False
    
    def send_trading_signal_notification(self, signal: Dict[str, Any]) -> bool:
        """发送交易信号通知"""
        symbol = signal.get('symbol', 'N/A')
        action = signal.get('action', 'N/A')
        price = signal.get('price', 0)
        confidence = signal.get('confidence', 0)
        reason = signal.get('reason', 'N/A')
        
        # 判断是否需要发送通知
        if confidence < self.config.get('min_notification_confidence', 0.7):
            self.logger.info(f"信号置信度过低，不发送通知: {confidence}")
            return False
        
        # 格式化消息
        action_emoji = {'BUY': '📈', 'SELL': '📉', 'HOLD': '🤝'}.get(action, '📊')
        
        message = f"""
{action_emoji} 交易信号提醒

股票代码: {symbol}
建议操作: {action}
当前价格: ${price:.2f}
置信度: {confidence*100:.1f}%
分析理由: {reason}
信号时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

请根据您的风险承受能力谨慎决策。
        """.strip()
        
        # 优先发送微信通知（更及时）
        wechat_success = self.send_wechat_notification(message)
        
        # 如果配置了邮件，也发送邮件
        email_success = False
        if self.email_config.get('enabled', False):
            email_subject = f"交易信号 - {symbol} {action}"
            email_success = self.send_email_notification(email_subject, message)
        
        return wechat_success or email_success
    
    def _create_email_html(self, subject: str, message: str) -> str:
        """创建HTML格式的邮件内容"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
                .content {{ padding: 30px; line-height: 1.6; }}
                .footer {{ background: #f8f9fa; padding: 20px; text-align: center; font-size: 12px; color: #666; }}
                .highlight {{ background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #2196f3; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 AI量化交易平台</h1>
                    <p>by Alvin - 专业级量化交易解决方案</p>
                </div>
                <div class="content">
                    <h2>{subject}</h2>
                    <div class="highlight">
                        {message.replace(chr(10), '<br>')}
                    </div>
                    <p><strong>发送时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                <div class="footer">
                    <p>© 2024 AI量化交易平台 by Alvin</p>
                    <p>免责声明: 本系统仅供学习研究使用，投资有风险，决策需谨慎。</p>
                </div>
            </div>
        </body>
        </html>
        """


def create_simple_notification_service(config: Dict[str, Any]) -> SimpleNotificationService:
    """
    创建简化通知服务实例
    """
    return SimpleNotificationService(config)
