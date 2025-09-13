"""
通知服务模块
Author: Alvin
支持邮件和微信通知功能
"""

import smtplib
import requests
import json
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, List, Optional
import base64
import urllib.parse

class NotificationService:
    """
    统一通知服务类
    支持邮件和微信通知
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
        发送邮件通知
        
        Args:
            subject: 邮件主题
            message: 邮件内容
            to_email: 收件人邮箱（可选，默认使用配置中的邮箱）
            
        Returns:
            bool: 发送是否成功
        """
        if not self.email_config.get('enabled', False):
            self.logger.info("邮件通知未启用")
            return False
            
        try:
            # 邮件配置
            smtp_server = self.email_config.get('smtp_host', 'smtp.gmail.com')
            smtp_port = self.email_config.get('smtp_port', 587)
            sender_email = self.email_config.get('username')
            sender_password = self.email_config.get('password')
            recipient_email = to_email or self.email_config.get('address')
            
            if not all([sender_email, sender_password, recipient_email]):
                self.logger.error("邮件配置不完整")
                return False
            
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"🚀 AI量化交易平台 - {subject}"
            
            # 创建HTML格式的邮件内容
            html_content = self._create_email_html(subject, message)
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            # 发送邮件
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            self.logger.info(f"邮件通知发送成功: {subject}")
            return True
            
        except Exception as e:
            self.logger.error(f"邮件发送失败: {e}")
            return False
    
    def send_wechat_notification(self, message: str, message_type: str = "text") -> bool:
        """
        发送微信通知
        
        Args:
            message: 消息内容
            message_type: 消息类型 (text, markdown)
            
        Returns:
            bool: 发送是否成功
        """
        if not self.wechat_config.get('enabled', False):
            self.logger.info("微信通知未启用")
            return False
            
        webhook_url = self.wechat_config.get('webhook_url')
        if not webhook_url:
            self.logger.error("微信Webhook URL未配置")
            return False
            
        try:
            # 构建微信消息
            if message_type == "markdown":
                payload = {
                    "msgtype": "markdown",
                    "markdown": {
                        "content": message
                    }
                }
            else:
                payload = {
                    "msgtype": "text",
                    "text": {
                        "content": f"🚀 AI量化交易平台通知\n\n{message}"
                    }
                }
            
            # 发送请求
            headers = {'Content-Type': 'application/json'}
            response = requests.post(webhook_url, data=json.dumps(payload), headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    self.logger.info("微信通知发送成功")
                    return True
                else:
                    self.logger.error(f"微信通知发送失败: {result.get('errmsg')}")
                    return False
            else:
                self.logger.error(f"微信通知请求失败: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"微信通知发送异常: {e}")
            return False
    
    def send_trading_signal_notification(self, signal: Dict[str, Any]) -> bool:
        """
        发送交易信号通知
        
        Args:
            signal: 交易信号数据
            
        Returns:
            bool: 发送是否成功
        """
        symbol = signal.get('symbol', 'N/A')
        action = signal.get('action', 'N/A')
        price = signal.get('price', 0)
        confidence = signal.get('confidence', 0)
        reason = signal.get('reason', 'N/A')
        
        # 判断是否需要发送通知
        if confidence < self.config.get('min_notification_confidence', 0.7):
            self.logger.info(f"信号置信度过低，不发送通知: {confidence}")
            return False
        
        # 邮件通知
        email_subject = f"交易信号 - {symbol} {action}"
        email_message = self._format_signal_message(signal)
        email_success = self.send_email_notification(email_subject, email_message)
        
        # 微信通知
        wechat_message = self._format_signal_wechat_message(signal)
        wechat_success = self.send_wechat_notification(wechat_message, "markdown")
        
        return email_success or wechat_success
    
    def send_portfolio_alert(self, portfolio_data: Dict[str, Any], alert_type: str) -> bool:
        """
        发送投资组合预警
        
        Args:
            portfolio_data: 投资组合数据
            alert_type: 预警类型 (gain, loss, rebalance)
            
        Returns:
            bool: 发送是否成功
        """
        total_value = portfolio_data.get('total_value', 0)
        total_return = portfolio_data.get('total_return', 0)
        daily_pnl = portfolio_data.get('daily_pnl', 0)
        
        if alert_type == "gain":
            subject = "投资组合收益预警"
            message = f"您的投资组合表现优异！\n\n总资产: ${total_value:,.2f}\n总收益率: {total_return*100:.2f}%\n今日盈亏: ${daily_pnl:,.2f}"
        elif alert_type == "loss":
            subject = "投资组合风险预警"
            message = f"您的投资组合出现较大亏损，请注意风险控制！\n\n总资产: ${total_value:,.2f}\n总收益率: {total_return*100:.2f}%\n今日盈亏: ${daily_pnl:,.2f}"
        else:
            subject = "投资组合再平衡提醒"
            message = f"建议对投资组合进行再平衡\n\n总资产: ${total_value:,.2f}\n总收益率: {total_return*100:.2f}%"
        
        # 发送邮件和微信通知
        email_success = self.send_email_notification(subject, message)
        wechat_success = self.send_wechat_notification(message)
        
        return email_success or wechat_success
    
    def send_system_alert(self, alert_message: str, severity: str = "info") -> bool:
        """
        发送系统预警
        
        Args:
            alert_message: 预警消息
            severity: 严重程度 (info, warning, error)
            
        Returns:
            bool: 发送是否成功
        """
        severity_icons = {
            "info": "ℹ️",
            "warning": "⚠️", 
            "error": "❌"
        }
        
        icon = severity_icons.get(severity, "📢")
        subject = f"系统{severity.upper()}预警"
        message = f"{icon} {alert_message}"
        
        # 根据严重程度决定发送方式
        if severity in ["warning", "error"]:
            # 严重问题同时发送邮件和微信
            email_success = self.send_email_notification(subject, message)
            wechat_success = self.send_wechat_notification(message)
            return email_success or wechat_success
        else:
            # 一般信息只发送微信
            return self.send_wechat_notification(message)
    
    def send_daily_summary(self, summary_data: Dict[str, Any]) -> bool:
        """
        发送每日总结报告
        
        Args:
            summary_data: 总结数据
            
        Returns:
            bool: 发送是否成功
        """
        if not self.email_config.get('send_daily_summary', True):
            return False
            
        date = datetime.now().strftime('%Y-%m-%d')
        subject = f"每日交易总结 - {date}"
        
        # 构建详细的每日总结
        message = self._format_daily_summary(summary_data)
        
        return self.send_email_notification(subject, message)
    
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
                .highlight {{ background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 AI量化交易平台</h1>
                    <p>by Alvin</p>
                </div>
                <div class="content">
                    <h2>{subject}</h2>
                    <div class="highlight">
                        {message.replace(chr(10), '<br>')}
                    </div>
                    <p>发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                <div class="footer">
                    <p>© 2024 AI量化交易平台 by Alvin | 专业级量化交易解决方案</p>
                    <p>免责声明: 本系统仅供学习研究使用，投资有风险，决策需谨慎。</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _format_signal_message(self, signal: Dict[str, Any]) -> str:
        """格式化交易信号消息"""
        symbol = signal.get('symbol', 'N/A')
        action = signal.get('action', 'N/A')
        price = signal.get('price', 0)
        confidence = signal.get('confidence', 0)
        reason = signal.get('reason', 'N/A')
        timestamp = signal.get('timestamp', datetime.now().isoformat())
        
        action_emoji = {
            'BUY': '📈 买入',
            'SELL': '📉 卖出',
            'HOLD': '🤝 持有'
        }.get(action, action)
        
        return f"""
        交易信号详情:
        
        股票代码: {symbol}
        建议操作: {action_emoji}
        当前价格: ${price:.2f}
        置信度: {confidence*100:.1f}%
        分析理由: {reason}
        信号时间: {timestamp}
        
        请根据您的风险承受能力和投资策略谨慎决策。
        """
    
    def _format_signal_wechat_message(self, signal: Dict[str, Any]) -> str:
        """格式化微信交易信号消息"""
        symbol = signal.get('symbol', 'N/A')
        action = signal.get('action', 'N/A')
        price = signal.get('price', 0)
        confidence = signal.get('confidence', 0)
        reason = signal.get('reason', 'N/A')
        
        action_emoji = {
            'BUY': '📈',
            'SELL': '📉',
            'HOLD': '🤝'
        }.get(action, '📊')
        
        return f"""# 🚀 AI量化交易信号
        
> **{symbol}** {action_emoji} **{action}**
> 
> **价格**: ${price:.2f}
> **置信度**: {confidence*100:.1f}%
> **理由**: {reason}
> 
> 📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
    
    def _format_daily_summary(self, summary_data: Dict[str, Any]) -> str:
        """格式化每日总结"""
        return f"""
        每日交易总结报告
        
        📊 市场表现:
        - 监控股票数量: {summary_data.get('total_stocks', 0)}
        - 生成信号数量: {summary_data.get('total_signals', 0)}
        - 高置信度信号: {summary_data.get('high_confidence_signals', 0)}
        
        💰 投资组合:
        - 总资产: ${summary_data.get('total_value', 0):,.2f}
        - 今日盈亏: ${summary_data.get('daily_pnl', 0):,.2f}
        - 总收益率: {summary_data.get('total_return', 0)*100:.2f}%
        
        🤖 AI模型表现:
        - 模型准确率: {summary_data.get('model_accuracy', 0)*100:.1f}%
        - 策略胜率: {summary_data.get('win_rate', 0)*100:.1f}%
        
        📈 热门信号:
        {self._format_top_signals(summary_data.get('top_signals', []))}
        
        明日关注:
        {summary_data.get('tomorrow_focus', '继续监控市场动态')}
        """
    
    def _format_top_signals(self, signals: List[Dict]) -> str:
        """格式化热门信号"""
        if not signals:
            return "暂无热门信号"
        
        result = []
        for i, signal in enumerate(signals[:3], 1):
            symbol = signal.get('symbol', 'N/A')
            action = signal.get('action', 'N/A')
            confidence = signal.get('confidence', 0)
            result.append(f"{i}. {symbol} - {action} ({confidence*100:.1f}%)")
        
        return "\n".join(result)


def create_notification_service(config: Dict[str, Any]) -> NotificationService:
    """
    创建通知服务实例
    
    Args:
        config: 配置字典
        
    Returns:
        NotificationService: 通知服务实例
    """
    return NotificationService(config)
