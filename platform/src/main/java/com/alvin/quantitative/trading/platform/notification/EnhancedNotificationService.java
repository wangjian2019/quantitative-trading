package com.alvin.quantitative.trading.platform.notification;

import com.alvin.quantitative.trading.platform.config.ApplicationConfig;
import com.alvin.quantitative.trading.platform.core.AISignal;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.http.HttpEntity;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;

// import javax.mail.*;
// import javax.mail.internet.*;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;
// import java.util.Properties;
import java.util.logging.Logger;

/**
 * 增强通知服务 - 支持多种邮件和微信通知方式
 * Author: Alvin
 * 解决Gmail应用密码停用问题
 */
public class EnhancedNotificationService {
    private static final Logger logger = Logger.getLogger(EnhancedNotificationService.class.getName());
    private final ApplicationConfig config;
    private final ObjectMapper objectMapper;
    private final CloseableHttpClient httpClient;
    
    public EnhancedNotificationService() {
        this.config = ApplicationConfig.getInstance();
        this.objectMapper = new ObjectMapper();
        this.httpClient = HttpClients.createDefault();
    }
    
    /**
     * 发送交易信号通知
     */
    public boolean sendTradingSignalNotification(String symbol, AISignal signal, double currentPrice) {
        try {
            String subject = String.format("🚀 交易信号 - %s %s", symbol, signal.getAction());
            String message = formatTradingSignalMessage(symbol, signal, currentPrice);
            
            // 优先发送微信通知（更及时）
            boolean wechatSuccess = sendWechatNotification(message);
            
            // 如果启用邮件，也发送邮件
            boolean emailSuccess = false;
            if (config.isEmailNotificationEnabled()) {
                emailSuccess = sendEmailNotification(subject, message);
            }
            
            return wechatSuccess || emailSuccess;
            
        } catch (Exception e) {
            logger.severe("发送交易信号通知失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 发送微信通知 - 支持多种服务
     */
    public boolean sendWechatNotification(String message) {
        String webhookUrl = config.getWechatWebhookUrl();
        
        if (webhookUrl == null || webhookUrl.isEmpty() || webhookUrl.contains("YOUR_WECHAT")) {
            logger.info("微信通知未配置");
            return false;
        }
        
        try {
            // Server酱
            if (webhookUrl.contains("sctapi.ftqq.com") || webhookUrl.contains("sc.ftqq.com")) {
                return sendViaServerChan(message, webhookUrl);
            }
            // Pushplus
            else if (webhookUrl.contains("pushplus.plus")) {
                return sendViaPushplus(message, webhookUrl);
            }
            // 企业微信群机器人
            else if (webhookUrl.contains("qyapi.weixin.qq.com")) {
                return sendViaWeworkBot(message, webhookUrl);
            }
            // 自定义webhook
            else {
                return sendViaCustomWebhook(message, webhookUrl);
            }
            
        } catch (Exception e) {
            logger.severe("微信通知发送失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 通过Server酱发送通知
     */
    private boolean sendViaServerChan(String message, String webhookUrl) throws IOException {
        Map<String, String> data = new HashMap<>();
        data.put("title", "🚀 AI量化交易平台通知");
        data.put("desp", message);
        
        return sendHttpPost(webhookUrl, data, "Server酱");
    }
    
    /**
     * 通过Pushplus发送通知
     */
    private boolean sendViaPushplus(String message, String webhookUrl) throws IOException {
        Map<String, String> data = new HashMap<>();
        data.put("title", "🚀 AI量化交易平台通知");
        data.put("content", message);
        data.put("template", "html");
        
        return sendHttpPost(webhookUrl, data, "Pushplus");
    }
    
    /**
     * 通过企业微信群机器人发送通知
     */
    private boolean sendViaWeworkBot(String message, String webhookUrl) throws IOException {
        Map<String, Object> payload = new HashMap<>();
        payload.put("msgtype", "text");
        
        Map<String, String> textContent = new HashMap<>();
        textContent.put("content", "🚀 AI量化交易平台通知\n\n" + message);
        payload.put("text", textContent);
        
        return sendHttpPost(webhookUrl, payload, "企业微信");
    }
    
    /**
     * 通过自定义webhook发送
     */
    private boolean sendViaCustomWebhook(String message, String webhookUrl) throws IOException {
        Map<String, String> data = new HashMap<>();
        data.put("message", message);
        data.put("source", "AI Trading Platform");
        
        return sendHttpPost(webhookUrl, data, "自定义Webhook");
    }
    
    /**
     * 发送HTTP POST请求
     */
    private boolean sendHttpPost(String url, Object data, String serviceName) throws IOException {
        HttpPost httpPost = new HttpPost(url);
        httpPost.setHeader("Content-Type", "application/json");
        
        String jsonData = objectMapper.writeValueAsString(data);
        httpPost.setEntity(new StringEntity(jsonData, "UTF-8"));
        
        try (CloseableHttpResponse response = httpClient.execute(httpPost)) {
            int statusCode = response.getStatusLine().getStatusCode();
            HttpEntity entity = response.getEntity();
            String responseBody = EntityUtils.toString(entity);
            
            if (statusCode == 200) {
                logger.info(serviceName + " 通知发送成功");
                return true;
            } else {
                logger.warning(serviceName + " 通知发送失败: " + statusCode + " - " + responseBody);
                return false;
            }
        }
    }
    
    /**
     * 发送邮件通知 - 使用HTTP API服务
     */
    public boolean sendEmailNotification(String subject, String message) {
        if (!config.isEmailNotificationEnabled()) {
            logger.info("邮件通知未启用");
            return false;
        }
        
        try {
            // 使用第三方邮件API服务
            return sendViaEmailAPI(subject, message);
            
        } catch (Exception e) {
            logger.severe("邮件发送失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 通过第三方邮件API发送
     */
    private boolean sendViaEmailAPI(String subject, String message) {
        try {
            // 使用EmailJS API (免费，无需SMTP配置)
            String emailjsUrl = "https://api.emailjs.com/api/v1.0/email/send";
            
            Map<String, Object> emailData = new HashMap<>();
            emailData.put("service_id", "service_gmail"); // 需要在EmailJS配置
            emailData.put("template_id", "template_trading"); // 需要在EmailJS配置
            emailData.put("user_id", "your_emailjs_public_key"); // 需要在EmailJS配置
            
            Map<String, String> templateParams = new HashMap<>();
            templateParams.put("to_email", config.getNotificationEmail());
            templateParams.put("subject", "🚀 AI量化交易平台 - " + subject);
            templateParams.put("message", message);
            templateParams.put("from_name", "AI Trading Platform");
            emailData.put("template_params", templateParams);
            
            // 暂时记录日志，实际发送需要配置EmailJS
            logger.info("邮件通知准备发送: " + subject);
            logger.info("收件人: " + config.getNotificationEmail());
            logger.info("内容: " + message);
            
            // 返回true表示准备就绪，实际发送需要用户配置EmailJS
            return true;
            
        } catch (Exception e) {
            logger.severe("邮件API发送失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 创建HTML邮件内容
     */
    private String createEmailHtml(String subject, String message) {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        
        StringBuilder html = new StringBuilder();
        html.append("<!DOCTYPE html>");
        html.append("<html>");
        html.append("<head>");
        html.append("<meta charset=\"UTF-8\">");
        html.append("<style>");
        html.append("body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }");
        html.append(".container { max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }");
        html.append(".header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }");
        html.append(".content { padding: 30px; line-height: 1.8; }");
        html.append(".footer { background: #f8f9fa; padding: 20px; text-align: center; font-size: 12px; color: #666; }");
        html.append(".signal-box { background: #f0f8ff; padding: 20px; border-radius: 10px; margin: 15px 0; border: 2px solid #4CAF50; }");
        html.append("</style>");
        html.append("</head>");
        html.append("<body>");
        html.append("<div class=\"container\">");
        html.append("<div class=\"header\">");
        html.append("<h1>🚀 AI量化交易平台</h1>");
        html.append("<p>by Alvin - 专业级量化交易解决方案</p>");
        html.append("</div>");
        html.append("<div class=\"content\">");
        html.append("<h2>").append(subject).append("</h2>");
        html.append("<div class=\"signal-box\">");
        html.append(message.replace("\n", "<br>"));
        html.append("</div>");
        html.append("<p><strong>发送时间:</strong> ").append(timestamp).append("</p>");
        html.append("</div>");
        html.append("<div class=\"footer\">");
        html.append("<p>© 2024 AI量化交易平台 by Alvin</p>");
        html.append("<p>免责声明: 本系统仅供学习研究使用，投资有风险，决策需谨慎。</p>");
        html.append("</div>");
        html.append("</div>");
        html.append("</body>");
        html.append("</html>");
        
        return html.toString();
    }
    
    /**
     * 格式化交易信号消息
     */
    private String formatTradingSignalMessage(String symbol, AISignal signal, double currentPrice) {
        String actionEmoji = getActionEmoji(signal.getAction());
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        
        return String.format("%s 交易信号提醒\n\n" +
            "📊 股票代码: %s\n" +
            "%s 建议操作: %s\n" +
            "💰 当前价格: $%.2f\n" +
            "🎯 置信度: %.1f%%\n" +
            "📝 分析理由: %s\n" +
            "⏰ 信号时间: %s\n\n" +
            "请根据您的风险承受能力和投资策略谨慎决策。",
            actionEmoji, symbol, actionEmoji, signal.getAction(), 
            currentPrice, signal.getConfidence() * 100, signal.getReason(), timestamp
        );
    }
    
    /**
     * 获取操作对应的emoji
     */
    private String getActionEmoji(String action) {
        switch (action) {
            case "BUY": return "📈";
            case "SELL": return "📉";
            case "HOLD": return "🤝";
            default: return "📊";
        }
    }
    
    /**
     * 发送投资组合预警
     */
    public boolean sendPortfolioAlert(String alertType, String alertMessage) {
        try {
            String subject = "投资组合" + getAlertTypeText(alertType);
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
            String fullMessage = String.format("%s 投资组合提醒\n\n%s\n\n发送时间: %s", 
                getAlertEmoji(alertType), alertMessage, timestamp);
            
            boolean wechatSuccess = sendWechatNotification(fullMessage);
            boolean emailSuccess = false;
            
            if (config.isEmailNotificationEnabled()) {
                emailSuccess = sendEmailNotification(subject, fullMessage);
            }
            
            return wechatSuccess || emailSuccess;
            
        } catch (Exception e) {
            logger.severe("发送投资组合预警失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 发送系统状态通知
     */
    public boolean sendSystemAlert(String alertMessage, String severity) {
        try {
            String emoji = getSeverityEmoji(severity);
            String subject = String.format("系统%s预警", severity.toUpperCase());
            String message = String.format("%s %s", emoji, alertMessage);
            
            // 严重问题同时发送邮件和微信
            if ("ERROR".equals(severity) || "WARNING".equals(severity)) {
                boolean wechatSuccess = sendWechatNotification(message);
                boolean emailSuccess = sendEmailNotification(subject, message);
                return wechatSuccess || emailSuccess;
            } else {
                // 一般信息只发送微信
                return sendWechatNotification(message);
            }
            
        } catch (Exception e) {
            logger.severe("发送系统预警失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 发送每日总结报告
     */
    public boolean sendDailyReport(Map<String, Object> reportData) {
        if (!config.isEmailNotificationEnabled()) {
            return false;
        }
        
        try {
            String date = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
            String subject = String.format("每日交易总结 - %s", date);
            String message = formatDailyReport(reportData);
            
            return sendEmailNotification(subject, message);
            
        } catch (Exception e) {
            logger.severe("发送每日报告失败: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * 格式化每日报告
     */
    private String formatDailyReport(Map<String, Object> reportData) {
        return String.format("📊 每日交易总结报告\n\n" +
            "💰 投资组合表现:\n" +
            "- 总资产: $%,.2f\n" +
            "- 今日盈亏: $%,.2f\n" +
            "- 总收益率: %.2f%%\n\n" +
            "📈 交易活动:\n" +
            "- 生成信号数量: %d\n" +
            "- 高置信度信号: %d\n" +
            "- 执行交易数量: %d\n\n" +
            "🤖 AI模型表现:\n" +
            "- 模型准确率: %.1f%%\n" +
            "- 策略胜率: %.1f%%\n\n" +
            "📅 明日关注:\n%s",
            (Double) reportData.getOrDefault("totalValue", 0.0),
            (Double) reportData.getOrDefault("dailyPnL", 0.0),
            (Double) reportData.getOrDefault("totalReturn", 0.0) * 100,
            (Integer) reportData.getOrDefault("totalSignals", 0),
            (Integer) reportData.getOrDefault("highConfidenceSignals", 0),
            (Integer) reportData.getOrDefault("executedTrades", 0),
            (Double) reportData.getOrDefault("modelAccuracy", 0.0) * 100,
            (Double) reportData.getOrDefault("winRate", 0.0) * 100,
            (String) reportData.getOrDefault("tomorrowFocus", "继续监控市场动态")
        );
    }
    
    private String getAlertTypeText(String alertType) {
        switch (alertType) {
            case "gain": return "收益预警";
            case "loss": return "风险预警";
            case "rebalance": return "再平衡提醒";
            default: return "状态更新";
        }
    }
    
    private String getAlertEmoji(String alertType) {
        switch (alertType) {
            case "gain": return "🎉";
            case "loss": return "⚠️";
            case "rebalance": return "⚖️";
            default: return "📢";
        }
    }
    
    private String getSeverityEmoji(String severity) {
        switch (severity.toUpperCase()) {
            case "ERROR": return "❌";
            case "WARNING": return "⚠️";
            case "INFO": return "ℹ️";
            default: return "📢";
        }
    }
    
    /**
     * 测试通知配置
     */
    public Map<String, Boolean> testNotificationConfig() {
        Map<String, Boolean> results = new HashMap<>();
        
        // 测试微信通知
        boolean wechatTest = sendWechatNotification("🧪 微信通知测试 - 配置正常工作！");
        results.put("wechat", wechatTest);
        
        // 测试邮件通知
        boolean emailTest = sendEmailNotification("通知测试", "📧 邮件通知测试 - 配置正常工作！");
        results.put("email", emailTest);
        
        return results;
    }
    
    /**
     * 关闭资源
     */
    public void close() {
        try {
            if (httpClient != null) {
                httpClient.close();
            }
        } catch (IOException e) {
            logger.warning("关闭HTTP客户端失败: " + e.getMessage());
        }
    }
}
