package com.alvin.quantitative.trading.platform.notification;

import com.alvin.quantitative.trading.platform.config.ApplicationConfig;
import com.alvin.quantitative.trading.platform.core.AISignal;
import com.alvin.quantitative.trading.platform.portfolio.PortfolioManager;

import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;

// Email imports temporarily disabled due to dependency issues
// import javax.mail.*;
// import javax.mail.internet.InternetAddress;
// import javax.mail.internet.MimeMessage;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Logger;

/**
 * Notification Service
 * Author: Alvin
 * Handles email and WeChat notifications for trading signals
 */
public class NotificationService {
    private static final Logger logger = Logger.getLogger(NotificationService.class.getName());
    private static NotificationService instance;
    
    private final PortfolioManager portfolioManager;
    private final ApplicationConfig configManager;
    private final ExecutorService executorService;
    private final CloseableHttpClient httpClient;
    
    // Email configuration - temporarily disabled
    // private Session emailSession;
    private boolean emailConfigured = false;
    
    private NotificationService() {
        this.portfolioManager = PortfolioManager.getInstance();
        this.configManager = ApplicationConfig.getInstance();
        this.executorService = Executors.newFixedThreadPool(2);
        this.httpClient = HttpClients.createDefault();
        
        setupEmailConfiguration();
    }
    
    public static synchronized NotificationService getInstance() {
        if (instance == null) {
            instance = new NotificationService();
        }
        return instance;
    }
    
    private void setupEmailConfiguration() {
        // Email functionality temporarily disabled due to dependency issues
        // Will use alternative notification methods
        emailConfigured = false;
        logger.info("Email notifications temporarily disabled - using WeChat and console notifications");
    }
    
    public void sendTradingSignalNotification(String symbol, AISignal signal, double currentPrice) {
        if (!portfolioManager.shouldSendNotification(symbol, signal)) {
            return;
        }
        
        PortfolioManager.SymbolConfig symbolConfig = portfolioManager.getSymbolConfig(symbol);
        if (symbolConfig == null) {
            return;
        }
        
        // Send notifications asynchronously
        executorService.submit(() -> {
            try {
                if (portfolioManager.isEmailNotificationEnabled() && emailConfigured) {
                    sendEmailNotification(symbol, signal, currentPrice, symbolConfig);
                }
                
                if (portfolioManager.isWeChatNotificationEnabled()) {
                    sendWeChatNotification(symbol, signal, currentPrice, symbolConfig);
                }
            } catch (Exception e) {
                logger.severe("Failed to send notifications: " + e.getMessage());
            }
        });
    }
    
    private void sendEmailNotification(String symbol, AISignal signal, double currentPrice, 
                                     PortfolioManager.SymbolConfig symbolConfig) {
        // Email functionality temporarily disabled
        // Print to console instead
        String action = signal.getAction();
        String emoji = getActionEmoji(action);
        
        System.out.println("\n" + repeat("=", 50));
        System.out.println(String.format("📧 [邮件通知] %s AI量化交易信号 - %s %s", emoji, symbol, action));
        System.out.println(String.format("📊 标的: %s (%s)", symbol, symbolConfig.getName()));
        System.out.println(String.format("💰 当前价格: $%.2f", currentPrice));
        System.out.println(String.format("🎯 操作建议: %s", action));
        System.out.println(String.format("🔥 置信度: %.1f%%", signal.getConfidence() * 100));
        System.out.println(String.format("💡 理由: %s", signal.getReason()));
        System.out.println(String.format("📧 发送至: %s", portfolioManager.getNotificationEmail()));
        System.out.println(repeat("=", 50) + "\n");
        
        logger.info(String.format("Console notification displayed for %s %s signal", symbol, action));
    }
    
    private void sendWeChatNotification(String symbol, AISignal signal, double currentPrice,
                                      PortfolioManager.SymbolConfig symbolConfig) {
        try {
            String webhookUrl = portfolioManager.getWeChatWebhookUrl();
            if (webhookUrl.isEmpty() || "YOUR_WECHAT_BOT_WEBHOOK".equals(webhookUrl)) {
                logger.warning("WeChat webhook URL not configured");
                return;
            }
            
            String action = signal.getAction();
            String emoji = getActionEmoji(action);
            
            String message = String.format(
                "%s **AI量化交易提醒**\n\n" +
                "📊 **标的**: %s (%s)\n" +
                "💰 **当前价格**: $%.2f\n" +
                "🎯 **操作建议**: %s\n" +
                "🔥 **置信度**: %.1f%%\n" +
                "💡 **理由**: %s\n" +
                "⏰ **时间**: %s\n\n" +
                "---\n" +
                "🤖 AI量化交易系统 by Alvin",
                emoji, symbol, symbolConfig.getName(), currentPrice, action,
                signal.getConfidence() * 100, signal.getReason(),
                LocalDateTime.now().format(DateTimeFormatter.ofPattern("MM-dd HH:mm:ss"))
            );
            
            // Create JSON payload for WeChat bot
            String jsonPayload = String.format(
                "{\"msgtype\":\"markdown\",\"markdown\":{\"content\":\"%s\"}}",
                message.replace("\"", "\\\"").replace("\n", "\\n")
            );
            
            HttpPost httpPost = new HttpPost(webhookUrl);
            httpPost.setHeader("Content-Type", "application/json");
            httpPost.setEntity(new StringEntity(jsonPayload, "UTF-8"));
            
            try (CloseableHttpResponse response = httpClient.execute(httpPost)) {
                int statusCode = response.getStatusLine().getStatusCode();
                if (statusCode == 200) {
                    logger.info(String.format("WeChat notification sent for %s %s signal", symbol, action));
                } else {
                    String responseBody = EntityUtils.toString(response.getEntity());
                    logger.warning(String.format("WeChat notification failed: %d - %s", statusCode, responseBody));
                }
            }
            
        } catch (Exception e) {
            logger.severe("Failed to send WeChat notification: " + e.getMessage());
        }
    }
    
    private String createEmailBody(String symbol, AISignal signal, double currentPrice, 
                                 PortfolioManager.SymbolConfig symbolConfig) {
        String action = signal.getAction();
        String emoji = getActionEmoji(action);
        String actionColor = getActionColor(action);
        
        return String.format(
            "<html><body style='font-family: Arial, sans-serif;'>" +
            "<div style='max-width: 600px; margin: 0 auto; padding: 20px;'>" +
            
            "<h2 style='color: #333; text-align: center;'>%s AI量化交易信号</h2>" +
            
            "<div style='background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%); " +
            "color: white; padding: 20px; border-radius: 10px; margin: 20px 0;'>" +
            "<h3 style='margin: 0 0 10px 0;'>📊 %s (%s)</h3>" +
            "<p style='margin: 5px 0; font-size: 14px;'>%s | %s</p>" +
            "</div>" +
            
            "<div style='background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;'>" +
            "<table style='width: 100%%; border-collapse: collapse;'>" +
            "<tr><td style='padding: 8px 0; font-weight: bold;'>💰 当前价格:</td><td>$%.2f</td></tr>" +
            "<tr><td style='padding: 8px 0; font-weight: bold;'>🎯 操作建议:</td>" +
            "<td><span style='color: %s; font-weight: bold; font-size: 16px;'>%s %s</span></td></tr>" +
            "<tr><td style='padding: 8px 0; font-weight: bold;'>🔥 置信度:</td><td>%.1f%%</td></tr>" +
            "<tr><td style='padding: 8px 0; font-weight: bold;'>📈 权重配置:</td><td>%.1f%%</td></tr>" +
            "<tr><td style='padding: 8px 0; font-weight: bold;'>⭐ 优先级:</td><td>%s</td></tr>" +
            "</table>" +
            "</div>" +
            
            "<div style='background: #e3f2fd; padding: 15px; border-left: 4px solid #2196f3; margin: 20px 0;'>" +
            "<h4 style='margin: 0 0 10px 0; color: #1976d2;'>💡 AI分析理由</h4>" +
            "<p style='margin: 0; color: #424242;'>%s</p>" +
            "</div>" +
            
            "<div style='background: #fff3e0; padding: 15px; border-left: 4px solid #ff9800; margin: 20px 0;'>" +
            "<h4 style='margin: 0 0 10px 0; color: #f57c00;'>📝 备注信息</h4>" +
            "<p style='margin: 0; color: #424242;'>%s</p>" +
            "</div>" +
            
            "<div style='text-align: center; margin-top: 30px; padding-top: 20px; " +
            "border-top: 1px solid #eee; color: #666; font-size: 12px;'>" +
            "<p>⏰ 生成时间: %s</p>" +
            "<p>🤖 AI量化交易系统 by Alvin</p>" +
            "<p style='color: #999;'>⚠️ 投资有风险，决策需谨慎。本信号仅供参考，不构成投资建议。</p>" +
            "</div>" +
            
            "</div></body></html>",
            
            emoji, symbol, symbolConfig.getName(), symbolConfig.getType().toUpperCase(), 
            symbolConfig.getSector(), currentPrice, actionColor, emoji, action, 
            signal.getConfidence() * 100, symbolConfig.getWeight() * 100, 
            symbolConfig.getPriority(), signal.getReason(), symbolConfig.getNotes(),
            LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))
        );
    }
    
    public void sendDailySummaryEmail(Map<String, Double> portfolioPerformance, 
                                    List<String> todaySignals) {
        if (!portfolioManager.isEmailNotificationEnabled()) {
            return;
        }
        
        // Print daily summary to console instead of email
        System.out.println("\n" + repeat("=", 60));
        System.out.println("📊 AI量化交易日报 - " + 
            LocalDateTime.now().format(DateTimeFormatter.ofPattern("MM月dd日")));
        System.out.println(repeat("=", 60));
        System.out.println("📈 今日交易信号 (" + todaySignals.size() + "个):");
        for (String signal : todaySignals) {
            System.out.println("  • " + signal);
        }
        System.out.println("📧 本应发送至: " + portfolioManager.getNotificationEmail());
        System.out.println(repeat("=", 60) + "\n");
        
        logger.info("Daily summary displayed in console (email temporarily disabled)");
    }
    
    private String createDailySummaryBody(Map<String, Double> portfolioPerformance, 
                                        List<String> todaySignals) {
        StringBuilder signalsHtml = new StringBuilder();
        for (String signal : todaySignals) {
            signalsHtml.append("<li style='margin: 5px 0;'>").append(signal).append("</li>");
        }
        
        return String.format(
            "<html><body style='font-family: Arial, sans-serif;'>" +
            "<div style='max-width: 600px; margin: 0 auto; padding: 20px;'>" +
            "<h2 style='color: #333; text-align: center;'>📊 AI量化交易日报</h2>" +
            "<h3 style='color: #666;'>%s</h3>" +
            
            "<div style='background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;'>" +
            "<h4>📈 今日交易信号 (%d个)</h4>" +
            "<ul style='padding-left: 20px;'>%s</ul>" +
            "</div>" +
            
            "<div style='text-align: center; margin-top: 30px; padding-top: 20px; " +
            "border-top: 1px solid #eee; color: #666; font-size: 12px;'>" +
            "<p>🤖 AI量化交易系统 by Alvin</p>" +
            "</div>" +
            
            "</div></body></html>",
            
            LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy年MM月dd日")),
            todaySignals.size(), signalsHtml.toString()
        );
    }
    
    private String getActionEmoji(String action) {
        switch (action) {
            case "BUY": return "🚀";
            case "SELL": return "📉";
            case "HOLD": return "⏸️";
            default: return "📊";
        }
    }
    
    private String getActionColor(String action) {
        switch (action) {
            case "BUY": return "#4caf50";
            case "SELL": return "#f44336";
            case "HOLD": return "#ff9800";
            default: return "#2196f3";
        }
    }
    
    private String repeat(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }
    
    public void shutdown() {
        executorService.shutdown();
        try {
            if (httpClient != null) {
                httpClient.close();
            }
        } catch (Exception e) {
            logger.warning("Error closing HTTP client: " + e.getMessage());
        }
    }
}
