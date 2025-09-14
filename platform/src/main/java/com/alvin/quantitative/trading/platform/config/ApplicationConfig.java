package com.alvin.quantitative.trading.platform.config;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Application Configuration Manager
 * Author: Alvin
 * Singleton pattern for managing application configuration
 */
public class ApplicationConfig {
    private static final Logger logger = Logger.getLogger(ApplicationConfig.class.getName());
    private static volatile ApplicationConfig instance;
    private final Properties properties;
    
    private ApplicationConfig() {
        properties = new Properties();
        loadConfiguration();
    }
    
    public static ApplicationConfig getInstance() {
        if (instance == null) {
            synchronized (ApplicationConfig.class) {
                if (instance == null) {
                    instance = new ApplicationConfig();
                }
            }
        }
        return instance;
    }
    
    private void loadConfiguration() {
        try (InputStream input = getClass().getClassLoader().getResourceAsStream("application.properties")) {
            if (input == null) {
                logger.warning("Unable to find application.properties, using defaults");
                loadDefaults();
                return;
            }
            
            properties.load(input);
            logger.info("Configuration loaded successfully");
            
        } catch (IOException e) {
            logger.log(Level.SEVERE, "Error loading configuration", e);
            loadDefaults();
        }
        
        overrideWithSystemProperties();
    }
    
    private void loadDefaults() {
        // AI Service Configuration
        properties.setProperty("ai.service.url", "http://localhost:5000");
        properties.setProperty("ai.service.timeout.connect", "10000");
        properties.setProperty("ai.service.timeout.socket", "30000");
        properties.setProperty("ai.service.retry.max", "3");
        
        // Trading Configuration
        properties.setProperty("trading.initial.capital", "100000.0");
        properties.setProperty("trading.symbols", "AAPL,TSLA,MSFT");
        properties.setProperty("trading.data.collection.interval", "60");
        properties.setProperty("trading.strategy.execution.interval", "300");
        properties.setProperty("trading.risk.check.interval", "30");
        
        // Risk Management
        properties.setProperty("risk.max.position.ratio", "0.3");
        properties.setProperty("risk.stop.loss.ratio", "0.05");
        properties.setProperty("risk.take.profit.ratio", "0.15");
        properties.setProperty("risk.max.daily.loss", "3000.0");
        properties.setProperty("risk.min.confidence", "0.6");
        
        // Data Management
        properties.setProperty("data.buffer.max.size", "500");
        properties.setProperty("data.history.max.days", "30");
        
        // UI Configuration
        properties.setProperty("ui.server.port", "8080");
        
        // 通知默认配置
        properties.setProperty("email.enabled", "false");
        properties.setProperty("email.username", "your_qq_email@qq.com");
        properties.setProperty("email.password", "your_qq_auth_code");
        properties.setProperty("email.notification.address", "wangjians8813@gmail.com");
        properties.setProperty("email.smtp.host", "smtp.qq.com");
        properties.setProperty("email.smtp.port", "587");
        properties.setProperty("wechat.enabled", "false");
        properties.setProperty("wechat.webhook.url", "YOUR_WECHAT_BOT_WEBHOOK");
        properties.setProperty("notification.min.confidence", "0.75");
        properties.setProperty("notification.send.daily.summary", "true");
        properties.setProperty("ui.server.host", "localhost");
        
        logger.info("Default configuration loaded");
    }
    
    private void overrideWithSystemProperties() {
        String[] systemOverrides = {
            "ai.service.url", "trading.initial.capital", "trading.symbols",
            "risk.max.position.ratio", "ui.server.port"
        };
        
        for (String key : systemOverrides) {
            String systemValue = System.getProperty(key);
            if (systemValue != null) {
                properties.setProperty(key, systemValue);
                logger.info("Override from system property: " + key + " = " + systemValue);
            }
        }
    }
    
    // AI Service Configuration
    public String getAiServiceUrl() { return properties.getProperty("ai.service.url", "http://localhost:5000"); }
    public int getAiServiceConnectTimeout() { return getIntProperty("ai.service.timeout.connect", 10000); }
    public int getAiServiceSocketTimeout() { return getIntProperty("ai.service.timeout.socket", 30000); }
    public int getAiServiceMaxRetry() { return getIntProperty("ai.service.retry.max", 3); }
    
    // Trading Configuration
    public double getInitialCapital() { return getDoubleProperty("trading.initial.capital", 100000.0); }
    public List<String> getTradingSymbols() {
        String symbols = properties.getProperty("trading.symbols", "AAPL,TSLA,MSFT");
        return Arrays.asList(symbols.split(","));
    }
    public int getDataCollectionInterval() { return getIntProperty("trading.data.collection.interval", 60); }
    public int getStrategyExecutionInterval() { return getIntProperty("trading.strategy.execution.interval", 300); }
    public int getRiskCheckInterval() { return getIntProperty("trading.risk.check.interval", 30); }
    
    // Risk Management Configuration
    public double getMaxPositionRatio() { return getDoubleProperty("risk.max.position.ratio", 0.3); }
    public double getStopLossRatio() { return getDoubleProperty("risk.stop.loss.ratio", 0.05); }
    public double getTakeProfitRatio() { return getDoubleProperty("risk.take.profit.ratio", 0.15); }
    public double getMaxDailyLoss() { return getDoubleProperty("risk.max.daily.loss", 3000.0); }
    public double getMinConfidence() { return getDoubleProperty("risk.min.confidence", 0.6); }
    
    // Data Management Configuration
    public int getMaxBufferSize() { return getIntProperty("data.buffer.max.size", 500); }
    public int getMaxHistoryDays() { return getIntProperty("data.history.max.days", 30); }
    
    // UI Configuration
    public int getUiServerPort() { return getIntProperty("ui.server.port", 8080); }
    public String getUiServerHost() { return properties.getProperty("ui.server.host", "localhost"); }
    
    // 通知配置获取方法
    public boolean isEmailNotificationEnabled() { return getBooleanProperty("email.enabled", false); }
    public String getEmailUsername() { return getStringProperty("email.username", ""); }
    public String getEmailPassword() { return getStringProperty("email.password", ""); }
    public String getNotificationEmail() { return getStringProperty("email.notification.address", ""); }
    public String getEmailSmtpHost() { return getStringProperty("email.smtp.host", "smtp.qq.com"); }
    public int getEmailSmtpPort() { return getIntProperty("email.smtp.port", 587); }
    
    public boolean isWechatNotificationEnabled() { return getBooleanProperty("wechat.enabled", false); }
    public String getWechatWebhookUrl() { return getStringProperty("wechat.webhook.url", ""); }
    
    public double getNotificationMinConfidence() { return getDoubleProperty("notification.min.confidence", 0.75); }
    public boolean isSendDailySummary() { return getBooleanProperty("notification.send.daily.summary", true); }
    
    // Utility methods
    public int getIntProperty(String key, int defaultValue) {
        try {
            return Integer.parseInt(properties.getProperty(key, String.valueOf(defaultValue)));
        } catch (NumberFormatException e) {
            logger.warning("Invalid integer value for " + key + ", using default: " + defaultValue);
            return defaultValue;
        }
    }
    
    public double getDoubleProperty(String key, double defaultValue) {
        try {
            return Double.parseDouble(properties.getProperty(key, String.valueOf(defaultValue)));
        } catch (NumberFormatException e) {
            logger.warning("Invalid double value for " + key + ", using default: " + defaultValue);
            return defaultValue;
        }
    }
    
    public boolean getBooleanProperty(String key, boolean defaultValue) {
        String value = properties.getProperty(key, String.valueOf(defaultValue));
        return Boolean.parseBoolean(value);
    }
    
    public String getStringProperty(String key, String defaultValue) {
        return properties.getProperty(key, defaultValue);
    }
    
    public String getProperty(String key, String defaultValue) {
        return properties.getProperty(key, defaultValue);
    }
    
    public void setProperty(String key, String value) {
        properties.setProperty(key, value);
    }
    
    public boolean validateConfiguration() {
        boolean isValid = true;
        
        // 生产环境关键配置验证
        if (getAiServiceUrl() == null || getAiServiceUrl().trim().isEmpty()) {
            logger.severe("🚨 PRODUCTION ERROR: AI service URL is not configured");
            isValid = false;
        }
        
        if (getInitialCapital() <= 0) {
            logger.severe("🚨 PRODUCTION ERROR: Initial capital must be positive");
            isValid = false;
        }
        
        if (getMaxPositionRatio() <= 0 || getMaxPositionRatio() > 1) {
            logger.severe("🚨 PRODUCTION ERROR: Max position ratio must be between 0 and 1");
            isValid = false;
        }
        
        // 新增：生产环境安全检查
        if (getInitialCapital() > 1000000 && getMaxDailyLoss() < getInitialCapital() * 0.02) {
            logger.warning("⚠️ PRODUCTION WARNING: For large capital, daily loss limit seems too low");
        }
        
        if (getStopLossRatio() < 0.02 || getStopLossRatio() > 0.1) {
            logger.warning("⚠️ PRODUCTION WARNING: Stop loss ratio should be between 2%-10%");
        }
        
        if (getTradingSymbols().isEmpty()) {
            logger.severe("🚨 PRODUCTION ERROR: No trading symbols configured");
            isValid = false;
        }
        
        // 验证AI服务连接
        try {
            // 这里可以添加AI服务连接测试
            logger.info("AI service URL configured: " + getAiServiceUrl());
        } catch (Exception e) {
            logger.warning("⚠️ PRODUCTION WARNING: Cannot verify AI service connection: " + e.getMessage());
        }
        
        if (isValid) {
            logger.info("✅ Production configuration validation passed");
        } else {
            logger.severe("❌ Production configuration validation failed - SYSTEM WILL NOT START");
        }
        
        return isValid;
    }
    
    public void printConfiguration() {
        logger.info("=== Trading Platform Configuration ===");
        logger.info("AI Service URL: " + getAiServiceUrl());
        logger.info("Initial Capital: $" + String.format("%.2f", getInitialCapital()));
        logger.info("Trading Symbols: " + getTradingSymbols());
        logger.info("Max Position Ratio: " + getMaxPositionRatio());
        logger.info("Stop Loss Ratio: " + getStopLossRatio());
        logger.info("Take Profit Ratio: " + getTakeProfitRatio());
        logger.info("Min Confidence: " + getMinConfidence());
        logger.info("UI Server: http://" + getUiServerHost() + ":" + getUiServerPort());
        logger.info("=====================================");
    }
}
