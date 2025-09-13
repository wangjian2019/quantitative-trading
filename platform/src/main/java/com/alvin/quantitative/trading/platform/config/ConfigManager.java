package com.alvin.quantitative.trading.platform.config;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Configuration Manager
 * Author: Alvin
 * Manages all configuration settings for the trading platform
 */
public class ConfigManager {
    private static final Logger logger = Logger.getLogger(ConfigManager.class.getName());
    private static ConfigManager instance;
    private final Properties properties;
    
    private ConfigManager() {
        properties = new Properties();
        loadConfiguration();
    }
    
    public static synchronized ConfigManager getInstance() {
        if (instance == null) {
            instance = new ConfigManager();
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
        
        // Override with system properties if available
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
        
        logger.info("Default configuration loaded");
    }
    
    private void overrideWithSystemProperties() {
        // Check for system property overrides
        String[] systemOverrides = {
            "ai.service.url",
            "trading.initial.capital",
            "trading.symbols",
            "risk.max.position.ratio",
            "risk.stop.loss.ratio",
            "risk.take.profit.ratio"
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
    public String getAiServiceUrl() {
        return properties.getProperty("ai.service.url", "http://localhost:5000");
    }
    
    public int getAiServiceConnectTimeout() {
        return getIntProperty("ai.service.timeout.connect", 10000);
    }
    
    public int getAiServiceSocketTimeout() {
        return getIntProperty("ai.service.timeout.socket", 30000);
    }
    
    public int getAiServiceMaxRetry() {
        return getIntProperty("ai.service.retry.max", 3);
    }
    
    // Trading Configuration
    public double getInitialCapital() {
        return getDoubleProperty("trading.initial.capital", 100000.0);
    }
    
    public List<String> getTradingSymbols() {
        String symbols = properties.getProperty("trading.symbols", "AAPL,TSLA,MSFT");
        return Arrays.asList(symbols.split(","));
    }
    
    public int getDataCollectionInterval() {
        return getIntProperty("trading.data.collection.interval", 60);
    }
    
    public int getStrategyExecutionInterval() {
        return getIntProperty("trading.strategy.execution.interval", 300);
    }
    
    public int getRiskCheckInterval() {
        return getIntProperty("trading.risk.check.interval", 30);
    }
    
    // Risk Management Configuration
    public double getMaxPositionRatio() {
        return getDoubleProperty("risk.max.position.ratio", 0.3);
    }
    
    public double getStopLossRatio() {
        return getDoubleProperty("risk.stop.loss.ratio", 0.05);
    }
    
    public double getTakeProfitRatio() {
        return getDoubleProperty("risk.take.profit.ratio", 0.15);
    }
    
    public double getMaxDailyLoss() {
        return getDoubleProperty("risk.max.daily.loss", 3000.0);
    }
    
    public double getMinConfidence() {
        return getDoubleProperty("risk.min.confidence", 0.6);
    }
    
    // Data Management Configuration
    public int getMaxBufferSize() {
        return getIntProperty("data.buffer.max.size", 500);
    }
    
    public int getMaxHistoryDays() {
        return getIntProperty("data.history.max.days", 30);
    }
    
    // Utility methods
    private int getIntProperty(String key, int defaultValue) {
        try {
            return Integer.parseInt(properties.getProperty(key, String.valueOf(defaultValue)));
        } catch (NumberFormatException e) {
            logger.warning("Invalid integer value for " + key + ", using default: " + defaultValue);
            return defaultValue;
        }
    }
    
    private double getDoubleProperty(String key, double defaultValue) {
        try {
            return Double.parseDouble(properties.getProperty(key, String.valueOf(defaultValue)));
        } catch (NumberFormatException e) {
            logger.warning("Invalid double value for " + key + ", using default: " + defaultValue);
            return defaultValue;
        }
    }
    
    private boolean getBooleanProperty(String key, boolean defaultValue) {
        String value = properties.getProperty(key, String.valueOf(defaultValue));
        return Boolean.parseBoolean(value);
    }
    
    public String getProperty(String key, String defaultValue) {
        return properties.getProperty(key, defaultValue);
    }
    
    public void setProperty(String key, String value) {
        properties.setProperty(key, value);
    }
    
    // Configuration validation
    public boolean validateConfiguration() {
        boolean isValid = true;
        
        // Validate AI service URL
        String aiUrl = getAiServiceUrl();
        if (aiUrl == null || aiUrl.trim().isEmpty()) {
            logger.severe("AI service URL is not configured");
            isValid = false;
        }
        
        // Validate capital
        if (getInitialCapital() <= 0) {
            logger.severe("Initial capital must be positive");
            isValid = false;
        }
        
        // Validate risk ratios
        if (getMaxPositionRatio() <= 0 || getMaxPositionRatio() > 1) {
            logger.severe("Max position ratio must be between 0 and 1");
            isValid = false;
        }
        
        if (getStopLossRatio() <= 0 || getStopLossRatio() > 1) {
            logger.severe("Stop loss ratio must be between 0 and 1");
            isValid = false;
        }
        
        if (getTakeProfitRatio() <= 0) {
            logger.severe("Take profit ratio must be positive");
            isValid = false;
        }
        
        // Validate confidence
        if (getMinConfidence() < 0 || getMinConfidence() > 1) {
            logger.severe("Min confidence must be between 0 and 1");
            isValid = false;
        }
        
        if (isValid) {
            logger.info("Configuration validation passed");
        } else {
            logger.severe("Configuration validation failed");
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
        logger.info("=====================================");
    }
}
