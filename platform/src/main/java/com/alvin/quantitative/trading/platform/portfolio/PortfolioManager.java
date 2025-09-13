package com.alvin.quantitative.trading.platform.portfolio;

import com.alvin.quantitative.trading.platform.core.AISignal;
import com.alvin.quantitative.trading.platform.core.Position;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Portfolio Manager
 * Author: Alvin
 * Manages portfolio configuration and trading rules
 */
public class PortfolioManager {
    private static final Logger logger = Logger.getLogger(PortfolioManager.class.getName());
    private static PortfolioManager instance;
    
    private JsonNode portfolioConfig;
    private final ObjectMapper objectMapper;
    private final Map<String, SymbolConfig> symbolConfigs;
    private final Map<String, Position> currentPositions;
    
    private PortfolioManager() {
        this.objectMapper = new ObjectMapper();
        this.symbolConfigs = new HashMap<>();
        this.currentPositions = new HashMap<>();
        loadPortfolioConfig();
    }
    
    public static synchronized PortfolioManager getInstance() {
        if (instance == null) {
            instance = new PortfolioManager();
        }
        return instance;
    }
    
    private void loadPortfolioConfig() {
        try {
            File configFile = new File("portfolio.json");
            if (!configFile.exists()) {
                logger.warning("Portfolio configuration file not found: portfolio.json");
                return;
            }
            
            portfolioConfig = objectMapper.readTree(configFile);
            
            // Parse symbols configuration
            JsonNode symbolsNode = portfolioConfig.get("symbols");
            if (symbolsNode != null && symbolsNode.isArray()) {
                for (JsonNode symbolNode : symbolsNode) {
                    SymbolConfig config = parseSymbolConfig(symbolNode);
                    symbolConfigs.put(config.getSymbol(), config);
                }
            }
            
            logger.info("Portfolio configuration loaded successfully");
            logger.info("Monitoring " + symbolConfigs.size() + " symbols: " + 
                String.join(", ", symbolConfigs.keySet()));
            
        } catch (IOException e) {
            logger.severe("Failed to load portfolio configuration: " + e.getMessage());
        }
    }
    
    private SymbolConfig parseSymbolConfig(JsonNode symbolNode) {
        SymbolConfig config = new SymbolConfig();
        config.setSymbol(symbolNode.get("symbol").asText());
        config.setName(symbolNode.get("name").asText());
        config.setType(symbolNode.get("type").asText());
        config.setSector(symbolNode.get("sector").asText());
        config.setWeight(symbolNode.get("weight").asDouble());
        config.setPriority(symbolNode.get("priority").asText());
        config.setMinConfidence(symbolNode.get("min_confidence").asDouble());
        config.setNotes(symbolNode.get("notes").asText(""));
        return config;
    }
    
    public List<String> getMonitoringSymbols() {
        return new ArrayList<>(symbolConfigs.keySet());
    }
    
    public List<SymbolConfig> getSymbolConfigs() {
        return new ArrayList<>(symbolConfigs.values());
    }
    
    public SymbolConfig getSymbolConfig(String symbol) {
        return symbolConfigs.get(symbol);
    }
    
    public boolean shouldTrade(String symbol, AISignal signal) {
        SymbolConfig config = symbolConfigs.get(symbol);
        if (config == null) {
            return false;
        }
        
        // Check minimum confidence threshold
        if (signal.getConfidence() < config.getMinConfidence()) {
            logger.info(String.format("Signal for %s rejected: confidence %.2f < required %.2f", 
                symbol, signal.getConfidence(), config.getMinConfidence()));
            return false;
        }
        
        // Check daily trade limits
        if (getTodayTradeCount() >= getMaxDailyTrades()) {
            logger.info("Daily trade limit reached, skipping trade for " + symbol);
            return false;
        }
        
        // Check portfolio risk limits
        if (wouldExceedRiskLimits(symbol, signal)) {
            logger.info("Trade would exceed risk limits for " + symbol);
            return false;
        }
        
        return true;
    }
    
    public boolean shouldSendNotification(String symbol, AISignal signal) {
        JsonNode notificationTriggers = portfolioConfig.path("trading_rules").path("notification_triggers");
        
        if ("BUY".equals(signal.getAction()) || "SELL".equals(signal.getAction())) {
            double threshold = "BUY".equals(signal.getAction()) ? 
                notificationTriggers.path("strong_buy").asDouble(0.85) :
                notificationTriggers.path("strong_sell").asDouble(0.85);
                
            return signal.getConfidence() >= threshold;
        }
        
        return false;
    }
    
    private int getTodayTradeCount() {
        // TODO: Implement trade counting logic
        return 0;
    }
    
    private int getMaxDailyTrades() {
        return portfolioConfig.path("trading_rules").path("max_daily_trades").asInt(3);
    }
    
    private boolean wouldExceedRiskLimits(String symbol, AISignal signal) {
        // TODO: Implement risk calculation
        return false;
    }
    
    public String getPortfolioName() {
        return portfolioConfig.path("portfolio").path("name").asText("AI量化投资组合");
    }
    
    public String getNotificationEmail() {
        return portfolioConfig.path("notification_settings").path("email").path("address").asText("");
    }
    
    public boolean isEmailNotificationEnabled() {
        return portfolioConfig.path("notification_settings").path("email").path("enabled").asBoolean(false);
    }
    
    public boolean isWeChatNotificationEnabled() {
        return portfolioConfig.path("notification_settings").path("wechat").path("enabled").asBoolean(false);
    }
    
    public String getWeChatWebhookUrl() {
        return portfolioConfig.path("notification_settings").path("wechat").path("webhook_url").asText("");
    }
    
    public int getBacktestPeriodYears() {
        return portfolioConfig.path("trading_rules").path("backtest_period_years").asInt(3);
    }
    
    public double getMaxPortfolioRisk() {
        return portfolioConfig.path("trading_rules").path("max_portfolio_risk").asDouble(0.15);
    }
    
    public void updatePosition(String symbol, Position position) {
        currentPositions.put(symbol, position);
        logger.info(String.format("Updated position for %s: %.2f shares at $%.2f", 
            symbol, position.getShares(), position.getAvgCost()));
    }
    
    public Map<String, Position> getCurrentPositions() {
        return new HashMap<>(currentPositions);
    }
    
    public void printPortfolioSummary() {
        System.out.println("\n" + repeat("=", 60));
        System.out.println("📊 " + getPortfolioName());
        System.out.println(repeat("=", 60));
        
        System.out.println("监控标的:");
        for (SymbolConfig config : symbolConfigs.values()) {
            String type = "stock".equals(config.getType()) ? "📈" : "📊";
            System.out.println(String.format("  %s %s (%s) - %s - 权重:%.1f%% - 最小置信度:%.1f%%", 
                type, config.getSymbol(), config.getName(), config.getSector(), 
                config.getWeight() * 100, config.getMinConfidence() * 100));
        }
        
        System.out.println("\n交易规则:");
        System.out.println("  • 每日最大交易次数: " + getMaxDailyTrades());
        System.out.println("  • 投资组合最大风险: " + String.format("%.1f%%", getMaxPortfolioRisk() * 100));
        System.out.println("  • 回测周期: " + getBacktestPeriodYears() + "年");
        
        System.out.println("\n通知设置:");
        System.out.println("  📧 邮件通知: " + (isEmailNotificationEnabled() ? "✅ " + getNotificationEmail() : "❌"));
        System.out.println("  💬 微信通知: " + (isWeChatNotificationEnabled() ? "✅" : "❌"));
        
        System.out.println(repeat("=", 60) + "\n");
    }
    
    private String repeat(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }
    
    // Inner class for symbol configuration
    public static class SymbolConfig {
        private String symbol;
        private String name;
        private String type;
        private String sector;
        private double weight;
        private String priority;
        private double minConfidence;
        private String notes;
        
        // Getters and setters
        public String getSymbol() { return symbol; }
        public void setSymbol(String symbol) { this.symbol = symbol; }
        
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        
        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        
        public String getSector() { return sector; }
        public void setSector(String sector) { this.sector = sector; }
        
        public double getWeight() { return weight; }
        public void setWeight(double weight) { this.weight = weight; }
        
        public String getPriority() { return priority; }
        public void setPriority(String priority) { this.priority = priority; }
        
        public double getMinConfidence() { return minConfidence; }
        public void setMinConfidence(double minConfidence) { this.minConfidence = minConfidence; }
        
        public String getNotes() { return notes; }
        public void setNotes(String notes) { this.notes = notes; }
    }
}
