package com.alvin.quantitative.trading.platform.ui;

import com.alvin.quantitative.trading.platform.engine.TradingEngine;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * UI Controller - MVC Pattern
 * Author: Alvin
 * Handles web UI requests and responses
 */
public class UIController {
    private static final Logger logger = Logger.getLogger(UIController.class.getName());
    
    private final TradingEngine tradingEngine;
    private final ObjectMapper objectMapper;
    
    public UIController(TradingEngine tradingEngine) {
        this.tradingEngine = tradingEngine;
        this.objectMapper = new ObjectMapper();
    }
    
    public String getSystemStatus() {
        try {
            Map<String, Object> status = new HashMap<String, Object>();
            
            // Get system health from trading engine
            Map<String, Object> healthReport = tradingEngine.getHealthReport();
            
            status.put("healthy", healthReport.get("system_healthy"));
            status.put("signalSuccessRate", 0.95); // TODO: Get from health monitor
            status.put("tradeSuccessRate", 0.87); // TODO: Get from health monitor
            status.put("activeThreads", healthReport.get("active_threads"));
            status.put("memoryUsage", 256); // TODO: Get actual memory usage
            status.put("lastUpdate", System.currentTimeMillis());
            
            return objectMapper.writeValueAsString(status);
            
        } catch (Exception e) {
            logger.severe("Failed to get system status: " + e.getMessage());
            return "{\"error\":\"Failed to get system status\"}";
        }
    }
    
    public String getPortfolioData() {
        try {
            Map<String, Object> portfolio = new HashMap<String, Object>();
            
            // TODO: Get actual portfolio data from trading engine
            portfolio.put("totalValue", 125000.0);
            portfolio.put("totalReturn", 0.25);
            portfolio.put("todayPnL", 1250.0);
            portfolio.put("positions", tradingEngine.getCurrentPositions());
            
            return objectMapper.writeValueAsString(portfolio);
            
        } catch (Exception e) {
            logger.severe("Failed to get portfolio data: " + e.getMessage());
            return "{\"error\":\"Failed to get portfolio data\"}";
        }
    }
    
    public String runBacktest() {
        try {
            // Run backtest asynchronously
            Map<String, Object> result = tradingEngine.runBacktestAnalysis();
            return objectMapper.writeValueAsString(result);
            
        } catch (Exception e) {
            logger.severe("Failed to run backtest: " + e.getMessage());
            return "{\"error\":\"Failed to run backtest: " + e.getMessage() + "\"}";
        }
    }
    
    public String getBacktestResults() {
        try {
            Map<String, Object> results = tradingEngine.getLatestBacktestResults();
            return objectMapper.writeValueAsString(results);
            
        } catch (Exception e) {
            logger.severe("Failed to get backtest results: " + e.getMessage());
            return "{\"error\":\"Failed to get backtest results\"}";
        }
    }
    
    public String getRecentSignals() {
        try {
            Map<String, Object> signals = tradingEngine.getRecentSignals();
            return objectMapper.writeValueAsString(signals);
            
        } catch (Exception e) {
            logger.severe("Failed to get recent signals: " + e.getMessage());
            return "{\"error\":\"Failed to get recent signals\"}";
        }
    }
    
    public String getHealthStatus() {
        try {
            Map<String, Object> health = tradingEngine.getHealthReport();
            return objectMapper.writeValueAsString(health);
            
        } catch (Exception e) {
            logger.severe("Failed to get health status: " + e.getMessage());
            return "{\"error\":\"Failed to get health status\"}";
        }
    }
}
