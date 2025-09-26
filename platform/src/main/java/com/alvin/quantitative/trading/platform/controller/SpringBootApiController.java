package com.alvin.quantitative.trading.platform.controller;

import com.alvin.quantitative.trading.platform.config.ApplicationConfig;
import com.alvin.quantitative.trading.platform.core.KlineData;
import com.alvin.quantitative.trading.platform.engine.TradingEngineInterface;
import com.alvin.quantitative.trading.platform.engine.ProfessionalTradingEngine;
import com.alvin.quantitative.trading.platform.TradingPlatformApplication;

import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * SpringBoot REST API控制器
 * Author: Alvin
 * 处理所有RESTful API请求
 */
@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class SpringBootApiController {
    private static final Logger logger = Logger.getLogger(SpringBootApiController.class.getName());
    
    private final ApplicationConfig config;
    
    // 注意：这里我们通过静态方式获取engine，因为它在main方法中创建
    // 在真实的SpringBoot应用中，应该使用依赖注入
    
    public SpringBootApiController() {
        this.config = ApplicationConfig.getInstance();
    }
    
    private TradingEngineInterface getEngine() {
        // 从TradingPlatformApplication获取engine实例
        return TradingPlatformApplication.getEngine();
    }
    
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        Map<String, Object> result = new HashMap<>();
        result.put("status", "healthy");
        result.put("service", "AI Trading Platform");
        result.put("version", "0.1");
        result.put("author", "Alvin");
        result.put("architecture", "SpringBoot + ProfessionalTradingEngine + Transformer AI");
        result.put("timestamp", LocalDateTime.now());
        result.put("ai_service_url", config.getAiServiceUrl());
        
        return ResponseEntity.ok(result);
    }
    
    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> status() {
        Map<String, Object> result = new HashMap<>();
        result.put("trading_engine", "running");
        result.put("web_server", "springboot");
        result.put("ai_service", checkAIServiceConnection() ? "connected" : "disconnected");
        result.put("monitored_symbols", config.getTradingSymbols().size());
        result.put("notification_email", config.isEmailNotificationEnabled());
        result.put("notification_wechat", config.isWechatNotificationEnabled());
        result.put("last_update", LocalDateTime.now());
        
        return ResponseEntity.ok(result);
    }
    
    @GetMapping("/portfolio")
    public ResponseEntity<Map<String, Object>> portfolio() {
        Map<String, Object> portfolio = new HashMap<>();
        
        try {
            TradingEngineInterface engine = getEngine();
            if (engine == null) {
                portfolio.put("error", "Trading engine not available");
                return ResponseEntity.ok(portfolio);
            }
            
            // 从交易引擎获取真实投资组合数据
            Map<String, com.alvin.quantitative.trading.platform.core.Position> currentPositions = engine.getCurrentPositions();
            java.util.List<String> symbolsList = config.getTradingSymbols();
            
            double totalValue = 0.0;
            double totalCost = 0.0;
            double dailyPnl = 0.0;
            
            // 构建持仓数组
            java.util.List<Map<String, Object>> positionsList = new ArrayList<>();
            
            for (String symbol : symbolsList) {
                try {
                    // 获取实时价格数据
                    com.alvin.quantitative.trading.platform.core.KlineData realTimeData = 
                        engine.getDataSource().getRealTimeData(symbol.trim());
                    double currentPrice = realTimeData.getClose();
                    
                    Map<String, Object> position = new HashMap<>();
                    position.put("symbol", symbol.trim());
                    
                    // 检查是否有实际持仓
                    com.alvin.quantitative.trading.platform.core.Position pos = currentPositions.get(symbol.trim());
                    if (pos != null) {
                        position.put("shares", pos.getShares());
                        position.put("avg_cost", pos.getAvgCost());
                        double value = pos.getShares() * currentPrice;
                        double pnl = value - (pos.getShares() * pos.getAvgCost());
                        position.put("value", value);
                        position.put("pnl", pnl);
                        totalValue += value;
                        totalCost += pos.getShares() * pos.getAvgCost();
                        dailyPnl += pnl;
                    } else {
                        // 如果没有持仓，显示当前价格信息
                        position.put("shares", 0);
                        position.put("avg_cost", 0.0);
                        position.put("value", 0.0);
                        position.put("pnl", 0.0);
                    }
                    
                    position.put("current_price", currentPrice);
                    positionsList.add(position);
                    
                } catch (Exception e) {
                    logger.warning("Failed to get real-time data for " + symbol + ": " + e.getMessage());
                    // 如果获取失败，添加错误信息
                    Map<String, Object> position = new HashMap<>();
                    position.put("symbol", symbol.trim());
                    position.put("shares", 0);
                    position.put("avg_cost", 0.0);
                    position.put("current_price", 0.0);
                    position.put("value", 0.0);
                    position.put("pnl", 0.0);
                    position.put("error", "Failed to fetch real-time data");
                    positionsList.add(position);
                }
            }
            
            // 计算总收益率
            double totalReturn = totalCost > 0 ? (totalValue - totalCost) / totalCost : 0.0;
            
            portfolio.put("total_value", totalValue);
            portfolio.put("total_return", totalReturn);
            portfolio.put("daily_pnl", dailyPnl);
            portfolio.put("positions_count", positionsList.size());
            portfolio.put("last_update", LocalDateTime.now());
            portfolio.put("positions", positionsList);
            portfolio.put("data_source", "Real-time from " + engine.getDataSource().getSourceName());
            
        } catch (Exception e) {
            logger.severe("Failed to get portfolio info: " + e.getMessage());
            portfolio.put("error", "Failed to retrieve portfolio data: " + e.getMessage());
        }
        
        return ResponseEntity.ok(portfolio);
    }
    
    @GetMapping("/indicators")
    public ResponseEntity<Map<String, Object>> indicators() {
        Map<String, Object> indicatorsResponse = new HashMap<>();
        
        try {
            TradingEngineInterface engine = getEngine();
            if (engine == null) {
                indicatorsResponse.put("error", "Trading engine not available");
                return ResponseEntity.ok(indicatorsResponse);
            }
            
            List<String> symbols = config.getTradingSymbols();
            Map<String, Object> allIndicators = new HashMap<>();
            
            for (String symbol : symbols) {
                String cleanSymbol = symbol.trim();
                Map<String, Double> indicators = engine.getRealTimeIndicators(cleanSymbol);
                
                if (!indicators.isEmpty()) {
                    // 获取最新价格数据
                    List<KlineData> recentData = engine.getRecentData(cleanSymbol, 1);
                    double currentPrice = 0.0;
                    if (!recentData.isEmpty()) {
                        currentPrice = recentData.get(recentData.size() - 1).getClose();
                    }
                    
                    Map<String, Object> symbolData = new HashMap<>();
                    symbolData.put("current_price", currentPrice);
                    symbolData.put("indicators", indicators);
                    symbolData.put("data_points", engine.getRecentData(cleanSymbol, 500).size());
                    symbolData.put("last_update", java.time.LocalDateTime.now());
                    
                    allIndicators.put(cleanSymbol, symbolData);
                } else {
                    Map<String, Object> symbolData = new HashMap<>();
                    symbolData.put("error", "Insufficient data for indicator calculation");
                    symbolData.put("message", "Need at least 20 data points");
                    allIndicators.put(cleanSymbol, symbolData);
                }
            }
            
            indicatorsResponse.put("indicators", allIndicators);
            indicatorsResponse.put("symbols_count", symbols.size());
            indicatorsResponse.put("data_source", "Real-time from " + engine.getDataSource().getSourceName());
            indicatorsResponse.put("timestamp", java.time.LocalDateTime.now());
            
        } catch (Exception e) {
            logger.severe("Failed to get real-time indicators: " + e.getMessage());
            indicatorsResponse.put("error", "Failed to retrieve indicators: " + e.getMessage());
        }
        
        return ResponseEntity.ok(indicatorsResponse);
    }
    
    @GetMapping("/trading-signals")
    public ResponseEntity<Map<String, Object>> tradingSignals() {
        Map<String, Object> tradingSignals = new HashMap<>();
        
        try {
            TradingEngineInterface engine = getEngine();
            if (engine == null) {
                tradingSignals.put("error", "Trading engine not available");
                return ResponseEntity.ok(tradingSignals);
            }
            
            List<String> symbols = config.getTradingSymbols();
            List<Map<String, Object>> signalsList = new ArrayList<>();
            
            for (String symbol : symbols) {
                try {
                    String cleanSymbol = symbol.trim();
                    
                    // 获取实时数据
                    KlineData realTimeData = engine.getDataSource().getRealTimeData(cleanSymbol);
                    
                    // 获取技术指标
                    Map<String, Double> indicators = engine.getRealTimeIndicators(cleanSymbol);
                    
                    // 获取历史数据
                    List<KlineData> history = engine.getRecentData(cleanSymbol, 100);
                    
                    if (realTimeData != null && !indicators.isEmpty() && !history.isEmpty()) {
                        Map<String, Object> tradingSignal = new HashMap<>();
                        tradingSignal.put("symbol", cleanSymbol);
                        tradingSignal.put("current_price", realTimeData.getClose());
                        tradingSignal.put("volume", realTimeData.getVolume());
                        tradingSignal.put("timestamp", realTimeData.getTimestamp());
                        
                        // 技术分析建议
                        String technicalRecommendation = analyzeTechnicalSignals(indicators, realTimeData.getClose());
                        tradingSignal.put("technical_analysis", technicalRecommendation);
                        
                        // 关键指标
                        tradingSignal.put("rsi", indicators.getOrDefault("RSI", 50.0));
                        tradingSignal.put("macd", indicators.getOrDefault("MACD", 0.0));
                        tradingSignal.put("ma5", indicators.getOrDefault("MA5", realTimeData.getClose()));
                        tradingSignal.put("ma20", indicators.getOrDefault("MA20", realTimeData.getClose()));
                        tradingSignal.put("volatility", indicators.getOrDefault("VOLATILITY", 0.02));
                        tradingSignal.put("volume_ratio", indicators.getOrDefault("VOLUME_RATIO", 1.0));
                        
                        // 风险评估
                        Map<String, Object> riskAssessment = assessTradingRisk(cleanSymbol, indicators, realTimeData);
                        tradingSignal.put("risk_assessment", riskAssessment);
                        

                        double suggestedPosition = calculateSuggestedPosition(indicators, realTimeData.getClose());
                        tradingSignal.put("suggested_position_million", suggestedPosition);
                        tradingSignal.put("suggested_position_percent", suggestedPosition / 10.0);
                        
                        signalsList.add(tradingSignal);
                    }
                    
                } catch (Exception e) {
                    logger.warning("Failed to generate trading signal for " + symbol + ": " + e.getMessage());
                }
            }
            
            tradingSignals.put("trading_signals", signalsList);
            tradingSignals.put("total_symbols", signalsList.size());
            tradingSignals.put("generated_at", java.time.LocalDateTime.now());
            tradingSignals.put("data_source", "Real-time analysis for manual trading");
            tradingSignals.put("disclaimer", "仅供参考，投资有风险，决策需谨慎");
            
        } catch (Exception e) {
            logger.severe("Failed to generate trading signals: " + e.getMessage());
            tradingSignals.put("error", "Failed to generate trading signals: " + e.getMessage());
        }
        
        return ResponseEntity.ok(tradingSignals);
    }
    
    @PostMapping("/backtest")
    public ResponseEntity<Map<String, Object>> backtest() {
        Map<String, Object> result = new HashMap<>();
        
        try {
            TradingEngineInterface engine = getEngine();
            if (engine == null) {
                result.put("error", "Trading engine not available");
                return ResponseEntity.ok(result);
            }
            
            Map<String, Object> backtestResult = engine.runBacktestAnalysis();
            result.putAll(backtestResult);
            result.put("data_source", "Real backtest from trading engine");
            result.put("generated_at", LocalDateTime.now());
            
        } catch (Exception e) {
            logger.warning("Failed to run backtest: " + e.getMessage());
            result.put("error", "Failed to run backtest: " + e.getMessage());
        }
        
        return ResponseEntity.ok(result);
    }
    
    @PostMapping("/test-notification")
    public ResponseEntity<Map<String, Object>> testNotification(@RequestParam(required = false) String type) {
        Map<String, Object> result = new HashMap<>();
        
        try {
            TradingEngineInterface engine = getEngine();
            if (engine == null) {
                result.put("error", "Trading engine not available");
                return ResponseEntity.ok(result);
            }
            
            Map<String, Boolean> testResults = engine.testNotificationConfig();
            boolean success = false;
            
            if ("email".equals(type)) {
                success = testResults.getOrDefault("email", false);
            } else if ("wechat".equals(type)) {
                success = testResults.getOrDefault("wechat", false);
            }
            
            result.put("success", success);
            result.put("type", type);
            result.put("message", success ? "通知发送成功" : "通知发送失败");
            
        } catch (Exception e) {
            logger.warning("Failed to test notification: " + e.getMessage());
            result.put("error", "Failed to test notification: " + e.getMessage());
        }
        
        return ResponseEntity.ok(result);
    }
    
    // 辅助方法
    private boolean checkAIServiceConnection() {
        try {
            // 这里可以添加实际的AI服务连接检查
            return true;
        } catch (Exception e) {
            return false;
        }
    }
    
    private String analyzeTechnicalSignals(Map<String, Double> indicators, double currentPrice) {
        List<String> signals = new ArrayList<>();
        
        double rsi = indicators.getOrDefault("RSI", 50.0);
        double macd = indicators.getOrDefault("MACD", 0.0);
        double ma5 = indicators.getOrDefault("MA5", currentPrice);
        double ma20 = indicators.getOrDefault("MA20", currentPrice);
        double volumeRatio = indicators.getOrDefault("VOLUME_RATIO", 1.0);
        
        // RSI分析
        if (rsi < 30) {
            signals.add("RSI超卖，可能反弹");
        } else if (rsi > 70) {
            signals.add("RSI超买，注意回调风险");
        }
        
        // MACD分析
        if (macd > 0) {
            signals.add("MACD金叉，上升趋势");
        } else {
            signals.add("MACD死叉，下降趋势");
        }
        
        // 均线分析
        if (currentPrice > ma5 && ma5 > ma20) {
            signals.add("多头排列，强势上涨");
        } else if (currentPrice < ma5 && ma5 < ma20) {
            signals.add("空头排列，弱势下跌");
        }
        
        // 成交量分析
        if (volumeRatio > 2.0) {
            signals.add("放量突破，关注度高");
        } else if (volumeRatio < 0.5) {
            signals.add("缩量整理，观望为主");
        }
        
        return String.join("; ", signals);
    }
    
    private Map<String, Object> assessTradingRisk(String symbol, Map<String, Double> indicators, KlineData currentData) {
        Map<String, Object> risk = new HashMap<>();
        
        double volatility = indicators.getOrDefault("VOLATILITY", 0.02);
        double rsi = indicators.getOrDefault("RSI", 50.0);
        double volumeRatio = indicators.getOrDefault("VOLUME_RATIO", 1.0);
        
        // 风险等级评估
        String riskLevel;
        if (volatility > 0.04 || Math.abs(rsi - 50) > 30) {
            riskLevel = "高风险";
        } else if (volatility > 0.02 || Math.abs(rsi - 50) > 20) {
            riskLevel = "中等风险";
        } else {
            riskLevel = "低风险";
        }
        
        risk.put("risk_level", riskLevel);
        risk.put("volatility_percent", volatility * 100);
        risk.put("liquidity_score", Math.min(1.0, volumeRatio / 2.0));
        
        // 止损建议
        double suggestedStopLoss = currentData.getClose() * (1 - Math.max(0.03, volatility * 2));
        risk.put("suggested_stop_loss", suggestedStopLoss);
        
        // 止盈建议  
        double suggestedTakeProfit = currentData.getClose() * (1 + Math.max(0.05, volatility * 3));
        risk.put("suggested_take_profit", suggestedTakeProfit);
        
        return risk;
    }
    
    private double calculateSuggestedPosition(Map<String, Double> indicators, double currentPrice) {
        double volatility = indicators.getOrDefault("VOLATILITY", 0.02);
        double rsi = indicators.getOrDefault("RSI", 50.0);
        double volumeRatio = indicators.getOrDefault("VOLUME_RATIO", 1.0);
        
        // 基础仓位：50%
        double basePosition = 5.0;
        
        // 基于波动率调整
        double volatilityAdjustment = Math.min(1.0, 0.02 / volatility);
        
        // 基于RSI调整
        double rsiAdjustment = 1.0;
        if (rsi < 30) {
            rsiAdjustment = 1.2; // RSI超卖，增加仓位
        } else if (rsi > 70) {
            rsiAdjustment = 0.8; // RSI超买，减少仓位
        }
        
        // 基于流动性调整
        double liquidityAdjustment = Math.min(1.0, volumeRatio / 1.5);
        
        double finalPosition = basePosition * volatilityAdjustment * rsiAdjustment * liquidityAdjustment;
        
        // 限制：单股票最大(100%)，最小 (5%)
        return Math.max(0.5, Math.min(10.0, finalPosition));
    }
}
