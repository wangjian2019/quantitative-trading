package com.alvin.quantitative.trading.platform.strategy;

import com.alvin.quantitative.trading.platform.config.ApplicationConfig;
import com.alvin.quantitative.trading.platform.core.AISignal;
import com.alvin.quantitative.trading.platform.core.KlineData;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.logging.Logger;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.apache.http.HttpEntity;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * AI Strategy Client
 * Author: Alvin
 * Communicates with Python AI service for strategy signals
 */
public class AIStrategyClient {
    private static final Logger logger = Logger.getLogger(AIStrategyClient.class.getName());
    
    private final CloseableHttpClient httpClient;
    private final String apiUrl;
    private final ObjectMapper objectMapper;
    private final ApplicationConfig config;
    private int retryCount = 0;
    
    public AIStrategyClient(String apiUrl) {
        this.config = ApplicationConfig.getInstance();
        
        RequestConfig requestConfig = RequestConfig.custom()
                .setConnectTimeout(config.getAiServiceConnectTimeout())
                .setSocketTimeout(config.getAiServiceSocketTimeout())
                .build();
        
        this.httpClient = HttpClients.custom()
                .setDefaultRequestConfig(requestConfig)
                .build();
        this.apiUrl = apiUrl;
        this.objectMapper = new ObjectMapper();
        this.objectMapper.registerModule(new JavaTimeModule());
    }
    
    public AISignal getSignal(String symbol, KlineData currentData, 
                             Map<String, Double> indicators, List<KlineData> history) {
        return getSignalWithRetry(symbol, currentData, indicators, history, 0);
    }
    
    private AISignal getSignalWithRetry(String symbol, KlineData currentData, 
                                       Map<String, Double> indicators, List<KlineData> history, 
                                       int attemptCount) {
        try {
            Map<String, Object> request = new HashMap<>();
            request.put("symbol", symbol);
            request.put("current_data", currentData);
            request.put("indicators", indicators);
            request.put("history", history.subList(Math.max(0, history.size() - 100), history.size()));
            
            String jsonRequest = objectMapper.writeValueAsString(request);
            
            HttpPost httpPost = new HttpPost(apiUrl + "/get_signal");
            httpPost.setHeader("Content-Type", "application/json");
            httpPost.setEntity(new StringEntity(jsonRequest, "UTF-8"));
            
            try (CloseableHttpResponse response = httpClient.execute(httpPost)) {
                int statusCode = response.getStatusLine().getStatusCode();
                HttpEntity entity = response.getEntity();
                String responseBody = EntityUtils.toString(entity);
                
                if (statusCode == 200) {
                    retryCount = 0; // Reset retry count on success
                    return objectMapper.readValue(responseBody, AISignal.class);
                } else {
                    System.err.println("AI service error: " + statusCode + " (attempt " + (attemptCount + 1) + ")");
                    return handleRetryOrFallback(symbol, currentData, indicators, history, attemptCount, 
                        "AI service error: " + statusCode);
                }
            }
            
        } catch (IOException e) {
            System.err.println("Failed to call AI service: " + e.getMessage() + " (attempt " + (attemptCount + 1) + ")");
            return handleRetryOrFallback(symbol, currentData, indicators, history, attemptCount, 
                "Network error: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("Unexpected error: " + e.getMessage() + " (attempt " + (attemptCount + 1) + ")");
            return handleRetryOrFallback(symbol, currentData, indicators, history, attemptCount, 
                "Unexpected error: " + e.getMessage());
        }
    }
    
    private AISignal handleRetryOrFallback(String symbol, KlineData currentData, 
                                          Map<String, Double> indicators, List<KlineData> history, 
                                          int attemptCount, String errorMessage) {
        if (attemptCount < config.getAiServiceMaxRetry() - 1) {
            // Wait before retry
            try {
                Thread.sleep(1000 * (attemptCount + 1)); // Exponential backoff
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
            return getSignalWithRetry(symbol, currentData, indicators, history, attemptCount + 1);
        } else {
            System.err.println("Max retries reached for " + symbol + ", using fallback strategy");
            return createFallbackSignal(errorMessage);
        }
    }
    
    private AISignal createFallbackSignal(String reason) {
        AISignal signal = new AISignal();
        signal.setAction("HOLD");
        signal.setConfidence(0.0);
        signal.setReason("AI服务故障，使用保守策略: " + reason);
        
        // 生产环境关键：记录错误并发送警报
        logger.severe("🚨 PRODUCTION ALERT: AI service failure - " + reason);
        
        return signal;
    }
    
    /**
     * 生产环境增强的回退策略
     */
    private AISignal createEnhancedFallbackSignal(String symbol, KlineData currentData, 
                                                 Map<String, Double> indicators, String errorMessage) {
        try {
            // 基于技术指标的智能回退策略
            double rsi = indicators.getOrDefault("RSI", 50.0);
            double macd = indicators.getOrDefault("MACD", 0.0);
            double ma5 = indicators.getOrDefault("MA5", currentData.getClose());
            double ma20 = indicators.getOrDefault("MA20", currentData.getClose());
            double volumeRatio = indicators.getOrDefault("VOLUME_RATIO", 1.0);
            double currentPrice = currentData.getClose();
            
            AISignal signal = new AISignal();
            
            // 强买入条件
            if (rsi < 20 && macd > 0 && currentPrice > ma5 && volumeRatio > 2.0) {
                signal.setAction("BUY");
                signal.setConfidence(0.75);
                signal.setReason("🚨 AI故障回退策略: RSI极度超卖(" + String.format("%.1f", rsi) + ") + MACD金叉 + 放量突破");
            }
            // 强卖出条件
            else if (rsi > 80 && macd < 0 && currentPrice < ma20 && volumeRatio > 2.0) {
                signal.setAction("SELL");
                signal.setConfidence(0.75);
                signal.setReason("🚨 AI故障回退策略: RSI极度超买(" + String.format("%.1f", rsi) + ") + MACD死叉 + 放量下跌");
            }
            // 保守持有
            else {
                signal.setAction("HOLD");
                signal.setConfidence(0.5);
                signal.setReason("🚨 AI故障回退策略: 技术指标无明确信号，保守持有");
            }
            
            logger.warning("Using enhanced fallback strategy for " + symbol + ": " + signal.getAction() + 
                          " (confidence: " + signal.getConfidence() + ")");
            
            return signal;
            
        } catch (Exception e) {
            logger.severe("Even fallback strategy failed: " + e.getMessage());
            AISignal emergencySignal = new AISignal();
            emergencySignal.setAction("HOLD");
            emergencySignal.setConfidence(0.0);
            emergencySignal.setReason("🚨 紧急模式: 所有策略失败，强制持有");
            return emergencySignal;
        }
    }
}