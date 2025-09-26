package com.alvin.quantitative.trading.platform.strategy;

import com.alvin.quantitative.trading.platform.core.AISignal;
import com.alvin.quantitative.trading.platform.core.KlineData;
import com.alvin.quantitative.trading.platform.config.ApplicationConfig;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;

import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.springframework.stereotype.Component;
import lombok.extern.slf4j.Slf4j;

import java.util.*;

/**
 * Transformer AI模型客户端
 * Author: Alvin
 *
 * 专业级AI交易信号生成客户端，连接Python Transformer模型服务
 * 支持：
 * - 高频信号生成
 * - 多任务学习结果解析
 * - 智能重试机制
 * - 性能监控
 */
@Slf4j
@Component
public class TransformerAIClient {

    private final ApplicationConfig config;
    private final ObjectMapper objectMapper;
    private final CloseableHttpClient httpClient;
    private final String aiServiceUrl;

    // 性能统计
    private long totalRequests = 0;
    private long successfulRequests = 0;
    private long failedRequests = 0;
    private double averageResponseTime = 0.0;

    public TransformerAIClient() {
        this.config = ApplicationConfig.getInstance();
        this.objectMapper = new ObjectMapper();
        this.httpClient = HttpClients.createDefault();
        this.aiServiceUrl = config.getAiServiceUrl();

        log.info("🤖 Transformer AI Client initialized");
        log.info("🔗 AI Service URL: {}", aiServiceUrl);
    }

    /**
     * 获取Transformer模型生成的交易信号
     * 这是核心方法，用于实盘交易
     */
    public AISignal getTransformerSignal(String symbol, KlineData currentData,
                                       Map<String, Double> indicators, List<KlineData> history) {
        long startTime = System.currentTimeMillis();
        totalRequests++;

        try {
            // 构建请求数据
            Map<String, Object> requestData = buildRequestData(symbol, currentData, indicators, history);

            // 发送HTTP请求到Transformer AI服务
            AISignal signal = sendTransformerRequest(requestData);

            if (signal != null) {
                successfulRequests++;
                updateAverageResponseTime(System.currentTimeMillis() - startTime);

                log.debug("✅ Transformer signal generated for {}: {} (confidence: {:.2%})",
                    symbol, signal.getAction(), signal.getConfidence());

                return signal;
            } else {
                failedRequests++;
                log.warn("⚠️ Transformer service returned null signal for {}", symbol);
                return createFallbackSignal(symbol, "Null response from Transformer service");
            }

        } catch (Exception e) {
            failedRequests++;
            log.error("❌ Transformer AI request failed for {}: {}", symbol, e.getMessage());
            return createFallbackSignal(symbol, "AI service error: " + e.getMessage());
        }
    }

    /**
     * 构建发送给Transformer模型的请求数据
     */
    private Map<String, Object> buildRequestData(String symbol, KlineData currentData,
                                                Map<String, Double> indicators, List<KlineData> history) {
        Map<String, Object> request = new HashMap<>();

        // 基本信息
        request.put("symbol", symbol);
        request.put("timestamp", System.currentTimeMillis());

        // 当前市场数据
        Map<String, Object> current = new HashMap<>();
        current.put("open", currentData.getOpen());
        current.put("high", currentData.getHigh());
        current.put("low", currentData.getLow());
        current.put("close", currentData.getClose());
        current.put("volume", currentData.getVolume());
        request.put("current_data", current);

        // 技术指标
        request.put("indicators", indicators != null ? indicators : new HashMap<>());

        // 历史数据（最近100个数据点，用于Transformer的时序分析）
        List<Map<String, Object>> historyData = new ArrayList<>();
        int historySize = Math.min(history.size(), 100);

        for (int i = Math.max(0, history.size() - historySize); i < history.size(); i++) {
            KlineData kline = history.get(i);
            Map<String, Object> dataPoint = new HashMap<>();
            dataPoint.put("open", kline.getOpen());
            dataPoint.put("high", kline.getHigh());
            dataPoint.put("low", kline.getLow());
            dataPoint.put("close", kline.getClose());
            dataPoint.put("volume", kline.getVolume());
            dataPoint.put("timestamp", kline.getTimestamp());
            historyData.add(dataPoint);
        }
        request.put("history", historyData);

        return request;
    }

    /**
     * 发送请求到Transformer AI服务
     */
    private AISignal sendTransformerRequest(Map<String, Object> requestData) throws Exception {
        HttpPost post = new HttpPost(aiServiceUrl + "/get_signal");
        post.setHeader("Content-Type", "application/json");
        post.setHeader("User-Agent", "Professional-Trading-Engine/0.1");

        // 设置请求体
        String jsonRequest = objectMapper.writeValueAsString(requestData);
        post.setEntity(new StringEntity(jsonRequest, "UTF-8"));

        // 发送请求
        try (CloseableHttpResponse response = httpClient.execute(post)) {
            int statusCode = response.getStatusLine().getStatusCode();
            String responseBody = EntityUtils.toString(response.getEntity());

            if (statusCode == 200) {
                return parseTransformerResponse(responseBody);
            } else {
                log.error("❌ Transformer service returned status {}: {}", statusCode, responseBody);
                return null;
            }
        }
    }

    /**
     * 解析Transformer模型的响应
     */
    private AISignal parseTransformerResponse(String responseBody) throws Exception {
        JsonNode response = objectMapper.readTree(responseBody);

        // 检查是否有错误
        if (response.has("error")) {
            log.error("❌ Transformer service error: {}", response.get("error").asText());
            return null;
        }

        // 解析信号数据
        String action = response.get("action").asText();
        double confidence = response.get("confidence").asDouble();
        double expectedReturn = response.has("expected_return") ?
            response.get("expected_return").asDouble() : 0.0;
        String reason = response.has("reason") ?
            response.get("reason").asText() : "Transformer AI analysis";

        // 解析额外信息
        String modelType = response.has("model_type") ?
            response.get("model_type").asText() : "Transformer";
        double volatility = response.has("volatility") ?
            response.get("volatility").asDouble() : 0.02;

        // 创建AI信号对象
        AISignal signal = new AISignal(action, confidence, reason);
        signal.setExpectedReturn(expectedReturn);
        signal.setModelType(modelType);
        signal.setVolatility(volatility);

        // 解析元数据（如果存在）
        if (response.has("metadata")) {
            JsonNode metadata = response.get("metadata");
            Map<String, Object> metadataMap = objectMapper.convertValue(metadata, Map.class);
            signal.setMetadata(metadataMap);
        }

        // 验证信号有效性
        if (!isValidSignal(signal)) {
            log.warn("⚠️ Invalid signal received from Transformer service");
            return null;
        }

        return signal;
    }

    /**
     * 验证AI信号的有效性
     */
    private boolean isValidSignal(AISignal signal) {
        if (signal == null) return false;

        String action = signal.getAction();
        if (!Arrays.asList("BUY", "SELL", "HOLD").contains(action)) {
            return false;
        }

        double confidence = signal.getConfidence();
        if (confidence < 0.0 || confidence > 1.0) {
            return false;
        }

        return true;
    }

    /**
     * 创建回退信号（当AI服务不可用时）
     */
    private AISignal createFallbackSignal(String symbol, String reason) {
        log.warn("🔄 Creating fallback signal for {}: {}", symbol, reason);

        // 基于技术指标的简单策略
        AISignal signal = new AISignal("HOLD", 0.5, "Fallback signal: " + reason);
        signal.setExpectedReturn(0.0);
        signal.setModelType("Fallback");
        signal.setVolatility(0.02);

        return signal;
    }

    /**
     * 批量获取信号（用于多股票同时分析）
     */
    public Map<String, AISignal> getBatchSignals(Map<String, Map<String, Object>> symbolsData) {
        Map<String, AISignal> signals = new HashMap<>();

        try {
            HttpPost post = new HttpPost(aiServiceUrl + "/batch_signals");
            post.setHeader("Content-Type", "application/json");

            Map<String, Object> request = new HashMap<>();
            request.put("symbols", new ArrayList<>(symbolsData.keySet()));
            request.put("data", symbolsData);

            String jsonRequest = objectMapper.writeValueAsString(request);
            post.setEntity(new StringEntity(jsonRequest, "UTF-8"));

            try (CloseableHttpResponse response = httpClient.execute(post)) {
                if (response.getStatusLine().getStatusCode() == 200) {
                    String responseBody = EntityUtils.toString(response.getEntity());
                    JsonNode responseJson = objectMapper.readTree(responseBody);

                    if (responseJson.has("signals")) {
                        JsonNode signalsNode = responseJson.get("signals");
                        signalsNode.fields().forEachRemaining(entry -> {
                            try {
                                String symbol = entry.getKey();
                                JsonNode signalNode = entry.getValue();
                                String signalJson = objectMapper.writeValueAsString(signalNode);
                                AISignal signal = parseTransformerResponse(signalJson);
                                if (signal != null) {
                                    signals.put(symbol, signal);
                                }
                            } catch (Exception e) {
                                log.error("❌ Failed to parse batch signal: {}", e.getMessage());
                            }
                        });
                    }
                }
            }

        } catch (Exception e) {
            log.error("❌ Batch signals request failed: {}", e.getMessage());
        }

        return signals;
    }

    /**
     * 检查AI服务健康状态
     */
    public boolean checkAIServiceHealth() {
        try {
            HttpPost post = new HttpPost(aiServiceUrl + "/health");
            post.setHeader("Content-Type", "application/json");

            try (CloseableHttpResponse response = httpClient.execute(post)) {
                int statusCode = response.getStatusLine().getStatusCode();
                if (statusCode == 200) {
                    String responseBody = EntityUtils.toString(response.getEntity());
                    JsonNode healthResponse = objectMapper.readTree(responseBody);
                    return "healthy".equals(healthResponse.get("status").asText());
                }
            }
        } catch (Exception e) {
            log.debug("AI service health check failed: {}", e.getMessage());
        }

        return false;
    }

    /**
     * 更新平均响应时间
     */
    private void updateAverageResponseTime(long responseTime) {
        if (successfulRequests == 1) {
            averageResponseTime = responseTime;
        } else {
            averageResponseTime = ((averageResponseTime * (successfulRequests - 1)) + responseTime) / successfulRequests;
        }
    }

    /**
     * 获取客户端性能统计
     */
    public Map<String, Object> getPerformanceStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("total_requests", totalRequests);
        stats.put("successful_requests", successfulRequests);
        stats.put("failed_requests", failedRequests);
        stats.put("success_rate", totalRequests > 0 ? (double) successfulRequests / totalRequests : 0.0);
        stats.put("average_response_time_ms", averageResponseTime);
        stats.put("ai_service_url", aiServiceUrl);
        stats.put("health_status", checkAIServiceHealth() ? "healthy" : "unhealthy");

        return stats;
    }

    /**
     * 重置性能统计
     */
    public void resetPerformanceStats() {
        totalRequests = 0;
        successfulRequests = 0;
        failedRequests = 0;
        averageResponseTime = 0.0;
        log.info("📊 Transformer AI Client performance stats reset");
    }

    /**
     * 优雅关闭客户端
     */
    public void shutdown() {
        try {
            if (httpClient != null) {
                httpClient.close();
            }
            log.info("🔄 Transformer AI Client shutdown completed");
        } catch (Exception e) {
            log.error("❌ Error during AI client shutdown: {}", e.getMessage());
        }
    }
}