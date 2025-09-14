package com.alvin.quantitative.trading.platform.data.impl;

import com.alvin.quantitative.trading.platform.core.KlineData;
import com.alvin.quantitative.trading.platform.data.DataSource;
import com.alvin.quantitative.trading.platform.data.DataSourceConfig;
import com.alvin.quantitative.trading.platform.data.DataSourceException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;

import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * Yahoo Finance Data Source Implementation - Real API Integration
 * Author: Alvin
 * Real Yahoo Finance API integration for live market data
 */
public class YahooFinanceDataSource implements DataSource {
    
    private static final Logger logger = Logger.getLogger(YahooFinanceDataSource.class.getName());
    
    private boolean initialized = false;
    private CloseableHttpClient httpClient;
    private ObjectMapper objectMapper;
    private DataSourceConfig config;
    
    // Yahoo Finance API endpoints
    private static final String YAHOO_CHART_API = "https://query1.finance.yahoo.com/v8/finance/chart";
    private static final String USER_AGENT = "Mozilla/5.0 (AI Trading Platform v0.1)";
    
    @Override
    public void initialize(DataSourceConfig config) throws DataSourceException {
        try {
            this.config = config;
            this.httpClient = HttpClients.createDefault();
            this.objectMapper = new ObjectMapper();
            this.objectMapper.findAndRegisterModules();
            this.initialized = true;
            
            logger.info("Yahoo Finance Data Source initialized successfully");
        } catch (Exception e) {
            throw new DataSourceException("Failed to initialize Yahoo Finance data source: " + e.getMessage());
        }
    }
    
    @Override
    public KlineData getRealTimeData(String symbol) throws DataSourceException {
        if (!initialized) {
            throw new DataSourceException("Data source not initialized");
        }
        
        try {
            // 1. 构建Yahoo Finance API URL - 生产环境优化
            String url = String.format("%s/%s?interval=1d&range=1d&includePrePost=true", YAHOO_CHART_API, symbol);
            
            // 2. 发送HTTP请求
            HttpGet request = new HttpGet(url);
            request.setHeader("User-Agent", USER_AGENT);
            request.setHeader("Accept", "application/json");
            
            try (CloseableHttpResponse response = httpClient.execute(request)) {
                if (response.getStatusLine().getStatusCode() != 200) {
                    throw new DataSourceException("Yahoo Finance API returned status: " + 
                        response.getStatusLine().getStatusCode());
                }
                
                String responseBody = EntityUtils.toString(response.getEntity());
                
                // 3. 解析JSON响应
                JsonNode rootNode = objectMapper.readTree(responseBody);
                JsonNode chartNode = rootNode.path("chart");
                
                if (chartNode.path("error").size() > 0) {
                    String error = chartNode.path("error").get(0).path("description").asText();
                    throw new DataSourceException("Yahoo Finance API error: " + error);
                }
                
                JsonNode resultNode = chartNode.path("result");
                if (resultNode.size() == 0) {
                    throw new DataSourceException("No data returned for symbol: " + symbol);
                }
                
                JsonNode dataNode = resultNode.get(0);
                JsonNode meta = dataNode.path("meta");
                
                // 4. 提取OHLCV数据
                double previousClose = meta.path("previousClose").asDouble(0.0);
                double regularMarketPrice = meta.path("regularMarketPrice").asDouble(previousClose);
                double dayHigh = meta.path("regularMarketDayHigh").asDouble(regularMarketPrice);
                double dayLow = meta.path("regularMarketDayLow").asDouble(regularMarketPrice);
                long volume = meta.path("regularMarketVolume").asLong(0L);
                long marketTime = meta.path("regularMarketTime").asLong();
                
                // 5. 创建时间戳
                LocalDateTime dateTime;
                if (marketTime > 0) {
                    dateTime = LocalDateTime.ofEpochSecond(marketTime, 0, ZoneOffset.UTC);
                } else {
                    dateTime = LocalDateTime.now();
                }
                
                // 6. 返回KlineData对象
                KlineData klineData = new KlineData(
                    dateTime, 
                    previousClose,      // open
                    dayHigh,           // high
                    dayLow,            // low
                    regularMarketPrice, // close
                    volume             // volume
                );
                
                logger.fine(String.format("Retrieved real-time data for %s: close=%.2f, volume=%d", 
                    symbol, regularMarketPrice, volume));
                
                return klineData;
                
            }
        } catch (DataSourceException e) {
            throw e;
        } catch (Exception e) {
            logger.warning("Failed to get real-time data for " + symbol + ": " + e.getMessage());
            throw new DataSourceException("获取实时数据失败: " + e.getMessage());
        }
    }
    
    @Override
    public List<KlineData> getHistoricalData(String symbol, int days) throws DataSourceException {
        if (!initialized) {
            throw new DataSourceException("Data source not initialized");
        }
        
        try {
            // 1. 计算时间范围
            long endTime = System.currentTimeMillis() / 1000;
            long startTime = endTime - (days * 24 * 60 * 60);
            
            // 2. 构建API URL
            String url = String.format("%s/%s?period1=%d&period2=%d&interval=1d",
                YAHOO_CHART_API, symbol, startTime, endTime);
            
            // 3. 发送请求获取历史数据
            HttpGet request = new HttpGet(url);
            request.setHeader("User-Agent", USER_AGENT);
            request.setHeader("Accept", "application/json");
            
            try (CloseableHttpResponse response = httpClient.execute(request)) {
                if (response.getStatusLine().getStatusCode() != 200) {
                    throw new DataSourceException("Yahoo Finance API returned status: " + 
                        response.getStatusLine().getStatusCode());
                }
                
                String responseBody = EntityUtils.toString(response.getEntity());
                
                // 4. 解析JSON数据
                JsonNode rootNode = objectMapper.readTree(responseBody);
                JsonNode chartNode = rootNode.path("chart");
                
                if (chartNode.path("error").size() > 0) {
                    String error = chartNode.path("error").get(0).path("description").asText();
                    throw new DataSourceException("Yahoo Finance API error: " + error);
                }
                
                JsonNode resultNode = chartNode.path("result");
                if (resultNode.size() == 0) {
                    return new ArrayList<>();
                }
                
                JsonNode dataNode = resultNode.get(0);
                
                // 5. 提取时间序列数据
                JsonNode timestamps = dataNode.path("timestamp");
                JsonNode indicators = dataNode.path("indicators").path("quote");
                
                if (indicators.size() == 0) {
                    return new ArrayList<>();
                }
                
                JsonNode quote = indicators.get(0);
                JsonNode opens = quote.path("open");
                JsonNode highs = quote.path("high");
                JsonNode lows = quote.path("low");
                JsonNode closes = quote.path("close");
                JsonNode volumes = quote.path("volume");
                
                // 6. 构建KlineData列表
                List<KlineData> historicalData = new ArrayList<>();
                int dataSize = Math.min(timestamps.size(), Math.min(opens.size(), closes.size()));
                
                for (int i = 0; i < dataSize; i++) {
                    // 检查数据有效性
                    if (opens.get(i).isNull() || closes.get(i).isNull()) {
                        continue;
                    }
                    
                    long timestamp = timestamps.get(i).asLong();
                    LocalDateTime dateTime = LocalDateTime.ofEpochSecond(timestamp, 0, ZoneOffset.UTC);
                    
                    double open = opens.get(i).asDouble();
                    double high = highs.get(i).asDouble(open);
                    double low = lows.get(i).asDouble(open);
                    double close = closes.get(i).asDouble();
                    long volume = volumes.get(i).asLong(0L);
                    
                    historicalData.add(new KlineData(dateTime, open, high, low, close, volume));
                }
                
                logger.info(String.format("Retrieved %d days of historical data for %s (%d records)", 
                    days, symbol, historicalData.size()));
                
                return historicalData;
            }
            
        } catch (DataSourceException e) {
            throw e;
        } catch (Exception e) {
            logger.warning("Failed to get historical data for " + symbol + ": " + e.getMessage());
            throw new DataSourceException("获取历史数据失败: " + e.getMessage());
        }
    }
    
    @Override
    public boolean isAvailable() {
        try {
            // 测试连接性 - 简单HTTP测试
            String testUrl = YAHOO_CHART_API + "/SPY?interval=1d&range=1d";
            HttpGet testRequest = new HttpGet(testUrl);
            testRequest.setHeader("User-Agent", USER_AGENT);
            
            try (CloseableHttpResponse response = httpClient.execute(testRequest)) {
                int statusCode = response.getStatusLine().getStatusCode();
                boolean available = statusCode == 200;
                
                if (!available) {
                    logger.warning("Yahoo Finance API test failed with status: " + statusCode);
                }
                
                return available;
            }
        } catch (Exception e) {
            logger.severe("🚨 PRODUCTION CRITICAL: Yahoo Finance data source unavailable: " + e.getMessage());
            return false;
        }
    }
    
    @Override
    public String getSourceName() {
        return "Yahoo Finance (Live)";
    }
    
    @Override
    public String getRateLimitInfo() {
        return "Free with reasonable use - No official rate limit";
    }
    
    @Override
    public void cleanup() {
        try {
            if (httpClient != null) {
                httpClient.close();
            }
            initialized = false;
            logger.info("Yahoo Finance data source cleaned up");
        } catch (Exception e) {
            logger.warning("Error during cleanup: " + e.getMessage());
        }
    }
}