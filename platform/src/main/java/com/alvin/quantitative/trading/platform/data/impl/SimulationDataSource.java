package com.alvin.quantitative.trading.platform.data.impl;

import com.alvin.quantitative.trading.platform.core.KlineData;
import com.alvin.quantitative.trading.platform.data.DataSource;
import com.alvin.quantitative.trading.platform.data.DataSourceConfig;
import com.alvin.quantitative.trading.platform.data.DataSourceException;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Simulation Data Source Implementation
 * Author: Alvin
 * Generates realistic simulated stock data for testing
 */
public class SimulationDataSource implements DataSource {
    
    private final Random random = new Random();
    private final Map<String, Double> lastPrices = new HashMap<String, Double>();
    private boolean initialized = false;
    
    // Base prices for common stocks
    private final Map<String, Double> basePrices = new HashMap<String, Double>() {{
        put("AAPL", 175.0);
        put("TSLA", 250.0);
        put("MSFT", 350.0);
        put("GOOGL", 140.0);
        put("AMZN", 145.0);
        put("NVDA", 450.0);
        put("SPY", 450.0);
        put("QQQ", 380.0);
        put("VTI", 220.0);
    }};
    
    @Override
    public void initialize(DataSourceConfig config) throws DataSourceException {
        this.initialized = true;
    }
    
    @Override
    public KlineData getRealTimeData(String symbol) throws DataSourceException {
        if (!initialized) {
            throw new DataSourceException("Data source not initialized");
        }
        
        double basePrice = basePrices.getOrDefault(symbol, 100.0);
        double lastPrice = lastPrices.getOrDefault(symbol, basePrice);
        
        // Generate realistic price movement
        double volatility = 0.02; // 2% volatility
        double drift = 0.0001; // Small upward drift
        
        // Random walk with drift
        double change = random.nextGaussian() * volatility + drift;
        double newPrice = lastPrice * (1 + change);
        
        // Ensure price doesn't go below $1
        newPrice = Math.max(newPrice, 1.0);
        
        // Generate OHLC data
        double open = lastPrice;
        double high = Math.max(open, newPrice) * (1 + Math.abs(random.nextGaussian()) * 0.005);
        double low = Math.min(open, newPrice) * (1 - Math.abs(random.nextGaussian()) * 0.005);
        double close = newPrice;
        
        // Generate realistic volume
        long baseVolume = 5000000L; // 5M base volume
        long volume = (long) (baseVolume * (0.5 + random.nextDouble() * 1.5));
        
        // Update tracking
        lastPrices.put(symbol, close);
        
        return new KlineData(LocalDateTime.now(), open, high, low, close, volume);
    }
    
    @Override
    public List<KlineData> getHistoricalData(String symbol, int days) throws DataSourceException {
        if (!initialized) {
            throw new DataSourceException("Data source not initialized");
        }
        
        List<KlineData> historicalData = new ArrayList<KlineData>();
        
        double basePrice = basePrices.getOrDefault(symbol, 100.0);
        double currentPrice = basePrice;
        
        LocalDateTime startDate = LocalDateTime.now().minusDays(days);
        
        for (int i = 0; i < days; i++) {
            LocalDateTime date = startDate.plusDays(i);
            
            // Generate daily price movement
            double volatility = 0.02; // 2% daily volatility
            double drift = 0.0005; // Small upward trend
            
            double change = random.nextGaussian() * volatility + drift;
            double newPrice = currentPrice * (1 + change);
            newPrice = Math.max(newPrice, 1.0);
            
            // Generate OHLC
            double open = currentPrice;
            double close = newPrice;
            
            // High and low with some randomness
            double high = Math.max(open, close) * (1 + Math.abs(random.nextGaussian()) * 0.01);
            double low = Math.min(open, close) * (1 - Math.abs(random.nextGaussian()) * 0.01);
            
            // Volume varies by day
            long baseVolume = 3000000L + random.nextInt(10000000); // 3M to 13M
            
            historicalData.add(new KlineData(date, open, high, low, close, baseVolume));
            
            currentPrice = close;
        }
        
        return historicalData;
    }
    
    @Override
    public boolean isAvailable() {
        return true; // Simulation is always available
    }
    
    @Override
    public String getSourceName() {
        return "Simulation Data Source";
    }
    
    @Override
    public String getRateLimitInfo() {
        return "No limits - simulated data";
    }
    
    @Override
    public void cleanup() {
        lastPrices.clear();
        initialized = false;
    }
}