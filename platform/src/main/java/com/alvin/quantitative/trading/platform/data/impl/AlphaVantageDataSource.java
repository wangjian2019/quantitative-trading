package com.alvin.quantitative.trading.platform.data.impl;

import com.alvin.quantitative.trading.platform.core.KlineData;
import com.alvin.quantitative.trading.platform.data.DataSource;
import com.alvin.quantitative.trading.platform.data.DataSourceConfig;
import com.alvin.quantitative.trading.platform.data.DataSourceException;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Alpha Vantage Data Source Implementation
 * Author: Alvin
 * Simplified implementation for demo purposes
 */
public class AlphaVantageDataSource implements DataSource {
    
    private boolean initialized = false;
    
    @Override
    public void initialize(DataSourceConfig config) throws DataSourceException {
        this.initialized = true;
    }
    
    @Override
    public KlineData getRealTimeData(String symbol) throws DataSourceException {
        if (!initialized) {
            throw new DataSourceException("Data source not initialized");
        }
        
        // Simplified implementation - returns simulated data
        return new KlineData(LocalDateTime.now(), 100.0, 102.0, 98.0, 101.0, 1000000L);
    }
    
    @Override
    public List<KlineData> getHistoricalData(String symbol, int days) throws DataSourceException {
        if (!initialized) {
            throw new DataSourceException("Data source not initialized");
        }
        
        List<KlineData> data = new ArrayList<KlineData>();
        // Simplified implementation - returns empty list for now
        return data;
    }
    
    @Override
    public boolean isAvailable() {
        return false; // Disabled for demo
    }
    
    @Override
    public String getSourceName() {
        return "Alpha Vantage (Demo)";
    }
    
    @Override
    public String getRateLimitInfo() {
        return "500 calls/day, 5 calls/minute";
    }
    
    @Override
    public void cleanup() {
        initialized = false;
    }
}