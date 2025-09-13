package com.alvin.quantitative.trading.platform.data.impl;

import com.alvin.quantitative.trading.platform.core.KlineData;
import com.alvin.quantitative.trading.platform.data.DataSource;
import com.alvin.quantitative.trading.platform.data.DataSourceConfig;
import com.alvin.quantitative.trading.platform.data.DataSourceException;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Yahoo Finance Data Source Implementation
 * Author: Alvin
 * Simplified implementation for demo purposes
 */
public class YahooFinanceDataSource implements DataSource {
    
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
        return new KlineData(LocalDateTime.now(), 150.0, 152.0, 148.0, 151.0, 2000000L);
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
        return true; // Always available for demo
    }
    
    @Override
    public String getSourceName() {
        return "Yahoo Finance (Demo)";
    }
    
    @Override
    public String getRateLimitInfo() {
        return "Free with reasonable use";
    }
    
    @Override
    public void cleanup() {
        initialized = false;
    }
}