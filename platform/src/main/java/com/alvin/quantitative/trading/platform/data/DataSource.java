package com.alvin.quantitative.trading.platform.data;

import com.alvin.quantitative.trading.platform.core.KlineData;
import java.util.List;

/**
 * Data Source Interface - Strategy Pattern
 * Author: Alvin
 * Abstraction for different market data sources
 */
public interface DataSource {
    
    /**
     * Get real-time stock data
     * @param symbol Stock symbol (e.g., "AAPL")
     * @return KlineData object with current price data
     */
    KlineData getRealTimeData(String symbol) throws DataSourceException;
    
    /**
     * Get historical stock data
     * @param symbol Stock symbol
     * @param days Number of days of historical data
     * @return List of historical KlineData
     */
    List<KlineData> getHistoricalData(String symbol, int days) throws DataSourceException;
    
    /**
     * Check if the data source is available
     * @return true if available, false otherwise
     */
    boolean isAvailable();
    
    /**
     * Get data source name
     * @return Name of the data source
     */
    String getSourceName();
    
    /**
     * Get rate limit information
     * @return Rate limit info string
     */
    String getRateLimitInfo();
    
    /**
     * Initialize the data source with configuration
     * @param config Configuration parameters
     */
    void initialize(DataSourceConfig config) throws DataSourceException;
    
    /**
     * Cleanup resources
     */
    void cleanup();
}
