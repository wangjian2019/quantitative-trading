package com.alvin.quantitative.trading.platform.engine;

import com.alvin.quantitative.trading.platform.core.KlineData;
import com.alvin.quantitative.trading.platform.core.Position;
import com.alvin.quantitative.trading.platform.data.DataSource;

import java.util.List;
import java.util.Map;

/**
 * Trading Engine Interface
 * Author: Alvin
 * Common interface for different trading engine implementations
 */
public interface TradingEngineInterface {
    
    /**
     * Start the trading engine
     */
    void start();
    
    /**
     * Stop the trading engine
     */
    void stop();
    
    /**
     * Get current positions
     */
    Map<String, Position> getCurrentPositions();
    
    /**
     * Get data source
     */
    DataSource getDataSource();
    
    /**
     * Get recent signals
     */
    Map<String, Object> getRecentSignals();
    
    /**
     * Run backtest analysis
     */
    Map<String, Object> runBacktestAnalysis();
    
    /**
     * Test notification configuration
     */
    Map<String, Boolean> testNotificationConfig();
    
    /**
     * Get health report
     */
    Map<String, Object> getHealthReport();
    
    /**
     * Get real-time indicators for a symbol
     */
    Map<String, Double> getRealTimeIndicators(String symbol);
    
    /**
     * Get recent data for a symbol
     */
    List<KlineData> getRecentData(String symbol, int count);
    
    /**
     * Restart the trading engine
     */
    void restart() throws Exception;
    
    /**
     * Print health summary
     */
    void printHealthSummary();
    
    /**
     * Run manual backtest
     */
    void runManualBacktest();
}
