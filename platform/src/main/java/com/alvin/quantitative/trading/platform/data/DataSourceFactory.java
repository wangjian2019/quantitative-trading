package com.alvin.quantitative.trading.platform.data;

import com.alvin.quantitative.trading.platform.config.ApplicationConfig;
import com.alvin.quantitative.trading.platform.data.impl.SimulationDataSource;
import com.alvin.quantitative.trading.platform.data.impl.YahooFinanceDataSource;

import java.util.logging.Logger;

/**
 * Data Source Factory - Factory Pattern
 * Author: Alvin
 * Creates appropriate data source based on configuration
 */
public class DataSourceFactory {
    private static final Logger logger = Logger.getLogger(DataSourceFactory.class.getName());
    
    public enum DataSourceType {
        YAHOO_FINANCE,
        SIMULATION
    }
    
    private DataSourceFactory() {
        // Private constructor to prevent instantiation
    }
    
    public static DataSource createDataSource(ApplicationConfig config) throws DataSourceException {
        String dataSourceTypeStr = config.getProperty("data.source.type", "YAHOO_FINANCE");
        DataSourceType dataSourceType;
        
        try {
            dataSourceType = DataSourceType.valueOf(dataSourceTypeStr.toUpperCase());
        } catch (IllegalArgumentException e) {
            logger.warning("Unknown data source type: " + dataSourceTypeStr + ", using YAHOO_FINANCE");
            dataSourceType = DataSourceType.YAHOO_FINANCE;
        }
        
        DataSource dataSource = createDataSourceByType(dataSourceType, config);
        
        // Test data source availability
        if (!dataSource.isAvailable()) {
            logger.warning("Primary data source not available, falling back to simulation");
            dataSource.cleanup();
            dataSource = createSimulationDataSource();
        }
        
        logger.info("Data source initialized: " + dataSource.getSourceName() + 
                   " (" + dataSource.getRateLimitInfo() + ")");
        
        return dataSource;
    }
    
    private static DataSource createDataSourceByType(DataSourceType type, ApplicationConfig config) 
            throws DataSourceException {
        
        switch (type) {
            case YAHOO_FINANCE:
                return createYahooFinanceDataSource(config);
                
            case SIMULATION:
            default:
                return createSimulationDataSource();
        }
    }
    
    private static DataSource createYahooFinanceDataSource(ApplicationConfig config) throws DataSourceException {
        String baseUrl = config.getProperty("data.source.yahoo.finance.base.url", 
            "https://query1.finance.yahoo.com/v8/finance/chart");
        
        DataSourceConfig dataSourceConfig = DataSourceConfig.builder()
            .setBaseUrl(baseUrl)
            .setTimeout(config.getIntProperty("data.fetch.timeout", 10000))
            .setRetryCount(config.getIntProperty("data.fetch.retry.max", 3))
            .build();
        
        DataSource dataSource = new YahooFinanceDataSource();
        dataSource.initialize(dataSourceConfig);
        return dataSource;
    }
    
    public static DataSource createSimulationDataSource() {
        try {
            DataSource dataSource = new SimulationDataSource();
            dataSource.initialize(new DataSourceConfig());
            return dataSource;
        } catch (DataSourceException e) {
            // This should never happen for simulation data source
            throw new RuntimeException("Failed to create simulation data source", e);
        }
    }
}