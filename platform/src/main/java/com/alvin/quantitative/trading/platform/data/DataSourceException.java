package com.alvin.quantitative.trading.platform.data;

/**
 * Data Source Exception
 * Author: Alvin
 * Custom exception for data source operations
 */
public class DataSourceException extends Exception {
    
    public enum ErrorType {
        NETWORK_ERROR,
        API_LIMIT_EXCEEDED,
        INVALID_SYMBOL,
        AUTHENTICATION_FAILED,
        DATA_NOT_AVAILABLE,
        CONFIGURATION_ERROR,
        UNKNOWN_ERROR
    }
    
    private final ErrorType errorType;
    private final String source;
    
    public DataSourceException(String message) {
        super(message);
        this.errorType = ErrorType.UNKNOWN_ERROR;
        this.source = "Unknown";
    }
    
    public DataSourceException(String message, Throwable cause) {
        super(message, cause);
        this.errorType = ErrorType.UNKNOWN_ERROR;
        this.source = "Unknown";
    }
    
    public DataSourceException(ErrorType errorType, String source, String message) {
        super(message);
        this.errorType = errorType;
        this.source = source;
    }
    
    public DataSourceException(ErrorType errorType, String source, String message, Throwable cause) {
        super(message, cause);
        this.errorType = errorType;
        this.source = source;
    }
    
    public ErrorType getErrorType() {
        return errorType;
    }
    
    public String getSource() {
        return source;
    }
    
    public boolean isRetryable() {
        return errorType == ErrorType.NETWORK_ERROR || 
               errorType == ErrorType.API_LIMIT_EXCEEDED;
    }
    
    @Override
    public String toString() {
        return String.format("DataSourceException{type=%s, source='%s', message='%s'}", 
                           errorType, source, getMessage());
    }
}
