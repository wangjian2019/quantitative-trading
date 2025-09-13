package com.alvin.quantitative.trading.platform.data;

import java.util.HashMap;
import java.util.Map;

/**
 * Data Source Configuration
 * Author: Alvin
 * Configuration object for data sources
 */
public class DataSourceConfig {
    private final Map<String, String> properties;
    
    public DataSourceConfig() {
        this.properties = new HashMap<String, String>();
    }
    
    public DataSourceConfig(Map<String, String> properties) {
        this.properties = new HashMap<String, String>(properties);
    }
    
    public String getProperty(String key) {
        return properties.get(key);
    }
    
    public String getProperty(String key, String defaultValue) {
        return properties.getOrDefault(key, defaultValue);
    }
    
    public void setProperty(String key, String value) {
        properties.put(key, value);
    }
    
    public int getIntProperty(String key, int defaultValue) {
        try {
            String value = properties.get(key);
            return value != null ? Integer.parseInt(value) : defaultValue;
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
    
    public long getLongProperty(String key, long defaultValue) {
        try {
            String value = properties.get(key);
            return value != null ? Long.parseLong(value) : defaultValue;
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
    
    public boolean getBooleanProperty(String key, boolean defaultValue) {
        String value = properties.get(key);
        return value != null ? Boolean.parseBoolean(value) : defaultValue;
    }
    
    public Map<String, String> getAllProperties() {
        return new HashMap<String, String>(properties);
    }
    
    public boolean hasProperty(String key) {
        return properties.containsKey(key);
    }
    
    public static class Builder {
        private final Map<String, String> properties = new HashMap<String, String>();
        
        public Builder setProperty(String key, String value) {
            properties.put(key, value);
            return this;
        }
        
        public Builder setApiKey(String apiKey) {
            return setProperty("api.key", apiKey);
        }
        
        public Builder setBaseUrl(String baseUrl) {
            return setProperty("base.url", baseUrl);
        }
        
        public Builder setTimeout(int timeout) {
            return setProperty("timeout", String.valueOf(timeout));
        }
        
        public Builder setRetryCount(int retryCount) {
            return setProperty("retry.count", String.valueOf(retryCount));
        }
        
        public DataSourceConfig build() {
            return new DataSourceConfig(properties);
        }
    }
    
    public static Builder builder() {
        return new Builder();
    }
}
