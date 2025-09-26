package com.alvin.quantitative.trading.platform.core;

import java.util.Map;

/**
 * AI Trading Signal Model
 * Author: Alvin
 * Simple signal data class
 */
public class AISignal {
    private String action;
    private double confidence;
    private String reason;
    private Map<String, Object> metadata;
    private double expectedReturn;
    private String modelType;
    private double volatility;
    private double suggestedPositionSize;
    
    public AISignal() {}
    
    public AISignal(String action, double confidence, String reason) {
        this.action = action;
        this.confidence = confidence;
        this.reason = reason;
    }
    
    // Getters and Setters
    public String getAction() { return action; }
    public void setAction(String action) { this.action = action; }
    
    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }
    
    public String getReason() { return reason; }
    public void setReason(String reason) { this.reason = reason; }
    
    public Map<String, Object> getMetadata() { return metadata; }
    public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }

    public double getExpectedReturn() { return expectedReturn; }
    public void setExpectedReturn(double expectedReturn) { this.expectedReturn = expectedReturn; }

    public String getModelType() { return modelType; }
    public void setModelType(String modelType) { this.modelType = modelType; }

    public double getVolatility() { return volatility; }
    public void setVolatility(double volatility) { this.volatility = volatility; }

    public double getSuggestedPositionSize() { return suggestedPositionSize; }
    public void setSuggestedPositionSize(double suggestedPositionSize) { this.suggestedPositionSize = suggestedPositionSize; }
    
    @Override
    public String toString() {
        return String.format("AISignal{action=%s, confidence=%.2f, reason='%s'}", 
                           action, confidence, reason);
    }
}