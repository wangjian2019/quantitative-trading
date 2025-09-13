package com.alvin.quantitative.trading.platform;

import java.util.Map;

public class AISignal {
    private String action;
    private double confidence;
    private String reason;
    private Map<String, Object> metadata;
    
    // Getters and Setters
    public String getAction() { return action; }
    public void setAction(String action) { this.action = action; }
    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }
    public String getReason() { return reason; }
    public void setReason(String reason) { this.reason = reason; }
    public Map<String, Object> getMetadata() { return metadata; }
    public void setMetadata(Map<String, Object> metadata) { this.metadata = metadata; }
}
