package com.alvin.quantitative.trading.platform.core;

import java.time.LocalDateTime;

/**
 * K线数据类
 * Author: Alvin
 * 存储股票K线数据，包括开盘价、最高价、最低价、收盘价和成交量
 */
public class KlineData {
    private LocalDateTime timestamp;
    private double open;
    private double high;
    private double low;
    private double close;
    private long volume;
    
    public KlineData() {}
    
    public KlineData(LocalDateTime timestamp, double open, double high, double low, double close, long volume) {
        this.timestamp = timestamp;
        this.open = open;
        this.high = high;
        this.low = low;
        this.close = close;
        this.volume = volume;
    }
    
    // Getters and Setters
    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
    
    public double getOpen() { return open; }
    public void setOpen(double open) { this.open = open; }
    
    public double getHigh() { return high; }
    public void setHigh(double high) { this.high = high; }
    
    public double getLow() { return low; }
    public void setLow(double low) { this.low = low; }
    
    public double getClose() { return close; }
    public void setClose(double close) { this.close = close; }
    
    public long getVolume() { return volume; }
    public void setVolume(long volume) { this.volume = volume; }
    
    @Override
    public String toString() {
        return String.format("KlineData{timestamp=%s, open=%.2f, high=%.2f, low=%.2f, close=%.2f, volume=%d}", 
                           timestamp, open, high, low, close, volume);
    }
}

