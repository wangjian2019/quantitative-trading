package com.alvin.quantitative.trading.platform.broker;

import com.alvin.quantitative.trading.platform.core.KlineData;
import java.util.List;
import java.util.Map;

/**
 * Broker Interface for Real Trading
 * Author: Alvin
 * Interface for connecting to real brokers like Interactive Brokers, TD Ameritrade, etc.
 */
public interface BrokerInterface {
    
    /**
     * Initialize broker connection
     */
    boolean connect(Map<String, String> credentials) throws BrokerException;
    
    /**
     * Disconnect from broker
     */
    void disconnect();
    
    /**
     * Check if connected to broker
     */
    boolean isConnected();
    
    /**
     * Get account information
     */
    AccountInfo getAccountInfo() throws BrokerException;
    
    /**
     * Place a market order
     */
    OrderResult placeMarketOrder(String symbol, String action, double quantity) throws BrokerException;
    
    /**
     * Place a limit order
     */
    OrderResult placeLimitOrder(String symbol, String action, double quantity, double limitPrice) throws BrokerException;
    
    /**
     * Place a stop loss order
     */
    OrderResult placeStopLossOrder(String symbol, double quantity, double stopPrice) throws BrokerException;
    
    /**
     * Cancel an order
     */
    boolean cancelOrder(String orderId) throws BrokerException;
    
    /**
     * Get order status
     */
    OrderStatus getOrderStatus(String orderId) throws BrokerException;
    
    /**
     * Get current positions
     */
    List<BrokerPosition> getPositions() throws BrokerException;
    
    /**
     * Get available cash
     */
    double getAvailableCash() throws BrokerException;
    
    /**
     * Get portfolio value
     */
    double getPortfolioValue() throws BrokerException;
    
    /**
     * Get real-time quote
     */
    Quote getRealTimeQuote(String symbol) throws BrokerException;
    
    /**
     * Subscribe to real-time data
     */
    void subscribeRealTimeData(List<String> symbols, DataCallback callback) throws BrokerException;
    
    /**
     * Unsubscribe from real-time data
     */
    void unsubscribeRealTimeData(List<String> symbols) throws BrokerException;
    
    // Data classes
    
    class AccountInfo {
        private double totalValue;
        private double availableCash;
        private double buyingPower;
        private double dayPnL;
        private double totalPnL;
        
        // Getters and setters
        public double getTotalValue() { return totalValue; }
        public void setTotalValue(double totalValue) { this.totalValue = totalValue; }
        
        public double getAvailableCash() { return availableCash; }
        public void setAvailableCash(double availableCash) { this.availableCash = availableCash; }
        
        public double getBuyingPower() { return buyingPower; }
        public void setBuyingPower(double buyingPower) { this.buyingPower = buyingPower; }
        
        public double getDayPnL() { return dayPnL; }
        public void setDayPnL(double dayPnL) { this.dayPnL = dayPnL; }
        
        public double getTotalPnL() { return totalPnL; }
        public void setTotalPnL(double totalPnL) { this.totalPnL = totalPnL; }
    }
    
    class OrderResult {
        private String orderId;
        private String status;
        private String message;
        private double filledQuantity;
        private double filledPrice;
        private long timestamp;
        
        // Getters and setters
        public String getOrderId() { return orderId; }
        public void setOrderId(String orderId) { this.orderId = orderId; }
        
        public String getStatus() { return status; }
        public void setStatus(String status) { this.status = status; }
        
        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
        
        public double getFilledQuantity() { return filledQuantity; }
        public void setFilledQuantity(double filledQuantity) { this.filledQuantity = filledQuantity; }
        
        public double getFilledPrice() { return filledPrice; }
        public void setFilledPrice(double filledPrice) { this.filledPrice = filledPrice; }
        
        public long getTimestamp() { return timestamp; }
        public void setTimestamp(long timestamp) { this.timestamp = timestamp; }
    }
    
    enum OrderStatus {
        PENDING, FILLED, PARTIALLY_FILLED, CANCELLED, REJECTED
    }
    
    class BrokerPosition {
        private String symbol;
        private double quantity;
        private double averageCost;
        private double currentPrice;
        private double unrealizedPnL;
        private double realizedPnL;
        
        // Getters and setters
        public String getSymbol() { return symbol; }
        public void setSymbol(String symbol) { this.symbol = symbol; }
        
        public double getQuantity() { return quantity; }
        public void setQuantity(double quantity) { this.quantity = quantity; }
        
        public double getAverageCost() { return averageCost; }
        public void setAverageCost(double averageCost) { this.averageCost = averageCost; }
        
        public double getCurrentPrice() { return currentPrice; }
        public void setCurrentPrice(double currentPrice) { this.currentPrice = currentPrice; }
        
        public double getUnrealizedPnL() { return unrealizedPnL; }
        public void setUnrealizedPnL(double unrealizedPnL) { this.unrealizedPnL = unrealizedPnL; }
        
        public double getRealizedPnL() { return realizedPnL; }
        public void setRealizedPnL(double realizedPnL) { this.realizedPnL = realizedPnL; }
    }
    
    class Quote {
        private String symbol;
        private double bid;
        private double ask;
        private double last;
        private long volume;
        private long timestamp;
        
        // Getters and setters
        public String getSymbol() { return symbol; }
        public void setSymbol(String symbol) { this.symbol = symbol; }
        
        public double getBid() { return bid; }
        public void setBid(double bid) { this.bid = bid; }
        
        public double getAsk() { return ask; }
        public void setAsk(double ask) { this.ask = ask; }
        
        public double getLast() { return last; }
        public void setLast(double last) { this.last = last; }
        
        public long getVolume() { return volume; }
        public void setVolume(long volume) { this.volume = volume; }
        
        public long getTimestamp() { return timestamp; }
        public void setTimestamp(long timestamp) { this.timestamp = timestamp; }
    }
    
    interface DataCallback {
        void onQuoteUpdate(Quote quote);
        void onError(String error);
    }
}

class BrokerException extends Exception {
    public BrokerException(String message) {
        super(message);
    }
    
    public BrokerException(String message, Throwable cause) {
        super(message, cause);
    }
}
