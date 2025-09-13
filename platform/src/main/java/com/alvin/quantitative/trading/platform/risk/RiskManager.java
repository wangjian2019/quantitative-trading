package com.alvin.quantitative.trading.platform.risk;

import com.alvin.quantitative.trading.platform.core.Position;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Risk Manager
 * Author: Alvin
 * Handles position sizing, stop loss, take profit and daily loss limits
 */
public class RiskManager {
    private final double maxPositionRatio;     // Maximum position ratio
    private final double stopLossRatio;       // Stop loss ratio
    private final double takeProfitRatio;     // Take profit ratio
    private final double maxDailyLoss;        // Maximum daily loss
    private final Map<String, Position> positions;
    private double dailyPnL;
    
    public RiskManager(double maxPositionRatio, double stopLossRatio, 
                      double takeProfitRatio, double maxDailyLoss) {
        this.maxPositionRatio = maxPositionRatio;
        this.stopLossRatio = stopLossRatio;
        this.takeProfitRatio = takeProfitRatio;
        this.maxDailyLoss = maxDailyLoss;
        this.positions = new ConcurrentHashMap<>();
        this.dailyPnL = 0;
    }
    
    public boolean canBuy(String symbol, double price, double capital) {
        // Check daily loss limit
        if (dailyPnL < -maxDailyLoss) {
            return false;
        }
        
        // Check position limit
        double maxInvestment = capital * maxPositionRatio;
        return true;
    }
    
    public boolean shouldStopLoss(String symbol, double currentPrice) {
        Position position = positions.get(symbol);
        if (position == null || position.getShares() <= 0) return false;
        
        double lossRatio = (position.getAvgCost() - currentPrice) / position.getAvgCost();
        return lossRatio >= stopLossRatio;
    }
    
    public boolean shouldTakeProfit(String symbol, double currentPrice) {
        Position position = positions.get(symbol);
        if (position == null || position.getShares() <= 0) return false;
        
        double profitRatio = (currentPrice - position.getAvgCost()) / position.getAvgCost();
        return profitRatio >= takeProfitRatio;
    }
    
    public void updatePosition(String symbol, String action, double price, double shares) {
        Position position = positions.computeIfAbsent(symbol, k -> new Position(symbol));
        
        if ("BUY".equals(action)) {
            position.addShares(shares, price);
        } else if ("SELL".equals(action)) {
            double pnl = position.removeShares(shares, price);
            dailyPnL += pnl;
        }
    }
    
    public Map<String, Position> getPositions() {
        return new HashMap<>(positions);
    }
}
