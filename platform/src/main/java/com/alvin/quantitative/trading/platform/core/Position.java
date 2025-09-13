package com.alvin.quantitative.trading.platform.core;

public class Position {
    private String symbol;
    private double shares;
    private double avgCost;
    private double totalCost;
    
    public Position(String symbol) {
        this.symbol = symbol;
        this.shares = 0;
        this.avgCost = 0;
        this.totalCost = 0;
    }
    
    public void addShares(double newShares, double price) {
        double newCost = newShares * price;
        this.totalCost += newCost;
        this.shares += newShares;
        this.avgCost = totalCost / shares;
    }
    
    public double removeShares(double sellShares, double price) {
        if (sellShares >= shares) {
            // 全部卖出
            double pnl = shares * (price - avgCost);
            shares = 0;
            avgCost = 0;
            totalCost = 0;
            return pnl;
        } else {
            // 部分卖出
            double pnl = sellShares * (price - avgCost);
            shares -= sellShares;
            totalCost -= sellShares * avgCost;
            return pnl;
        }
    }
    
    // Getters
    public String getSymbol() { return symbol; }
    public double getShares() { return shares; }
    public double getAvgCost() { return avgCost; }
}