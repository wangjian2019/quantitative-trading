package com.alvin.quantitative.trading.platform;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Smart Trading Engine
 * Author: Alvin
 * Main trading engine that coordinates data, AI strategy, and risk management
 */
public class SmartTradingEngine {
    private final MarketDataManager dataManager;
    private final AIStrategyClient aiClient;
    private final RiskManager riskManager;
    private final ScheduledExecutorService scheduler;
    private final Map<String, String> watchList;
    private volatile boolean isRunning;
    private double totalCapital;
    
    public SmartTradingEngine(String aiServiceUrl, double initialCapital) {
        this.dataManager = new MarketDataManager(500);
        this.aiClient = new AIStrategyClient(aiServiceUrl);
        this.riskManager = new RiskManager(0.3, 0.05, 0.15, initialCapital * 0.03);
        this.scheduler = Executors.newScheduledThreadPool(4);
        this.watchList = new HashMap<>();
        this.totalCapital = initialCapital;
        this.isRunning = false;
        
        // Add stocks to watchlist
        addToWatchList("AAPL", "Apple Inc");
        addToWatchList("TSLA", "Tesla Inc");
        addToWatchList("MSFT", "Microsoft Corp");
    }
    
    public void start() {
        isRunning = true;
        
        // Data collection thread (every minute)
        scheduler.scheduleAtFixedRate(this::collectMarketData, 0, 1, TimeUnit.MINUTES);
        
        // Strategy execution thread (every 5 minutes)
        scheduler.scheduleAtFixedRate(this::executeStrategy, 1, 5, TimeUnit.MINUTES);
        
        // Risk check thread (every 30 seconds)
        scheduler.scheduleAtFixedRate(this::checkRisk, 30, 30, TimeUnit.SECONDS);
        
        // Daily reset (every 24 hours)
        scheduler.scheduleAtFixedRate(this::dailyReset, 0, 24, TimeUnit.HOURS);
        
        System.out.println("Smart Trading Engine started");
    }
    
    private void collectMarketData() {
        if (!isRunning) return;
        
        for (String symbol : watchList.keySet()) {
            try {
                // Call actual market data API here
                KlineData data = fetchRealTimeData(symbol);
                if (data != null) {
                    dataManager.addKlineData(symbol, data);
                }
            } catch (Exception e) {
                System.err.println("Failed to get data for " + symbol + ": " + e.getMessage());
            }
        }
    }
    
    private void executeStrategy() {
        if (!isRunning) return;
        
        for (String symbol : watchList.keySet()) {
            try {
                executeStrategyForSymbol(symbol);
            } catch (Exception e) {
                System.err.println("Failed to execute strategy for " + symbol + ": " + e.getMessage());
            }
        }
    }
    
    private void executeStrategyForSymbol(String symbol) {
        List<KlineData> history = dataManager.getRecentData(symbol, 100);
        if (history.isEmpty()) return;
        
        KlineData currentData = history.get(history.size() - 1);
        Map<String, Double> indicators = dataManager.getIndicators(symbol);
        
        // Call AI strategy
        AISignal signal = aiClient.getSignal(symbol, currentData, indicators, history);
        
        // Risk check
        if (!passRiskCheck(symbol, signal, currentData.getClose())) {
            System.out.println(symbol + " signal rejected by risk control: " + signal.getAction());
            return;
        }
        
        // Execute order
        executeOrder(symbol, signal, currentData.getClose());
    }
    
    private boolean passRiskCheck(String symbol, AISignal signal, double price) {
        // Confidence check
        if (signal.getConfidence() < 0.6) {
            return false;
        }
        
        // Risk management check
        switch (signal.getAction()) {
            case "BUY":
                return riskManager.canBuy(symbol, price, totalCapital);
            case "SELL":
                return riskManager.shouldStopLoss(symbol, price) || 
                       riskManager.shouldTakeProfit(symbol, price) ||
                       signal.getConfidence() > 0.8;
            default:
                return true;
        }
    }
    
    private void executeOrder(String symbol, AISignal signal, double price) {
        System.out.println(String.format("[%s] %s: %s@%.2f Confidence:%.2f Reason:%s", 
            LocalDateTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss")),
            symbol, signal.getAction(), price, signal.getConfidence(), signal.getReason()));
        
        // Connect to actual trading API here
        // e.g., Interactive Brokers, TD Ameritrade, etc.
        
        // Simulate trade execution
        switch (signal.getAction()) {
            case "BUY":
                double buyAmount = totalCapital * 0.1; // Buy 10% each time
                double shares = buyAmount / price;
                riskManager.updatePosition(symbol, "BUY", price, shares);
                System.out.println("Bought " + symbol + " " + shares + " shares");
                break;
                
            case "SELL":
                Position position = riskManager.getPositions().get(symbol);
                if (position != null && position.getShares() > 0) {
                    riskManager.updatePosition(symbol, "SELL", price, position.getShares());
                    System.out.println("Sold " + symbol + " " + position.getShares() + " shares");
                }
                break;
        }
    }
    
    private void checkRisk() {
        if (!isRunning) return;
        
        for (Map.Entry<String, Position> entry : riskManager.getPositions().entrySet()) {
            String symbol = entry.getKey();
            Position position = entry.getValue();
            
            if (position.getShares() <= 0) continue;
            
            List<KlineData> recent = dataManager.getRecentData(symbol, 1);
            if (recent.isEmpty()) continue;
            
            double currentPrice = recent.get(0).getClose();
            
            // Check stop loss
            if (riskManager.shouldStopLoss(symbol, currentPrice)) {
                System.out.println("Stop loss triggered: " + symbol + " @" + currentPrice);
                executeOrder(symbol, createSellSignal("Stop Loss"), currentPrice);
            }
            
            // Check take profit
            if (riskManager.shouldTakeProfit(symbol, currentPrice)) {
                System.out.println("Take profit triggered: " + symbol + " @" + currentPrice);
                executeOrder(symbol, createSellSignal("Take Profit"), currentPrice);
            }
        }
    }
    
    private AISignal createSellSignal(String reason) {
        AISignal signal = new AISignal();
        signal.setAction("SELL");
        signal.setConfidence(1.0);
        signal.setReason(reason);
        return signal;
    }
    
    private void dailyReset() {
        System.out.println("Performing daily reset...");
        // Reset daily statistics
        // Can add summary and reporting here
    }
    
    private KlineData fetchRealTimeData(String symbol) {
        // Connect to actual market data source here
        // e.g., Alpha Vantage, Yahoo Finance, IEX Cloud, etc.
        
        // Simulate data
        Random random = new Random();
        double basePrice = 100 + random.nextGaussian() * 10;
        
        return new KlineData(
            LocalDateTime.now(),
            basePrice,
            basePrice + random.nextDouble() * 2,
            basePrice - random.nextDouble() * 2,
            basePrice + random.nextGaussian(),
            1000 + random.nextInt(9000)
        );
    }
    
    public void addToWatchList(String symbol, String name) {
        watchList.put(symbol, name);
    }
    
    public void stop() {
        isRunning = false;
        scheduler.shutdown();
        System.out.println("Trading engine stopped");
    }
}
