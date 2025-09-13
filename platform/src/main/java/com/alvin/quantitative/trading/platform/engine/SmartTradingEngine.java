package com.alvin.quantitative.trading.platform.engine;

import com.alvin.quantitative.trading.platform.config.ApplicationConfig;
import com.alvin.quantitative.trading.platform.core.AISignal;
import com.alvin.quantitative.trading.platform.core.KlineData;
import com.alvin.quantitative.trading.platform.core.Position;
import com.alvin.quantitative.trading.platform.data.DataSource;
import com.alvin.quantitative.trading.platform.data.DataSourceFactory;
import com.alvin.quantitative.trading.platform.data.DataSourceException;
import com.alvin.quantitative.trading.platform.data.MarketDataManager;
import com.alvin.quantitative.trading.platform.notification.NotificationService;
import com.alvin.quantitative.trading.platform.portfolio.PortfolioManager;
import com.alvin.quantitative.trading.platform.risk.RiskManager;
import com.alvin.quantitative.trading.platform.strategy.AIStrategyClient;
import com.alvin.quantitative.trading.platform.util.HealthMonitor;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * Smart Trading Engine
 * Author: Alvin
 * Main trading engine that coordinates data, AI strategy, and risk management
 */
public class SmartTradingEngine {
    private static final Logger logger = Logger.getLogger(SmartTradingEngine.class.getName());
    private final MarketDataManager dataManager;
    private final AIStrategyClient aiClient;
    private final RiskManager riskManager;
    private final ScheduledExecutorService scheduler;
    private final Map<String, String> watchList;
    private final ApplicationConfig config;
    private final HealthMonitor healthMonitor;
    private final DataSource dataSource;
    private final PortfolioManager portfolioManager;
    private final NotificationService notificationService;
    private final BacktestEngine backtestEngine;
    private volatile boolean isRunning;
    private double totalCapital;
    
    public SmartTradingEngine() {
        this.config = ApplicationConfig.getInstance();
        this.healthMonitor = HealthMonitor.getInstance();
        
        // Validate configuration
        if (!config.validateConfiguration()) {
            throw new IllegalStateException("Invalid configuration detected");
        }
        
        // Initialize data source
        try {
            this.dataSource = DataSourceFactory.createDataSource(config);
        } catch (DataSourceException e) {
            throw new IllegalStateException("Failed to initialize data source", e);
        }
        
        // Initialize portfolio and notification services
        this.portfolioManager = PortfolioManager.getInstance();
        this.notificationService = NotificationService.getInstance();
        
        // Initialize components with configuration
        this.dataManager = new MarketDataManager(config.getMaxBufferSize());
        this.aiClient = new AIStrategyClient(config.getAiServiceUrl());
        this.riskManager = new RiskManager(
            config.getMaxPositionRatio(),
            config.getStopLossRatio(),
            config.getTakeProfitRatio(),
            config.getMaxDailyLoss()
        );
        this.backtestEngine = new BacktestEngine(dataSource, aiClient, portfolioManager);
        this.scheduler = Executors.newScheduledThreadPool(6); // Increased for more tasks
        this.watchList = new HashMap<>();
        this.totalCapital = config.getInitialCapital();
        this.isRunning = false;
        
        // Add configured stocks to watchlist from portfolio
        initializePortfolioWatchList();
        
        // Print configuration
        config.printConfiguration();
    }
    
    private void initializePortfolioWatchList() {
        // Load symbols from portfolio configuration
        for (String symbol : portfolioManager.getMonitoringSymbols()) {
            PortfolioManager.SymbolConfig symbolConfig = portfolioManager.getSymbolConfig(symbol);
            addToWatchList(symbol, symbolConfig.getName());
        }
        
        // Print portfolio summary
        portfolioManager.printPortfolioSummary();
    }
    
    private String getCompanyName(String symbol) {
        // Simple mapping for common symbols
        switch (symbol) {
            case "AAPL": return "Apple Inc";
            case "TSLA": return "Tesla Inc";
            case "MSFT": return "Microsoft Corp";
            case "GOOGL": return "Alphabet Inc";
            case "AMZN": return "Amazon Inc";
            default: return symbol + " Corp";
        }
    }
    
    public void start() {
        isRunning = true;
        
        // Data collection thread (configurable interval)
        scheduler.scheduleAtFixedRate(this::collectMarketData, 0, 
            config.getDataCollectionInterval(), TimeUnit.SECONDS);
        
        // Strategy execution thread (configurable interval)
        scheduler.scheduleAtFixedRate(this::executeStrategy, 
            config.getDataCollectionInterval(), 
            config.getStrategyExecutionInterval(), TimeUnit.SECONDS);
        
        // Risk check thread (configurable interval)
        scheduler.scheduleAtFixedRate(this::checkRisk, 
            config.getRiskCheckInterval(), 
            config.getRiskCheckInterval(), TimeUnit.SECONDS);
        
        // Daily reset (every 24 hours)
        scheduler.scheduleAtFixedRate(this::dailyReset, 0, 24, TimeUnit.HOURS);
        
        // Health monitoring (every 60 seconds)
        scheduler.scheduleAtFixedRate(this::performHealthCheck, 60, 60, TimeUnit.SECONDS);
        
        // Daily summary (every 24 hours at 6 PM)
        scheduler.scheduleAtFixedRate(this::sendDailySummary, 0, 24, TimeUnit.HOURS);
        
        // Weekly backtest (every Sunday)
        scheduler.scheduleAtFixedRate(this::runWeeklyBacktest, 0, 7, TimeUnit.DAYS);
        
        System.out.println("Smart Trading Engine started with configuration:");
        System.out.println("- Data collection: every " + config.getDataCollectionInterval() + " seconds");
        System.out.println("- Strategy execution: every " + config.getStrategyExecutionInterval() + " seconds");
        System.out.println("- Risk check: every " + config.getRiskCheckInterval() + " seconds");
        System.out.println("- Health monitoring: every 60 seconds");
        System.out.println("- Daily summary: every 24 hours");
        System.out.println("- Weekly backtest: every 7 days");
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
        healthMonitor.incrementActiveThreads();
        try {
            List<KlineData> history = dataManager.getRecentData(symbol, 100);
            if (history.isEmpty()) {
                healthMonitor.setDataManagerHealth(false);
                return;
            }
            
            healthMonitor.setDataManagerHealth(true);
            KlineData currentData = history.get(history.size() - 1);
            Map<String, Double> indicators = dataManager.getIndicators(symbol);
            
            // Call AI strategy
            healthMonitor.recordSignalRequest();
            AISignal signal = aiClient.getSignal(symbol, currentData, indicators, history);
            
            if (signal != null && !"HOLD".equals(signal.getAction()) && signal.getConfidence() > 0) {
                healthMonitor.recordSuccessfulSignal();
                healthMonitor.setAiServiceHealth(true);
            } else if (signal == null || signal.getConfidence() == 0) {
                healthMonitor.recordFailedSignal();
                healthMonitor.setAiServiceHealth(false);
            }
            
            // Risk check
            if (!passRiskCheck(symbol, signal, currentData.getClose())) {
                System.out.println(symbol + " signal rejected by risk control: " + signal.getAction());
                healthMonitor.recordRejectedTrade();
                return;
            }
            
            // Execute order
            healthMonitor.recordTradeAttempt();
            executeOrder(symbol, signal, currentData.getClose());
            healthMonitor.recordSuccessfulTrade();
            
            // Send notification for significant signals
            notificationService.sendTradingSignalNotification(symbol, signal, currentData.getClose());
            
        } catch (Exception e) {
            healthMonitor.recordError("Strategy execution failed for " + symbol + ": " + e.getMessage());
            System.err.println("Strategy execution failed for " + symbol + ": " + e.getMessage());
        } finally {
            healthMonitor.decrementActiveThreads();
        }
    }
    
    private boolean passRiskCheck(String symbol, AISignal signal, double price) {
        // Confidence check using configured minimum
        if (signal.getConfidence() < config.getMinConfidence()) {
            System.out.println(symbol + " signal rejected: confidence " + 
                String.format("%.2f", signal.getConfidence()) + " < " + config.getMinConfidence());
            return false;
        }
        
        // Risk management check
        switch (signal.getAction()) {
            case "BUY":
                boolean canBuy = riskManager.canBuy(symbol, price, totalCapital);
                if (!canBuy) {
                    System.out.println(symbol + " BUY signal rejected by risk manager");
                }
                return canBuy;
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
        // Print daily health summary
        healthMonitor.printHealthSummary();
        // Reset daily statistics
        healthMonitor.reset();
        // Can add summary and reporting here
    }
    
    private void performHealthCheck() {
        if (!isRunning) return;
        
        try {
            healthMonitor.updateHealthCheck();
            
            // Check if system is healthy and print summary if not
            if (!healthMonitor.isSystemHealthy()) {
                System.out.println("⚠️  System health issue detected!");
                healthMonitor.printHealthSummary();
            }
            
            // Log basic metrics every 5 minutes (300 seconds / 60 = 5 calls)
            if (System.currentTimeMillis() % (5 * 60 * 1000) < 60000) {
                System.out.println(String.format("📊 Health Check - Signal Success: %.1f%%, Trade Success: %.1f%%, Active Threads: %d",
                    healthMonitor.getSignalSuccessRate() * 100,
                    healthMonitor.getTradeSuccessRate() * 100,
                    healthMonitor.getHealthReport().get("performance")));
            }
            
        } catch (Exception e) {
            healthMonitor.recordError("Health check failed: " + e.getMessage());
            System.err.println("Health check failed: " + e.getMessage());
        }
    }
    
    private KlineData fetchRealTimeData(String symbol) {
        try {
            // Use configured data source to get real data
            KlineData data = dataSource.getRealTimeData(symbol);
            
            if (data != null) {
                System.out.println(String.format("📊 [%s] Real-time data: $%.2f (Vol: %,d) from %s", 
                    symbol, data.getClose(), data.getVolume(), dataSource.getSourceName()));
                return data;
            }
            
        } catch (Exception e) {
            System.err.println("⚠️ Failed to fetch real-time data for " + symbol + ": " + e.getMessage());
            healthMonitor.recordError("Data fetch failed for " + symbol + ": " + e.getMessage());
        }
        
        // Fallback to simulation if real data fails
        try {
            DataSource fallbackSource = DataSourceFactory.createSimulationDataSource();
            KlineData data = fallbackSource.getRealTimeData(symbol);
            System.out.println("🔄 Using simulated data for " + symbol);
            return data;
        } catch (Exception e) {
            System.err.println("❌ Even simulation data failed for " + symbol + ": " + e.getMessage());
            return null;
        }
    }
    
    public void addToWatchList(String symbol, String name) {
        watchList.put(symbol, name);
    }
    
    private String repeat(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }
    
    private void sendDailySummary() {
        if (!isRunning) return;
        
        try {
            // Collect today's trading signals and performance
            Map<String, Double> portfolioPerformance = new HashMap<>();
            List<String> todaySignals = new ArrayList<String>();
            
            // TODO: Collect actual performance data
            todaySignals.add("AAPL - BUY signal at $175.32 (85% confidence)");
            todaySignals.add("TSLA - SELL signal at $250.15 (78% confidence)");
            
            notificationService.sendDailySummaryEmail(portfolioPerformance, todaySignals);
            logger.info("Daily summary sent successfully");
            
        } catch (Exception e) {
            logger.severe("Failed to send daily summary: " + e.getMessage());
        }
    }
    
    private void runWeeklyBacktest() {
        if (!isRunning) return;
        
        try {
            System.out.println("🔄 Starting weekly portfolio backtest...");
            
            BacktestEngine.BacktestResult result = backtestEngine.runPortfolioBacktest();
            
            if (result.getError() != null) {
                System.err.println("❌ Backtest failed: " + result.getError());
                return;
            }
            
            // Print backtest results
            System.out.println("\n" + repeat("=", 60));
            System.out.println("📈 3年历史回测结果 - " + portfolioManager.getPortfolioName());
            System.out.println(repeat("=", 60));
            System.out.println(String.format("📅 回测期间: %s 至 %s", 
                result.getStartDate().format(DateTimeFormatter.ofPattern("yyyy-MM-dd")),
                result.getEndDate().format(DateTimeFormatter.ofPattern("yyyy-MM-dd"))));
            System.out.println(String.format("💰 初始资金: $%,.2f", result.getInitialCapital()));
            System.out.println(String.format("💰 最终资金: $%,.2f", result.getFinalCapital()));
            System.out.println(String.format("📊 总收益率: %.2f%%", result.getTotalReturn() * 100));
            System.out.println(String.format("📊 年化收益率: %.2f%%", result.getAnnualizedReturn() * 100));
            System.out.println(String.format("📊 夏普比率: %.2f", result.getSharpeRatio()));
            System.out.println(String.format("📊 最大回撤: %.2f%%", result.getMaxDrawdown() * 100));
            System.out.println(String.format("📊 胜率: %.2f%%", result.getWinRate() * 100));
            System.out.println(String.format("🔢 总交易次数: %d", result.getTrades().size()));
            
            // Performance evaluation
            if (result.getTotalReturn() > 0.2) { // > 20% total return
                System.out.println("✅ 策略表现优秀！");
            } else if (result.getTotalReturn() > 0.1) { // > 10% total return
                System.out.println("👍 策略表现良好");
            } else if (result.getTotalReturn() > 0) {
                System.out.println("⚠️ 策略表现一般，需要优化");
            } else {
                System.out.println("❌ 策略表现不佳，需要重新评估");
            }
            
            System.out.println(repeat("=", 60) + "\n");
            
            logger.info(String.format("Weekly backtest completed: %.2f%% total return, %.2f Sharpe ratio",
                result.getTotalReturn() * 100, result.getSharpeRatio()));
            
        } catch (Exception e) {
            logger.severe("Weekly backtest failed: " + e.getMessage());
            System.err.println("❌ Weekly backtest failed: " + e.getMessage());
        }
    }
    
    public void runManualBacktest() {
        System.out.println("🚀 Starting manual portfolio backtest...");
        runWeeklyBacktest();
    }
    
    public void stop() {
        isRunning = false;
        scheduler.shutdown();
        
        // Cleanup notification service
        notificationService.shutdown();
        
        System.out.println("Trading engine stopped");
    }
}
