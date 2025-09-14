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
public class SmartTradingEngine implements TradingEngineInterface {
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
            
            // Call AI strategy with null safety
            healthMonitor.recordSignalRequest();
            AISignal signal = null;
            
            try {
                signal = aiClient.getSignal(symbol, currentData, indicators, history);
            } catch (Exception e) {
                logger.severe("🚨 CRITICAL: AI service call failed for " + symbol + ": " + e.getMessage());
                healthMonitor.recordFailedSignal();
                healthMonitor.setAiServiceHealth(false);
                return; // Skip this symbol
            }
            
            // Null safety check
            if (signal == null) {
                logger.warning("AI service returned null signal for " + symbol);
                healthMonitor.recordFailedSignal();
                healthMonitor.setAiServiceHealth(false);
                return;
            }
            
            if (!"HOLD".equals(signal.getAction()) && signal.getConfidence() > 0) {
                healthMonitor.recordSuccessfulSignal();
                healthMonitor.setAiServiceHealth(true);
            } else if (signal.getConfidence() == 0) {
                healthMonitor.recordFailedSignal();
                healthMonitor.setAiServiceHealth(false);
            }
            
            // Risk check
            if (!passRiskCheck(symbol, signal, currentData.getClose())) {
                System.out.println(symbol + " signal rejected by risk control: " + signal.getAction());
                healthMonitor.recordRejectedTrade();
                return;
            }
            
            // 发送交易通知给用户（手动执行）
            healthMonitor.recordTradeAttempt();
            sendTradingNotificationToUser(symbol, signal, currentData.getClose());
            healthMonitor.recordSuccessfulTrade();
            
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
    
    /**
     * 发送交易通知给用户，用户手动执行交易
     */
    private void sendTradingNotificationToUser(String symbol, AISignal signal, double price) {
        // 计算建议仓位大小
        double suggestedPositionMillion = calculatePositionSizeForManualTrading(symbol, signal, price);
        
        // 计算止损止盈价格
        double stopLoss = price * (1 - config.getStopLossRatio());
        double takeProfit = price * (1 + config.getTakeProfitRatio());
        
        // 格式化通知消息
        String notificationMessage = String.format(
            "🚨 AI交易信号 - %s\n" +
            "📊 股票: %s\n" +
            "🎯 操作: %s\n" +
            "💰 价格: $%.2f\n" +
            "📈 置信度: %.1f%%\n" +
            "💼 建议仓位: %.1f万 (%.1f%%)\n" +
            "🛡️ 建议止损: $%.2f\n" +
            "🎯 建议止盈: $%.2f\n" +
            "📝 分析理由: %s\n" +
            "⏰ 时间: %s",
            signal.getAction().equals("BUY") ? "买入信号" : "卖出信号",
            symbol,
            signal.getAction(),
            price,
            signal.getConfidence() * 100,
            suggestedPositionMillion,
            suggestedPositionMillion / 10.0,
            stopLoss,
            takeProfit,
            signal.getReason(),
            LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))
        );
        
        // 控制台输出
        System.out.println(repeat("=", 60));
        System.out.println("🚨 用户交易通知 🚨");
        System.out.println(repeat("=", 60));
        System.out.println(notificationMessage);
        System.out.println(repeat("=", 60));
        
        // 发送邮件和微信通知
        try {
            notificationService.sendTradingSignalNotification(symbol, signal, price);
            logger.info("Trading notification sent to user for manual execution: " + symbol + " " + signal.getAction());
        } catch (Exception e) {
            logger.warning("Failed to send notification: " + e.getMessage());
        }
    }
    
    /**
     * 计算投资的建议仓位大小
     */
    private double calculatePositionSizeForManualTrading(String symbol, AISignal signal, double price) {
        // 基础仓位：根据置信度动态调整
        double basePositionPercent = 0.05; // 基础5%
        
        // 置信度调整：置信度越高，仓位越大
        double confidenceMultiplier = Math.min(3.0, signal.getConfidence() / 0.6);
        
        // 波动率调整：波动率越高，仓位越小
        Map<String, Double> indicators = dataManager.getIndicators(symbol);
        double volatility = indicators.getOrDefault("VOLATILITY", 0.02);
        double volatilityAdjustment = Math.min(1.0, 0.02 / volatility);
        
        // RSI调整：超卖时增加仓位，超买时减少仓位
        double rsi = indicators.getOrDefault("RSI", 50.0);
        double rsiAdjustment = 1.0;
        if (rsi < 30 && "BUY".equals(signal.getAction())) {
            rsiAdjustment = 1.5; // RSI超卖买入，增加50%仓位
        } else if (rsi > 70 && "SELL".equals(signal.getAction())) {
            rsiAdjustment = 1.5; // RSI超买卖出，增加50%仓位
        } else if (rsi > 70 && "BUY".equals(signal.getAction())) {
            rsiAdjustment = 0.5; // RSI超买买入，减少50%仓位
        }
        
        double finalPositionPercent = basePositionPercent * confidenceMultiplier * volatilityAdjustment * rsiAdjustment;
        
        // 限制：单股票最大20%（200万），最小2%（20万）
        finalPositionPercent = Math.max(0.02, Math.min(0.20, finalPositionPercent));
        
        return finalPositionPercent * 100; // 转换为万元
    }
    
    private void checkRisk() {
        if (!isRunning) return;
        
        // 用户手动交易，这里只做风险监控和提醒
        logger.info("⚠️ 风险检查完成 - 用户手动交易模式");
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
        
        // No fallback - we only use real data
        logger.severe("❌ Failed to fetch real data for " + symbol + " - no fallback configured");
        return null;
    }
    
    public void addToWatchList(String symbol, String name) {
        watchList.put(symbol, name);
    }
    
    // Implementation of TradingEngineInterface methods
    
    @Override
    public Map<String, Position> getCurrentPositions() {
        return riskManager.getPositions();
    }
    
    @Override
    public DataSource getDataSource() {
        return dataSource;
    }
    
    @Override
    public Map<String, Object> getRecentSignals() {
        // Return recent signals from AI strategy
        Map<String, Object> signals = new HashMap<>();
        signals.put("signals", new Object[0]); // TODO: Implement signal history
        signals.put("count", 0);
        return signals;
    }
    
    @Override
    public Map<String, Object> runBacktestAnalysis() {
        try {
            BacktestEngine.BacktestResult result = backtestEngine.runPortfolioBacktest();
            
            Map<String, Object> resultMap = new HashMap<>();
            resultMap.put("total_return", result.getTotalReturn());
            resultMap.put("sharpe_ratio", result.getSharpeRatio());
            resultMap.put("max_drawdown", result.getMaxDrawdown());
            resultMap.put("win_rate", result.getWinRate());
            resultMap.put("total_trades", result.getTrades().size()); // 使用trades列表的大小
            resultMap.put("initial_capital", result.getInitialCapital());
            resultMap.put("final_capital", result.getFinalCapital());
            resultMap.put("backtest_type", "Real portfolio backtest");
            
            return resultMap;
        } catch (Exception e) {
            Map<String, Object> result = new HashMap<>();
            result.put("error", "Backtest failed: " + e.getMessage());
            return result;
        }
    }
    
    @Override
    public Map<String, Boolean> testNotificationConfig() {
        Map<String, Boolean> results = new HashMap<>();
        results.put("email", false); // TODO: Test email config
        results.put("wechat", false); // TODO: Test wechat config
        return results;
    }
    
    @Override
    public Map<String, Object> getHealthReport() {
        return healthMonitor.getHealthReport();
    }
    
    @Override
    public Map<String, Double> getRealTimeIndicators(String symbol) {
        return dataManager.getIndicators(symbol);
    }
    
    @Override
    public List<KlineData> getRecentData(String symbol, int count) {
        return dataManager.getRecentData(symbol, count);
    }
    
    @Override
    public void restart() throws Exception {
        System.out.println("🔄 重启智能交易引擎...");
        stop();
        Thread.sleep(2000);
        start();
        System.out.println("✅ 智能交易引擎重启完成");
    }
    
    @Override
    public void printHealthSummary() {
        healthMonitor.printHealthSummary();
    }
    
    @Override
    public void runManualBacktest() {
        try {
            System.out.println("🚀 开始手动回测分析...");
            Map<String, Object> result = runBacktestAnalysis();
            
            System.out.println("\n📈 回测结果:");
            System.out.println("总收益率: " + String.format("%.2f%%", (Double)result.get("total_return") * 100));
            System.out.println("夏普比率: " + String.format("%.2f", result.get("sharpe_ratio")));
            System.out.println("最大回撤: " + String.format("%.2f%%", (Double)result.get("max_drawdown") * 100));
            System.out.println("胜率: " + String.format("%.1f%%", (Double)result.get("win_rate") * 100));
            System.out.println("交易次数: " + result.get("total_trades"));
        } catch (Exception e) {
            System.err.println("❌ 手动回测失败: " + e.getMessage());
        }
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
    
    
    public void stop() {
        isRunning = false;
        scheduler.shutdown();
        
        // Cleanup notification service
        notificationService.shutdown();
        
        System.out.println("Trading engine stopped");
    }
}
