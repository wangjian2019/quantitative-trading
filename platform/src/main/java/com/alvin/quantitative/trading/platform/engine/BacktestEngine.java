package com.alvin.quantitative.trading.platform.engine;

import com.alvin.quantitative.trading.platform.core.AISignal;
import com.alvin.quantitative.trading.platform.core.KlineData;
import com.alvin.quantitative.trading.platform.core.Position;
import com.alvin.quantitative.trading.platform.data.DataSource;
import com.alvin.quantitative.trading.platform.portfolio.PortfolioManager;
import com.alvin.quantitative.trading.platform.strategy.AIStrategyClient;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.logging.Logger;

/**
 * Backtest Engine
 * Author: Alvin
 * Comprehensive backtesting engine for portfolio performance analysis
 */
public class BacktestEngine {
    private static final Logger logger = Logger.getLogger(BacktestEngine.class.getName());
    
    private final DataSource dataSource;
    private final AIStrategyClient aiClient;
    private final PortfolioManager portfolioManager;
    
    public BacktestEngine(DataSource dataSource, AIStrategyClient aiClient, PortfolioManager portfolioManager) {
        this.dataSource = dataSource;
        this.aiClient = aiClient;
        this.portfolioManager = portfolioManager;
    }
    
    public BacktestResult runPortfolioBacktest() {
        logger.info("Starting comprehensive portfolio backtest...");
        
        List<String> symbols = portfolioManager.getMonitoringSymbols();
        int backtestYears = portfolioManager.getBacktestPeriodYears();
        
        BacktestResult result = new BacktestResult();
        result.setStartDate(LocalDateTime.now().minusYears(backtestYears));
        result.setEndDate(LocalDateTime.now());
        result.setSymbols(symbols);
        result.setInitialCapital(100000.0); // $100,000 initial capital
        
        try {
            // Get historical data for all symbols
            Map<String, List<KlineData>> historicalDataMap = new HashMap<>();
            for (String symbol : symbols) {
                try {
                    List<KlineData> historicalData = dataSource.getHistoricalData(symbol, backtestYears * 365);
                    if (!historicalData.isEmpty()) {
                        historicalDataMap.put(symbol, historicalData);
                        logger.info(String.format("Retrieved %d data points for %s", 
                            historicalData.size(), symbol));
                    }
                } catch (Exception e) {
                    logger.warning(String.format("Failed to get historical data for %s: %s", 
                        symbol, e.getMessage()));
                }
            }
            
            if (historicalDataMap.isEmpty()) {
                throw new RuntimeException("No historical data available for backtesting");
            }
            
            // Run simulation
            runBacktestSimulation(historicalDataMap, result);
            
            // Calculate performance metrics
            calculatePerformanceMetrics(result);
            
            logger.info("Portfolio backtest completed successfully");
            
        } catch (Exception e) {
            logger.severe("Backtest failed: " + e.getMessage());
            result.setError("Backtest failed: " + e.getMessage());
        }
        
        return result;
    }
    
    private void runBacktestSimulation(Map<String, List<KlineData>> historicalDataMap, 
                                     BacktestResult result) {
        double currentCapital = result.getInitialCapital();
        Map<String, Double> positions = new HashMap<>(); // symbol -> shares
        Map<String, Double> positionCosts = new HashMap<>(); // symbol -> avg cost
        List<BacktestTrade> trades = new ArrayList<>();
        List<Double> dailyPortfolioValues = new ArrayList<>();
        
        // Find the minimum data length to ensure all symbols have data
        int minDataLength = historicalDataMap.values().stream()
            .mapToInt(List::size)
            .min()
            .orElse(0);
        
        if (minDataLength < 100) {
            throw new RuntimeException("Insufficient historical data for meaningful backtest");
        }
        
        // Simulate trading day by day
        for (int dayIndex = 50; dayIndex < minDataLength - 1; dayIndex++) {
            LocalDateTime currentDate = null;
            double portfolioValue = currentCapital;
            
            // Process each symbol for this day
            for (String symbol : historicalDataMap.keySet()) {
                List<KlineData> symbolData = historicalDataMap.get(symbol);
                
                if (dayIndex >= symbolData.size()) continue;
                
                KlineData currentDayData = symbolData.get(dayIndex);
                currentDate = currentDayData.getTimestamp();
                
                // Get historical data up to current day for AI analysis
                List<KlineData> historicalSubset = symbolData.subList(
                    Math.max(0, dayIndex - 100), dayIndex + 1);
                
                // Calculate indicators
                Map<String, Double> indicators = calculateIndicatorsForBacktest(historicalSubset);
                
                try {
                    // Get AI signal
                    AISignal signal = aiClient.getSignal(symbol, currentDayData, indicators, historicalSubset);
                    
                    if (signal != null && portfolioManager.shouldTrade(symbol, signal)) {
                        BacktestTrade trade = processBacktestTrade(
                            symbol, signal, currentDayData, currentCapital, positions, positionCosts);
                        
                        if (trade != null) {
                            trades.add(trade);
                            
                            // Update capital and positions
                            if ("BUY".equals(signal.getAction())) {
                                double tradeAmount = currentCapital * 0.1; // 10% per trade
                                double shares = tradeAmount / currentDayData.getClose();
                                positions.put(symbol, positions.getOrDefault(symbol, 0.0) + shares);
                                positionCosts.put(symbol, currentDayData.getClose());
                                currentCapital -= tradeAmount;
                            } else if ("SELL".equals(signal.getAction())) {
                                double shares = positions.getOrDefault(symbol, 0.0);
                                if (shares > 0) {
                                    currentCapital += shares * currentDayData.getClose();
                                    positions.put(symbol, 0.0);
                                    positionCosts.remove(symbol);
                                }
                            }
                        }
                    }
                    
                } catch (Exception e) {
                    logger.warning(String.format("Error processing signal for %s on %s: %s", 
                        symbol, currentDate, e.getMessage()));
                }
                
                // Add to portfolio value
                double positionValue = positions.getOrDefault(symbol, 0.0) * currentDayData.getClose();
                portfolioValue += positionValue;
            }
            
            // Record daily portfolio value
            if (currentDate != null) {
                dailyPortfolioValues.add(portfolioValue);
                
                // Log progress every 30 days
                if (dayIndex % 30 == 0) {
                    double returnPct = ((portfolioValue - result.getInitialCapital()) / result.getInitialCapital()) * 100;
                    logger.info(String.format("Backtest progress: %s, Portfolio value: $%.2f (%.2f%%)", 
                        currentDate.format(DateTimeFormatter.ofPattern("yyyy-MM-dd")), 
                        portfolioValue, returnPct));
                }
            }
        }
        
        // Calculate final portfolio value
        double finalValue = currentCapital;
        for (String symbol : positions.keySet()) {
            double shares = positions.get(symbol);
            if (shares > 0) {
                List<KlineData> symbolData = historicalDataMap.get(symbol);
                double finalPrice = symbolData.get(symbolData.size() - 1).getClose();
                finalValue += shares * finalPrice;
            }
        }
        
        result.setFinalCapital(finalValue);
        result.setTrades(trades);
        result.setDailyPortfolioValues(dailyPortfolioValues);
        
        logger.info(String.format("Backtest simulation completed. Final value: $%.2f", finalValue));
    }
    
    private BacktestTrade processBacktestTrade(String symbol, AISignal signal, KlineData currentData,
                                             double currentCapital, Map<String, Double> positions,
                                             Map<String, Double> positionCosts) {
        
        PortfolioManager.SymbolConfig symbolConfig = portfolioManager.getSymbolConfig(symbol);
        if (symbolConfig == null) return null;
        
        BacktestTrade trade = new BacktestTrade();
        trade.setSymbol(symbol);
        trade.setAction(signal.getAction());
        trade.setPrice(currentData.getClose());
        trade.setTimestamp(currentData.getTimestamp());
        trade.setConfidence(signal.getConfidence());
        trade.setReason(signal.getReason());
        
        if ("BUY".equals(signal.getAction())) {
            double tradeAmount = currentCapital * symbolConfig.getWeight();
            double shares = tradeAmount / currentData.getClose();
            trade.setShares(shares);
            trade.setTradeValue(tradeAmount);
            
        } else if ("SELL".equals(signal.getAction())) {
            double shares = positions.getOrDefault(symbol, 0.0);
            if (shares > 0) {
                trade.setShares(shares);
                trade.setTradeValue(shares * currentData.getClose());
                
                // Calculate P&L
                double avgCost = positionCosts.getOrDefault(symbol, currentData.getClose());
                double pnl = (currentData.getClose() - avgCost) * shares;
                trade.setPnl(pnl);
            } else {
                return null; // No position to sell
            }
        }
        
        return trade;
    }
    
    private Map<String, Double> calculateIndicatorsForBacktest(List<KlineData> data) {
        Map<String, Double> indicators = new HashMap<>();
        
        if (data.size() < 20) return indicators;
        
        List<Double> closes = new ArrayList<>();
        List<Long> volumes = new ArrayList<>();
        
        for (KlineData kline : data) {
            closes.add(kline.getClose());
            volumes.add(kline.getVolume());
        }
        
        // Simple moving averages
        indicators.put("MA5", calculateSMA(closes, 5));
        indicators.put("MA10", calculateSMA(closes, 10));
        indicators.put("MA20", calculateSMA(closes, 20));
        
        // RSI
        indicators.put("RSI", calculateRSI(closes, 14));
        
        // MACD
        indicators.put("MACD", calculateMACD(closes));
        
        // Volume ratio
        if (volumes.size() >= 20) {
            long currentVolume = volumes.get(volumes.size() - 1);
            double avgVolume = volumes.subList(volumes.size() - 20, volumes.size() - 1)
                .stream().mapToLong(Long::longValue).average().orElse(1.0);
            indicators.put("VOLUME_RATIO", currentVolume / avgVolume);
        }
        
        // ATR and volatility
        indicators.put("ATR", calculateATR(data));
        indicators.put("VOLATILITY", calculateVolatility(closes, 20));
        indicators.put("PRICE_POSITION", calculatePricePosition(closes, 20));
        
        return indicators;
    }
    
    private void calculatePerformanceMetrics(BacktestResult result) {
        double initialCapital = result.getInitialCapital();
        double finalCapital = result.getFinalCapital();
        List<BacktestTrade> trades = result.getTrades();
        List<Double> dailyValues = result.getDailyPortfolioValues();
        
        // Total return
        double totalReturn = (finalCapital - initialCapital) / initialCapital;
        result.setTotalReturn(totalReturn);
        
        // Annualized return
        int years = portfolioManager.getBacktestPeriodYears();
        double annualizedReturn = Math.pow(1 + totalReturn, 1.0 / years) - 1;
        result.setAnnualizedReturn(annualizedReturn);
        
        // Trade statistics
        if (!trades.isEmpty()) {
            long winningTrades = trades.stream()
                .filter(t -> "SELL".equals(t.getAction()) && t.getPnl() > 0)
                .count();
            
            double winRate = (double) winningTrades / trades.size();
            result.setWinRate(winRate);
            
            double totalPnL = trades.stream()
                .filter(t -> "SELL".equals(t.getAction()))
                .mapToDouble(BacktestTrade::getPnl)
                .sum();
            
            result.setTotalPnL(totalPnL);
        }
        
        // Volatility and Sharpe ratio
        if (dailyValues.size() > 1) {
            List<Double> dailyReturns = new ArrayList<>();
            for (int i = 1; i < dailyValues.size(); i++) {
                double dailyReturn = (dailyValues.get(i) - dailyValues.get(i - 1)) / dailyValues.get(i - 1);
                dailyReturns.add(dailyReturn);
            }
            
            double avgDailyReturn = dailyReturns.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            double volatility = calculateStandardDeviation(dailyReturns);
            
            result.setVolatility(volatility);
            
            // Sharpe ratio (assuming 2% risk-free rate)
            double riskFreeRate = 0.02 / 252; // Daily risk-free rate
            double sharpeRatio = volatility > 0 ? (avgDailyReturn - riskFreeRate) / volatility * Math.sqrt(252) : 0;
            result.setSharpeRatio(sharpeRatio);
            
            // Maximum drawdown
            double maxDrawdown = calculateMaxDrawdown(dailyValues);
            result.setMaxDrawdown(maxDrawdown);
        }
        
        logger.info(String.format("Performance metrics calculated: Total Return: %.2f%%, Sharpe: %.2f, Max DD: %.2f%%",
            totalReturn * 100, result.getSharpeRatio(), result.getMaxDrawdown() * 100));
    }
    
    // Helper methods for technical indicators
    private double calculateSMA(List<Double> prices, int period) {
        if (prices.size() < period) return 0;
        return prices.subList(prices.size() - period, prices.size())
            .stream().mapToDouble(Double::doubleValue).average().orElse(0);
    }
    
    private double calculateRSI(List<Double> prices, int period) {
        if (prices.size() < period + 1) return 50;
        
        double totalGain = 0, totalLoss = 0;
        for (int i = prices.size() - period; i < prices.size(); i++) {
            double change = prices.get(i) - prices.get(i - 1);
            if (change > 0) totalGain += change;
            else totalLoss -= change;
        }
        
        double avgGain = totalGain / period;
        double avgLoss = totalLoss / period;
        
        if (avgLoss == 0) return 100;
        return 100 - (100 / (1 + avgGain / avgLoss));
    }
    
    private double calculateMACD(List<Double> prices) {
        if (prices.size() < 26) return 0;
        double ema12 = calculateEMA(prices, 12);
        double ema26 = calculateEMA(prices, 26);
        return ema12 - ema26;
    }
    
    private double calculateEMA(List<Double> prices, int period) {
        if (prices.isEmpty()) return 0;
        double multiplier = 2.0 / (period + 1);
        double ema = prices.get(Math.max(0, prices.size() - period));
        
        for (int i = Math.max(1, prices.size() - period + 1); i < prices.size(); i++) {
            ema = (prices.get(i) * multiplier) + (ema * (1 - multiplier));
        }
        return ema;
    }
    
    private double calculateATR(List<KlineData> data) {
        if (data.size() < 2) return 0;
        
        double atr = 0;
        int count = Math.min(14, data.size() - 1);
        
        for (int i = data.size() - count; i < data.size(); i++) {
            KlineData current = data.get(i);
            KlineData previous = data.get(i - 1);
            
            double tr1 = current.getHigh() - current.getLow();
            double tr2 = Math.abs(current.getHigh() - previous.getClose());
            double tr3 = Math.abs(current.getLow() - previous.getClose());
            
            atr += Math.max(tr1, Math.max(tr2, tr3));
        }
        
        return atr / count;
    }
    
    private double calculateVolatility(List<Double> prices, int period) {
        if (prices.size() < period) return 0;
        
        List<Double> returns = new ArrayList<>();
        for (int i = prices.size() - period + 1; i < prices.size(); i++) {
            double ret = (prices.get(i) - prices.get(i - 1)) / prices.get(i - 1);
            returns.add(ret);
        }
        
        return calculateStandardDeviation(returns);
    }
    
    private double calculatePricePosition(List<Double> prices, int period) {
        if (prices.size() < period) return 0.5;
        
        List<Double> recentPrices = prices.subList(prices.size() - period, prices.size());
        double high = recentPrices.stream().mapToDouble(Double::doubleValue).max().orElse(0);
        double low = recentPrices.stream().mapToDouble(Double::doubleValue).min().orElse(0);
        double current = prices.get(prices.size() - 1);
        
        return (high - low) == 0 ? 0.5 : (current - low) / (high - low);
    }
    
    private double calculateStandardDeviation(List<Double> values) {
        if (values.isEmpty()) return 0;
        
        double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double variance = values.stream()
            .mapToDouble(v -> Math.pow(v - mean, 2))
            .average().orElse(0);
        
        return Math.sqrt(variance);
    }
    
    private double calculateMaxDrawdown(List<Double> portfolioValues) {
        double maxDrawdown = 0;
        double peak = portfolioValues.get(0);
        
        for (double value : portfolioValues) {
            if (value > peak) {
                peak = value;
            }
            double drawdown = (peak - value) / peak;
            maxDrawdown = Math.max(maxDrawdown, drawdown);
        }
        
        return maxDrawdown;
    }
    
    // Result classes
    public static class BacktestResult {
        private LocalDateTime startDate;
        private LocalDateTime endDate;
        private List<String> symbols;
        private double initialCapital;
        private double finalCapital;
        private double totalReturn;
        private double annualizedReturn;
        private double sharpeRatio;
        private double maxDrawdown;
        private double volatility;
        private double winRate;
        private double totalPnL;
        private List<BacktestTrade> trades;
        private List<Double> dailyPortfolioValues;
        private String error;
        
        // Getters and setters
        public LocalDateTime getStartDate() { return startDate; }
        public void setStartDate(LocalDateTime startDate) { this.startDate = startDate; }
        
        public LocalDateTime getEndDate() { return endDate; }
        public void setEndDate(LocalDateTime endDate) { this.endDate = endDate; }
        
        public List<String> getSymbols() { return symbols; }
        public void setSymbols(List<String> symbols) { this.symbols = symbols; }
        
        public double getInitialCapital() { return initialCapital; }
        public void setInitialCapital(double initialCapital) { this.initialCapital = initialCapital; }
        
        public double getFinalCapital() { return finalCapital; }
        public void setFinalCapital(double finalCapital) { this.finalCapital = finalCapital; }
        
        public double getTotalReturn() { return totalReturn; }
        public void setTotalReturn(double totalReturn) { this.totalReturn = totalReturn; }
        
        public double getAnnualizedReturn() { return annualizedReturn; }
        public void setAnnualizedReturn(double annualizedReturn) { this.annualizedReturn = annualizedReturn; }
        
        public double getSharpeRatio() { return sharpeRatio; }
        public void setSharpeRatio(double sharpeRatio) { this.sharpeRatio = sharpeRatio; }
        
        public double getMaxDrawdown() { return maxDrawdown; }
        public void setMaxDrawdown(double maxDrawdown) { this.maxDrawdown = maxDrawdown; }
        
        public double getVolatility() { return volatility; }
        public void setVolatility(double volatility) { this.volatility = volatility; }
        
        public double getWinRate() { return winRate; }
        public void setWinRate(double winRate) { this.winRate = winRate; }
        
        public double getTotalPnL() { return totalPnL; }
        public void setTotalPnL(double totalPnL) { this.totalPnL = totalPnL; }
        
        public List<BacktestTrade> getTrades() { return trades; }
        public void setTrades(List<BacktestTrade> trades) { this.trades = trades; }
        
        public List<Double> getDailyPortfolioValues() { return dailyPortfolioValues; }
        public void setDailyPortfolioValues(List<Double> dailyPortfolioValues) { 
            this.dailyPortfolioValues = dailyPortfolioValues; 
        }
        
        public String getError() { return error; }
        public void setError(String error) { this.error = error; }
    }
    
    public static class BacktestTrade {
        private String symbol;
        private String action;
        private double price;
        private double shares;
        private double tradeValue;
        private double pnl;
        private double confidence;
        private String reason;
        private LocalDateTime timestamp;
        
        // Getters and setters
        public String getSymbol() { return symbol; }
        public void setSymbol(String symbol) { this.symbol = symbol; }
        
        public String getAction() { return action; }
        public void setAction(String action) { this.action = action; }
        
        public double getPrice() { return price; }
        public void setPrice(double price) { this.price = price; }
        
        public double getShares() { return shares; }
        public void setShares(double shares) { this.shares = shares; }
        
        public double getTradeValue() { return tradeValue; }
        public void setTradeValue(double tradeValue) { this.tradeValue = tradeValue; }
        
        public double getPnl() { return pnl; }
        public void setPnl(double pnl) { this.pnl = pnl; }
        
        public double getConfidence() { return confidence; }
        public void setConfidence(double confidence) { this.confidence = confidence; }
        
        public String getReason() { return reason; }
        public void setReason(String reason) { this.reason = reason; }
        
        public LocalDateTime getTimestamp() { return timestamp; }
        public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
    }
}
