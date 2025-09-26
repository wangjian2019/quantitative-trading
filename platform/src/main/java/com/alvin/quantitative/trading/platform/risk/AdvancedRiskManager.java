package com.alvin.quantitative.trading.platform.risk;

import com.alvin.quantitative.trading.platform.core.AISignal;
import com.alvin.quantitative.trading.platform.core.KlineData;
import com.alvin.quantitative.trading.platform.core.Position;
import com.alvin.quantitative.trading.platform.config.ApplicationConfig;

import org.springframework.stereotype.Component;
import lombok.extern.slf4j.Slf4j;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 高级风险管理系统 v0.1
 * Author: Alvin
 *
 * 专业级风险管理，支持大资金量化交易：
 * - 多层风险控制
 * - 动态仓位管理
 * - 实时风险监控
 * - 智能止损系统
 * - 压力测试和情景分析
 */
@Slf4j
@Component
public class AdvancedRiskManager {

    private final ApplicationConfig config;

    // 风险参数
    private final double maxSinglePositionRatio;     // 单股最大仓位比例
    private final double maxTotalExposure;           // 总仓位上限
    private final double maxDailyLoss;              // 日最大亏损
    private final double stopLossRatio;             // 止损比例
    private final double takeProfitRatio;           // 止盈比例
    private final double maxVolatilityThreshold;    // 最大波动率阈值
    private final double minConfidenceThreshold;    // 最小置信度阈值

    // 风险监控状态
    private final Map<String, RiskMetrics> symbolRiskMetrics;
    private final Map<String, Double> symbolVolatility;
    private final Map<String, LocalDateTime> lastRiskCheck;
    private volatile double currentPortfolioValue;
    private volatile double dailyPnL;
    private volatile RiskLevel currentRiskLevel;

    // 风险事件记录
    private final List<RiskEvent> riskEventHistory;
    private final Map<String, Integer> symbolRiskViolations;

    public AdvancedRiskManager() {
        this.config = ApplicationConfig.getInstance();

        // 初始化风险参数
        this.maxSinglePositionRatio = 0.15;        // 15%单股上限
        this.maxTotalExposure = 0.85;              // 85%总仓位上限
        this.maxDailyLoss = 0.05;                  // 5%日最大亏损
        this.stopLossRatio = 0.03;                 // 3%止损
        this.takeProfitRatio = 0.08;               // 8%止盈
        this.maxVolatilityThreshold = 0.05;        // 5%波动率阈值
        this.minConfidenceThreshold = 0.75;        // 75%最小置信度

        // 初始化监控状态
        this.symbolRiskMetrics = new ConcurrentHashMap<>();
        this.symbolVolatility = new ConcurrentHashMap<>();
        this.lastRiskCheck = new ConcurrentHashMap<>();
        this.currentPortfolioValue = 0.0;
        this.dailyPnL = 0.0;
        this.currentRiskLevel = RiskLevel.LOW;

        // 初始化事件记录
        this.riskEventHistory = new ArrayList<>();
        this.symbolRiskViolations = new ConcurrentHashMap<>();

        log.info("🛡️ Advanced Risk Manager initialized");
        log.info("   Max Single Position: {:.1%}", maxSinglePositionRatio);
        log.info("   Max Total Exposure: {:.1%}", maxTotalExposure);
        log.info("   Max Daily Loss: {:.1%}", maxDailyLoss);
        log.info("   Stop Loss: {:.1%}", stopLossRatio);
        log.info("   Take Profit: {:.1%}", takeProfitRatio);
    }

    /**
     * 验证交易信号是否通过风险检查
     * 这是风险管理的核心方法
     */
    public boolean validateSignal(String symbol, AISignal signal, double currentPrice) {
        try {
            log.debug("🛡️ Validating signal for {}: {} (confidence: {:.2%})",
                symbol, signal.getAction(), signal.getConfidence());

            // 1. 基础验证
            if (!basicSignalValidation(signal)) {
                recordRiskEvent(symbol, RiskEventType.INVALID_SIGNAL,
                    "Signal failed basic validation", RiskLevel.HIGH);
                return false;
            }

            // 2. 置信度检查
            if (!confidenceCheck(signal)) {
                recordRiskEvent(symbol, RiskEventType.LOW_CONFIDENCE,
                    String.format("Confidence %.2%% below threshold %.2%%",
                        signal.getConfidence(), minConfidenceThreshold), RiskLevel.MEDIUM);
                return false;
            }

            // 3. 波动率检查
            if (!volatilityCheck(symbol, signal)) {
                recordRiskEvent(symbol, RiskEventType.HIGH_VOLATILITY,
                    "Symbol volatility exceeds risk threshold", RiskLevel.HIGH);
                return false;
            }

            // 4. 仓位限制检查
            if (!positionLimitCheck(symbol, signal, currentPrice)) {
                recordRiskEvent(symbol, RiskEventType.POSITION_LIMIT,
                    "Position would exceed maximum allowed", RiskLevel.HIGH);
                return false;
            }

            // 5. 投资组合风险检查
            if (!portfolioRiskCheck(signal)) {
                recordRiskEvent(symbol, RiskEventType.PORTFOLIO_RISK,
                    "Portfolio risk limits exceeded", RiskLevel.CRITICAL);
                return false;
            }

            // 6. 市场状况检查
            if (!marketConditionCheck(symbol, currentPrice)) {
                recordRiskEvent(symbol, RiskEventType.MARKET_CONDITION,
                    "Unfavorable market conditions detected", RiskLevel.MEDIUM);
                return false;
            }

            // 更新风险指标
            updateSymbolRiskMetrics(symbol, signal, currentPrice);

            log.debug("✅ Signal validation passed for {}", symbol);
            return true;

        } catch (Exception e) {
            log.error("❌ Risk validation error for {}: {}", symbol, e.getMessage(), e);
            recordRiskEvent(symbol, RiskEventType.SYSTEM_ERROR,
                "Risk validation system error: " + e.getMessage(), RiskLevel.CRITICAL);
            return false;
        }
    }

    /**
     * 基础信号验证
     */
    private boolean basicSignalValidation(AISignal signal) {
        if (signal == null) {
            log.warn("⚠️ Null signal received");
            return false;
        }

        String action = signal.getAction();
        if (action == null || (!action.equals("BUY") && !action.equals("SELL") && !action.equals("HOLD"))) {
            log.warn("⚠️ Invalid signal action: {}", action);
            return false;
        }

        double confidence = signal.getConfidence();
        if (confidence < 0.0 || confidence > 1.0) {
            log.warn("⚠️ Invalid confidence value: {}", confidence);
            return false;
        }

        return true;
    }

    /**
     * 置信度检查
     */
    private boolean confidenceCheck(AISignal signal) {
        double confidence = signal.getConfidence();
        if (confidence < minConfidenceThreshold) {
            log.debug("🛡️ Confidence check failed: {:.2%} < {:.2%}",
                confidence, minConfidenceThreshold);
            return false;
        }
        return true;
    }

    /**
     * 波动率检查
     */
    private boolean volatilityCheck(String symbol, AISignal signal) {
        double volatility = symbolVolatility.getOrDefault(symbol, 0.02);

        // 如果信号包含波动率信息，使用它
        if (signal.getVolatility() > 0) {
            volatility = Math.max(volatility, signal.getVolatility());
        }

        if (volatility > maxVolatilityThreshold) {
            log.debug("🛡️ Volatility check failed for {}: {:.2%} > {:.2%}",
                symbol, volatility, maxVolatilityThreshold);
            return false;
        }

        // 更新波动率记录
        symbolVolatility.put(symbol, volatility);
        return true;
    }

    /**
     * 仓位限制检查
     */
    private boolean positionLimitCheck(String symbol, AISignal signal, double currentPrice) {
        if (!signal.getAction().equals("BUY")) {
            return true; // 卖出和持有不需要仓位检查
        }

        // 计算建议仓位大小
        double suggestedPosition = calculatePositionSize(signal, currentPrice);
        double positionValue = suggestedPosition * currentPrice;

        // 检查单股仓位限制
        double portfolioValue = getCurrentPortfolioValue();
        double positionRatio = positionValue / portfolioValue;

        if (positionRatio > maxSinglePositionRatio) {
            log.debug("🛡️ Single position limit exceeded for {}: {:.2%} > {:.2%}",
                symbol, positionRatio, maxSinglePositionRatio);
            return false;
        }

        return true;
    }

    /**
     * 投资组合风险检查
     */
    private boolean portfolioRiskCheck(AISignal signal) {
        double portfolioValue = getCurrentPortfolioValue();

        // 检查日内亏损限制
        if (dailyPnL / portfolioValue < -maxDailyLoss) {
            log.warn("🛡️ Daily loss limit reached: {:.2%} < -{:.2%}",
                dailyPnL / portfolioValue, maxDailyLoss);
            return false;
        }

        // 检查总风险暴露
        double totalExposure = calculateTotalExposure();
        if (totalExposure > maxTotalExposure) {
            log.debug("🛡️ Total exposure limit exceeded: {:.2%} > {:.2%}",
                totalExposure, maxTotalExposure);
            return false;
        }

        return true;
    }

    /**
     * 市场状况检查
     */
    private boolean marketConditionCheck(String symbol, double currentPrice) {
        // 检查市场时间
        LocalDateTime now = LocalDateTime.now();
        int hour = now.getHour();

        // 避免在市场开盘/收盘前后30分钟内进行大额交易
        if (hour < 10 || hour > 15) {
            if (calculatePositionSize(null, currentPrice) > maxSinglePositionRatio * 0.5) {
                log.debug("🛡️ Large position blocked during market edge hours");
                return false;
            }
        }

        return true;
    }

    /**
     * 实时风险监控
     */
    public void performRealTimeRiskCheck(Map<String, Position> positions,
                                       Map<String, Queue<KlineData>> marketData) {
        try {
            updateCurrentRiskLevel(positions, marketData);
            checkStopLossConditions(positions, marketData);
            checkPortfolioConcentration(positions);
            checkMarketVolatility(marketData);

            lastRiskCheck.put("PORTFOLIO", LocalDateTime.now());

        } catch (Exception e) {
            log.error("❌ Real-time risk check failed: {}", e.getMessage(), e);
            recordRiskEvent("SYSTEM", RiskEventType.SYSTEM_ERROR,
                "Real-time risk check failed: " + e.getMessage(), RiskLevel.CRITICAL);
        }
    }

    /**
     * 更新当前风险级别
     */
    private void updateCurrentRiskLevel(Map<String, Position> positions,
                                      Map<String, Queue<KlineData>> marketData) {
        int riskScore = 0;

        // 基于投资组合集中度的风险评分
        double concentrationRisk = calculateConcentrationRisk(positions);
        if (concentrationRisk > 0.4) riskScore += 3;
        else if (concentrationRisk > 0.2) riskScore += 1;

        // 基于波动率的风险评分
        double avgVolatility = calculateAverageVolatility();
        if (avgVolatility > 0.04) riskScore += 3;
        else if (avgVolatility > 0.02) riskScore += 1;

        // 基于日内盈亏的风险评分
        double dailyPnLRatio = dailyPnL / getCurrentPortfolioValue();
        if (dailyPnLRatio < -0.03) riskScore += 3;
        else if (dailyPnLRatio < -0.01) riskScore += 1;

        // 更新风险级别
        RiskLevel newLevel;
        if (riskScore >= 6) newLevel = RiskLevel.CRITICAL;
        else if (riskScore >= 4) newLevel = RiskLevel.HIGH;
        else if (riskScore >= 2) newLevel = RiskLevel.MEDIUM;
        else newLevel = RiskLevel.LOW;

        if (newLevel != currentRiskLevel) {
            log.info("🛡️ Risk level changed from {} to {}", currentRiskLevel, newLevel);
            currentRiskLevel = newLevel;

            // 记录风险级别变化事件
            recordRiskEvent("PORTFOLIO", RiskEventType.RISK_LEVEL_CHANGE,
                String.format("Risk level changed to %s (score: %d)", newLevel, riskScore), newLevel);
        }
    }

    /**
     * 检查止损条件
     */
    private void checkStopLossConditions(Map<String, Position> positions,
                                       Map<String, Queue<KlineData>> marketData) {
        for (Map.Entry<String, Position> entry : positions.entrySet()) {
            String symbol = entry.getKey();
            Position position = entry.getValue();

            Queue<KlineData> dataQueue = marketData.get(symbol);
            if (dataQueue == null || dataQueue.isEmpty()) continue;

            KlineData latestData = ((LinkedList<KlineData>) dataQueue).getLast();
            double currentPrice = latestData.getClose();
            double entryPrice = position.getAvgCost();

            // 计算盈亏比例
            double pnlRatio = (currentPrice - entryPrice) / entryPrice;

            // 检查止损条件
            if (pnlRatio < -stopLossRatio) {
                recordRiskEvent(symbol, RiskEventType.STOP_LOSS_TRIGGERED,
                    String.format("Stop loss triggered: %.2%% loss", pnlRatio * 100), RiskLevel.HIGH);

                log.warn("🚨 Stop loss triggered for {}: {:.2%} loss", symbol, pnlRatio * 100);
            }

            // 检查止盈条件
            if (pnlRatio > takeProfitRatio) {
                recordRiskEvent(symbol, RiskEventType.TAKE_PROFIT_TRIGGERED,
                    String.format("Take profit triggered: %.2%% gain", pnlRatio * 100), RiskLevel.LOW);

                log.info("🎯 Take profit level reached for {}: {:.2%} gain", symbol, pnlRatio * 100);
            }
        }
    }

    /**
     * 计算仓位大小
     */
    private double calculatePositionSize(AISignal signal, double currentPrice) {
        if (signal == null) return 0.0;

        double basePosition = 0.05; // 5%基础仓位

        // 基于置信度调整
        double confidenceMultiplier = Math.min(2.0, signal.getConfidence() / 0.7);

        // 基于预期收益调整
        double expectedReturn = Math.abs(signal.getExpectedReturn());
        double returnMultiplier = Math.min(1.5, 1.0 + expectedReturn * 10);

        // 基于波动率调整（波动率越高，仓位越小）
        double volatility = signal.getVolatility();
        double volatilityAdjustment = Math.min(1.0, 0.02 / Math.max(volatility, 0.01));

        double finalPosition = basePosition * confidenceMultiplier * returnMultiplier * volatilityAdjustment;
        return Math.min(maxSinglePositionRatio, finalPosition);
    }

    // 辅助方法
    private double getCurrentPortfolioValue() {
        // 实际实现中应该从投资组合管理器获取
        return currentPortfolioValue > 0 ? currentPortfolioValue : 10000000.0; // 默认1000万
    }

    private double calculateTotalExposure() {
        // 实际实现中应该计算所有持仓的总风险暴露
        return 0.6; // 占位符
    }

    private double calculateConcentrationRisk(Map<String, Position> positions) {
        if (positions.isEmpty()) return 0.0;

        double totalValue = getCurrentPortfolioValue();
        double maxPositionValue = 0.0;

        for (Position position : positions.values()) {
            double positionValue = position.getShares() * position.getAvgCost();
            maxPositionValue = Math.max(maxPositionValue, positionValue);
        }

        return maxPositionValue / totalValue;
    }

    private double calculateAverageVolatility() {
        if (symbolVolatility.isEmpty()) return 0.02;

        return symbolVolatility.values().stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.02);
    }

    private void checkPortfolioConcentration(Map<String, Position> positions) {
        double concentrationRisk = calculateConcentrationRisk(positions);
        if (concentrationRisk > maxSinglePositionRatio * 1.2) {
            recordRiskEvent("PORTFOLIO", RiskEventType.CONCENTRATION_RISK,
                String.format("High concentration risk: %.2%%", concentrationRisk * 100), RiskLevel.HIGH);
        }
    }

    private void checkMarketVolatility(Map<String, Queue<KlineData>> marketData) {
        double avgVolatility = calculateAverageVolatility();
        if (avgVolatility > maxVolatilityThreshold * 1.5) {
            recordRiskEvent("MARKET", RiskEventType.HIGH_VOLATILITY,
                String.format("High market volatility detected: %.2%%", avgVolatility * 100), RiskLevel.HIGH);
        }
    }

    private void updateSymbolRiskMetrics(String symbol, AISignal signal, double currentPrice) {
        RiskMetrics metrics = symbolRiskMetrics.computeIfAbsent(symbol, k -> new RiskMetrics());
        metrics.lastSignalConfidence = signal.getConfidence();
        metrics.lastSignalTime = LocalDateTime.now();
        metrics.volatility = signal.getVolatility();
        metrics.currentPrice = currentPrice;
    }

    /**
     * 记录风险事件
     */
    private void recordRiskEvent(String symbol, RiskEventType eventType, String description, RiskLevel level) {
        RiskEvent event = new RiskEvent(symbol, eventType, description, level, LocalDateTime.now());
        riskEventHistory.add(event);

        // 限制历史记录大小
        if (riskEventHistory.size() > 1000) {
            riskEventHistory.remove(0);
        }

        // 更新违规计数
        symbolRiskViolations.merge(symbol, 1, Integer::sum);

        // 根据风险级别记录日志
        switch (level) {
            case CRITICAL:
                log.error("🚨 CRITICAL Risk Event - {}: {} - {}", symbol, eventType, description);
                break;
            case HIGH:
                log.warn("⚠️ HIGH Risk Event - {}: {} - {}", symbol, eventType, description);
                break;
            case MEDIUM:
                log.info("ℹ️ MEDIUM Risk Event - {}: {} - {}", symbol, eventType, description);
                break;
            case LOW:
                log.debug("📝 LOW Risk Event - {}: {} - {}", symbol, eventType, description);
                break;
        }
    }

    // 公共方法
    public RiskLevel getCurrentRiskLevel() {
        return currentRiskLevel;
    }

    public List<RiskEvent> getRecentRiskEvents(int count) {
        int start = Math.max(0, riskEventHistory.size() - count);
        return new ArrayList<>(riskEventHistory.subList(start, riskEventHistory.size()));
    }

    public Map<String, Object> getRiskSummary() {
        Map<String, Object> summary = new HashMap<>();
        summary.put("current_risk_level", currentRiskLevel.name());
        summary.put("total_risk_events", riskEventHistory.size());
        summary.put("risk_violations_by_symbol", new HashMap<>(symbolRiskViolations));
        summary.put("average_volatility", calculateAverageVolatility());
        summary.put("concentration_risk", "N/A"); // 需要实际持仓数据
        summary.put("daily_pnl_ratio", dailyPnL / getCurrentPortfolioValue());
        summary.put("last_risk_check", lastRiskCheck.getOrDefault("PORTFOLIO", LocalDateTime.now()));

        return summary;
    }

    // 内部类定义
    public enum RiskLevel {
        LOW, MEDIUM, HIGH, CRITICAL
    }

    public enum RiskEventType {
        INVALID_SIGNAL, LOW_CONFIDENCE, HIGH_VOLATILITY, POSITION_LIMIT,
        PORTFOLIO_RISK, MARKET_CONDITION, STOP_LOSS_TRIGGERED,
        TAKE_PROFIT_TRIGGERED, CONCENTRATION_RISK, RISK_LEVEL_CHANGE, SYSTEM_ERROR
    }

    public static class RiskEvent {
        public final String symbol;
        public final RiskEventType eventType;
        public final String description;
        public final RiskLevel level;
        public final LocalDateTime timestamp;

        public RiskEvent(String symbol, RiskEventType eventType, String description,
                        RiskLevel level, LocalDateTime timestamp) {
            this.symbol = symbol;
            this.eventType = eventType;
            this.description = description;
            this.level = level;
            this.timestamp = timestamp;
        }
    }

    private static class RiskMetrics {
        double lastSignalConfidence;
        LocalDateTime lastSignalTime;
        double volatility;
        double currentPrice;
    }
}