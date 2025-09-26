package com.alvin.quantitative.trading.platform.portfolio;

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
 * 智能投资组合管理器 v0.1
 * Author: Alvin
 *
 * 专业级投资组合管理，针对大资金量化交易优化：
 * - 动态仓位分配
 * - 智能再平衡
 * - 风险调整收益优化
 * - 多因子模型
 * - 实时性能监控
 */
@Slf4j
@Component
public class IntelligentPortfolioManager {

    private final ApplicationConfig config;

    // 投资组合参数
    private final double maxSinglePositionRatio = 0.20;    // 单股最大20%
    private final double minPositionRatio = 0.02;          // 单股最小2%
    private final double targetVolatility = 0.15;          // 目标波动率15%
    private final double rebalanceThreshold = 0.05;        // 5%偏离触发再平衡

    // 投资组合状态
    private final Map<String, PortfolioWeight> targetWeights;
    private final Map<String, PortfolioWeight> currentWeights;
    private final Map<String, PerformanceMetrics> symbolMetrics;
    private final Map<String, Double> expectedReturns;
    private final Map<String, Double> riskContributions;

    // 性能跟踪
    private volatile double totalPortfolioValue;
    private volatile double totalPnL;
    private volatile double sharpeRatio;
    private volatile LocalDateTime lastRebalance;
    private final List<PerformanceRecord> performanceHistory;

    public IntelligentPortfolioManager() {
        this.config = ApplicationConfig.getInstance();

        // 初始化数据结构
        this.targetWeights = new ConcurrentHashMap<>();
        this.currentWeights = new ConcurrentHashMap<>();
        this.symbolMetrics = new ConcurrentHashMap<>();
        this.expectedReturns = new ConcurrentHashMap<>();
        this.riskContributions = new ConcurrentHashMap<>();
        this.performanceHistory = new ArrayList<>();

        // 初始化状态
        this.totalPortfolioValue = 0.0;
        this.totalPnL = 0.0;
        this.sharpeRatio = 0.0;
        this.lastRebalance = LocalDateTime.now();

        log.info("💼 Intelligent Portfolio Manager initialized");
        log.info("   Max Single Position: {:.1%}", maxSinglePositionRatio);
        log.info("   Min Position Size: {:.1%}", minPositionRatio);
        log.info("   Target Volatility: {:.1%}", targetVolatility);
        log.info("   Rebalance Threshold: {:.1%}", rebalanceThreshold);
    }

    /**
     * 优化交易信号 - 核心方法
     * 基于投资组合理论和风险管理优化信号
     */
    public AISignal optimizeSignal(String symbol, AISignal originalSignal, KlineData currentData) {
        try {
            log.debug("💼 Optimizing signal for {}: {} (confidence: {:.2%})",
                symbol, originalSignal.getAction(), originalSignal.getConfidence());

            // 创建优化后的信号副本
            AISignal optimizedSignal = new AISignal(
                originalSignal.getAction(),
                originalSignal.getConfidence(),
                originalSignal.getReason()
            );

            // 设置基本属性
            optimizedSignal.setExpectedReturn(originalSignal.getExpectedReturn());
            optimizedSignal.setVolatility(originalSignal.getVolatility());
            optimizedSignal.setModelType(originalSignal.getModelType());

            // 1. 计算优化后的仓位大小
            double optimizedPosition = calculateOptimalPosition(symbol, originalSignal, currentData);
            optimizedSignal.setSuggestedPositionSize(optimizedPosition);

            // 2. 调整信号强度（基于投资组合上下文）
            double adjustedConfidence = adjustConfidenceForPortfolio(symbol, originalSignal);
            optimizedSignal.setConfidence(adjustedConfidence);

            // 3. 计算风险调整后的预期收益
            double riskAdjustedReturn = calculateRiskAdjustedReturn(symbol, originalSignal);
            optimizedSignal.setExpectedReturn(riskAdjustedReturn);

            // 4. 生成投资组合优化的交易理由
            String optimizedReason = generateOptimizedReason(symbol, originalSignal, optimizedSignal);
            optimizedSignal.setReason(optimizedReason);

            // 5. 更新投资组合状态
            updatePortfolioMetrics(symbol, optimizedSignal, currentData);

            log.debug("✅ Signal optimized for {}: position {:.1%}, confidence {:.2%}",
                symbol, optimizedPosition, adjustedConfidence);

            return optimizedSignal;

        } catch (Exception e) {
            log.error("❌ Signal optimization failed for {}: {}", symbol, e.getMessage(), e);
            return originalSignal; // 返回原始信号作为回退
        }
    }

    /**
     * 计算最优仓位大小
     * 使用现代投资组合理论和Kelly公式
     */
    public double calculatePositionSize(String symbol, AISignal signal, double currentPrice) {
        try {
            if (!signal.getAction().equals("BUY")) {
                return 0.0; // 非买入信号不分配仓位
            }

            // 1. 基础仓位计算（Kelly公式的简化版本）
            double expectedReturn = Math.abs(signal.getExpectedReturn());
            double confidence = signal.getConfidence();
            double winRate = confidence; // 将置信度作为胜率代理
            double avgWin = expectedReturn;
            double avgLoss = Math.min(0.05, expectedReturn * 0.5); // 假设平均损失

            // Kelly公式: f = (bp - q) / b，其中b是赔率，p是胜率，q是败率
            double kellyFraction = (winRate * avgWin - (1 - winRate) * avgLoss) / avgWin;
            kellyFraction = Math.max(0, Math.min(0.25, kellyFraction)); // 限制在0-25%之间

            // 2. 基于波动率的调整
            double volatility = Math.max(signal.getVolatility(), 0.01);
            double volatilityAdjustment = targetVolatility / volatility;
            volatilityAdjustment = Math.min(2.0, Math.max(0.2, volatilityAdjustment));

            // 3. 基于相关性的调整（简化实现）
            double correlationAdjustment = calculateCorrelationAdjustment(symbol);

            // 4. 基于当前投资组合集中度的调整
            double concentrationAdjustment = calculateConcentrationAdjustment(symbol);

            // 5. 综合计算最终仓位
            double basePosition = kellyFraction * volatilityAdjustment * correlationAdjustment * concentrationAdjustment;

            // 6. 应用最大/最小仓位限制
            double finalPosition = Math.max(minPositionRatio, Math.min(maxSinglePositionRatio, basePosition));

            log.debug("💼 Position calculation for {}: Kelly={:.3f}, Vol Adj={:.3f}, Final={:.3f}",
                symbol, kellyFraction, volatilityAdjustment, finalPosition);

            return finalPosition;

        } catch (Exception e) {
            log.error("❌ Position size calculation failed for {}: {}", symbol, e.getMessage());
            return minPositionRatio; // 返回最小仓位作为回退
        }
    }

    private double calculateOptimalPosition(String symbol, AISignal signal, KlineData currentData) {
        return calculatePositionSize(symbol, signal, currentData.getClose());
    }

    /**
     * 基于投资组合上下文调整信号置信度
     */
    private double adjustConfidenceForPortfolio(String symbol, AISignal signal) {
        double originalConfidence = signal.getConfidence();

        // 1. 基于历史表现调整
        PerformanceMetrics metrics = symbolMetrics.get(symbol);
        double performanceAdjustment = 1.0;
        if (metrics != null && metrics.totalTrades > 5) {
            double successRate = metrics.successfulTrades / (double) metrics.totalTrades;
            performanceAdjustment = Math.min(1.2, Math.max(0.8, successRate / 0.6));
        }

        // 2. 基于投资组合分散化需求调整
        double diversificationAdjustment = calculateDiversificationAdjustment(symbol);

        // 3. 基于市场制度调整
        double regimeAdjustment = calculateMarketRegimeAdjustment();

        double adjustedConfidence = originalConfidence * performanceAdjustment *
                                  diversificationAdjustment * regimeAdjustment;

        return Math.min(1.0, Math.max(0.0, adjustedConfidence));
    }

    /**
     * 计算风险调整后收益
     */
    private double calculateRiskAdjustedReturn(String symbol, AISignal signal) {
        double expectedReturn = signal.getExpectedReturn();
        double volatility = Math.max(signal.getVolatility(), 0.01);

        // 使用夏普比率概念进行风险调整
        double riskFreeRate = 0.02; // 2%无风险利率
        double excessReturn = expectedReturn - riskFreeRate;
        double riskAdjustedReturn = excessReturn / volatility;

        // 转换回收益率，并限制在合理范围内
        return Math.min(0.20, Math.max(-0.10, riskAdjustedReturn * volatility + riskFreeRate));
    }

    /**
     * 生成优化后的交易理由
     */
    private String generateOptimizedReason(String symbol, AISignal originalSignal, AISignal optimizedSignal) {
        StringBuilder reason = new StringBuilder();
        reason.append(originalSignal.getReason());

        // 添加投资组合优化信息
        if (Math.abs(optimizedSignal.getConfidence() - originalSignal.getConfidence()) > 0.05) {
            reason.append(" | 📊 投资组合优化调整");
        }

        if (optimizedSignal.getSuggestedPositionSize() > 0) {
            reason.append(String.format(" | 💼 建议仓位: %.1f%%",
                optimizedSignal.getSuggestedPositionSize() * 100));
        }

        // 添加风险信息
        PerformanceMetrics metrics = symbolMetrics.get(symbol);
        if (metrics != null && metrics.totalTrades > 0) {
            double successRate = metrics.successfulTrades / (double) metrics.totalTrades;
            reason.append(String.format(" | 🎯 历史胜率: %.1f%%", successRate * 100));
        }

        return reason.toString();
    }

    /**
     * 更新投资组合指标
     */
    private void updatePortfolioMetrics(String symbol, AISignal signal, KlineData currentData) {
        // 更新预期收益
        expectedReturns.put(symbol, signal.getExpectedReturn());

        // 更新目标权重（基于信号强度和风险）
        double targetWeight = calculateTargetWeight(symbol, signal);
        targetWeights.put(symbol, new PortfolioWeight(targetWeight, LocalDateTime.now()));

        // 更新风险贡献
        double riskContribution = calculateRiskContribution(symbol, signal);
        riskContributions.put(symbol, riskContribution);
    }

    /**
     * 投资组合再平衡检查
     */
    public boolean shouldRebalance() {
        try {
            // 检查是否需要再平衡
            double maxDeviation = 0.0;

            for (Map.Entry<String, PortfolioWeight> entry : targetWeights.entrySet()) {
                String symbol = entry.getKey();
                double targetWeight = entry.getValue().weight;
                double currentWeight = getCurrentWeight(symbol);

                double deviation = Math.abs(targetWeight - currentWeight);
                maxDeviation = Math.max(maxDeviation, deviation);
            }

            boolean shouldRebalance = maxDeviation > rebalanceThreshold;

            if (shouldRebalance) {
                log.info("⚖️ Rebalancing required: max deviation {:.2%} > threshold {:.2%}",
                    maxDeviation, rebalanceThreshold);
            }

            return shouldRebalance;

        } catch (Exception e) {
            log.error("❌ Rebalance check failed: {}", e.getMessage());
            return false;
        }
    }

    /**
     * 执行投资组合再平衡
     */
    public Map<String, Double> calculateRebalancingTrades() {
        Map<String, Double> rebalancingTrades = new HashMap<>();

        try {
            for (Map.Entry<String, PortfolioWeight> entry : targetWeights.entrySet()) {
                String symbol = entry.getKey();
                double targetWeight = entry.getValue().weight;
                double currentWeight = getCurrentWeight(symbol);

                double weightDifference = targetWeight - currentWeight;
                if (Math.abs(weightDifference) > rebalanceThreshold) {
                    rebalancingTrades.put(symbol, weightDifference);
                }
            }

            if (!rebalancingTrades.isEmpty()) {
                lastRebalance = LocalDateTime.now();
                log.info("⚖️ Calculated rebalancing trades for {} symbols", rebalancingTrades.size());
            }

        } catch (Exception e) {
            log.error("❌ Rebalancing calculation failed: {}", e.getMessage());
        }

        return rebalancingTrades;
    }

    // 辅助方法
    private double calculateCorrelationAdjustment(String symbol) {
        // 简化实现：基于资产类别的相关性调整
        // 实际实现中应该使用历史收益率计算相关性矩阵
        return 1.0;
    }

    private double calculateConcentrationAdjustment(String symbol) {
        double currentWeight = getCurrentWeight(symbol);
        if (currentWeight > maxSinglePositionRatio * 0.8) {
            return 0.5; // 如果接近最大仓位，减少新增仓位
        }
        return 1.0;
    }

    private double calculateDiversificationAdjustment(String symbol) {
        // 基于当前投资组合的分散化程度调整
        int totalPositions = currentWeights.size();
        if (totalPositions < 5) {
            return 1.2; // 鼓励分散化
        } else if (totalPositions > 15) {
            return 0.8; // 避免过度分散
        }
        return 1.0;
    }

    private double calculateMarketRegimeAdjustment() {
        // 基于市场制度的调整（简化实现）
        // 实际实现中应该基于VIX、市场趋势等指标
        return 1.0;
    }

    private double calculateTargetWeight(String symbol, AISignal signal) {
        double suggestedPosition = signal.getSuggestedPositionSize();
        return Math.min(maxSinglePositionRatio, Math.max(minPositionRatio, suggestedPosition));
    }

    private double calculateRiskContribution(String symbol, AISignal signal) {
        double volatility = Math.max(signal.getVolatility(), 0.01);
        double weight = calculateTargetWeight(symbol, signal);
        return weight * volatility; // 简化的风险贡献计算
    }

    private double getCurrentWeight(String symbol) {
        PortfolioWeight weight = currentWeights.get(symbol);
        return weight != null ? weight.weight : 0.0;
    }

    /**
     * 计算投资组合性能指标
     */
    public Map<String, Object> calculatePerformanceMetrics() {
        Map<String, Object> metrics = new HashMap<>();

        try {
            // 基本指标
            metrics.put("total_portfolio_value", totalPortfolioValue);
            metrics.put("total_pnl", totalPnL);
            metrics.put("total_return", totalPortfolioValue > 0 ? totalPnL / totalPortfolioValue : 0.0);

            // 风险指标
            double portfolioVolatility = calculatePortfolioVolatility();
            metrics.put("portfolio_volatility", portfolioVolatility);
            metrics.put("sharpe_ratio", calculateSharpeRatio());

            // 分散化指标
            metrics.put("position_count", currentWeights.size());
            metrics.put("max_position_weight", currentWeights.values().stream()
                .mapToDouble(w -> w.weight)
                .max().orElse(0.0));

            // 最后再平衡时间
            metrics.put("last_rebalance", lastRebalance);
            metrics.put("needs_rebalancing", shouldRebalance());

            // 各股票权重
            Map<String, Double> weights = new HashMap<>();
            currentWeights.forEach((symbol, weight) -> weights.put(symbol, weight.weight));
            metrics.put("current_weights", weights);

        } catch (Exception e) {
            log.error("❌ Performance metrics calculation failed: {}", e.getMessage());
        }

        return metrics;
    }

    private double calculatePortfolioVolatility() {
        // 简化实现：计算加权平均波动率
        double weightedVolatility = 0.0;
        double totalWeight = 0.0;

        for (Map.Entry<String, Double> entry : riskContributions.entrySet()) {
            String symbol = entry.getKey();
            double riskContrib = entry.getValue();
            double weight = getCurrentWeight(symbol);

            weightedVolatility += weight * riskContrib;
            totalWeight += weight;
        }

        return totalWeight > 0 ? weightedVolatility / totalWeight : 0.02;
    }

    private double calculateSharpeRatio() {
        double totalReturn = totalPortfolioValue > 0 ? totalPnL / totalPortfolioValue : 0.0;
        double riskFreeRate = 0.02; // 2%
        double excessReturn = totalReturn - riskFreeRate;
        double volatility = calculatePortfolioVolatility();

        return volatility > 0 ? excessReturn / volatility : 0.0;
    }

    // 公共接口方法
    public void updatePortfolioValue(double newValue) {
        this.totalPortfolioValue = newValue;
    }

    public void updatePnL(double pnl) {
        this.totalPnL = pnl;
    }

    public void updateCurrentWeight(String symbol, double weight) {
        currentWeights.put(symbol, new PortfolioWeight(weight, LocalDateTime.now()));
    }

    public Map<String, Double> getTargetWeights() {
        Map<String, Double> weights = new HashMap<>();
        targetWeights.forEach((symbol, weight) -> weights.put(symbol, weight.weight));
        return weights;
    }

    // 内部类
    private static class PortfolioWeight {
        final double weight;
        final LocalDateTime timestamp;

        PortfolioWeight(double weight, LocalDateTime timestamp) {
            this.weight = weight;
            this.timestamp = timestamp;
        }
    }

    private static class PerformanceMetrics {
        int totalTrades = 0;
        int successfulTrades = 0;
        double totalReturn = 0.0;
        double averageReturn = 0.0;
        double maxDrawdown = 0.0;
        LocalDateTime lastUpdate = LocalDateTime.now();
    }

    private static class PerformanceRecord {
        final LocalDateTime timestamp;
        final double portfolioValue;
        final double totalReturn;
        final double sharpeRatio;

        PerformanceRecord(LocalDateTime timestamp, double portfolioValue,
                         double totalReturn, double sharpeRatio) {
            this.timestamp = timestamp;
            this.portfolioValue = portfolioValue;
            this.totalReturn = totalReturn;
            this.sharpeRatio = sharpeRatio;
        }
    }
}