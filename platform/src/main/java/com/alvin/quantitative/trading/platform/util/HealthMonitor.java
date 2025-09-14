package com.alvin.quantitative.trading.platform.util;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

/**
 * Health Monitor
 * Author: Alvin
 * Monitors system health and performance metrics
 */
public class HealthMonitor {
    private static final Logger logger = Logger.getLogger(HealthMonitor.class.getName());
    private static HealthMonitor instance;
    
    // Performance metrics
    private final AtomicLong totalSignalRequests = new AtomicLong(0);
    private final AtomicLong successfulSignals = new AtomicLong(0);
    private final AtomicLong failedSignals = new AtomicLong(0);
    private final AtomicInteger activeThreads = new AtomicInteger(0);
    private final AtomicLong lastHealthCheck = new AtomicLong(System.currentTimeMillis());
    
    // System status
    private volatile boolean aiServiceHealthy = true;
    private volatile boolean dataManagerHealthy = true;
    private volatile boolean riskManagerHealthy = true;
    private volatile String lastError = null;
    private volatile LocalDateTime lastErrorTime = null;
    
    // Trading metrics
    private final AtomicInteger totalTrades = new AtomicInteger(0);
    private final AtomicInteger successfulTrades = new AtomicInteger(0);
    private final AtomicInteger rejectedTrades = new AtomicInteger(0);
    
    private HealthMonitor() {}
    
    public static synchronized HealthMonitor getInstance() {
        if (instance == null) {
            instance = new HealthMonitor();
        }
        return instance;
    }
    
    // Signal metrics
    public void recordSignalRequest() {
        totalSignalRequests.incrementAndGet();
    }
    
    public void recordSuccessfulSignal() {
        successfulSignals.incrementAndGet();
    }
    
    public void recordFailedSignal() {
        failedSignals.incrementAndGet();
    }
    
    // Trading metrics
    public void recordTradeAttempt() {
        totalTrades.incrementAndGet();
    }
    
    public void recordSuccessfulTrade() {
        successfulTrades.incrementAndGet();
    }
    
    public void recordRejectedTrade() {
        rejectedTrades.incrementAndGet();
    }
    
    // Thread tracking
    public void incrementActiveThreads() {
        activeThreads.incrementAndGet();
    }
    
    public void decrementActiveThreads() {
        activeThreads.decrementAndGet();
    }
    
    // Component health
    public void setAiServiceHealth(boolean healthy) {
        this.aiServiceHealthy = healthy;
        if (!healthy) {
            recordError("AI Service unhealthy");
        }
    }
    
    public void setDataManagerHealth(boolean healthy) {
        this.dataManagerHealthy = healthy;
        if (!healthy) {
            recordError("Data Manager unhealthy");
        }
    }
    
    public void setRiskManagerHealth(boolean healthy) {
        this.riskManagerHealthy = healthy;
        if (!healthy) {
            recordError("Risk Manager unhealthy");
        }
    }
    
    public void recordError(String error) {
        this.lastError = error;
        this.lastErrorTime = LocalDateTime.now();
        logger.warning("Health Monitor recorded error: " + error);
    }
    
    public void updateHealthCheck() {
        lastHealthCheck.set(System.currentTimeMillis());
    }
    
    // Health status
    public boolean isSystemHealthy() {
        return aiServiceHealthy && dataManagerHealthy && riskManagerHealthy;
    }
    
    public double getSignalSuccessRate() {
        long total = totalSignalRequests.get();
        if (total == 0) return 1.0;
        return (double) successfulSignals.get() / total;
    }
    
    public double getTradeSuccessRate() {
        int total = totalTrades.get();
        if (total == 0) return 1.0;
        return (double) successfulTrades.get() / total;
    }
    
    public Map<String, Object> getHealthReport() {
        Map<String, Object> report = new HashMap<>();
        
        // Overall status
        report.put("system_healthy", isSystemHealthy());
        try {
            report.put("last_health_check", LocalDateTime.ofEpochSecond(
                lastHealthCheck.get() / 1000, 0, java.time.ZoneOffset.UTC)
                .format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        } catch (Exception e) {
            report.put("last_health_check", LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            logger.warning("Health check timestamp conversion error: " + e.getMessage());
        }
        
        // Component status
        Map<String, Boolean> components = new HashMap<>();
        components.put("ai_service", aiServiceHealthy);
        components.put("data_manager", dataManagerHealthy);
        components.put("risk_manager", riskManagerHealthy);
        report.put("components", components);
        
        // Performance metrics
        Map<String, Object> performance = new HashMap<>();
        performance.put("total_signal_requests", totalSignalRequests.get());
        performance.put("successful_signals", successfulSignals.get());
        performance.put("failed_signals", failedSignals.get());
        performance.put("signal_success_rate", getSignalSuccessRate());
        performance.put("active_threads", activeThreads.get());
        report.put("performance", performance);
        
        // Trading metrics
        Map<String, Object> trading = new HashMap<>();
        trading.put("total_trades", totalTrades.get());
        trading.put("successful_trades", successfulTrades.get());
        trading.put("rejected_trades", rejectedTrades.get());
        trading.put("trade_success_rate", getTradeSuccessRate());
        report.put("trading", trading);
        
        // Error information
        if (lastError != null) {
            Map<String, String> errorInfo = new HashMap<>();
            errorInfo.put("last_error", lastError);
            errorInfo.put("last_error_time", lastErrorTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            report.put("error_info", errorInfo);
        }
        
        // System information
        Map<String, Object> system = new HashMap<>();
        Runtime runtime = Runtime.getRuntime();
        system.put("total_memory_mb", runtime.totalMemory() / (1024 * 1024));
        system.put("free_memory_mb", runtime.freeMemory() / (1024 * 1024));
        system.put("used_memory_mb", (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024));
        system.put("max_memory_mb", runtime.maxMemory() / (1024 * 1024));
        system.put("available_processors", runtime.availableProcessors());
        report.put("system", system);
        
        return report;
    }
    
    public void printHealthSummary() {
        System.out.println("\n=== System Health Summary ===");
        System.out.println("Overall Status: " + (isSystemHealthy() ? "HEALTHY" : "UNHEALTHY"));
        System.out.println("AI Service: " + (aiServiceHealthy ? "OK" : "FAILED"));
        System.out.println("Data Manager: " + (dataManagerHealthy ? "OK" : "FAILED"));
        System.out.println("Risk Manager: " + (riskManagerHealthy ? "OK" : "FAILED"));
        System.out.println("Signal Success Rate: " + String.format("%.2f%%", getSignalSuccessRate() * 100));
        System.out.println("Trade Success Rate: " + String.format("%.2f%%", getTradeSuccessRate() * 100));
        System.out.println("Active Threads: " + activeThreads.get());
        
        Runtime runtime = Runtime.getRuntime();
        long usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024);
        long maxMemory = runtime.maxMemory() / (1024 * 1024);
        System.out.println("Memory Usage: " + usedMemory + "MB / " + maxMemory + "MB");
        
        if (lastError != null) {
            System.out.println("Last Error: " + lastError + " at " + 
                lastErrorTime.format(DateTimeFormatter.ofPattern("HH:mm:ss")));
        }
        System.out.println("=============================\n");
    }
    
    public void reset() {
        totalSignalRequests.set(0);
        successfulSignals.set(0);
        failedSignals.set(0);
        totalTrades.set(0);
        successfulTrades.set(0);
        rejectedTrades.set(0);
        activeThreads.set(0);
        aiServiceHealthy = true;
        dataManagerHealthy = true;
        riskManagerHealthy = true;
        lastError = null;
        lastErrorTime = null;
        logger.info("Health Monitor metrics reset");
    }
}
