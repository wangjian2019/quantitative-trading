package com.alvin.quantitative.trading.platform;

import com.alvin.quantitative.trading.platform.config.ApplicationConfig;
import com.alvin.quantitative.trading.platform.engine.SmartTradingEngine;
import com.alvin.quantitative.trading.platform.util.HealthMonitor;

import java.util.Scanner;

/**
 * 量化交易平台主程序
 * Author: Alvin
 * 启动AI量化交易系统
 */
public class TradingPlatformMain {
    
    public static void main(String[] args) {
        System.out.println(repeat("=", 60));
        System.out.println("AI量化交易平台");
        System.out.println("Author: Alvin");
        System.out.println(repeat("=", 60));
        
        // 创建交易引擎（使用配置管理器）
        SmartTradingEngine engine;
        try {
            engine = new SmartTradingEngine();
        } catch (IllegalStateException e) {
            System.err.println("配置错误: " + e.getMessage());
            System.err.println("请检查 application.properties 配置文件");
            return;
        }
        
        System.out.println("启动AI交易引擎...");
        System.out.println();
        
        // 启动引擎
        engine.start();
        
        System.out.println("交易引擎已启动！");
        System.out.println("监控股票: AAPL, TSLA, MSFT");
        System.out.println("按 'q' 退出程序...");
        System.out.println();
        
        // 等待用户输入退出
        Scanner scanner = new Scanner(System.in);
        String input;
        do {
            System.out.print("输入命令 (q=退出, s=状态, b=回测, h=帮助): ");
            input = scanner.nextLine().trim().toLowerCase();
            
            switch (input) {
                case "s":
                case "status":
                    printStatus(engine);
                    break;
                case "b":
                case "backtest":
                    System.out.println("启动3年历史回测分析...");
                    engine.runManualBacktest();
                    break;
                case "h":
                case "help":
                    printHelp();
                    break;
                case "q":
                case "quit":
                case "exit":
                    System.out.println("正在停止交易引擎...");
                    break;
                default:
                    if (!input.isEmpty()) {
                        System.out.println("未知命令: " + input + " (输入 'h' 查看帮助)");
                    }
                    break;
            }
        } while (!input.equals("q") && !input.equals("quit") && !input.equals("exit"));
        
        // 停止引擎
        engine.stop();
        scanner.close();
        
        System.out.println("交易引擎已停止。再见！");
    }
    
    private static String repeat(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }
    
    private static void printStatus(SmartTradingEngine engine) {
        System.out.println();
        System.out.println("=== 交易引擎状态 ===");
        System.out.println("引擎状态: 运行中");
        
        // Get configuration for display
        ApplicationConfig config = ApplicationConfig.getInstance();
        System.out.println("监控股票: " + String.join(", ", config.getTradingSymbols()));
        System.out.println("数据收集: 每" + config.getDataCollectionInterval() + "秒");
        System.out.println("策略执行: 每" + config.getStrategyExecutionInterval() + "秒");
        System.out.println("风险检查: 每" + config.getRiskCheckInterval() + "秒");
        System.out.println("初始资金: $" + String.format("%.2f", config.getInitialCapital()));
        System.out.println("最小置信度: " + config.getMinConfidence());
        
        // Display health information
        HealthMonitor healthMonitor = HealthMonitor.getInstance();
        healthMonitor.printHealthSummary();
    }
    
    private static void printHelp() {
        System.out.println();
        System.out.println("=== 可用命令 ===");
        System.out.println("s, status   - 显示交易引擎状态");
        System.out.println("b, backtest - 运行3年历史回测分析");
        System.out.println("h, help     - 显示此帮助信息");
        System.out.println("q, quit     - 退出程序");
        System.out.println();
        System.out.println("=== 系统功能 ===");
        System.out.println("• 📊 可配置的股票/ETF投资组合监控");
        System.out.println("• 🤖 AI驱动的交易信号生成");
        System.out.println("• 🛡️ 智能风险管理和仓位控制");
        System.out.println("• 📈 3年历史数据回测分析");
        System.out.println("• 📧 邮件和微信通知提醒");
        System.out.println("• 🔄 AI模型持续优化改进");
        System.out.println();
    }
}
