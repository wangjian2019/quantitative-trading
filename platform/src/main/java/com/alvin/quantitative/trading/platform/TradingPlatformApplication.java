package com.alvin.quantitative.trading.platform;

import com.alvin.quantitative.trading.platform.config.ApplicationConfig;
import com.alvin.quantitative.trading.platform.engine.SmartTradingEngine;
import com.alvin.quantitative.trading.platform.engine.TradingEngineInterface;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.Map;
import java.util.logging.Logger;
import java.util.Scanner;

/**
 * AI量化交易平台主应用 - SpringBoot架构
 * Author: Alvin
 * 使用SpringBoot替代原生HTTP服务器
 */
@SpringBootApplication
public class TradingPlatformApplication {
    private static final Logger logger = Logger.getLogger(TradingPlatformApplication.class.getName());
    
    private static TradingEngineInterface engine;
    private static ConfigurableApplicationContext context;
    
    // 静态方法供SpringBoot控制器使用
    public static TradingEngineInterface getEngine() {
        return engine;
    }
    
    public static void main(String[] args) {
        printWelcomeBanner();
        
        try {
            // 启动SpringBoot应用
            context = SpringApplication.run(TradingPlatformApplication.class, args);
            
            // 初始化配置
            ApplicationConfig config = ApplicationConfig.getInstance();
            
            if (!config.validateConfiguration()) {
                System.err.println("❌ 配置验证失败，请检查 application.properties");
                System.exit(1);
            }
            
            // 创建智能交易引擎（功能完整版）
            engine = new SmartTradingEngine();
            
            // 添加监控股票到观察列表
            for (String symbol : config.getTradingSymbols()) {
                ((SmartTradingEngine)engine).addToWatchList(symbol.trim(), symbol.trim());
            }
            
            // 启动交易引擎
            System.out.println("🚀 启动交易引擎...");
            engine.start();
            
            System.out.println("✅ 系统启动完成！");
            System.out.println();
            System.out.println("🌐 Web界面: http://" + config.getUiServerHost() + ":" + config.getUiServerPort());
            System.out.println("📊 健康检查: http://" + config.getUiServerHost() + ":" + config.getUiServerPort() + "/actuator/health");
            System.out.println("🔧 AI服务: " + config.getAiServiceUrl());
            System.out.println("📈 API文档: http://" + config.getUiServerHost() + ":" + config.getUiServerPort() + "/api/status");
            System.out.println();
            System.out.println("按 Enter 键查看选项菜单...");
            
            // 交互式菜单
            runInteractiveMenu(engine);
            
        } catch (Exception e) {
            logger.severe("应用启动失败: " + e.getMessage());
            System.err.println("❌ 应用启动失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void runInteractiveMenu(TradingEngineInterface engine) {
        Scanner scanner = new Scanner(System.in);
        
        while (true) {
            scanner.nextLine(); // Wait for Enter
            
            System.out.println();
            System.out.println(repeat("=", 60));
            System.out.println("🎛️  AI量化交易平台控制台 v0.1");
            System.out.println(repeat("=", 60));
            System.out.println("1. 📊 查看系统状态");
            System.out.println("2. 📈 运行回测分析");
            System.out.println("3. 🔧 重启交易引擎");
            System.out.println("4. 🌐 重启Web服务");
            System.out.println("5. 📧 测试通知功能");
            System.out.println("6. 🤖 测试AI模型连接");
            System.out.println("7. 📋 查看API端点");
            System.out.println("8. ❓ 帮助信息");
            System.out.println("0. 🚪 退出系统");
            System.out.println(repeat("=", 60));
            System.out.print("请选择操作 (0-8): ");
            
            String choice = scanner.nextLine().trim();
            
            switch (choice) {
                case "1":
                    showSystemStatus(engine);
                    break;
                case "2":
                    runBacktestAnalysis(engine);
                    break;
                case "3":
                    restartTradingEngine(engine);
                    break;
                case "4":
                    restartSpringBootApp();
                    break;
                case "5":
                    testNotifications(engine);
                    break;
                case "6":
                    testAIConnection(engine);
                    break;
                case "7":
                    showAPIEndpoints();
                    break;
                case "8":
                    showHelp();
                    break;
                case "0":
                    System.out.println("🛑 正在停止系统...");
                    engine.stop();
                    SpringApplication.exit(context, () -> 0);
                    scanner.close();
                    System.out.println("✅ 系统已安全停止。再见！");
                    return;
                default:
                    System.out.println("❌ 无效选择，请输入 0-8");
                    break;
            }
            
            System.out.println();
            System.out.println("按 Enter 键返回主菜单...");
        }
    }
    
    private static void showSystemStatus(TradingEngineInterface engine) {
        System.out.println();
        System.out.println("📊 系统状态报告");
        System.out.println(repeat("=", 40));
        
        try {
            // 真实的健康检查
            engine.printHealthSummary();
            
            // 真实的Web服务状态检查
            boolean webServiceOk = checkWebServiceStatus();
            System.out.println("🌐 Web服务状态: " + (webServiceOk ? "SpringBoot运行中" : "异常"));
            
            // 真实的AI服务状态检查
            boolean aiServiceOk = checkAIService();
            System.out.println("🤖 AI服务状态: " + (aiServiceOk ? "正常" : "异常"));
            
            // 真实的数据源状态检查
            boolean dataSourceOk = engine.getDataSource().isAvailable();
            System.out.println("💾 数据源状态: " + (dataSourceOk ? "正常" : "异常"));
            
            // 真实的系统综合状态
            boolean systemOk = webServiceOk && aiServiceOk && dataSourceOk;
            System.out.println("🎯 系统综合状态: " + (systemOk ? "✅ 完全正常" : "❌ 存在问题"));
            
        } catch (Exception e) {
            System.err.println("❌ 获取系统状态失败: " + e.getMessage());
        }
    }
    
    private static void runBacktestAnalysis(TradingEngineInterface engine) {
        System.out.println();
        System.out.println("📈 启动回测分析...");
        System.out.println(repeat("=", 40));
        
        try {
            engine.runManualBacktest();
        } catch (Exception e) {
            System.err.println("❌ 回测分析失败: " + e.getMessage());
        }
    }
    
    private static void restartTradingEngine(TradingEngineInterface engine) {
        System.out.println();
        System.out.println("🔄 重启交易引擎...");
        try {
            engine.restart();
            System.out.println("✅ 交易引擎重启成功");
        } catch (Exception e) {
            System.err.println("❌ 交易引擎重启失败: " + e.getMessage());
        }
    }
    
    private static void restartSpringBootApp() {
        System.out.println();
        System.out.println("🔄 重启SpringBoot应用...");
        try {
            System.out.println("⚠️ SpringBoot应用重启需要手动重新启动程序");
            System.out.println("请按 Ctrl+C 停止当前程序，然后重新运行启动脚本");
        } catch (Exception e) {
            System.err.println("❌ 重启操作失败: " + e.getMessage());
        }
    }
    
    private static void testNotifications(TradingEngineInterface engine) {
        System.out.println();
        System.out.println("📧 测试通知功能...");
        try {
            Map<String, Boolean> results = engine.testNotificationConfig();
            System.out.println("📧 邮件通知: " + (results.get("email") ? "✅ 成功" : "❌ 失败"));
            System.out.println("💬 微信通知: " + (results.get("wechat") ? "✅ 成功" : "❌ 失败"));
        } catch (Exception e) {
            System.err.println("❌ 测试通知失败: " + e.getMessage());
        }
    }
    
    private static void testAIConnection(TradingEngineInterface engine) {
        System.out.println();
        System.out.println("🤖 测试AI模型连接...");
        try {
            boolean connected = checkAIService();
            if (connected) {
                System.out.println("✅ AI模型服务连接正常");
                // 可以添加更详细的AI服务测试
            } else {
                System.out.println("❌ AI模型服务连接失败");
                System.out.println("请检查: http://localhost:5001/health");
            }
        } catch (Exception e) {
            System.err.println("❌ 测试AI连接失败: " + e.getMessage());
        }
    }
    
    private static void showAPIEndpoints() {
        System.out.println();
        System.out.println("📋 API端点列表");
        System.out.println(repeat("=", 40));
        System.out.println("🌐 Web界面:");
        System.out.println("  GET  /                    - 主页面");
        System.out.println("  GET  /web/*              - 静态资源");
        System.out.println();
        System.out.println("📊 API端点:");
        System.out.println("  GET  /api/health         - 健康检查");
        System.out.println("  GET  /api/status         - 系统状态");
        System.out.println("  GET  /api/portfolio      - 投资组合");
        System.out.println("  GET  /api/signals        - 交易信号");
        System.out.println("  POST /api/backtest       - 运行回测");
        System.out.println("  POST /api/analyze/{symbol} - 分析股票");
        System.out.println("  POST /api/test-notification - 测试通知");
        System.out.println();
        System.out.println("🤖 AI模型服务:");
        System.out.println("  GET  http://localhost:5001/health     - AI健康检查");
        System.out.println("  POST http://localhost:5001/get_signal - 获取交易信号");
    }
    
    private static void showHelp() {
        System.out.println();
        System.out.println("❓ 系统帮助");
        System.out.println(repeat("=", 40));
        System.out.println("🌐 Web界面访问: http://localhost:8080");
        System.out.println("📊 主要功能:");
        System.out.println("  • 实时股票数据监控");
        System.out.println("  • AI驱动的交易信号生成");
        System.out.println("  • 投资组合管理和分析");
        System.out.println("  • 历史数据回测");
        System.out.println("  • 风险管理和通知");
        System.out.println();
        System.out.println("🔧 配置文件:");
        System.out.println("  • application.properties - 系统配置");
        System.out.println("  • portfolio.json - 投资组合配置");
        System.out.println("  • config.py - Python AI配置");
        System.out.println();
        System.out.println("📞 技术支持: 查看README.md或联系开发者");
    }
    
    private static boolean checkAIService() {
        try {
            // 真实的AI服务连接检查
            java.net.URL url = new java.net.URL("http://localhost:5001/health");
            java.net.HttpURLConnection connection = (java.net.HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(5000);
            connection.setReadTimeout(5000);
            
            int responseCode = connection.getResponseCode();
            return responseCode == 200;
        } catch (Exception e) {
            return false;
        }
    }
    
    private static boolean checkWebServiceStatus() {
        try {
            // 真实的SpringBoot Web服务检查
            java.net.URL url = new java.net.URL("http://localhost:8080/actuator/health");
            java.net.HttpURLConnection connection = (java.net.HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(3000);
            connection.setReadTimeout(3000);
            
            int responseCode = connection.getResponseCode();
            return responseCode == 200;
        } catch (Exception e) {
            return false;
        }
    }
    
    private static void printWelcomeBanner() {
        System.out.println(repeat("=", 70));
        System.out.println("🚀 AI量化交易平台 v0.1");
        System.out.println("👨‍💻 作者: Alvin");
        System.out.println("🏗️  架构: SmartTradingEngine + AI信号系统");
        System.out.println(repeat("=", 70));
        System.out.println("✨ 核心功能:");
        System.out.println("  • 🤖 AI交易信号生成");
        System.out.println("  • 📊 实时数据收集");
        System.out.println("  • 📧 智能通知系统");
        System.out.println("  • 🛡️ 风险管理控制");
        System.out.println("  • 🎯 手动交易模式");
        System.out.println(repeat("=", 70));
    }
    
    private static String repeat(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }
}