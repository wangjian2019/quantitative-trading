package com.alvin.quantitative.trading.platform.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.StreamUtils;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.logging.Logger;

/**
 * SpringBoot Web控制器
 * Author: Alvin
 * 处理Web界面请求
 */
@Controller
public class SpringBootWebController {
    private static final Logger logger = Logger.getLogger(SpringBootWebController.class.getName());
    
    @GetMapping("/")
    @ResponseBody
    public ResponseEntity<String> home() {
        try {
            // 尝试从web目录读取index.html
            Resource resource = new ClassPathResource("static/index.html");
            if (resource.exists()) {
                String content = StreamUtils.copyToString(resource.getInputStream(), StandardCharsets.UTF_8);
                return ResponseEntity.ok()
                    .contentType(MediaType.TEXT_HTML)
                    .body(content);
            } else {
                // 如果没有找到静态文件，返回内置的HTML
                return ResponseEntity.ok()
                    .contentType(MediaType.TEXT_HTML)
                    .body(getBuiltInHomePage());
            }
        } catch (IOException e) {
            logger.warning("Failed to load index.html: " + e.getMessage());
            return ResponseEntity.ok()
                .contentType(MediaType.TEXT_HTML)
                .body(getBuiltInHomePage());
        }
    }
    
    @GetMapping("/portfolio.json")
    @ResponseBody
    public ResponseEntity<String> portfolioConfig() {
        try {
            Resource resource = new ClassPathResource("portfolio.json");
            if (resource.exists()) {
                String content = StreamUtils.copyToString(resource.getInputStream(), StandardCharsets.UTF_8);
                return ResponseEntity.ok()
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(content);
            } else {
                return ResponseEntity.notFound().build();
            }
        } catch (IOException e) {
            logger.warning("Failed to load portfolio.json: " + e.getMessage());
            return ResponseEntity.notFound().build();
        }
    }
    
    private String getBuiltInHomePage() {
        return "<!DOCTYPE html>" +
            "<html lang=\"zh-CN\">" +
            "<head>" +
            "<meta charset=\"UTF-8\">" +
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">" +
            "<title>🚀 AI量化交易平台 v0.1</title>" +
            "<style>" +
            "body { font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 0; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center; min-height: 100vh; }" +
            ".container { max-width: 900px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 40px; border-radius: 20px; backdrop-filter: blur(10px); box-shadow: 0 20px 40px rgba(0,0,0,0.2); }" +
            "h1 { font-size: 3.5rem; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }" +
            ".subtitle { font-size: 1.3rem; margin-bottom: 40px; opacity: 0.9; }" +
            ".features { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 30px; margin: 40px 0; }" +
            ".feature { background: rgba(255,255,255,0.1); padding: 30px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2); }" +
            ".feature h3 { margin-top: 0; font-size: 1.5rem; }" +
            ".feature p { margin-bottom: 0; line-height: 1.6; }" +
            ".links { margin-top: 40px; }" +
            ".link { display: inline-block; margin: 10px 20px; padding: 15px 30px; background: rgba(255,255,255,0.2); text-decoration: none; color: white; border-radius: 25px; transition: all 0.3s; }" +
            ".link:hover { background: rgba(255,255,255,0.3); transform: translateY(-2px); }" +
            ".status { margin-top: 30px; padding: 20px; background: rgba(0,255,0,0.1); border-radius: 10px; border: 1px solid rgba(0,255,0,0.3); }" +
            "</style>" +
            "</head>" +
            "<body>" +
            "<div class=\"container\">" +
            "<h1>🚀 AI量化交易平台</h1>" +
            "<p class=\"subtitle\">v0.1 SpringBoot架构 - by Alvin</p>" +
            "<div class=\"features\">" +
            "<div class=\"feature\"><h3>🤖 AI智能分析</h3><p>多模型集成学习预测<br>RandomForest + GradientBoosting + LogisticRegression</p></div>" +
            "<div class=\"feature\"><h3>📊 实时监控</h3><p>支持美股、港股、ETF<br>ASML, TSLA, QQQ, VOO等</p></div>" +
            "<div class=\"feature\"><h3>🔔 智能通知</h3><p>邮件+微信实时推送<br>高置信度信号自动通知</p></div>" +
            "<div class=\"feature\"><h3>📈 专业回测</h3><p>多维度性能评估<br>夏普比率、最大回撤分析</p></div>" +
            "<div class=\"feature\"><h3>🛡️ 风险管理</h3><p>智能止损止盈<br>多层风险控制体系</p></div>" +
            "<div class=\"feature\"><h3>💰 大额投资</h3><p>投资支持<br>专业级资金管理</p></div>" +
            "</div>" +
            "<div class=\"links\">" +
            "<a href=\"/api/health\" class=\"link\">📊 系统健康</a>" +
            "<a href=\"/api/status\" class=\"link\">🔧 运行状态</a>" +
            "<a href=\"/api/portfolio\" class=\"link\">💼 投资组合</a>" +
            "<a href=\"/api/indicators\" class=\"link\">📈 技术指标</a>" +
            "<a href=\"/api/trading-signals\" class=\"link\">🎯 交易信号</a>" +
            "</div>" +
            "<div class=\"status\">" +
            "<h3>✅ SpringBoot服务运行中</h3>" +
            "<p>AI量化交易平台v0.1已启动，使用SpringBoot架构提供稳定的Web服务</p>" +
            "</div>" +
            "</div>" +
            "</body>" +
            "</html>";
    }
}
