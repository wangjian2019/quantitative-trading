#!/bin/bash

echo "============================================================"
echo "🚀 AI量化交易平台 v0.1 - 生产环境启动"
echo "作者: Alvin"
echo "============================================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查运行环境
echo -e "${BLUE}🔍 检查运行环境...${NC}"

# 检查Java
if ! command -v java &> /dev/null; then
    echo -e "${RED}❌ 错误: 未找到Java，请先安装Java 8+${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Java环境: $(java -version 2>&1 | head -n 1)${NC}"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ 错误: 未找到Python3，请先安装Python 3.8+${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python环境: $(python3 --version)${NC}"

# 检查Maven
if ! command -v mvn &> /dev/null; then
    echo -e "${RED}❌ 错误: 未找到Maven，请先安装Maven 3.6+${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Maven环境: $(mvn --version | head -n 1)${NC}"

# 检查端口占用
echo -e "${BLUE}🔍 检查端口占用...${NC}"
if lsof -i:8080 &> /dev/null; then
    echo -e "${YELLOW}⚠️ 警告: 端口8080被占用，正在尝试释放...${NC}"
    pkill -f "TradingPlatformApplication" 2>/dev/null
    sleep 3
fi

if lsof -i:5001 &> /dev/null; then
    echo -e "${YELLOW}⚠️ 警告: 端口5001被占用，正在尝试释放...${NC}"
    pkill -f "ai_model_service" 2>/dev/null
    sleep 3
fi

echo -e "${GREEN}✅ 端口检查完成${NC}"

# 配置验证
echo -e "${BLUE}🔍 验证生产环境配置...${NC}"

# 检查配置文件
if [ ! -f "platform/src/main/resources/application.properties" ]; then
    echo -e "${RED}❌ 错误: 配置文件不存在${NC}"
    exit 1
fi

# 验证关键配置
INITIAL_CAPITAL=$(grep "trading.initial.capital" platform/src/main/resources/application.properties | cut -d'=' -f2)
if [ "$INITIAL_CAPITAL" != "10000000.0" ]; then
    echo -e "${RED}❌ 错误: 初始资金配置错误，应为10000000.0，当前为: $INITIAL_CAPITAL${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 初始资金配置正确: $INITIAL_CAPITAL ${NC}"

# 检查风险配置
MAX_DAILY_LOSS=$(grep "risk.max.daily.loss" platform/src/main/resources/application.properties | cut -d'=' -f2)
echo -e "${GREEN}✅ 最大日亏损配置: $MAX_DAILY_LOSS (50万)${NC}"

# 安装Python依赖
echo -e "${BLUE}📦 安装Python依赖...${NC}"
cd strategy
pip3 install -r requirements.txt -q
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Python依赖安装失败${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python依赖安装完成${NC}"

# 编译Java项目
echo -e "${BLUE}🔨 编译Java项目...${NC}"
cd ../platform
mvn clean compile -q
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Java项目编译失败${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Java项目编译成功${NC}"

# 复制依赖
echo -e "${BLUE}📦 准备运行时依赖...${NC}"
mvn dependency:copy-dependencies -DoutputDirectory=target/lib -q
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 依赖复制失败${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 运行时依赖准备完成${NC}"

# 启动Python AI服务
echo -e "${BLUE}🐍 启动Python AI服务...${NC}"
cd ../strategy

# 检查Python依赖
echo -e "${BLUE}📦 检查Python依赖...${NC}"
if ! python3 -c "import flask, pandas, numpy, sklearn" 2>/dev/null; then
    echo -e "${YELLOW}⚠️ 安装Python依赖...${NC}"
    pip3 install -r requirements.txt -q
fi

python3 ai_model_service.py &
AI_PID=$!
echo -e "${GREEN}✅ Python AI服务已启动 (PID: $AI_PID)${NC}"

# 等待AI服务启动
echo -e "${BLUE}⏳ 等待AI服务初始化...${NC}"
sleep 15

# 验证AI服务
echo -e "${BLUE}🔍 验证AI服务连接...${NC}"
for i in {1..5}; do
    if curl -s http://localhost:5001/health > /dev/null; then
        echo -e "${GREEN}✅ AI服务健康检查通过${NC}"
        break
    else
        echo -e "${YELLOW}⏳ 等待AI服务启动... (尝试 $i/5)${NC}"
        sleep 5
    fi
    
    if [ $i -eq 5 ]; then
        echo -e "${RED}❌ AI服务启动失败，请检查Python环境${NC}"
        kill $AI_PID 2>/dev/null
        exit 1
    fi
done

# 启动Java平台
echo -e "${BLUE}☕ 启动Java交易平台...${NC}"
cd ../platform

# 使用trap处理中断信号
trap 'echo -e "\n${YELLOW}🛑 正在停止服务...${NC}"; kill $AI_PID 2>/dev/null; echo -e "${GREEN}✅ 服务已停止${NC}"; exit 0' INT TERM

echo "============================================================"
echo -e "${GREEN}🚀 生产环境启动完成！${NC}"
echo "============================================================"
echo -e "${BLUE}📊 系统信息:${NC}"
echo "• 最大单股仓位: 15% (150万)"
echo "• 日最大亏损: 50万"
echo "• 紧急止损: 3%"
echo "• 目标年化收益: 60%+"
echo ""
echo -e "${BLUE}🌐 访问地址:${NC}"
echo "• Web界面: http://localhost:8080"
echo "• API健康检查: http://localhost:8080/api/health"
echo "• AI服务: http://localhost:5001"
echo "• 交易信号: http://localhost:8080/api/trading-signals"
echo ""
echo -e "${YELLOW}⚠️ 重要提醒:${NC}"
echo "• 系统只提供交易信号，不执行真实订单"
echo "• 请根据通知手动执行交易"
echo "• 严格执行止损止盈策略"
echo "• 建议先小额测试验证准确性"
echo ""
echo -e "${BLUE}📱 监控建议:${NC}"
echo "• 配置邮件通知: 修改application.properties中的邮箱设置"
echo "• 配置微信通知: 设置企业微信机器人Webhook"
echo "• 每日检查系统健康状态"
echo "• 记录每笔交易结果以验证AI准确性"
echo ""
echo -e "${GREEN}🎯 按 Ctrl+C 安全停止系统${NC}"
echo "============================================================"

# 使用SpringBoot方式运行Java主程序 (带项目级settings.xml)
mvn spring-boot:run -s settings.xml

# 清理
echo -e "\n${YELLOW}🛑 正在清理资源...${NC}"
kill $AI_PID 2>/dev/null
echo -e "${GREEN}✅ 所有服务已安全停止${NC}"
