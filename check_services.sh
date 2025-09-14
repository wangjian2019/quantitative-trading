#!/bin/bash

echo "============================================================"
echo "🔍 AI量化交易平台 v0.1 - 服务状态检查"
echo "作者: Alvin"
echo "============================================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📊 检查服务状态...${NC}"

# 检查Python AI服务 (端口5001)
echo -e "${BLUE}🐍 检查Python AI服务 (端口5001)...${NC}"
if curl -s http://localhost:5001/health > /dev/null; then
    echo -e "${GREEN}✅ Python AI服务运行正常${NC}"
    AI_STATUS=$(curl -s http://localhost:5001/health | python3 -c "import json, sys; d=json.load(sys.stdin); print(f'状态: {d[\"status\"]}, 模型: {len(d[\"models_available\"])}个')")
    echo -e "   $AI_STATUS"
else
    echo -e "${RED}❌ Python AI服务未运行${NC}"
    echo -e "${YELLOW}💡 启动命令: cd strategy && python3 ai_model_service.py${NC}"
fi

echo ""

# 检查Java SpringBoot服务 (端口8080)
echo -e "${BLUE}☕ 检查Java SpringBoot服务 (端口8080)...${NC}"
if curl -s http://localhost:8080/api/health > /dev/null; then
    echo -e "${GREEN}✅ Java SpringBoot服务运行正常${NC}"
    JAVA_STATUS=$(curl -s http://localhost:8080/api/health | python3 -c "import json, sys; d=json.load(sys.stdin); print(f'版本: {d[\"version\"]}, 架构: {d[\"architecture\"]}')")
    echo -e "   $JAVA_STATUS"
else
    echo -e "${RED}❌ Java SpringBoot服务未运行${NC}"
    echo -e "${YELLOW}💡 启动命令: cd platform && mvn spring-boot:run -s settings.xml${NC}"
fi

echo ""

# 检查服务连接
echo -e "${BLUE}🔗 检查服务间连接...${NC}"
if curl -s http://localhost:8080/api/status > /dev/null; then
    AI_CONNECTION=$(curl -s http://localhost:8080/api/status | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('ai_service', 'unknown'))")
    if [ "$AI_CONNECTION" = "connected" ]; then
        echo -e "${GREEN}✅ Java ↔ Python AI服务连接正常${NC}"
    else
        echo -e "${RED}❌ Java ↔ Python AI服务连接失败${NC}"
    fi
else
    echo -e "${RED}❌ 无法检查服务连接${NC}"
fi

echo ""

# 检查端口占用
echo -e "${BLUE}🔌 检查端口占用...${NC}"
PORT_5001=$(lsof -i:5001 | grep LISTEN | wc -l)
PORT_8080=$(lsof -i:8080 | grep LISTEN | wc -l)

if [ $PORT_5001 -gt 0 ]; then
    echo -e "${GREEN}✅ 端口5001: Python AI服务${NC}"
else
    echo -e "${RED}❌ 端口5001: 无服务${NC}"
fi

if [ $PORT_8080 -gt 0 ]; then
    echo -e "${GREEN}✅ 端口8080: Java SpringBoot服务${NC}"
else
    echo -e "${RED}❌ 端口8080: 无服务${NC}"
fi

echo ""

# 系统健康总结
echo -e "${BLUE}📋 系统健康总结${NC}"
echo "============================================================"

PYTHON_OK=false
JAVA_OK=false
CONNECTION_OK=false

if curl -s http://localhost:5001/health > /dev/null; then
    PYTHON_OK=true
fi

if curl -s http://localhost:8080/api/health > /dev/null; then
    JAVA_OK=true
fi

if curl -s http://localhost:8080/api/status > /dev/null; then
    AI_CONN=$(curl -s http://localhost:8080/api/status | python3 -c "import json, sys; d=json.load(sys.stdin); print(d.get('ai_service', 'unknown'))")
    if [ "$AI_CONN" = "connected" ]; then
        CONNECTION_OK=true
    fi
fi

if $PYTHON_OK && $JAVA_OK && $CONNECTION_OK; then
    echo -e "${GREEN}🎉 系统状态: 完全正常${NC}"
    echo ""
    echo -e "${BLUE}📱 访问地址:${NC}"
    echo "• Web界面: http://localhost:8080"
    echo "• 交易信号: http://localhost:8080/api/trading-signals"
    echo "• 技术指标: http://localhost:8080/api/indicators"
    echo "• 系统监控: http://localhost:8080/actuator/health"
else
    echo -e "${RED}⚠️ 系统状态: 异常${NC}"
    echo -e "${YELLOW}请检查服务启动状态并重新启动${NC}"
fi

echo "============================================================"
