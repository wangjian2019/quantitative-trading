#!/bin/bash

# AI量化交易平台 - 新架构启动脚本
# Author: Alvin

echo "🚀 AI量化交易平台 v2.0 - 新架构启动"
echo "Author: Alvin"
echo "========================================"
echo ""

# 检查环境
echo "🔍 检查环境..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

if ! command -v java &> /dev/null; then
    echo "❌ Java 未安装"
    exit 1
fi

if ! command -v mvn &> /dev/null; then
    echo "❌ Maven 未安装"
    exit 1
fi

echo "✅ 环境检查通过"
echo ""

# 启动Python AI模型服务
echo "🤖 启动Python AI模型服务..."
cd strategy

# 检查依赖
if [ -f "requirements.txt" ]; then
    echo "📦 检查Python依赖..."
    pip3 install -r requirements.txt > /dev/null 2>&1
fi

# 启动AI模型服务
echo "🚀 启动AI模型服务 (端口5000)..."
python3 ai_model_service.py &
AI_PID=$!

# 等待服务启动
echo "⏳ 等待AI模型服务启动..."
sleep 5

# 检查AI服务状态
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✅ AI模型服务启动成功"
else
    echo "❌ AI模型服务启动失败"
    kill $AI_PID 2>/dev/null
    exit 1
fi

echo ""

# 启动Java平台服务
echo "☕ 启动Java平台服务..."
cd ../platform

echo "📦 编译Java项目..."
mvn clean compile -q

if [ $? -ne 0 ]; then
    echo "❌ Java项目编译失败"
    kill $AI_PID 2>/dev/null
    exit 1
fi

echo "🚀 启动Java平台服务 (端口8080)..."
echo ""
echo "🌐 Web界面: http://localhost:8080"
echo "🤖 AI服务: http://localhost:5000"
echo ""
echo "⚠️  按 Ctrl+C 停止所有服务"
echo ""

# 启动Java服务
mvn exec:java -Dexec.mainClass="com.alvin.quantitative.trading.platform.TradingPlatformApplication" -q

# 清理
echo ""
echo "🛑 停止服务..."
kill $AI_PID 2>/dev/null
echo "✅ 所有服务已停止"
