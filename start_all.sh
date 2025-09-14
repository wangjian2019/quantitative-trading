#!/bin/bash

echo "=========================================="
echo "🚀 AI量化交易平台 - 一键启动"
echo "Author: Alvin"
echo "=========================================="

# 检查环境
echo "检查运行环境..."

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查Java
if ! command -v java &> /dev/null; then
    echo "❌ 错误: 未找到Java，请先安装Java 8+"
    exit 1
fi

# 检查Maven
if ! command -v mvn &> /dev/null; then
    echo "❌ 错误: 未找到Maven，请先安装Maven"
    exit 1
fi

echo "✅ 环境检查通过"
echo ""

# 安装Python依赖
echo "📦 安装Python依赖..."
cd strategy
pip3 install -r requirements.txt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Python依赖安装成功"
else
    echo "⚠️  Python依赖安装失败，尝试继续运行..."
fi
cd ..

# 编译Java项目
echo "🔨 编译Java项目..."
cd platform
mvn clean compile -q
if [ $? -ne 0 ]; then
    echo "❌ Java项目编译失败"
    exit 1
fi
echo "✅ Java项目编译成功"
cd ..

# 创建日志目录
mkdir -p logs strategy/logs strategy/models

echo ""
echo "=========================================="
echo "🎯 启动服务"
echo "=========================================="

# 启动Python AI服务（后台运行）
echo "🐍 启动Python AI服务..."
cd strategy
python3 app.py &
AI_PID=$!
cd ..

# 等待AI服务启动
echo "⏳ 等待AI服务启动（10秒）..."
sleep 10

# 检查AI服务是否运行
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ AI服务启动成功 - http://localhost:5000"
else
    echo "⚠️  AI服务可能未完全启动，继续启动Java平台..."
fi

echo ""
echo "☕ 启动Java交易平台..."
echo "=========================================="

# 启动Java交易平台
cd platform

# 使用trap处理中断信号，确保清理Python进程
trap 'echo ""; echo "🛑 正在停止服务..."; kill $AI_PID 2>/dev/null; echo "✅ 服务已停止"; exit 0' INT TERM

# 使用SpringBoot方式启动Java主程序 (带项目级settings.xml)
mvn spring-boot:run -s settings.xml

# 清理Python进程
kill $AI_PID 2>/dev/null
echo "✅ 所有服务已停止"
