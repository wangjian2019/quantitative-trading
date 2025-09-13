#!/bin/bash

# AI量化交易平台 - Java服务启动脚本
# Author: Alvin

echo "======================================"
echo "启动 Java 量化交易平台服务"
echo "Author: Alvin"
echo "======================================"

# 进入Java项目目录
cd "$(dirname "$0")/platform"

# 检查Maven是否可用
if ! command -v mvn &> /dev/null; then
    echo "❌ Maven 未安装或不在PATH中"
    exit 1
fi

# 检查Java是否可用
if ! command -v java &> /dev/null; then
    echo "❌ Java 未安装或不在PATH中"
    exit 1
fi

echo "📦 编译Java项目..."
mvn clean compile -q

if [ $? -ne 0 ]; then
    echo "❌ Java项目编译失败"
    exit 1
fi

echo "🚀 启动Java服务..."
echo "📍 Web界面将在 http://localhost:8080 启动"
echo "🔧 AI服务地址: http://localhost:5000"
echo "⚠️  请确保Python AI服务已经启动"
echo ""

# 使用Maven执行插件启动应用
mvn exec:java -Dexec.mainClass="com.alvin.quantitative.trading.platform.TradingPlatformApplication" -Dexec.args="" -q

echo ""
echo "Java服务已停止"
