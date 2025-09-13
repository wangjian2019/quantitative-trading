#!/bin/bash

echo "=========================================="
echo "启动Java量化交易平台"
echo "Author: Alvin"
echo "=========================================="

# 检查Java环境
if ! command -v java &> /dev/null; then
    echo "错误: 未找到Java，请先安装Java 8+"
    exit 1
fi

# 检查Maven环境
if ! command -v mvn &> /dev/null; then
    echo "错误: 未找到Maven，请先安装Maven"
    exit 1
fi

# 进入平台目录
cd platform

echo "编译Java项目..."
mvn clean compile -q

if [ $? -ne 0 ]; then
    echo "错误: Java项目编译失败"
    exit 1
fi

echo "编译成功！"
echo ""
echo "启动Java交易平台..."
echo "确保Python AI服务已在 http://localhost:5000 运行"
echo ""

# 运行Java主程序（新架构）
mvn exec:java -Dexec.mainClass="com.alvin.quantitative.trading.platform.TradingPlatformApplication" -q

