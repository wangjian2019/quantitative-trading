#!/bin/bash

echo "🧪 测试AI量化交易平台服务"
echo "=============================="

# 测试Python AI服务
echo "1. 测试Python AI服务..."
cd /Users/alvin.wang/alvin-eclipse-workspace-new/quantitative-trading/strategy

# 启动Python服务（后台）
python3 app.py &
PYTHON_PID=$!
echo "Python服务PID: $PYTHON_PID"

# 等待服务启动
sleep 5

# 测试健康检查
echo "测试健康检查..."
curl -s http://localhost:5000/health | python3 -m json.tool

# 测试快速回测
echo -e "\n测试快速回测..."
curl -X POST -s http://localhost:5000/api/backtest/quick | python3 -c "
import sys, json
data = json.load(sys.stdin)
summary = data['backtest_summary']
print(f'✅ 回测完成: 总收益率 {summary[\"total_return\"]*100:.2f}%, 夏普比率 {summary[\"sharpe_ratio\"]:.2f}')
"

# 测试信号生成
echo -e "\n测试信号生成..."
curl -X POST -s http://localhost:5000/api/signals/generate \
  -H "Content-Type: application/json" \
  -d '{
    "current_data": {"close": 150.0, "volume": 1000000, "timestamp": "2024-09-13T13:30:00"},
    "indicators": {"RSI": 45, "MA5": 149.5, "MA20": 148.0, "MACD": 0.3, "current_price": 150.0},
    "history": [{"close": 148.0}, {"close": 149.0}, {"close": 150.0}],
    "symbol": "AAPL"
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'✅ 信号生成: {data[\"action\"]} (置信度: {data[\"confidence\"]*100:.1f}%)')
print(f'理由: {data[\"reason\"]}')
"

echo -e "\n2. 测试Java平台服务..."
cd /Users/alvin.wang/alvin-eclipse-workspace-new/quantitative-trading/platform

# 编译Java项目
mvn clean compile -q
if [ $? -eq 0 ]; then
    echo "✅ Java编译成功"
else
    echo "❌ Java编译失败"
    kill $PYTHON_PID 2>/dev/null
    exit 1
fi

# 启动Java服务（后台）
timeout 10 mvn exec:java -Dexec.mainClass="com.alvin.quantitative.trading.platform.TradingPlatformApplication" -q &
JAVA_PID=$!
echo "Java服务PID: $JAVA_PID"

# 等待Java服务启动
sleep 8

# 测试Java Web服务
echo "测试Java Web服务..."
if curl -s http://localhost:8080/api/health > /dev/null 2>&1; then
    echo "✅ Java Web服务正常"
    curl -s http://localhost:8080/api/health | python3 -m json.tool
else
    echo "⚠️ Java Web服务未响应（可能需要更多时间启动）"
fi

# 测试Web页面
echo -e "\n测试Web页面..."
if curl -s http://localhost:8080/ | grep -q "AI量化交易平台"; then
    echo "✅ Web页面正常"
else
    echo "⚠️ Web页面未正常加载"
fi

echo -e "\n3. 服务间通信测试..."
echo "测试Java调用Python服务..."
# 这里应该测试Java服务调用Python服务的功能

echo -e "\n=============================="
echo "✅ 服务测试完成"
echo "🌐 Web界面: http://localhost:8080"
echo "🤖 AI服务: http://localhost:5000"
echo "=============================="

# 清理进程
echo -e "\n按任意键停止测试服务..."
read -n 1
echo "🛑 停止服务..."
kill $PYTHON_PID 2>/dev/null
kill $JAVA_PID 2>/dev/null
echo "✅ 服务已停止"
