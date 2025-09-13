#!/bin/bash

echo "=========================================="
echo "启动AI量化交易平台"
echo "Author: Alvin"
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 进入策略目录
cd strategy

# 检查并安装Python依赖
echo "检查Python依赖..."
if [ -f "requirements.txt" ]; then
    echo "安装Python依赖包..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "警告: 依赖安装失败，尝试继续运行..."
    fi
else
    echo "警告: 未找到requirements.txt文件"
fi

echo ""
echo "启动Python AI策略服务..."
echo "服务地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务"
echo ""

# 启动Python AI模型服务（只负责策略计算）
python3 ai_model_service.py

