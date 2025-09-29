#!/bin/bash
# 手动执行数据更新和模型重训练
# Author: Alvin

STRATEGY_DIR="/Users/alvin/eclipse-workspace-new/quantitative-trading/strategy"
cd "$STRATEGY_DIR"

echo "🎯 量化交易手动更新工具"
echo "当前目录: $(pwd)"
echo "时间: $(date)"
echo ""

# 创建必要目录
mkdir -p logs status models/backup

function show_menu() {
    echo "请选择操作:"
    echo "1) 立即执行数据更新"
    echo "2) 立即执行模型重训练"
    echo "3) 执行系统健康检查"
    echo "4) 查看系统状态"
    echo "5) 查看最近日志"
    echo "6) 完整更新 (数据+模型)"
    echo "0) 退出"
    echo ""
}

function run_data_update() {
    echo "🚀 开始手动数据更新..."
    echo "======================================"
    python3 daily_data_update.py
    echo ""
    echo "✅ 数据更新完成"
}

function run_model_retrain() {
    echo "🤖 开始手动模型重训练..."
    echo "======================================"
    python3 weekly_model_retrain.py
    echo ""
    echo "✅ 模型重训练完成"
}

function run_health_check() {
    echo "🔍 执行系统健康检查..."
    echo "======================================"
    python3 system_health_check.py
    echo ""
    echo "✅ 健康检查完成"
}

function show_system_status() {
    echo "📊 系统状态概览"
    echo "======================================"

    # 数据状态
    if [ -f "data/training_data/summary.txt" ]; then
        echo "📈 数据状态:"
        cat data/training_data/summary.txt
        echo ""
    else
        echo "❌ 未找到数据摘要文件"
    fi

    # 模型状态
    if [ -f "tiny_model_info.pkl" ]; then
        echo "🤖 模型状态:"
        python3 -c "
import pickle
try:
    with open('tiny_model_info.pkl', 'rb') as f:
        info = pickle.load(f)
    print(f'模型类型: {info.get(\"model_class\", \"未知\")}')
    print(f'准确率: {info.get(\"accuracy\", 0):.4f}')
    print(f'训练样本: {info.get(\"num_samples\", 0)}')
    print(f'序列长度: {info.get(\"seq_len\", 0)}')
except Exception as e:
    print(f'读取模型信息失败: {e}')
"
        echo ""
    else
        echo "❌ 未找到模型文件"
    fi

    # 更新状态
    if [ -f "status/daily_update_status.json" ]; then
        echo "📅 最近更新:"
        python3 -c "
import json
from datetime import datetime
try:
    with open('status/daily_update_status.json', 'r') as f:
        status = json.load(f)
    last_update = datetime.fromisoformat(status['last_daily_update'])
    print(f'数据更新: {last_update.strftime(\"%Y-%m-%d %H:%M:%S\")}')
except Exception as e:
    print(f'读取更新状态失败: {e}')
"
    fi

    if [ -f "status/model_retrain_status.json" ]; then
        python3 -c "
import json
from datetime import datetime
try:
    with open('status/model_retrain_status.json', 'r') as f:
        status = json.load(f)
    last_retrain = datetime.fromisoformat(status['last_retrain_date'])
    print(f'模型重训练: {last_retrain.strftime(\"%Y-%m-%d %H:%M:%S\")}')
except Exception as e:
    print(f'读取重训练状态失败: {e}')
"
    fi
    echo ""
}

function show_recent_logs() {
    echo "📝 最近日志 (最后20行)"
    echo "======================================"

    # 最新的每日更新日志
    LATEST_DAILY_LOG=$(ls -t logs/daily_update_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_DAILY_LOG" ]; then
        echo "🔍 最新每日更新日志: $LATEST_DAILY_LOG"
        tail -20 "$LATEST_DAILY_LOG"
        echo ""
    fi

    # 最新的周度重训练日志
    LATEST_WEEKLY_LOG=$(ls -t logs/weekly_retrain_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_WEEKLY_LOG" ]; then
        echo "🔍 最新重训练日志: $LATEST_WEEKLY_LOG"
        tail -20 "$LATEST_WEEKLY_LOG"
        echo ""
    fi

    # cron日志
    if [ -f "logs/daily_cron.log" ]; then
        echo "🔍 定时任务日志:"
        tail -10 logs/daily_cron.log
    fi
}

function run_full_update() {
    echo "🎯 执行完整更新 (数据 + 模型)"
    echo "======================================"

    # 1. 数据更新
    echo "第1步: 数据更新"
    python3 daily_data_update.py
    if [ $? -eq 0 ]; then
        echo "✅ 数据更新成功"
    else
        echo "❌ 数据更新失败，终止操作"
        return 1
    fi

    echo ""

    # 2. 模型重训练
    echo "第2步: 模型重训练"
    python3 weekly_model_retrain.py
    if [ $? -eq 0 ]; then
        echo "✅ 模型重训练成功"
    else
        echo "❌ 模型重训练失败"
        return 1
    fi

    echo ""

    # 3. 健康检查
    echo "第3步: 健康检查"
    python3 system_health_check.py

    echo ""
    echo "🎉 完整更新完成!"
}

# 主循环
while true; do
    show_menu
    read -p "请输入选择 (0-6): " choice

    case $choice in
        1)
            run_data_update
            ;;
        2)
            run_model_retrain
            ;;
        3)
            run_health_check
            ;;
        4)
            show_system_status
            ;;
        5)
            show_recent_logs
            ;;
        6)
            run_full_update
            ;;
        0)
            echo "👋 退出"
            exit 0
            ;;
        *)
            echo "❌ 无效选择，请重新输入"
            ;;
    esac

    echo ""
    read -p "按回车键继续..."
    echo ""
done