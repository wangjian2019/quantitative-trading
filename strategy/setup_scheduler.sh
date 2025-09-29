#!/bin/bash
# 设置自动化调度任务
# Author: Alvin

STRATEGY_DIR="/Users/alvin/eclipse-workspace-new/quantitative-trading/strategy"

echo "🔧 配置量化交易自动化调度任务"

# 创建必要目录
mkdir -p "$STRATEGY_DIR/logs"
mkdir -p "$STRATEGY_DIR/status"
mkdir -p "$STRATEGY_DIR/models/backup"

# 检查Python环境
echo "🐍 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

python3 --version
echo "✅ Python环境正常"

# 设置脚本权限
echo "🔒 设置脚本权限..."
chmod +x "$STRATEGY_DIR/daily_data_update.py"
chmod +x "$STRATEGY_DIR/weekly_model_retrain.py"
chmod +x "$STRATEGY_DIR/enhanced_data_collector.py"
chmod +x "$STRATEGY_DIR/memory_efficient_train.py"

# 生成crontab配置
echo "⏰ 生成定时任务配置..."

CRON_CONFIG="# 量化交易自动化任务
# 每日数据更新 - 工作日晚上22:00 (美股收盘后)
0 22 * * 1-5 cd $STRATEGY_DIR && /usr/bin/python3 daily_data_update.py >> logs/daily_cron.log 2>&1

# 周度模型重训练 - 每周日早上6:00
0 6 * * 0 cd $STRATEGY_DIR && /usr/bin/python3 weekly_model_retrain.py >> logs/weekly_cron.log 2>&1

# 每日系统健康检查 - 每天早上8:00
0 8 * * * cd $STRATEGY_DIR && /usr/bin/python3 system_health_check.py >> logs/health_check.log 2>&1
"

echo "📋 定时任务配置:"
echo "$CRON_CONFIG"

# 询问是否安装到crontab
echo ""
echo "是否要安装这些定时任务到crontab? (y/N)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    # 备份当前crontab
    echo "💾 备份当前crontab..."
    crontab -l > "$STRATEGY_DIR/crontab_backup_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || echo "# 无现有crontab" > "$STRATEGY_DIR/crontab_backup_$(date +%Y%m%d_%H%M%S).txt"

    # 安装新的定时任务
    echo "⚙️ 安装定时任务..."
    echo "$CRON_CONFIG" | crontab -

    echo "✅ 定时任务安装完成"
    echo "📅 查看当前定时任务:"
    crontab -l
else
    # 保存配置文件供手动安装
    echo "$CRON_CONFIG" > "$STRATEGY_DIR/crontab_config.txt"
    echo "💾 定时任务配置已保存到: $STRATEGY_DIR/crontab_config.txt"
    echo "📝 手动安装命令: crontab $STRATEGY_DIR/crontab_config.txt"
fi

echo ""
echo "🎯 调度任务说明:"
echo "- 每日数据更新: 工作日晚上22:00自动执行"
echo "- 周度模型重训练: 每周日早上6:00自动执行"
echo "- 系统健康检查: 每天早上8:00自动执行"
echo ""
echo "📁 日志文件位置:"
echo "- 每日更新: $STRATEGY_DIR/logs/daily_update_YYYYMMDD.log"
echo "- 周度重训练: $STRATEGY_DIR/logs/weekly_retrain_YYYYMMDD.log"
echo "- 系统状态: $STRATEGY_DIR/status/"
echo ""
echo "🔍 监控命令:"
echo "- 查看定时任务: crontab -l"
echo "- 监控日志: tail -f $STRATEGY_DIR/logs/daily_cron.log"
echo "- 检查状态: python3 $STRATEGY_DIR/system_health_check.py"

echo ""
echo "✅ 自动化调度配置完成!"