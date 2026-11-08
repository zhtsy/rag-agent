#!/bin/bash
# QA 记录定时清理 - Cron 配置脚本
# 默认每天凌晨3点清理超过30天的记录

RETENTION_DAYS=${1:-30}
SCRIPT_PATH="/root/rag_deploy/qa_logger.py"
CRON_CMD="0 3 * * * cd /root/rag_deploy && python3 ${SCRIPT_PATH} cleanup ${RETENTION_DAYS} --run >> /root/rag_deploy/qa_cleanup.log 2>&1"

echo "📋 即将添加以下 cron 任务:"
echo "   ${CRON_CMD}"
echo ""

# 检查是否已存在相同任务
if crontab -l 2>/dev/null | grep -q "qa_logger.py cleanup"; then
    echo "⚠️  已存在 qa_logger 清理任务，先移除旧条目..."
    crontab -l 2>/dev/null | grep -v "qa_logger.py cleanup" | crontab -
fi

# 添加新任务
(crontab -l 2>/dev/null; echo "${CRON_CMD}") | crontab -

echo "✅ Cron 任务已设置！保留最近 ${RETENTION_DAYS} 天记录，每天凌晨3点自动清理"
echo "   查看日志: tail -f /root/rag_deploy/qa_cleanup.log"
echo "   手动测试: python3 ${SCRIPT_PATH} cleanup ${RETENTION_DAYS}      (dry-run)"
echo "             python3 ${SCRIPT_PATH} cleanup ${RETENTION_DAYS} --run (实际执行)"
