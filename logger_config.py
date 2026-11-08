import sys
from loguru import logger

# 移除默认的控制台输出
logger.remove()

# 1. 输出到控制台（带颜色，格式更清爽）
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
)

# 2. 输出到文件（每天自动切分，保留 7 天，防止日志文件撑爆硬盘）
logger.add(
    "logs/rag_agent_{time:YYYY-MM-DD}.log",
    rotation="00:00",    # 每天 0 点切分
    retention="7 days",  # 只保留最近 7 天
    level="INFO",
    encoding="utf-8"
)

# 导出 logger 供其他文件使用
__all__ = ["logger"]