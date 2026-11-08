#!/usr/bin/env bash
# RAG 系统备份：只备份"重建很贵或不可重建"的东西。
#
# 分工：
#   MySQL  —— 会话、反馈、评测记录。**不可重建**，是备份的重点。
#   Neo4j  —— 由 build_graph.py 从知识库源文件生成。备份源文件即可，
#             比 dump 数据库更小、更可读，恢复时直接重建。
#   Chroma —— 同理，由源文件向量化而来。
#
# 用法: ./backup.sh          正常备份
#       ./backup.sh --check  只打印将要备份什么，不落盘
set -euo pipefail

APP_DIR=/root/rag_deploy
BACKUP_DIR=/root/backups
KEEP_DAYS=7
TS=$(date +%Y%m%d_%H%M%S)

CHECK=0
if [ "${1:-}" = "--check" ]; then CHECK=1; fi

mkdir -p "$BACKUP_DIR"

# 口令从容器实际环境读，不从 .env 猜：容器里生效的才是真的。
# 用 MYSQL_PWD 传给客户端，避免密码出现在进程参数里（ps 可见）。
MYSQL_PW=$(docker exec rag_mysql printenv MYSQL_ROOT_PASSWORD || true)
MYSQL_DB=$(docker exec rag_mysql printenv MYSQL_DATABASE || true)
if [ -z "${MYSQL_PW}" ] || [ -z "${MYSQL_DB}" ]; then
    echo "FATAL: 拿不到 MySQL 口令/库名，检查 rag_mysql 环境变量" >&2
    exit 1
fi

echo "[1/2] MySQL ($MYSQL_DB)"
if [ "$CHECK" = "1" ]; then
    echo "  would dump -> $BACKUP_DIR/mysql_$TS.sql.gz"
else
    # --single-transaction: InnoDB 一致性快照，不锁表，线上可跑。
    # 不用 --databases：那样 dump 里会带 CREATE DATABASE / USE `rag_qa`，
    # 恢复演练时一不留神就写进生产库。库名交给恢复方显式指定。
    docker exec -e MYSQL_PWD="$MYSQL_PW" rag_mysql mysqldump -uroot \
        --single-transaction --quick --routines "$MYSQL_DB" \
        | gzip > "$BACKUP_DIR/mysql_$TS.sql.gz"
    echo "  -> $(du -h "$BACKUP_DIR/mysql_$TS.sql.gz" | cut -f1)"
fi

echo "[2/2] 知识库源文件（Neo4j / Chroma 的重建原料）"
if [ "$CHECK" = "1" ]; then
    echo "  would tar -> $BACKUP_DIR/kb_$TS.tar.gz"
else
    # 只打源数据与图谱构建脚本，不打 .env、日志、缓存
    FILES=$(cd "$APP_DIR" && ls *.txt *.json build_graph.py 2>/dev/null || true)
    if [ -n "$FILES" ]; then
        tar -czf "$BACKUP_DIR/kb_$TS.tar.gz" -C "$APP_DIR" $FILES
        echo "  -> $(du -h "$BACKUP_DIR/kb_$TS.tar.gz" | cut -f1)"
    else
        echo "  WARN: 没找到源文件，跳过" >&2
    fi
fi

if [ "$CHECK" = "0" ]; then
    # 轮转：只留最近 KEEP_DAYS 天
    find "$BACKUP_DIR" -name 'mysql_*.sql.gz' -mtime +"$KEEP_DAYS" -delete
    find "$BACKUP_DIR" -name 'kb_*.tar.gz'    -mtime +"$KEEP_DAYS" -delete
    echo "保留策略：$KEEP_DAYS 天，当前 $(ls -1 "$BACKUP_DIR" | wc -l) 个文件"
    # 空文件说明 dump 失败，宁可报错也不要留下"看起来有备份"的假象
    if [ ! -s "$BACKUP_DIR/mysql_$TS.sql.gz" ]; then
        echo "FATAL: dump 为空" >&2
        exit 1
    fi
fi
