#!/usr/bin/env bash
# 恢复演练：把最近一次 MySQL 备份导入临时库，逐表比对行数。
# 没验证过的备份等于没有备份，这个脚本就是"验证"那一步。
set -euo pipefail

DUMP=$(ls -t /root/backups/mysql_*.sql.gz 2>/dev/null | head -1)
if [ -z "$DUMP" ]; then
    echo "FATAL: /root/backups 下没有 dump" >&2
    exit 1
fi

PW=$(docker exec rag_mysql printenv MYSQL_ROOT_PASSWORD)
SRC=$(docker exec rag_mysql printenv MYSQL_DATABASE)
TMP=rag_restore_test

echo "dump   : $DUMP ($(du -h "$DUMP" | cut -f1))"
echo "tables : dump 内 CREATE TABLE x $(zcat "$DUMP" | grep -c 'CREATE TABLE')"

docker exec -e MYSQL_PWD="$PW" rag_mysql mysql -uroot \
    -e "DROP DATABASE IF EXISTS $TMP; CREATE DATABASE $TMP CHARACTER SET utf8mb4;"

zcat "$DUMP" | docker exec -i -e MYSQL_PWD="$PW" rag_mysql mysql -uroot "$TMP"

# 逐表比对行数，不只看表数量——表在但数据没进去是最常见的"假成功"
docker exec -e MYSQL_PWD="$PW" rag_mysql mysql -uroot -N -B -e \
    "select table_name from information_schema.tables where table_schema='$SRC' order by table_name" \
    > /tmp/_tabs.txt

FAIL=0
while read -r t; do
    [ -z "$t" ] && continue
    a=$(docker exec -e MYSQL_PWD="$PW" rag_mysql mysql -uroot -N -B -e "select count(*) from \`$SRC\`.\`$t\`")
    b=$(docker exec -e MYSQL_PWD="$PW" rag_mysql mysql -uroot -N -B -e "select count(*) from \`$TMP\`.\`$t\`")
    if [ "$a" = "$b" ]; then
        echo "  $t: $a == $b  OK"
    else
        echo "  $t: $a != $b  MISMATCH"
        FAIL=1
    fi
done < /tmp/_tabs.txt

docker exec -e MYSQL_PWD="$PW" rag_mysql mysql -uroot -e "DROP DATABASE $TMP"

if [ "$FAIL" = "0" ]; then
    echo "RESTORE_OK（临时库已清理）"
else
    echo "RESTORE_FAILED" >&2
    exit 1
fi
