"""
QA Logger - 无侵入式问答记录模块 (MySQL版)
功能：持久化存储、反馈关联、多格式导出、定时清理
使用 SQLAlchemy + PyMySQL 替代 sqlite3，支持多进程并发写入
"""
import json
import csv
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# 从环境变量读取数据库连接，回退到本地 MySQL
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "mysql+pymysql://rag_user:051585913885w@127.0.0.1:3306/rag_qa?charset=utf8mb4"
)

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=False,
)
SessionLocal = sessionmaker(bind=engine)


def _init_db():
    """初始化数据库表结构"""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS qa_records (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                timestamp VARCHAR(64) NOT NULL,
                question LONGTEXT NOT NULL,
                answer LONGTEXT NOT NULL,
                sources LONGTEXT,
                contexts LONGTEXT,
                confidence DOUBLE,
                feedback TEXT,
                metadata LONGTEXT,
                INDEX idx_session (session_id),
                INDEX idx_timestamp (timestamp),
                INDEX idx_qa_match (session_id, question(255), answer(255))
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """))
        conn.commit()


_init_db()


# ==================== 核心写入 ====================
def log_qa(
    session_id: str,
    question: str,
    answer: str,
    sources: Optional[List[str]] = None,
    contexts: Optional[List[Dict[str, Any]]] = None,
    confidence: float = 0.0,
    feedback: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> int:
    """记录一条问答，返回记录ID"""
    with SessionLocal() as session:
        result = session.execute(
            text("""INSERT INTO qa_records
               (session_id, timestamp, question, answer, sources, contexts, confidence, feedback, metadata)
               VALUES (:session_id, :timestamp, :question, :answer, :sources, :contexts, :confidence, :feedback, :metadata)"""),
            {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
                "sources": json.dumps(sources or [], ensure_ascii=False),
                "contexts": json.dumps(contexts or [], ensure_ascii=False),
                "confidence": confidence,
                "feedback": feedback,
                "metadata": json.dumps(metadata or {}, ensure_ascii=False),
            }
        )
        record_id = result.lastrowid
        session.commit()
        return record_id


# ==================== 扩展1: 反馈关联 ====================
def update_feedback_by_content(session_id: str, question: str, answer: str, feedback: str) -> bool:
    """
    通过 session_id + question + answer 三元组匹配记录并更新反馈。
    Returns: 是否成功更新了至少一条记录
    """
    with SessionLocal() as session:
        # MySQL 不支持 UPDATE ... ORDER BY ... LIMIT 的子查询写法，用两步实现
        row = session.execute(
            text("""SELECT id FROM qa_records
                    WHERE session_id = :sid AND question = :q AND answer = :a AND feedback IS NULL
                    ORDER BY timestamp DESC LIMIT 1"""),
            {"sid": session_id, "q": question, "a": answer}
        ).fetchone()
        if not row:
            return False
        session.execute(
            text("UPDATE qa_records SET feedback = :fb WHERE id = :id"),
            {"fb": feedback, "id": row[0]}
        )
        session.commit()
        return True


def update_feedback_by_id(record_id: int, feedback: str) -> bool:
    """通过记录ID直接更新反馈"""
    with SessionLocal() as session:
        result = session.execute(
            text("UPDATE qa_records SET feedback = :fb WHERE id = :id"),
            {"fb": feedback, "id": record_id}
        )
        session.commit()
        return result.rowcount > 0


# ==================== 扩展2: 多格式导出 ====================
def export_to_csv(output_path: str, session_id: Optional[str] = None,
                  start_date: Optional[str] = None, end_date: Optional[str] = None) -> int:
    """导出为 CSV 文件，Returns: 导出记录数"""
    records = _query_records(session_id, start_date, end_date)
    if not records:
        return 0
    fieldnames = ["id", "session_id", "timestamp", "question", "answer",
                  "sources", "contexts", "confidence", "feedback", "metadata"]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    return len(records)


def export_to_json(output_path: str, session_id: Optional[str] = None,
                   start_date: Optional[str] = None, end_date: Optional[str] = None) -> int:
    """导出为 JSON 文件，Returns: 导出记录数"""
    records = _query_records(session_id, start_date, end_date)
    if not records:
        return 0
    for r in records:
        for field in ("sources", "contexts", "metadata"):
            if isinstance(r.get(field), str):
                try:
                    r[field] = json.loads(r[field])
                except (json.JSONDecodeError, TypeError):
                    pass
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return len(records)


def _query_records(session_id: Optional[str] = None,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> List[Dict]:
    """通用查询，支持按会话和时间范围过滤"""
    query = "SELECT * FROM qa_records WHERE 1=1"
    params: dict = {}
    if session_id:
        query += " AND session_id = :session_id"
        params["session_id"] = session_id
    if start_date:
        query += " AND timestamp >= :start_date"
        params["start_date"] = start_date
    if end_date:
        query += " AND timestamp <= :end_date"
        params["end_date"] = end_date
    query += " ORDER BY timestamp ASC"
    with SessionLocal() as session:
        rows = session.execute(text(query), params).fetchall()
        columns = rows[0]._mapping.keys() if rows else []
        return [dict(zip(columns, row)) for row in rows]


# ==================== 扩展3: 定时清理 ====================
def cleanup_old_records(retention_days: int = 30, dry_run: bool = False) -> Dict[str, Any]:
    """清理超过 retention_days 天的旧记录"""
    cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
    with SessionLocal() as session:
        count_row = session.execute(
            text("SELECT COUNT(*) as cnt FROM qa_records WHERE timestamp < :cutoff"),
            {"cutoff": cutoff}
        ).fetchone()
        to_delete = count_row[0]
        if not dry_run and to_delete > 0:
            session.execute(
                text("DELETE FROM qa_records WHERE timestamp < :cutoff"),
                {"cutoff": cutoff}
            )
            session.execute(text("OPTIMIZE TABLE qa_records"))
            session.commit()
        remaining = session.execute(
            text("SELECT COUNT(*) as cnt FROM qa_records")
        ).fetchone()[0]
    return {
        "deleted": to_delete if not dry_run else 0,
        "would_delete": to_delete if dry_run else 0,
        "remaining": remaining,
        "cutoff_date": cutoff
    }


# ==================== 查询接口 ====================
def get_records(session_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
    """查询历史记录"""
    with SessionLocal() as session:
        if session_id:
            rows = session.execute(
                text("SELECT * FROM qa_records WHERE session_id = :sid ORDER BY timestamp DESC LIMIT :lim"),
                {"sid": session_id, "lim": limit}
            ).fetchall()
        else:
            rows = session.execute(
                text("SELECT * FROM qa_records ORDER BY timestamp DESC LIMIT :lim"),
                {"lim": limit}
            ).fetchall()
        if not rows:
            return []
        columns = rows[0]._mapping.keys()
        return [dict(zip(columns, row)) for row in rows]


def get_stats() -> Dict[str, Any]:
    """获取数据库统计信息"""
    with SessionLocal() as session:
        total = session.execute(text("SELECT COUNT(*) FROM qa_records")).scalar()
        with_fb = session.execute(
            text("SELECT COUNT(*) FROM qa_records WHERE feedback IS NOT NULL")
        ).scalar()
        sessions = session.execute(
            text("SELECT COUNT(DISTINCT session_id) FROM qa_records")
        ).scalar()
        oldest = session.execute(text("SELECT MIN(timestamp) FROM qa_records")).scalar()
        newest = session.execute(text("SELECT MAX(timestamp) FROM qa_records")).scalar()
        # MySQL 表大小估算
        size_row = session.execute(
            text("SELECT data_length + index_length FROM information_schema.tables "
                 "WHERE table_schema = DATABASE() AND table_name = 'qa_records'")
        ).fetchone()
        db_size = size_row[0] if size_row else 0
    return {
        "total_records": total,
        "with_feedback": with_fb,
        "unique_sessions": sessions,
        "oldest_record": oldest,
        "newest_record": newest,
        "db_size_bytes": db_size
    }


# ==================== CLI 入口 ====================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法:")
        print("  python3 qa_logger.py test          # 写入测试数据")
        print("  python3 qa_logger.py stats          # 查看统计")
        print("  python3 qa_logger.py export-csv [path]  # 导出CSV")
        print("  python3 qa_logger.py export-json [path] # 导出JSON")
        print("  python3 qa_logger.py cleanup [days]     # 清理旧数据(dry-run)")
        print("  python3 qa_logger.py cleanup [days] --run # 实际清理")
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "test":
        rid = log_qa("test-session", "测试问题", "测试回答", ["doc1.txt"], confidence=0.95)
        print(f"✅ 测试记录已写入，ID={rid}")
        update_feedback_by_content("test-session", "测试问题", "测试回答", "good")
        print("✅ 反馈已关联")
    elif cmd == "stats":
        s = get_stats()
        print(f"📊 总记录: {s['total_records']} | 有反馈: {s['with_feedback']} | 会话数: {s['unique_sessions']}")
        print(f"   最早: {s['oldest_record']} | 最新: {s['newest_record']}")
        print(f"   数据库大小: {s['db_size_bytes'] / 1024:.1f} KB")
    elif cmd == "export-csv":
        path = sys.argv[2] if len(sys.argv) > 2 else "qa_export.csv"
        n = export_to_csv(path)
        print(f"✅ 已导出 {n} 条记录到 {path}")
    elif cmd == "export-json":
        path = sys.argv[2] if len(sys.argv) > 2 else "qa_export.json"
        n = export_to_json(path)
        print(f"✅ 已导出 {n} 条记录到 {path}")
    elif cmd == "cleanup":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        run = "--run" in sys.argv
        result = cleanup_old_records(retention_days=days, dry_run=not run)
        if run:
            print(f"🗑️  已删除 {result['deleted']} 条超过 {days} 天的记录，剩余 {result['remaining']} 条")
        else:
            print(f"🔍 [Dry-run] 将删除 {result['would_delete']} 条超过 {days} 天的记录，剩余 {result['remaining']} 条")
            print(f"   截止时间: {result['cutoff_date']}")
            print(f"   加 --run 参数执行实际清理")
    else:
        print(f"未知命令: {cmd}")
