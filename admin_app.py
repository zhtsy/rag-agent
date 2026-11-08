"""
QA History 数据管理后台 (MySQL版)
功能：列表查看、搜索、详情、删除、反馈标注、导出CSV
技术栈：Flask + SQLAlchemy + PyMySQL + Bootstrap5
"""
import os
import csv
import hmac
import json
from datetime import datetime
from flask import (
    Flask, render_template_string, request, redirect,
    url_for, flash, Response
)
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# 口令原来以默认值的形式硬编码在这里（等于把生产库密码提交进 git），
# 现在统一走 settings.py；没配就明确报错，而不是静默连一个错地址再超时。
from settings import require_database_url

# 后端 / 前端 / 评测 / 本管理台共用同一个密钥来源（api_auth 是唯一读取点）。
# 之前这个台子没有任何鉴权，而它能导出全部对话记录、删记录、改标注，
# 还是公网可达——等于"给数据开了个后门"。鉴权就挂在 before_request 上，
# 这样新增路由不会忘记加保护（默认拒绝，而不是默认放行）。
from api_auth import api_key

app = Flask(__name__)
app.secret_key = os.urandom(24)


@app.before_request
def _require_api_key():
    """全站鉴权：密钥缺失则拒绝（fail-closed），比对用常量时间。"""
    expected = api_key()
    if not expected:
        # 密钥文件丢了也照样拒绝：这个台子能导出全库，
        # 宁可让运维去补密钥，也不能默认敞开。
        return Response(
            "管理后台未配置访问密钥：请检查 /root/rag_deploy/.rag_api_key。",
            status=503,
            mimetype="text/plain; charset=utf-8",
        )

    # 浏览器直接打开时用 ?key=...；脚本调用优先走请求头。
    supplied = request.headers.get("X-API-Key") or request.args.get("key") or ""
    if not hmac.compare_digest(supplied, expected):
        msg = "缺少或错误的访问密钥：请在 URL 加 ?key=<密钥>，或用请求头 X-API-Key。"
        return Response(msg, status=401, mimetype="text/plain; charset=utf-8")
    return None

DATABASE_URL = require_database_url()

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine)


def get_session():
    return SessionLocal()


# ─── HTML Template (Single-file, Bootstrap5) ───────────────────────
TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QA History Admin</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  .truncate { max-width:300px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .confidence-high { color:#198754; font-weight:bold; }
  .confidence-mid  { color:#fd7e14; font-weight:bold; }
  .confidence-low  { color:#dc3545; font-weight:bold; }
</style>
</head>
<body class="bg-light">
<nav class="navbar navbar-dark bg-dark mb-4">
  <div class="container-fluid"><span class="navbar-brand mb-0 h1">📋 QA History 管理后台</span></div>
</nav>
<div class="container-fluid">
  {% with messages = get_flashed_messages(with_categories=true) %}
  {% if messages %}{% for cat,msg in messages %}
  <div class="alert alert-{{ cat }} alert-dismissible fade show">{{ msg }}
    <button type="button" class="btn-close" data-bs-dismiss="alert"></button></div>
  {% endfor %}{% endif %}{% endwith %}
  <!-- Search & Export Bar -->
  <form method="get" class="row g-2 mb-3 align-items-end">
    <div class="col-md-3"><input name="q" class="form-control" placeholder="搜索问题/答案/session_id" value="{{ q }}"></div>
    <div class="col-md-2"><input name="session_id" class="form-control" placeholder="Session ID" value="{{ session_filter }}"></div>
    <div class="col-md-2"><select name="feedback" class="form-select">
      <option value="">全部反馈</option>
      <option value="good" {{ 'selected' if feedback_filter=='good' }}>👍 Good</option>
      <option value="bad"  {{ 'selected' if feedback_filter=='bad'  }}>👎 Bad</option>
      <option value="none" {{ 'selected' if feedback_filter=='none' }}>无反馈</option>
    </select></div>
    <div class="col-md-2"><button class="btn btn-primary w-100">🔍 搜索</button></div>
    <div class="col-md-2"><a href="{{ url_for('export_csv') }}" class="btn btn-outline-success w-100">📥 导出CSV</a></div>
  </form>
  <!-- Stats -->
  <div class="row mb-3">
    <div class="col-auto"><span class="badge bg-secondary fs-6">总计: {{ total }}</span></div>
    <div class="col-auto"><span class="badge bg-info fs-6">当前页: {{ records|length }}</span></div>
  </div>
  <!-- Table -->
  <div class="table-responsive">
  <table class="table table-hover table-bordered bg-white">
    <thead class="table-dark">
      <tr>
        <th>ID</th><th>Session</th><th>时间</th><th>问题</th><th>答案</th>
        <th>置信度</th><th>反馈</th><th>Sources</th><th>操作</th>
      </tr>
    </thead>
    <tbody>
    {% for r in records %}
    <tr>
      <td>{{ r.id }}</td>
      <td><code>{{ r.session_id[:12] }}</code></td>
      <td>{{ r.timestamp }}</td>
      <td class="truncate" title="{{ r.question }}">{{ r.question }}</td>
      <td class="truncate" title="{{ r.answer }}">{{ r.answer }}</td>
      <td>
        {% if r.confidence is not none %}
          <span class="{{ 'confidence-high' if r.confidence>=0.8 else ('confidence-mid' if r.confidence>=0.5 else 'confidence-low') }}">
            {{ "%.2f"|format(r.confidence) }}
          </span>
        {% else %}-{% endif %}
      </td>
      <td>
        {% if r.feedback == 'good' %}👍{% elif r.feedback == 'bad' %}👎{% else %}<span class="text-muted">-</span>{% endif %}
      </td>
      <td class="truncate" title="{{ r.sources }}">{{ r.sources or '-' }}</td>
      <td>
        <a href="{{ url_for('detail', record_id=r.id) }}" class="btn btn-sm btn-outline-primary">详情</a>
        <form method="post" action="{{ url_for('delete', record_id=r.id) }}" class="d-inline"
              onsubmit="return confirm('确认删除 #{{ r.id }}?')">
          <button class="btn btn-sm btn-outline-danger">删除</button>
        </form>
      </td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
  </div>
  <!-- Pagination -->
  {% if total_pages > 1 %}
  <nav><ul class="pagination justify-content-center">
    {% for p in range(1, total_pages+1) %}
    <li class="page-item {{ 'active' if p==page }}">
      <a class="page-link" href="?page={{ p }}&q={{ q }}&session_id={{ session_filter }}&feedback={{ feedback_filter }}">{{ p }}</a>
    </li>{% endfor %}
  </ul></nav>
  {% endif %}
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body></html>
"""

DETAIL_TEMPLATE = """
<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>QA Record #{{ r.id }}</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head><body class="bg-light">
<nav class="navbar navbar-dark bg-dark mb-4">
  <div class="container-fluid">
    <a class="navbar-brand" href="{{ url_for('index') }}">← 返回列表</a>
    <span class="navbar-text text-white">QA Record #{{ r.id }}</span>
  </div>
</nav>
<div class="container">
<div class="card mb-3"><div class="card-body">
  <h5 class="card-title">❓ Question</h5>
  <p class="card-text">{{ r.question }}</p>
</div></div>
<div class="card mb-3"><div class="card-body">
  <h5 class="card-title">💡 Answer</h5>
  <p class="card-text" style="white-space:pre-wrap">{{ r.answer }}</p>
</div></div>
<div class="row">
  <div class="col-md-6"><div class="card mb-3"><div class="card-body">
    <h6>📎 Sources</h6><pre class="mb-0">{{ r.sources or 'N/A' }}</pre>
  </div></div></div>
  <div class="col-md-6"><div class="card mb-3"><div class="card-body">
    <h6>📚 Contexts</h6><pre class="mb-0">{{ r.contexts or 'N/A' }}</pre>
  </div></div></div>
</div>
<div class="row">
  <div class="col-md-4"><div class="card mb-3"><div class="card-body">
    <h6>Session ID</h6><code>{{ r.session_id }}</code>
  </div></div></div>
  <div class="col-md-4"><div class="card mb-3"><div class="card-body">
    <h6>Timestamp</h6>{{ r.timestamp }}
  </div></div></div>
  <div class="col-md-4"><div class="card mb-3"><div class="card-body">
    <h6>Confidence</h6>{{ r.confidence if r.confidence is not none else 'N/A' }}
  </div></div></div>
</div>
<div class="card mb-3"><div class="card-body">
  <h6>Metadata</h6><pre class="mb-0">{{ r.metadata or 'N/A' }}</pre>
</div></div>
<!-- Feedback Form -->
<div class="card mb-3"><div class="card-body">
  <h6>✏️ 标注反馈</h6>
  <form method="post" action="{{ url_for('update_feedback', record_id=r.id) }}" class="d-flex gap-2">
    <select name="feedback" class="form-select w-auto">
      <option value="good" {{ 'selected' if r.feedback=='good' }}>👍 Good</option>
      <option value="bad"  {{ 'selected' if r.feedback=='bad'  }}>👎 Bad</option>
      <option value=""     {{ 'selected' if not r.feedback     }}>清除</option>
    </select>
    <button class="btn btn-primary">保存</button>
  </form>
</div></div>
<form method="post" action="{{ url_for('delete', record_id=r.id) }}"
      onsubmit="return confirm('确认删除?')">
  <button class="btn btn-danger">🗑️ 删除此记录</button>
</form>
</div>
</body></html>
"""

# ─── Routes ─────────────────────────────────────────────────────────
PER_PAGE = 20


@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    session_filter = request.args.get("session_id", "").strip()
    feedback_filter = request.args.get("feedback", "").strip()
    page = max(1, request.args.get("page", 1, type=int))

    conditions, params = [], {}
    param_idx = [0]

    def next_param(val):
        key = f"p{param_idx[0]}"
        param_idx[0] += 1
        params[key] = val
        return f":{key}"

    if q:
        p1 = next_param(f"%{q}%")
        p2 = next_param(f"%{q}%")
        p3 = next_param(f"%{q}%")
        conditions.append(f"(question LIKE {p1} OR answer LIKE {p2} OR session_id LIKE {p3})")
    if session_filter:
        conditions.append(f"session_id = {next_param(session_filter)}")
    if feedback_filter == "good":
        conditions.append("feedback = 'good'")
    elif feedback_filter == "bad":
        conditions.append("feedback = 'bad'")
    elif feedback_filter == "none":
        conditions.append("(feedback IS NULL OR feedback = '')")

    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    with SessionLocal() as session:
        total = session.execute(
            text(f"SELECT COUNT(*) FROM qa_records{where}"), params
        ).scalar()
        total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
        offset = (page - 1) * PER_PAGE
        rows = session.execute(
            text(f"SELECT * FROM qa_records{where} ORDER BY id DESC LIMIT :limit OFFSET :offset"),
            {**params, "limit": PER_PAGE, "offset": offset}
        ).fetchall()
        columns = rows[0]._mapping.keys() if rows else []
        records = [dict(zip(columns, row)) for row in rows]

    return render_template_string(
        TEMPLATE, records=records, total=total, page=page,
        total_pages=total_pages, q=q,
        session_filter=session_filter, feedback_filter=feedback_filter
    )


@app.route("/detail/<int:record_id>")
def detail(record_id):
    with SessionLocal() as session:
        row = session.execute(
            text("SELECT * FROM qa_records WHERE id = :id"), {"id": record_id}
        ).fetchone()
    if not row:
        flash("记录不存在", "danger")
        return redirect(url_for("index"))
    columns = row._mapping.keys()
    r = dict(zip(columns, row))
    return render_template_string(DETAIL_TEMPLATE, r=r)


@app.route("/update_feedback/<int:record_id>", methods=["POST"])
def update_feedback(record_id):
    fb = request.form.get("feedback", "")
    with SessionLocal() as session:
        session.execute(
            text("UPDATE qa_records SET feedback = :fb WHERE id = :id"),
            {"fb": fb, "id": record_id}
        )
        session.commit()
    flash("反馈已更新", "success")
    return redirect(url_for("detail", record_id=record_id))


@app.route("/delete/<int:record_id>", methods=["POST"])
def delete(record_id):
    with SessionLocal() as session:
        session.execute(
            text("DELETE FROM qa_records WHERE id = :id"), {"id": record_id}
        )
        session.commit()
    flash(f"记录 #{record_id} 已删除", "warning")
    return redirect(url_for("index"))


@app.route("/export")
def export_csv():
    with SessionLocal() as session:
        rows = session.execute(text("SELECT * FROM qa_records ORDER BY id")).fetchall()
        columns = rows[0]._mapping.keys() if rows else []
        records = [dict(zip(columns, row)) for row in rows]

    output = Response(content_type="text/csv; charset=utf-8-sig")
    output.headers["Content-Disposition"] = f"attachment; filename=qa_history_{datetime.now():%Y%m%d_%H%M%S}.csv"
    writer = csv.writer(output)
    writer.writerow(["id","session_id","timestamp","question","answer","sources","contexts","confidence","feedback","metadata"])
    for r in records:
        writer.writerow([r["id"],r["session_id"],r["timestamp"],r["question"],r["answer"],
                         r["sources"],r["contexts"],r["confidence"],r["feedback"],r["metadata"]])
    return output


if __name__ == "__main__":
    print(f"✅ QA Admin running at http://0.0.0.0:5001")
    print(f"🗄️  DB: MySQL ({DATABASE_URL.split('@')[1].split('/')[0] if '@' in DATABASE_URL else 'unknown'})")
    # 绝不能开 debug：Werkzeug 的交互式调试器会在异常页面上给出可执行代码的控制台，
    # 而这个服务没有鉴权、5001 又是对外映射的，等于把远程代码执行直接挂在公网上。
    # 顺带一提，debug 的 reloader 会额外 fork 一个进程常驻，白占几十 MB 内存。
    app.run(host="0.0.0.0", port=5001, debug=False)
