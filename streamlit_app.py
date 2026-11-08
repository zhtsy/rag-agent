# -*- coding: utf-8 -*-
"""
streamlit_app.py —— 前端（流式对话 + 点赞/踩反馈）

1. requests.post(..., stream=True) + 手工按行解析 SSE（不用 iter_lines，它有 512 字节攒包）
2. st.write_stream 直接渲染 token → 真·打字机
3. session_id 写进 URL query_params，刷新不丢记忆
4. 历史只存在后端（MemorySaver），前端只发 session_id
5. 【新增】每条 AI 回答下方渲染 👍 / 👎，点击后 POST 到后端 /feedback
6. 【证据链】后端收尾帧会下发 contexts（本轮检索原文），这里挂到消息上，
   点踩时随反馈一起回传 —— 离线归因靠它做事实比对，而不是靠模型猜

关于反馈按钮"不报重复 key 错"的设计：
    - 每条消息在 append 进 st.session_state.messages 时就分到一个唯一 id（uuid），
      按钮 key 用 f"fb_good_{msg_id}" / f"fb_bad_{msg_id}"，天然唯一（历史消息也一样）。
    - 点击逻辑放在 on_click 回调里：Streamlit 会在"脚本重新执行之前"先跑回调，
      这样就不存在"写 session_state 时 widget 已实例化 → StreamlitAPIException"的坑。
    - 反馈结果存 st.session_state.fb_status[msg_id]，回调和渲染都只认这个字典，
      因此刷新/翻历史都不会重复提交，按钮也能保持"已反馈"态。
"""

import os
import re
import json
import uuid
from pathlib import Path

import requests
from qa_logger import log_qa, update_feedback_by_content
import streamlit as st

try:
    from api_auth import api_key as _server_api_key
except Exception:  # noqa: BLE001 - 老部署没有这个模块时，退化为"手填令牌"
    def _server_api_key() -> str:
        return ""

st.set_page_config(page_title="本地 RAG Agent", page_icon="🤖", layout="centered")
st.title("🤖 本地 RAG 智能助手")


def _auth_headers() -> dict:
    """统一的鉴权头。令牌由侧边栏写入 session_state["api_token"]。

    放在模块顶部而不是紧跟侧边栏：侧边栏那段代码在"本次脚本执行"时就会
    调用它取 /feedback/stats，函数必须在这之前已经定义好。
    回调（on_click）在下一轮 rerun 的最前面执行，那时 session_state 里仍是
    上一轮写入的令牌，所以回调里取到的值是对的。
    """
    token = (st.session_state.get("api_token") or "").strip()
    return {"X-API-Key": token} if token else {}


# =====================================================================
# 一、终极清洗（只在"整段答案为空"的判定里使用）
# =====================================================================
def clean_stream_text(text):
    if not text:
        return ""
    text = text.replace('"', '').replace('{', '').replace('}', '')
    text = re.sub(r'-answer:|answer:|quiry:|query:|sources:|confidence:', '', text)
    return text


# =====================================================================
# 二、会话 ID（刷新页面也不丢记忆）
# =====================================================================
def _get_or_create_session_id() -> str:
    try:
        qp = st.query_params
        sid = qp.get("sid")
        if isinstance(sid, list):
            sid = sid[0] if sid else None
        if sid:
            return sid
        new_sid = str(uuid.uuid4())
        qp["sid"] = new_sid
        return new_sid
    except Exception:  # noqa: BLE001 - 老版本 Streamlit 没有 query_params
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id


def _new_msg_id() -> str:
    """给每条消息发一个短唯一 ID，专门用来做按钮 key / 反馈去重。"""
    return uuid.uuid4().hex[:12]


def _default_api_base() -> str:
    """后端根地址的默认值：环境变量 → 本地/云端分口径。

    写死公网 IP 的后果是"本地把前端跑起来、点一下却打到云端"：
    要么连不上（云服务器安全组只放行特定来源），要么白烧 DeepSeek 额度。
    反过来直接把默认值改成 127.0.0.1 又会打坏云端前端——那边 streamlit 跑在
    容器里，本机 8000 不是后端。所以按 RAG_LOCAL_MODE 分叉，云端行为不变。
    """
    env = (os.getenv("RAG_API_BASE") or "").strip()
    if env:
        return env.rstrip("/")
    local = (os.getenv("RAG_LOCAL_MODE") or "0").strip().lower()
    if local in ("1", "true", "yes", "on"):
        return "http://127.0.0.1:8000"
    return "http://47.239.14.225:8000"


def _derive_feedback_url(chat_url: str) -> str:
    """从 /chat/stream 推导出 /feedback，省得用户手填两个地址。

    http://127.0.0.1:8000/chat/stream  →  http://127.0.0.1:8000/feedback
    """
    url = (chat_url or "").strip().rstrip("/")
    if url.endswith("/chat/stream"):
        return url[: -len("/chat/stream")] + "/feedback"
    return _default_api_base() + "/feedback"


def _render_contexts(msg: dict) -> None:
    """调试用：把这条回答当时的检索原文摊开看（默认收起，避免干扰阅读）。

    为什么要能看到：如果这里显示"（本轮没有检索到任何上下文）"，
    那这条 👎 大概率是"检索未命中"，而不用等离线报告。
    """
    contexts = msg.get("contexts") or []
    label = f"🔍 查看检索上下文（{len(contexts)} 块）"
    with st.expander(label, expanded=False):
        if not contexts:
            st.caption("（本轮没有检索到任何上下文）")
        for i, ctx in enumerate(contexts, 1):
            st.markdown(f"**上下文 {i}**")
            st.text(ctx[:1200] + ("…" if len(ctx) > 1200 else ""))


if "session_id" not in st.session_state:
    st.session_state.session_id = _get_or_create_session_id()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "fb_status" not in st.session_state:
    # msg_id -> "good" / "bad"，记录已反馈过的消息
    st.session_state.fb_status = {}


# =====================================================================
# 三、侧边栏
# =====================================================================
with st.sidebar:
    st.header("配置")
    api_url = st.text_input("后端 API", _default_api_base() + "/chat/stream")
    # 后端 /chat/stream 已加鉴权：密钥默认从服务端 .rag_api_key 读，
    # 输入框只是给"临时换后端/换密钥"留的口子，type="password" 不显示明文。
    api_token = st.text_input(
        "访问令牌",
        value=_server_api_key(),
        type="password",
        help="后端 /chat/stream 的 X-API-Key。默认自动读取服务器上的 .rag_api_key，一般不用改。",
    )
    feedback_url = st.text_input("反馈接口", _derive_feedback_url(api_url))
    # 反馈接口也走同一把钥匙：存进 session_state，_auth_headers() 统一取，
    # 免得把 api_token 作为参数串进 4 层函数签名。
    st.session_state["api_token"] = api_token
    show_sources = st.checkbox("显示引用来源", value=True)
    show_contexts = st.checkbox("显示检索上下文（调试用）", value=False)

    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.session_state.fb_status = {}
        new_sid = str(uuid.uuid4())
        st.session_state.session_id = new_sid
        try:
            st.query_params["sid"] = new_sid
        except Exception:  # noqa: BLE001
            pass
        st.rerun()

    st.caption(f"会话 ID: {st.session_state.session_id[:8]}")

    # 顺手展示一下反馈计数（失败就静默，不影响主流程）
    if st.button("📊 查看反馈统计"):
        try:
            _r = requests.get(feedback_url.replace("/feedback", "/feedback/stats"),
                              headers=_auth_headers(), timeout=3)
            if _r.ok:
                _d = _r.json()
                st.caption(f"总计 {_d.get('total', 0)} 条 ｜ 👍 {_d.get('good', 0)} ｜ 👎 {_d.get('bad', 0)}")
            else:
                st.caption(f"统计接口返回 {_r.status_code}")
        except Exception as _e:  # noqa: BLE001
            st.caption(f"统计失败：{_e}")


# =====================================================================
# 四、反馈相关：发送 + 渲染
# =====================================================================
def _post_feedback(feedback_url: str, session_id: str, question: str, answer: str,
                   feedback: str, sources=None, message_id=None, contexts=None):
    """POST 到后端 /feedback。返回 (是否成功, 错误信息)。

    contexts 是这条回答当时的【检索原始文档块】——归因脚本靠它判断
    "回答是忠于上下文" 还是 "无中生有"，没有它就只能瞎猜。
    """
    payload = {
        "session_id": session_id,
        "question": question or "",
        "answer": answer or "",
        "feedback": feedback,          # "good" / "bad"
        "contexts": contexts or [],    # ★ 证据
        "sources": sources or [],
        "message_id": message_id,
    }
    try:
        resp = requests.post(feedback_url, json=payload,
                             headers={"Content-Type": "application/json", **_auth_headers()},
                             timeout=(3, 15))
        if resp.status_code in (401, 403):
            return False, "反馈接口鉴权失败，请检查侧边栏的访问令牌"
        resp.raise_for_status()
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _make_feedback_callback(msg_id: str, question: str, answer: str,
                            sources, feedback_url: str, value: str, contexts=None):
    """生成一个 on_click 回调（闭包捕获这条消息的上下文，保证不会点错行）。

    注意：回调里不要 st.rerun()，Streamlit 跑完回调后本来就会重跑脚本。
    """
    def _cb():
        ok, err = _post_feedback(
            feedback_url=feedback_url,
            session_id=st.session_state.session_id,
            question=question,
            answer=answer,
            feedback=value,
            sources=sources,
            message_id=msg_id,
            contexts=contexts,
        )
        if ok:
            st.session_state.fb_status[msg_id] = value          # 记住这条已反馈
            # ★ 同步更新问答记录中的反馈
            try:
                update_feedback_by_content(
                    session_id=st.session_state.session_id,
                    question=question,
                    answer=answer,
                    feedback=value
                )
            except Exception:
                pass  # 反馈记录失败不影响主流程
            st.session_state["_pending_toast"] = "感谢反馈！"     # 交给主脚本去 toast
        else:
            st.session_state["_pending_toast"] = f"反馈提交失败：{err}"

    return _cb


def render_feedback_buttons(msg: dict, question: str, feedback_url: str):
    """在一条 AI 回答下方渲染 👍 / 👎。

    - msg["id"] 必须已存在（历史消息会在渲染前补发 ID）
    - 已反馈过的消息按钮置灰 + 文案变成"已赞 / 已踩"，防止重复刷日志
    """
    msg_id = msg.get("id") or _new_msg_id()
    msg["id"] = msg_id                       # 兜底回写，老数据也能用

    status = st.session_state.fb_status.get(msg_id)
    answer = msg.get("content", "")
    sources = msg.get("sources") or []
    contexts = msg.get("contexts") or []      # ★ 这条回答当时的检索原文，点踩时一起回传

    col_like, col_dislike, col_hint = st.columns([1, 1, 6])
    with col_like:
        st.button(
            "👍 已赞" if status == "good" else "👍 赞",
            key=f"fb_good_{msg_id}",                      # ← 唯一 key，历史消息也不会撞
            disabled=status is not None,                  # 已反馈则锁定
            on_click=_make_feedback_callback(
                msg_id, question, answer, sources, feedback_url, "good", contexts
            ),
        )
    with col_dislike:
        st.button(
            "👎 已踩" if status == "bad" else "👎 踩",
            key=f"fb_bad_{msg_id}",
            disabled=status is not None,
            on_click=_make_feedback_callback(
                msg_id, question, answer, sources, feedback_url, "bad", contexts
            ),
        )
    with col_hint:
        if status:
            st.caption("✅ 已记录你的反馈，谢谢！")


# =====================================================================
# 五、历史消息渲染（逐条渲染 + 每条 AI 消息都带反馈按钮）
# =====================================================================
# 关键：反馈按钮要知道"这条回答是回答哪个问题的"，
# 所以边遍历边记住"最近一条 user 消息"的内容，它就是当前 AI 消息的 question。
last_user_content = ""

for msg in st.session_state.messages:
    role = msg.get("role")
    msg.setdefault("id", _new_msg_id())      # 兼容升级前存下的老消息

    if role == "user":
        last_user_content = msg.get("content", "")
        with st.chat_message("user"):
            st.markdown(last_user_content)
    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg.get("content", ""))
            if msg.get("sources"):
                st.caption("📚 来源：" + "、".join(msg["sources"]))
            elif msg.get("status") == "out_of_scope":
                st.caption("📭 知识库中未检索到相关内容（本轮无引用来源）")
            if show_contexts:
                _render_contexts(msg)
            render_feedback_buttons(msg, last_user_content, feedback_url)


# =====================================================================
# 六、SSE 消费 + 打字机渲染
# =====================================================================
def sse_stream(api_url: str, payload: dict, meta_out: dict, api_key: str = ""):
    """
    生成器：消费后端 SSE，每拿到一个 data: {"text": ...} 就原样 yield。

    为什么不用 resp.iter_lines()：
    它默认 chunk_size=512，短回答（<512 字节）会一直攒到连接结束才吐出来，
    正好表现为"等全部生成完再一次性显示"。改用 iter_content(chunk_size=None)，
    有多少读多少，配合手动按 \\n 切行，才是真流式。

    鉴权：后端 /chat/stream 要求 X-API-Key。401 要单独给提示，
    否则只会显示成没头没尾的"请求失败：401 Client Error"。
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        with requests.post(api_url, json=payload, headers=headers,
                           stream=True, timeout=(10, 600)) as resp:
            if resp.status_code == 401:
                yield "\n\n❌ 访问令牌无效或缺失（401）。请在左侧「访问令牌」填入后端密钥。"
                return
            if resp.status_code == 503:
                yield "\n\n❌ 后端未配置访问密钥（503），请联系管理员检查 .rag_api_key。"
                return
            if resp.status_code == 429:
                # 并发闸门：同时在跑的请求已到上限，后端选择快速失败而不是无限排队。
                # 这里要把"等 5 秒重试"明说，否则用户会以为系统坏了。
                retry = resp.headers.get("Retry-After", "5")
                yield f"\n\n⏳ 服务繁忙（429）：当前排队已满，请约 {retry} 秒后重试。"
                return
            resp.raise_for_status()

            buf = ""
            for raw in resp.iter_content(chunk_size=None):
                if not raw:
                    continue
                buf += raw.decode("utf-8", errors="replace")

                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        return

                    try:
                        data = json.loads(data_str)
                    except Exception:  # noqa: BLE001
                        continue

                    if "text" in data:
                        # 原样 yield，不做任何缓冲
                        yield data["text"]
                    elif "sources" in data:
                        meta_out["sources"] = data.get("sources") or []
                        # 别用 0.0 兜底：拒答时后端给的就是 null（置信度不适用）
                        meta_out["confidence"] = data.get("confidence")
                        meta_out["status"] = data.get("status", "answered")
                        # ★ 收尾帧里同时带了原始文档块，一起收进 meta，稍后挂到消息上
                        meta_out["contexts"] = data.get("contexts") or []
                    elif "error" in data:
                        yield f"\n\n❌ 后端异常：{data['error']}"

    except requests.exceptions.RequestException as e:
        yield f"\n\n❌ 请求失败：{e}"


# =====================================================================
# 七、本轮对话
# =====================================================================
if prompt := st.chat_input("请输入您的问题..."):
    # 用户消息也带 id（虽然不渲染按钮，但保持结构统一，方便以后做"追问/重答"）
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "id": _new_msg_id()}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        meta = {}
        payload = {
            "message": prompt,
            "session_id": st.session_state.session_id,   # 只发 ID，历史在后端
        }

        answer = st.write_stream(sse_stream(api_url, payload, meta, api_token)) or ""
        if isinstance(answer, list):                     # 极老版本可能返回 chunk 列表
            answer = "".join(str(x) for x in answer)
        answer = answer.strip()

        if not clean_stream_text(answer).strip():
            answer = "⚠️ 后端没有返回有效内容，请查看服务端日志。"
            st.markdown(answer)

        sources = meta.get("sources") or []
        contexts = meta.get("contexts") or []     # ★ 后端收尾帧下发的原始文档块
        status = meta.get("status") or "answered"
        if show_sources and sources:
            st.caption("📚 来源：" + "、".join(sources))
        elif show_sources and status == "out_of_scope":
            # 明确告诉用户"没有来源"是结论的一部分，而不是漏渲染：
            # 旧版本这里什么都不显示，后端却会把 0.00x 分的无关块当来源挂上来看起来像有据可依。
            st.caption("📭 知识库中未检索到相关内容（本轮无引用来源）")

        # ---- 先入库，再渲染按钮（题目要求：存进 messages 之后）----
        # contexts 必须跟着消息一起存：否则刷新页面后，历史消息点踩就没有证据可传了
        assistant_msg = {
            "role": "assistant",
            "content": answer,
            "sources": sources if show_sources else [],
            "contexts": contexts,
            "confidence": meta.get("confidence"),
            "status": status,
            "id": _new_msg_id(),
        }
        st.session_state.messages.append(assistant_msg)
        # ★ 持久化存储本轮问答
        try:
            log_qa(
                session_id=st.session_state.session_id,
                question=prompt,
                answer=answer,
                sources=sources,
                contexts=contexts,
                confidence=meta.get("confidence")
            )
        except Exception as _log_err:
            st.toast(f"⚠️ 问答记录失败：{_log_err}")

        if show_contexts:
            _render_contexts(assistant_msg)

        # question 用本轮 prompt（而不是 last_user_content），语义最准确
        render_feedback_buttons(assistant_msg, prompt, feedback_url)


# =====================================================================
# 八、统一的 toast 出口
# =====================================================================
# 回调里 st.toast 在某些版本里会"闪一下就被 rerun 冲掉"，
# 所以把提示语塞进 session_state，等脚本跑完再统一弹，稳得多。
_pending_toast = st.session_state.pop("_pending_toast", None)
if _pending_toast:
    try:
        st.toast(_pending_toast)
    except Exception:  # noqa: BLE001 - 老版本没有 st.toast
        st.info(_pending_toast)
