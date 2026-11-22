/* ============================================================
   法问 Lex — 前端逻辑（原生 JS，无构建、无依赖）
   结构：api 层 → 渲染层 → 事件层。
   流式用 fetch + ReadableStream 手解 SSE，而不是 EventSource：
   EventSource 不能带请求头，也没法发 POST 体。
   ============================================================ */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = {
  user: null,
  convs: [],
  currentId: null,
  streaming: false,
  authMode: 'login',
};

/* ---------------- 基础请求 ---------------- */
async function api(path, { method = 'GET', body } = {}) {
  const res = await fetch(path, {
    method,
    credentials: 'include',
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });

  let data = {};
  try { data = await res.json(); } catch { /* 空响应体 */ }

  if (res.status === 401) {
    showAuth();
    throw new Error('未登录');
  }
  if (!res.ok) {
    throw new Error(data.detail || `请求失败（${res.status}）`);
  }
  return data;
}

/* ---------------- 文本渲染 ---------------- */
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

/* 极简 markdown：标题 / 列表 / 粗体 / 行内代码 / 围栏代码 / 链接。
   不引 marked.js 是因为这里只需要这几种，而多一个 CDN 就多一个断网就白屏的风险。 */
function md(src) {
  const lines = esc(src).split('\n');
  const out = [];
  let list = null;
  let code = null;
  let para = [];

  const flushPara = () => {
    if (para.length) { out.push(`<p>${para.join('<br>')}</p>`); para = []; }
  };
  const flushList = () => {
    if (list) { out.push(`</${list}>`); list = null; }
  };
  const inline = (t) => t
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  for (const line of lines) {
    if (line.trim().startsWith('```')) {
      if (code === null) { flushPara(); flushList(); code = []; }
      else { out.push(`<pre><code>${code.join('\n')}</code></pre>`); code = null; }
      continue;
    }
    if (code !== null) { code.push(line); continue; }

    if (!line.trim()) { flushPara(); flushList(); continue; }

    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      flushPara(); flushList();
      const lv = Math.min(h[1].length + 2, 6);
      out.push(`<h${lv}>${inline(h[2])}</h${lv}>`);
      continue;
    }

    const ul = line.match(/^\s*[-*•]\s+(.*)$/);
    if (ul) {
      flushPara();
      if (list !== 'ul') { flushList(); out.push('<ul>'); list = 'ul'; }
      out.push(`<li>${inline(ul[1])}</li>`);
      continue;
    }

    const ol = line.match(/^\s*(\d+)[.、)]\s+(.*)$/);
    if (ol) {
      flushPara();
      if (list !== 'ol') { flushList(); out.push('<ol>'); list = 'ol'; }
      out.push(`<li>${inline(ol[2])}</li>`);
      continue;
    }

    if (line.startsWith('&gt;')) { flushPara(); flushList(); out.push(`<blockquote>${inline(line.slice(4))}</blockquote>`); continue; }

    flushList();
    para.push(inline(line));
  }

  if (code !== null) out.push(`<pre><code>${code.join('\n')}</code></pre>`);
  flushPara();
  flushList();
  return out.join('');
}

/* ---------------- 视图切换 ---------------- */
function showAuth() {
  state.user = null;
  $('#app-screen').classList.add('is-hidden');
  $('#auth-screen').classList.remove('is-hidden');
  setTimeout(() => $('#username')?.focus(), 60);
}

function showApp(user) {
  state.user = user;
  $('#auth-screen').classList.add('is-hidden');
  $('#app-screen').classList.remove('is-hidden');
  $('#user-name').textContent = user.username;
  $('#user-avatar').textContent = user.username.slice(0, 1).toUpperCase();
  $('#user-sub').textContent = user.email || '已登录';
  loadConversations();
  checkHealth();
  setTimeout(() => $('#input')?.focus(), 80);
}

/* ---------------- 登录 / 注册 ---------------- */
$$('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    state.authMode = tab.dataset.mode;
    $$('.tab').forEach((t) => t.classList.toggle('is-on', t === tab));
    $('#email-field').classList.toggle('is-hidden', state.authMode !== 'register');
    $('#auth-submit').textContent = state.authMode === 'register' ? '创建账号' : '登录';
    $('#auth-head').textContent = state.authMode === 'register' ? '创建账号' : '欢迎回来';
    $('#auth-sub').textContent = state.authMode === 'register'
      ? '注册后，对话会自动保存在你的账号下'
      : '登录后可以保存并回看你的全部对话';
    $('#password').autocomplete = state.authMode === 'register' ? 'new-password' : 'current-password';
    $('#auth-error').textContent = '';
  });
});

$('#auth-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = $('#auth-submit');
  const errBox = $('#auth-error');
  const username = $('#username').value.trim();
  const password = $('#password').value;

  errBox.textContent = '';
  btn.disabled = true;
  btn.textContent = state.authMode === 'register' ? '创建中…' : '登录中…';

  try {
    const path = state.authMode === 'register' ? '/auth/register' : '/auth/login';
    const body = { username, password };
    if (state.authMode === 'register') body.email = $('#email').value.trim();

    const data = await api(path, { method: 'POST', body });
    $('#password').value = '';
    showApp(data.user);
  } catch (err) {
    errBox.textContent = err.message === '未登录' ? '登录已过期，请重新登录' : err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = state.authMode === 'register' ? '创建账号' : '登录';
  }
});

$('#logout').addEventListener('click', async () => {
  try { await api('/auth/logout', { method: 'POST' }); } catch { /* 忽略 */ }
  state.convs = [];
  state.currentId = null;
  $('#stream').innerHTML = '';
  showAuth();
});

/* ---------------- 会话列表 ---------------- */
function dayGroup(ts) {
  const d = new Date(ts * 1000);
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const t = d.getTime();
  if (t >= startOfToday) return '今天';
  if (t >= startOfToday - 86400000) return '昨天';
  if (t >= startOfToday - 7 * 86400000) return '最近 7 天';
  return '更早';
}

function renderConvs() {
  const box = $('#conv-list');
  box.innerHTML = '';
  if (!state.convs.length) {
    box.innerHTML = '<p class="group-label" style="letter-spacing:.06em">还没有对话</p>';
    return;
  }

  const groups = new Map();
  for (const c of state.convs) {
    const g = dayGroup(c.updated_at);
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(c);
  }

  for (const [label, items] of groups) {
    const h = document.createElement('div');
    h.className = 'group-label';
    h.textContent = label;
    box.appendChild(h);

    for (const c of items) {
      const el = document.createElement('div');
      el.className = 'conv' + (c.id === state.currentId ? ' is-on' : '');
      el.dataset.id = c.id;

      const title = document.createElement('span');
      title.className = 'conv-title';
      title.textContent = c.title || '新对话';
      title.title = c.title || '新对话';

      const more = document.createElement('button');
      more.className = 'conv-more';
      more.type = 'button';
      more.textContent = '⋯';
      more.title = '更多';
      more.addEventListener('click', (ev) => {
        ev.stopPropagation();
        openMenu(more, c);
      });

      el.append(title, more);
      el.addEventListener('click', () => openConversation(c.id));
      el.addEventListener('dblclick', () => startRename(el, c));
      box.appendChild(el);
    }
  }
}

async function loadConversations() {
  try {
    const data = await api('/conversations');
    state.convs = data.items || [];
    renderConvs();
  } catch { /* 已由 api 处理 401 */ }
}

function closeMenu() {
  $$('.menu').forEach((m) => m.remove());
}

function openMenu(anchor, conv) {
  closeMenu();
  const rect = anchor.getBoundingClientRect();
  const menu = document.createElement('div');
  menu.className = 'menu';
  menu.style.top = `${rect.bottom + 4}px`;
  menu.style.left = `${Math.max(8, rect.right - 136)}px`;

  const rename = document.createElement('button');
  rename.textContent = '重命名';
  rename.addEventListener('click', () => {
    closeMenu();
    const el = $(`.conv[data-id="${conv.id}"]`);
    if (el) startRename(el, conv);
  });

  const del = document.createElement('button');
  del.className = 'danger';
  del.textContent = '删除对话';
  del.addEventListener('click', async () => {
    closeMenu();
    if (!confirm(`删除「${conv.title || '新对话'}」？此操作不可撤销。`)) return;
    try {
      await api(`/conversations/${conv.id}`, { method: 'DELETE' });
      state.convs = state.convs.filter((c) => c.id !== conv.id);
      if (state.currentId === conv.id) {
        state.currentId = null;
        $('#conv-title').textContent = '新对话';
        $('#conv-meta').textContent = '尚未开始';
        renderEmpty();
      }
      renderConvs();
    } catch (err) { alert(err.message); }
  });

  menu.append(rename, del);
  document.body.appendChild(menu);
}

document.addEventListener('click', closeMenu);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeMenu(); });

function startRename(el, conv) {
  const title = $('.conv-title', el);
  if (!title || el.querySelector('input')) return;

  const input = document.createElement('input');
  input.type = 'text';
  input.value = conv.title || '';
  input.maxLength = 60;
  input.style.cssText = 'flex:1;min-width:0;padding:3px 6px;border-radius:6px;' +
    'border:1px solid rgba(220,192,138,.4);background:rgba(0,0,0,.5);' +
    'color:inherit;font:inherit;font-size:12.5px;outline:none';

  title.replaceWith(input);
  input.focus();
  input.select();

  let done = false;
  const commit = async () => {
    if (done) return;
    done = true;
    const name = input.value.trim() || '新对话';
    try {
      await api(`/conversations/${conv.id}`, { method: 'PATCH', body: { title: name } });
      conv.title = name;
    } catch (err) { alert(err.message); }
    renderConvs();
    if (conv.id === state.currentId) $('#conv-title').textContent = name;
  };

  input.addEventListener('blur', commit);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
    if (e.key === 'Escape') { done = true; renderConvs(); }
  });
}

/* ---------------- 打开历史会话 ---------------- */
async function openConversation(id) {
  if (state.streaming) return;
  state.currentId = id;
  // 深链：刷新后还能停在同一个对话，链接也能直接发给别人（仍受 404 归属校验保护）
  history.replaceState(null, '', `#/c/${id}`);
  renderConvs();
  $('#sidebar').classList.remove('is-open');

  const stream = $('#stream');
  stream.innerHTML = '<div class="skeleton"><div class="sk-line" style="width:56%"></div>' +
    '<div class="sk-line" style="width:84%"></div><div class="sk-line" style="width:72%"></div></div>';

  try {
    const data = await api(`/conversations/${id}`);
    const conv = data.conversation;
    $('#conv-title').textContent = conv.title || '新对话';
    $('#conv-meta').textContent = `${conv.msg_count} 条消息`;

    stream.innerHTML = '';
    if (!data.messages.length) { renderEmpty(); return; }

    for (const m of data.messages) {
      if (m.role === 'user') {
        appendUser(m.content);
      } else {
        const node = appendAssistant();
        node.body.innerHTML = md(m.content);
        node.finish(m);
      }
    }
    scrollToBottom(true);
  } catch (err) {
    stream.innerHTML = '';
    renderEmpty();
    if (err.message && err.message !== '未登录' && !err.message.includes('会话不存在')) {
      alert(err.message);
    }
    state.currentId = null;
    renderConvs();
  }
}

/* ---------------- 消息渲染 ---------------- */
const STATUS_TEXT = {
  answered: '有据可依',
  out_of_scope: '知识库未覆盖',
  missed: '命中但未作答',
  ungrounded: '缺少依据',
  hedged: '数值待核实',
};

function renderEmpty() {
  $('#stream').innerHTML = `
    <div class="empty" id="empty-state">
      <div class="empty-mark">§</div>
      <h3>从一个具体问题开始</h3>
      <p>回答会标注命中的法条与原文片段；知识库没有的内容会直接说明。</p>
      <div class="samples">
        <button type="button" class="sample">借款合同的印花税税率是多少？</button>
        <button type="button" class="sample">商业银行的经营原则是什么？</button>
        <button type="button" class="sample">车船税中挂车按什么比例计税？</button>
        <button type="button" class="sample">土地增值税的扣除项目包括哪些？</button>
      </div>
    </div>`;
  bindSamples();
}

function bindSamples() {
  $$('.sample').forEach((btn) => {
    btn.addEventListener('click', () => {
      $('#input').value = btn.textContent;
      autoGrow();
      $('#input').focus();
    });
  });
}

function appendUser(text) {
  const el = document.createElement('div');
  el.className = 'msg user';
  const bubble = document.createElement('div');
  bubble.className = 'bubble-user';
  bubble.textContent = text;
  el.appendChild(bubble);
  $('#stream').appendChild(el);
  return el;
}

function appendAssistant() {
  const el = document.createElement('div');
  el.className = 'msg assistant';

  const mark = document.createElement('div');
  mark.className = 'who-mark';
  mark.textContent = '§';

  const right = document.createElement('div');
  right.className = 'answer';

  el.append(mark, right);
  $('#stream').appendChild(el);

  return {
    body: right,
    finish(meta) {
      const tools = document.createElement('div');
      tools.className = 'msg-tools';
      const copy = document.createElement('button');
      copy.type = 'button';
      copy.textContent = '复制';
      copy.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(right.dataset.raw || '');
          copy.textContent = '已复制';
          setTimeout(() => { copy.textContent = '复制'; }, 1400);
        } catch { copy.textContent = '复制失败'; }
      });
      tools.appendChild(copy);
      right.appendChild(tools);

      if (meta) right.appendChild(metaRow(meta));
    },
  };
}

function metaRow(meta) {
  const wrap = document.createElement('div');
  wrap.className = 'msg-meta';

  const status = meta.status || 'answered';
  if (STATUS_TEXT[status]) {
    const pill = document.createElement('span');
    pill.className = 'pill';
    pill.dataset.s = status;
    pill.textContent = STATUS_TEXT[status];
    pill.title = '本轮的作答状态由后端判定（五态，见 README）';
    wrap.appendChild(pill);
  }

  if (typeof meta.confidence === 'number') {
    const conf = document.createElement('span');
    conf.className = 'conf';
    const pct = Math.round(meta.confidence * 100);
    conf.innerHTML = `<span class="conf-bar"><span class="conf-fill" style="width:${pct}%"></span></span>` +
      `<span>相关性 ${pct}%</span>`;
    conf.title = '本轮最相关证据的重排得分，不是“答对的概率”';
    wrap.appendChild(conf);
  }

  if (meta.sources && meta.sources.length) {
    const srcs = [...new Set(meta.sources)];
    const box = document.createElement('div');
    box.className = 'src';
    for (const s of srcs) {
      const chip = document.createElement('span');
      chip.className = 'chip';
      const b = document.createElement('b');
      b.textContent = s;
      b.title = s;
      chip.appendChild(b);
      box.appendChild(chip);
    }
    wrap.appendChild(box);
  }

  if (meta.contexts && meta.contexts.length) {
    const det = document.createElement('details');
    det.className = 'evidence';
    const sum = document.createElement('summary');
    sum.textContent = `查看 ${meta.contexts.length} 段原文依据`;
    const body = document.createElement('div');
    body.className = 'ev-body';
    for (const c of meta.contexts) {
      const d = document.createElement('div');
      d.className = 'ev';
      d.textContent = c;
      body.appendChild(d);
    }
    det.append(sum, body);
    wrap.appendChild(det);
  }

  return wrap;
}

function scrollToBottom(force = false) {
  const s = $('#stream');
  const nearBottom = s.scrollHeight - s.scrollTop - s.clientHeight < 120;
  if (force || nearBottom) s.scrollTop = s.scrollHeight;
}

/* ---------------- 发送与流式接收 ---------------- */
async function ensureConversation() {
  if (state.currentId) return state.currentId;
  const conv = await api('/conversations', { method: 'POST', body: {} });
  state.currentId = conv.id;
  state.convs.unshift(conv);
  renderConvs();
  $('#conv-title').textContent = conv.title || '新对话';
  $('#conv-meta').textContent = '0 条消息';
  return conv.id;
}

async function send() {
  const input = $('#input');
  const text = input.value.trim();
  if (!text || state.streaming) return;

  const empty = $('#empty-state');
  if (empty) $('#stream').innerHTML = '';

  state.streaming = true;
  $('#send').disabled = true;
  $('#conn-status').classList.add('is-busy');

  try {
    const cid = await ensureConversation();
    appendUser(text);
    input.value = '';
    autoGrow();
    scrollToBottom(true);

    const node = appendAssistant();
    const caret = '<span class="caret"></span>';
    let raw = '';
    let pending = false;

    const paint = () => {
      pending = false;
      node.body.innerHTML = md(raw) + caret;
      scrollToBottom();
    };

    const res = await fetch('/chat/stream', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, conversation_id: cid }),
    });

    if (!res.ok) {
      let detail = `请求失败（${res.status}）`;
      try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
      throw new Error(detail);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let meta = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 2);
        if (!frame || frame.startsWith(':')) continue;

        for (const line of frame.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (payload === '[DONE]') continue;

          let obj;
          try { obj = JSON.parse(payload); } catch { continue; }

          if (typeof obj.text === 'string') {
            raw += obj.text;
            if (!pending) { pending = true; requestAnimationFrame(paint); }
          }
          if (obj.error) throw new Error(obj.error);
          if (obj.sources || obj.status) {
            meta = {
              sources: obj.sources || [],
              contexts: obj.contexts || [],
              confidence: obj.confidence,
              status: obj.status || 'answered',
            };
          }
        }
      }
    }

    node.body.innerHTML = md(raw);
    node.body.dataset.raw = raw;
    node.finish(meta || { status: 'answered' });
    scrollToBottom();

    // 标题可能刚被后端按首句改过，刷新一次列表
    await loadConversations();
    $('#conv-meta').textContent = `${state.convs.find((c) => c.id === cid)?.msg_count ?? 0} 条消息`;
  } catch (err) {
    const node = appendAssistant();
    node.body.innerHTML = `<p style="color:#e0776f">出错了：${esc(err.message)}</p>`;
  } finally {
    state.streaming = false;
    $('#send').disabled = false;
    $('#conn-status').classList.remove('is-busy');
    $('#input').focus();
  }
}

/* ---------------- 输入框 ---------------- */
const inputEl = $('#input');

function autoGrow() {
  inputEl.style.height = 'auto';
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 168)}px`;
}

inputEl.addEventListener('input', autoGrow);
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send();
  }
});
$('#send').addEventListener('click', send);
$('#new-chat').addEventListener('click', () => {
  if (state.streaming) return;
  state.currentId = null;
  history.replaceState(null, '', location.pathname);
  renderConvs();
  $('#conv-title').textContent = '新对话';
  $('#conv-meta').textContent = '尚未开始';
  renderEmpty();
  $('#sidebar').classList.remove('is-open');
  inputEl.focus();
});

document.addEventListener('click', (e) => {
  const s = $('#sidebar');
  if (s.classList.contains('is-open') && !s.contains(e.target) && e.target.id !== 'menu-btn') {
    s.classList.remove('is-open');
  }
});
$('#menu-btn').addEventListener('click', (e) => {
  e.stopPropagation();
  $('#sidebar').classList.toggle('is-open');
});

/* ---------------- 健康检查 ---------------- */
async function checkHealth() {
  const el = $('#conn-status');
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error('bad');
    const data = await res.json();
    const chunks = data?.kb?.chunks;
    el.classList.remove('is-down');
    el.innerHTML = `<span class="dot"></span><span>知识库已连接${chunks ? ` · ${chunks.toLocaleString()} 块` : ''}</span>`;
  } catch {
    el.classList.add('is-down');
    el.innerHTML = '<span class="dot"></span><span>后端未连接</span>';
  }
}

/* ---------------- 启动 ---------------- */
(async function init() {
  bindSamples();
  try {
    const data = await api('/auth/me');
    showApp(data.user);

    // 深链：#/c/<id> → 直接打开那个对话（登录态由 /auth/me 确认过）
    const m = location.hash.match(/^#\/c\/([0-9a-f]+)$/);
    if (m) await openConversation(m[1]);
  } catch {
    showAuth();
  }
})();
