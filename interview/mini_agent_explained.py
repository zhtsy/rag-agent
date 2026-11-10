#!/usr/bin/env python3
"""
mini_agent_explained.py —— 人话注释版

这不是另一个程序，就是 mini_agent.py 的同一套逻辑，
只是把每段代码旁边的注释，换成了"说人话"的版本。

怎么读这份文件（建议三遍）：
  第一遍：只看【这段在干嘛】的方框，跳过所有代码，先建立整体印象
  第二遍：跟着代码一行一行看，遇到不懂的语法回开头查那 6 个语法点
  第三遍：把代码折起来，只看方框，试着自己把代码复述出来

跑起来看效果：
  python3 mini_agent_explained.py
"""

# ============================================================
# 读之前，先认识这 7 个 Python 写法
# （不认识它们，看代码会一直卡住。都很短，两分钟能过一遍）
# ============================================================
#
# 1) f"文字{变量}文字"
#    字符串前面带个 f，里面用大括号就能把变量的值"塞"进去。
#    例：name = "小明";  print(f"你好{name}")  →  你好小明
#
# 2) @xxx 写在函数上面
#    这叫"装饰器"。你可以理解成"给下面这个函数盖个章"。
#    盖完章之后，函数本身功能不变，但额外做了一件事（比如被登记进本子）。
#
# 3) def f(**字典)
#    两个星号的意思是：把字典"拆开"成一个个参数传进去。
#    例：f(**{"a": 1, "b": 2})  等价于  f(a=1, b=2)
#
# 4) try: ... except 错误类型: ...
#    "先试一下，如果出了这种错，就走另一条路"。
#    不写 except 的话，出错程序就整个停掉了。
#
# 5) if 键 in 字典:
#    查这个字典里有没有这一项。
#
# 6) isinstance(东西, 类型)
#    判断"这个东西是不是这种类型"。比如 isinstance(3, int) 是 True。
#
# 7) 字典.get(键, 默认值)
#    去字典里取值；如果没有这个键，就返回你给的默认值，不会报错。
# ============================================================


# ============================================================
# 第 0 段：整件事的骨架（先看这个，别急着看代码）
#
#   把问题写在草稿纸第一行
#   重复最多 N 次：
#       把草稿纸整段发给模型
#       看它递回来的纸条：
#           ├─ 它说"我想好了"      → 把答案交出去，结束
#           ├─ 它说"帮我做某件事"  → 检查它参数填得对不对
#           │                        对 → 真的去做，把结果写到草稿纸末尾
#           │                        不对 → 把"你哪里错了"写回草稿纸
#           └─ 它什么都没说清      → 写一句"格式不对，重写"
#   如果 N 次还没搞定 → 老实告诉用户没搞定
#
#   整个文件，就是把这 10 行伪代码写成真的 Python。
#   下面每一段，你都能在上面这 10 行里找到对应位置。
# ============================================================

import ast          # 用来"读"算术表达式，而不是傻乎乎地 eval
import json         # 用来解析 {"键": "值"} 这种文本
import operator     # 提供加减乘除这些运算函数
import re           # 正则表达式：按固定花样在文字里找东西
import sys          # 拿命令行参数
import time         # 计时
import urllib.request   # 发网络请求（调 ollama 用）
from dataclasses import dataclass   # 自动生成"构造函数"的小工具
from typing import Any, Callable     # 只是类型标注，运行时不影响


# ============================================================
# 第 1 段：一个"工具"长什么样
#
# 【这段在干嘛】
# 给"工具"定一个形状。每个工具必须有四样东西：
#   名字、说明、参数要求、真正干活的函数。
# 就像招聘要求上写清楚岗位名称、职责、任职要求、以及这个人本身。
# ============================================================

@dataclass
# @dataclass 是 Python 自带的小帮手：
# 它会自动帮你生成"构造函数"，你不用自己写 __init__。
# 有它之后，Tool("calc", "算数", {...}, 函数) 就能直接创建出一个工具。
class Tool:
    name: str            # 名字，比如 "calc"
    description: str     # 说明，给模型看的，它靠这句话决定要不要用
    parameters: dict     # 参数要求，说清楚"我要什么参数、什么类型、哪些必须给"
    func: Callable       # 真正干活的 Python 函数


class ToolError(Exception):
    """一个特殊的"错误"。
    普通错误会让程序崩掉；这个错误不会 —— 它会被抓住，
    然后原封不动地写回纸条给模型看。所以它是"给模型看的错误"。"""


# ============================================================
# 第 2 段：登记本
#
# 【这段在干嘛】
# 记住"我们有哪些工具可以用"。
# 好比给新员工一张权限表：表上没有的，一律不批。
#
# 为什么要有这一步？
# 因为模型是"客人"，不能让它凭空调一个没登记的函数。
# 这是安全边界，不是便利设施。
# ============================================================

REGISTRY = {}   # 空字典。键是工具名（字符串），值是上面那个 Tool 对象


def tool(name, description, parameters):
    """一个"盖章机"。

    用法：
        @tool("calc", "算数", {...})
        def calc(...): ...

    @tool(...) 这一行的作用，是把下面那个函数
    连同名字、说明、参数要求，一起塞进上面的 REGISTRY 本子里。
    函数本身还是原来那个函数，一点没变。

    这里返回值 deco 也是一个函数（函数可以当返回值，Python 允许这么玩），
    Python 会把"被盖章的那个函数"当作参数传给它。
    """

    def deco(fn):                        # fn 就是被盖章的那个函数
        REGISTRY[name] = Tool(name, description, parameters, fn)   # 登记进本子
        return fn                        # 原样还回去，函数正常可用

    return deco                          # 把 deco 交出去，等 Python 来调用


# ============================================================
# 第 3 段：检查单子 + 真的去做
#
# 【这段在干嘛】
# 模型说"帮我用 calc 算一下"，你不能直接照做。
# 先核对三件事：栏位对不对、必填的有没有漏、填的内容类型对不对。
# 核对通过，才真的执行。
#
# 这一段是整个 Agent 里最容易被新手做错的地方，重点看下面的注释。
# ============================================================

# 一张对照表：模型写的类型名 → Python 里对应的类型
_PYTYPE = {
    "string": str,
    "integer": int,
    "number": (int, float),     # 整数小数都算
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate(schema, args):
    """核对模型递来的参数。有问题就 raise ToolError。

    schema = 这个工具要求什么参数（说明书里的那部分）
    args   = 模型实际填的内容
    """

    # 先确认它给的是一整张单子，而不是一个数字或者一串文字
    if not isinstance(args, dict):
        raise ToolError(f"参数必须是一张单子，你给的是 {type(args).__name__}")

    props = schema.get("properties", {})     # 所有栏位的说明；没有就当空字典

    # 第一查：有没有填了单子上没有的栏位
    # 为什么把这个放第一个？因为模型常把 expression 写成 expr。
    # 如果先查"必填"，它收到的错误是"缺少 expression"，
    # 它会以为自己忘了填，不会想到是名字写错了。
    for k in args:
        if k not in props:
            raise ToolError(f"未知参数 '{k}'，可用参数：{list(props)}")

    # 第二查：该填的栏位有没有空着
    for k in schema.get("required", []):     # required 里列的是"必须给"的
        if k not in args:
            raise ToolError(f"缺少必填参数 '{k}'")

    # 第三查：填的内容类型对不对、是不是允许的取值
    for k, v in args.items():
        t = props[k].get("type")             # 说明书上写的类型，比如 "integer"
        py = _PYTYPE.get(t)                  # 换成 Python 的类型，比如 int

        ok = isinstance(v, py) if py else True    # 有要求就检查，没写要求就放行

        # 这里是个坑：Python 里 True 也算"整数"（bool 是 int 的子类），
        # 不管的话，模型传 {"top_k": true} 也能蒙混过关。所以单独拦一下。
        if t == "integer" and isinstance(v, bool):
            ok = False

        if not ok:
            raise ToolError(f"参数 '{k}' 应该是 {t}，你给的是 {type(v).__name__}")

        # 如果说明书里写了"只能是这几个值"，再查一遍
        if "enum" in props[k] and v not in props[k]["enum"]:
            raise ToolError(f"参数 '{k}' 只能是 {props[k]['enum']} 里的一个")


def call_tool(name, args, slow_warn=5.0):
    """拿着模型写的名字和参数，真的去执行。"""

    # 第一件事：查登记本。没登记过的名字，直接拒。
    if name not in REGISTRY:
        raise ToolError(f"没有'{name}'这个工具，可用的是：{list(REGISTRY)}")

    t = REGISTRY[name]          # 从本子里把这个工具取出来
    validate(t.parameters, args)    # 核对参数，不过就抛错，不会往下走

    start = time.time()         # 记一下开始时间，待会儿算耗时

    try:
        # **args 是把字典拆开成一个个参数。
        # 比如 args = {"expression": "1+1"}，这句就等于 calc(expression="1+1")
        out = t.func(**args)
    except ToolError:
        raise                   # 工具自己抛的"给模型看的错误"，原样往上交
    except Exception as e:
        # 其他任何错误（工具内部写崩了、网络断了……）也转成同一种错误。
        # 目的：别让一个工具的 bug 把整个 Agent 弄挂。
        # 对上层来说，"参数写错"和"工具崩了"都是"可以重试的错误"，处理方式一样。
        raise ToolError(f"{type(e).__name__}: {e}") from e

    cost = time.time() - start

    # 只是"提醒"，不是"强制打断"。
    # 真要强制打断一个卡住的函数，Python 里得用别的手段（signal 或多线程），
    # 这里故意没做——面试被问到，能说清"我只做了告警、硬超时要怎么做"，比含糊带过强。
    if cost > slow_warn:
        out = f"{out}\n[提醒：这个工具花了 {cost:.1f} 秒，有点慢]"

    return str(out)             # 统一转成文字，因为纸条上只能写字


# ============================================================
# 第 4 段：三个真的能干活的工具
#
# 【这段在干嘛】
# 给 Agent 装三只手：会算数、会查资料、会看时间。
# 这三个都是普通 Python 函数，你会写函数就会写工具。
# ============================================================

# 一个小到不能再小的"资料库"，就当它是数据库
_DOCS = {
    "D1": "评测集共 48 条：修复前答对 44 条（91.67%），修复后答对 48 条（100%）。",
    "D2": "检索层共 145 个文本块；TOP_K=5；rerank 候选 10 个；相关度阈值 0.15。",
    "D3": "图谱固定两跳：人物 -养了-> 宠物 -属于-> 品种 -属于类别-> 类别。",
    "D4": "后端只暴露 POST /chat/stream（SSE），需带 X-API-Key。",
}

# 加减乘除这些运算，在 Python 内部其实是"函数"，这里做个对照表
_OPS = {
    ast.Add: operator.add,        # 加
    ast.Sub: operator.sub,        # 减
    ast.Mult: operator.mul,       # 乘
    ast.Div: operator.truediv,    # 除
    ast.Pow: operator.pow,        # 次方
    ast.Mod: operator.mod,        # 取余
    ast.USub: operator.neg,       # 负号
}


def _eval(node):
    """把表达式"拆开看"，只允许数字和四则运算，别的一律不认。

    为什么不直接 eval(expression)？
    因为这段表达式是模型写的，属于"不可信输入"。
    模型完全可能写一句 __import__('os').system('rm -rf /')，
    eval 一执行，你电脑就没了。
    所以这里只认"数字 + 加减乘除"，别的一律报错。
    """

    # 情况一：就是一个数字（比如 48）
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value

    # 情况二：两个东西在做运算（比如 a + b）
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))

    # 情况三：一个东西前面加了负号（比如 -5）
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))

    # 其他一切（函数调用、变量名、文字……）统统拒绝
    raise ToolError("只能算数字和 + - * / % **")


@tool("calc", "计算一个算术表达式，支持 + - * / % ** 和括号。",
      {"type": "object", "required": ["expression"],
       "properties": {"expression": {"type": "string"}}})
def calc(expression):
    """算一个算式。注意：模型必须把算式写成文字交给它，不能自己心算。"""

    try:
        tree = ast.parse(expression, mode="eval")    # 把 "1+1" 变成一棵"语法树"
    except SyntaxError as e:
        raise ToolError(f"算式写错了：{e.msg}")

    v = _eval(tree.body)        # 按照上面那张对照表，安全地算出来

    # 结果整理得好看一点：8.330000000000005 这种尾巴，截到 4 位
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else str(round(v, 4))
    return str(v)


@tool("search_docs",
      "按关键词查本地资料。query 请用 2-4 个字的短词，不要写一整句。",
      {"type": "object", "required": ["query"],
       "properties": {"query": {"type": "string"},
                      "top_k": {"type": "integer"}}})
def search_docs(query, top_k=2):
    """在一个小资料库里按关键词找。

    做法很土：把问题和资料都切成"两个字一组"的小碎片，
    数一数重合了多少片。重合得多，就算命中。

    中文没空格，不能像英文那样按空格切词，所以用这种"两字碎片"的办法。
    """

    def pieces(s):
        s = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]+", "", s)   # 先去掉标点空格
        return {s[i:i + 2] for i in range(len(s) - 1)} or ({s} if s else set())
        # 上面这行：从第 i 位取两个字符，拼成一个集合（集合自动去重）

    q = pieces(query)       # 问题切成碎片
    scored = []

    for did, text in _DOCS.items():
        cover = len(q & pieces(text)) / len(q) if q else 0.0
        # q & pieces(text) 是"两边的碎片重合的部分"，len 数一下有几个
        if cover >= 0.15:   # 重合度超过 15% 就算命中
            scored.append((cover, did, text))

    if not scored:
        # 没找到时，别只说"没找到"，要告诉它下一步该怎么办。
        # 否则模型很可能就此放弃，直接回一句"资料里没有"。
        return "没找到。请换更短的关键词（2-4 个字）再试，比如「答对」「评测」。"

    scored.sort(reverse=True)       # 从高到低排
    return "\n".join(f"[{did}] {text}" for _, did, text in scored[:max(1, top_k)])


@tool("now", "获取当前时间。",
      {"type": "object", "required": ["fmt"],
       "properties": {"fmt": {"type": "string", "enum": ["date", "datetime"]}}})
def now(fmt):
    """看时间。fmt 只能填 date 或 datetime，填别的会被上面的检查拦下来。"""
    return time.strftime("%Y-%m-%d" if fmt == "date" else "%Y-%m-%d %H:%M:%S")


# ============================================================
# 第 5 段：写说明书
#
# 【这段在干嘛】
# 把"我们有哪些工具、该怎么递纸条"写成一段文字。
# 这段文字会放在每次发给模型的内容最前面。
# 相当于你请了个新同事，先给他一份工作手册。
# ============================================================

def render_tools():
    """把登记本里的工具，自动列成一张清单。

    注意"自动"两个字：这里没有手写工具名。
    你以后新加一个 @tool，这段清单会自己多一行。
    手写的话迟早忘了更新，模型就会调一个你以为不存在的工具。
    """

    lines = []

    for t in REGISTRY.values():     # 把本子里所有工具过一遍
        props = t.parameters.get("properties", {})
        req = set(t.parameters.get("required", []))

        # 把参数拼成 "expression: string" 这样，可选的就标上（可选）
        sig = ", ".join(
            f"{k}: {v.get('type', 'any')}" + ("" if k in req else "（可选）")
            for k, v in props.items()
        )

        lines.append(f"- {t.name}({sig})：{t.description}")

    return "\n".join(lines)     # 一行一个工具，拼成一段文字


def build_prompt(scratchpad, max_steps):
    """拼出这一次要发给模型的完整内容。

    scratchpad 就是那张草稿纸：里面已经有问题、有前面几轮说过的话、
    有我们查到的结果。这里把"说明书"和"草稿纸"接在一起发出去。
    """

    return f"""你可以调用工具来回答问题。

可用工具：
{render_tools()}

严格按这个格式写，一次只写一个动作：

Thought: 你的想法
Action: 要用哪个工具，必须是 [{", ".join(REGISTRY)}] 里的一个
Action Input: 参数，写成 {{"参数名": "值"}} 这样
Observation: 这一行由系统填，你不要自己写

如果已经想清楚了，就改成写：
Thought: 信息够了
Final Answer: 给用户的答案

必须遵守：
1. 最多 {max_steps} 步，超了会被强制停下。
2. Action Input 必须写成 {{"键": "值"}}，键名要和工具说明一致。
3. 不要自己编 Observation；工具报错就照着错误改参数。
4. 算数必须用 calc，不要自己心算。

{scratchpad}"""


# ============================================================
# 第 6 段：读它递回来的纸条
#
# 【这段在干嘛】
# 模型回的是"一段没有任何格式保证的文字"，你要从里面
# 准确地判断出：它是想干活，还是已经答完了。
#
# 做法就是"找关键字"，但要加几道筛子，因为模型真的不听话。
# ============================================================

# 预先编好三个"找字花样"
_RE_ACTION = re.compile(r"Action\s*[:：]\s*([A-Za-z_]\w*)")
# 找 "Action:" 后面跟着的英文名字。只认英文字母下划线开头，
# 是因为工具名都是英文。模型要是写 "Action: 搜索文档"，这里就抓不到，
# 后面会走"格式不对"那条路让它重写，而不是瞎猜一个不存在的工具。

_RE_INPUT = re.compile(r"Action\s*Input\s*[:：]\s*(.*)", re.S | re.I)
# 找 "Action Input:" 后面的全部内容。re.S 的作用是让 . 能匹配换行，
# 因为参数有可能被写成好几行。

_RE_FINAL = re.compile(r"Final\s*Answer\s*[:：]\s*(.*)", re.S | re.I)
# 找 "Final Answer:" 后面的全部内容，也就是答案。


def _strip_fence(s):
    """去掉模型爱加的 ```json 外壳。

    模型经常把参数写成这样：
        ```json
        {"expression": "1+1"}
        ```
    这三行里真正有用的是中间那行，前后两行是废话，要削掉。
    """
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[A-Za-z]*\s*", "", s)   # 削掉开头那行
        s = re.sub(r"\s*```$", "", s)            # 削掉结尾那行
    return s.strip()


def _load_json(raw):
    """把一段文字变成 Python 的字典。变不出来就抛错。"""

    s = _strip_fence(raw)

    try:
        return json.loads(s)        # 正常情况：直接能解析
    except json.JSONDecodeError:
        # 兜底：模型有时会在参数前后加一堆废话，
        # 比如"好的，我来算：{"expression": "1+1"}，请稍等"
        # 那就把大括号中间那段抠出来再试一次
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            raise                   # 实在抠不出来，就抛错，让循环去处理
        return json.loads(m.group(0))


def parse(text):
    """读纸条。返回三个东西：类型、内容、原始参数。

    类型只有三种：
      "action" → 它要用工具。内容 = 工具名，原始参数 = 它填的参数
      "final"  → 它答完了。  内容 = 答案
      "none"   → 什么都没说清（格式乱），需要让它重写
    """

    m_action = _RE_ACTION.search(text)
    m_input = _RE_INPUT.search(text)

    # 先看有没有"要用工具"。必须两个都找到才算数。
    if m_action and m_input:
        # 重要：模型有时候会自己往下编一行 "Observation: 我猜结果是 8.33"。
        # 这段是我们该填的，它填的就是瞎猜。所以在 Observation 那里切开，
        # 后面的内容全部扔掉。
        raw = re.split(r"\n\s*Observation\s*[:：]", m_input.group(1), maxsplit=1)[0]
        return "action", m_action.group(1), raw.strip()

    # 没找到工具调用，再看看是不是给答案了
    m = _RE_FINAL.search(text)
    if m:
        return "final", m.group(1).strip(), ""

    # 都没有 → 格式乱了
    return "none", "", ""

    # 为什么先看 Action、后看 Final？
    # 因为模型经常在想法里写一句"最终答案应该是……"，
    # 如果先找 Final Answer，就会误判成"它答完了"，提前结束。


# ============================================================
# 第 7 段：主循环（整个程序的心脏）
#
# 【这段在干嘛】
# 就是开头那 10 行伪代码。跟模型来回传纸条，直到它说想好了，
# 或者传到次数上限为止。
# ============================================================

@dataclass
class Step:
    """记录一步：模型说了什么、我们回了什么。事后排查问题全靠它。"""
    i: int              # 第几步
    output: str         # 模型原话
    observation: str    # 我们回给它的内容


@dataclass
class Trace:
    """一次完整运行的记录。"""
    answer: str | None  # 最终答案；没有就是 None（Python 里表示"空"）
    steps: list         # 每一步的明细
    stop_reason: str    # 为什么停下：答完了，还是次数用光了


def _clip(out):
    """把模型自己编的 Observation 切掉。

    和上面 parse 里那个切法是同一个目的，只是位置不同：
    一个管"读懂"，一个管"回写"。两处都要切，漏一处就会出问题。
    """
    return re.split(r"\n\s*Observation\s*[:：]", out, maxsplit=1)[0].rstrip()


def run(question, llm, max_steps=6, verbose=False):
    """跑一次完整问答。

    question  = 用户的问题
    llm       = 一个函数，吃一段文字、吐一段文字（真的模型或假模型都行）
    max_steps = 最多来回几轮
    """

    # 草稿纸的第一行：问题。
    # 整张草稿纸就是这个 Agent 全部的"记忆"，没有别的。
    scratch = f"Question: {question}\n"
    steps = []
    recent = []             # 记住最近要过什么，用来发现"它在原地打转"

    for i in range(1, max_steps + 1):        # 从 1 数到 max_steps

        # ① 拼好这次要发的内容，发给模型，拿到它的回话
        out = _clip(llm(build_prompt(scratch, max_steps)).strip())

        # ② 读纸条，看它想干嘛
        kind, value, raw = parse(out)

        # 情况一：它答完了 → 直接结束
        if kind == "final":
            steps.append(Step(i, out, ""))
            return Trace(value or None, steps, "final")

        # 情况二：格式乱了 → 告诉它按格式重写，这一轮不算干活
        if kind == "none":
            obs = "格式不对：要么写 'Action:' + 'Action Input:'，要么写 'Final Answer:'。"

        # 情况三：它要用工具
        else:
            try:
                # 它填的参数是文字，得先变成字典
                args = _load_json(raw) if raw and raw != "null" else {}
            except json.JSONDecodeError as e:
                # 参数根本不是合法格式 → 把"哪里不对"告诉它
                obs = f"参数不是合法的格式（{e.msg}），请写成 {{\"名字\": \"值\"}} 这样。"
                args = None

            if args is not None:
                # 做一个"指纹"：工具名 + 排好序的参数。
                # 排序很关键：{"a":1,"b":2} 和 {"b":2,"a":1} 是同一个请求，
                # 不排序的话会被当成两次不同的调用，重复检测就失效了。
                key = (value, json.dumps(args, sort_keys=True, ensure_ascii=False))

                if key in recent:
                    # 连着要两次一样的，说明它卡住了
                    obs = "这个请求你刚才已经要过了，换个思路，或者直接给 Final Answer。"
                else:
                    recent = (recent + [key])[-2:]      # 只留最近两条
                    # 只记最近两条的原因：够挡住"原地打转"，
                    # 又不会误伤"隔了几步再查一次同样的东西"（那有时是合理的）

                    try:
                        obs = call_tool(value, args)    # 真的去做
                    except ToolError as e:
                        obs = f"工具调用失败：{e}"       # 把错误原样写回去，让它自己改

        # ③ 把这一轮的记录存下来，然后把结果贴到草稿纸末尾
        steps.append(Step(i, out, obs))

        if verbose:
            print(f"--- 第 {i} 步 ---\n{out}\n>>> 我们回给它：{obs}")

        scratch += f"{out}\nObservation: {obs}\n"

    # 循环跑完还没答出来 → 老实承认
    # 生产环境这里不能静默返回空，要拿已有的信息给个"部分回答"，并说明没查完
    return Trace(None, steps, "max_steps")


# ============================================================
# 第 8 段：换成真模型
#
# 【这段在干嘛】
# 上面的 run() 只要求 llm 是一个"吃文字、吐文字"的函数。
# 至于这个函数背后是真模型还是假脚本，run() 根本不关心。
#
# 好处：测试的时候可以塞一个假的，秒出结果、不花钱。
# ============================================================

class MockLLM:
    """假模型：按你写好的台词，一句一句往外吐。

    用来测循环、测解析、测校验，不用联网、不花一分钱、马上出结果。
    能不能写单元测试，就取决于你有没有把模型抽成可替换的一层。
    """

    def __init__(self, script):
        self.script = list(script)     # 台词本
        self.i = 0                     # 说到第几句了

    def __call__(self, prompt):        # 加了这个方法，对象就能像函数一样被调用
        if self.i >= len(self.script):
            return "Final Answer: 台词用完了"
        self.i += 1
        return self.script[self.i - 1]


class OllamaLLM:
    """真模型：把内容发给本机的 ollama，拿回它写的话。"""

    def __init__(self, model="qwen2.5:7b",
                 url="http://localhost:11434/api/generate"):
        self.model = model
        self.url = url

    def __call__(self, prompt):
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,                        # 一次性拿完，不要一段一段吐
            "options": {"temperature": 0.0,         # 0 = 尽量别自由发挥，要稳定
                        "num_ctx": 4096},           # 一次最多能看多少字
            # 这一行是整段里最关键的：
            # 告诉模型"写到 Observation 就停"。
            # 不设的话，它会顺着往下自己编一行假的结果，
            # 你就得到一份"看起来特别合理、其实是编的"数据。
            "stop": ["\nObservation:", "\nQuestion:"],
        }

        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["response"]


# 假模型要说的台词：查资料 → 算数 → 给答案
MOCK_SCRIPT = [
    'Thought: 得先查一下修复前后的答对数\n'
    'Action: search_docs\nAction Input: {"query": "修复前 修复后 答对"}',

    'Thought: 44/48 和 48/48，差值除以 48\n'
    'Action: calc\nAction Input: {"expression": "(48-44)/48*100"}',

    'Thought: 提升 8.33 个百分点\nFinal Answer: 答对率从 91.67% 提升到 100%，提升了 8.33 个百分点。',
]

DEFAULT_Q = "修复后比修复前，答对率提升了几个百分点？"


# ============================================================
# 第 9 段：命令行入口
#
# 【这段在干嘛】
# 让这个文件能直接用 python 跑起来，并决定用真模型还是假模型。
# 这一段跟 Agent 原理无关，看不懂可以跳。
# ============================================================

def main(argv):
    real = "--real" in argv             # 带了 --real 就用真模型
    verbose = "-v" in argv              # 带了 -v 就打印每一步的细节

    q = DEFAULT_Q
    if "--question" in argv:
        q = argv[argv.index("--question") + 1]
    elif argv and not argv[-1].startswith("-") and argv[-1] not in ("--real", "-v"):
        q = argv[-1]                    # 直接把问题写在最后也行

    llm = OllamaLLM() if real else MockLLM(MOCK_SCRIPT)

    start = time.time()
    tr = run(q, llm, max_steps=6, verbose=verbose)
    cost = time.time() - start

    print(f"\n问题: {q}")
    print(f"模型: {'真模型' if real else '假脚本'}   步数: {len(tr.steps)}   "
          f"停止原因: {tr.stop_reason}   耗时: {cost:.2f} 秒")

    if tr.answer:
        print(f"答案: {tr.answer}")
    else:
        print("答案: （次数用完了还没答出来，实际产品里应该给个部分回答并说明）")

    if not verbose:
        for s in tr.steps:
            first_line = s.output.strip().splitlines()[0][:60]
            print(f"  第{s.i}步: {first_line}  ->  {s.observation[:50]}")

    return 0 if tr.answer else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
