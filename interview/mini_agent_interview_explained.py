#!/usr/bin/env python3
"""
mini_agent_interview_explained.py —— 面试 42 行的人话注释版

这就是 mini_agent_interview.py 的同一个程序，
只是把每一行代码旁边都写上了"这句话到底在干嘛"。

跑起来看效果：
    python3 mini_agent_interview_explained.py

读的顺序（别跳）：
    1. 先看下面的【骨架】—— 背下来，代码就是照着它填的
    2. 再看【4 个语法点】—— 不认识这几个，看代码会一直卡
    3. 最后逐行看 react() —— 那是全部的核心
"""

# ============================================================
# 【骨架】背这段，别背代码
#
#   准备一份对话记录，第一条是"问题 + 工具说明书"
#   最多重复 6 次：
#       把整份对话记录发给模型，让它说完就停
#       它说的话，记进对话记录
#       如果它说"我想好了"     → 交出答案，结束
#       如果它说"帮我做某事"   → 去做，把结果也记进对话记录
#       如果它说得不对         → 把"哪里不对"记进对话记录
#   6 次还没好 → 老实地返回"没结果"
#
#   就这么点事。下面所有代码都是这段话的展开。
# ============================================================


# ============================================================
# 【4 个语法点】两分钟过一遍
#
# 1) 字典 {"role": "user", "content": "你好"}
#    大括号包起来的"键: 值"，用来描述一条消息：谁说的、说了什么。
#
# 2) 列表.append(东西)
#    往列表最后面塞一个东西。对话记录就是这样一条条加上去的。
#
# 3) re.search(花样, 文字)
#    在文字里找符合"花样"的第一处。找到返回一个对象，找不到返回 None。
#    找到之后，用 xxx.group(1) 把括号里圈住的那段取出来。
#
# 4) try / except / else
#    try 里"试着做"，出错走 except，没出错才走 else。
#    注意 else：它在"没出错"的时候执行，不是"无论对错都执行"。
# ============================================================

import ast          # 用来看懂算式，而不是直接执行
import json         # 用来解析 {"键": "值"} 这种文字
import operator     # 提供加减乘除
import re           # 正则：按花样找字

# ============================================================
# 工具部分：一张"名字 -> 函数"的对照表
# 面试时工具写什么不重要，面试官看的是下面的循环。
# ============================================================

DOCS = {"D1": "修复前答对 44/48（91.67%），修复后答对 48/48（100%）"}

# 把 Python 内部的运算函数做成对照表，供下面安全地算数用
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _ev(n):
    """把算式"拆开算"，只认数字和四则运算。

    为什么不直接 eval(表达式)？
    因为表达式是模型写的，模型可能被诱导写出
    __import__('os').system('rm -rf /') 这种东西。
    eval 一执行，电脑就没了。所以只放行四则运算。
    """
    if isinstance(n, ast.Constant):         # 就是一个数字
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in OPS:    # 两个数在做运算
        return OPS[type(n.op)](_ev(n.left), _ev(n.right))
    raise ValueError("只支持 + - * /")       # 别的统统拒绝


def calc(expression):
    """算一个算式，比如 "100-91.67"。"""
    return round(_ev(ast.parse(expression, mode="eval").body), 4)


def search(query, top_k=1):
    """在资料里按关键词找，找不到就说找不到。"""
    hits = [v for k, v in DOCS.items() if any(w in v for w in query.split())]
    return "\n".join(hits) if hits else "没找到"


# 这就是"工具清单"：名字 -> 真正干活的函数。
# 模型只能从这张表的键里选。表上没有的名字，后面会被直接拒掉。
TOOLS = {
    "calc": calc,
    "search": search,
}


# 说明书：每次都会放在对话最前面，告诉模型有哪些工具、该怎么写。
PROMPT = """你可以用工具。可用：calc(expression), search(query)

严格按格式输出，一次一个动作：
Thought: 想法
Action: 工具名
Action Input: {"参数名": "值"}

想清楚了就输出：
Final Answer: 答案

问题：__Q__"""
# 注意最后那个 __Q__：它是个占位符，待会儿会被换成真正的问题。
# 这里没用 .format() 而用 .replace()，因为 format 会把 JSON 里的大括号
# 当成占位符，一跑就报错。replace 更笨，但不会踩这个坑。


# ============================================================
# 核心：跟模型来回传纸条
# ============================================================
def react(question, tools, llm, max_steps=6):
    """跑一次完整的问答。

    question  : 用户的问题
    tools     : 工具清单（一个字典）
    llm       : 一个函数。调用它、把对话给它，它就返回模型说的话
    max_steps : 最多来回几次
    """

    # ① 准备对话记录。真实的大模型接口就长这个样：
    #    一个列表，每一项说明"谁说了什么"。
    #    现在只有一条：用户的问题（前面还挂着工具说明书）。
    msgs = [{"role": "user", "content": PROMPT.replace("__Q__", question)}]

    # ② 开始循环，最多 6 次。
    #    range(6) 会依次给你 0,1,2,3,4,5。
    #    前面的下划线 _ 意思是"这个数字我用不上，只是要循环 6 次"。
    for _ in range(max_steps):

        # ③ 把目前的整份对话发给模型，拿到它这一次的回话。
        #
        #    stop 的意思是"说到这儿就闭嘴"。
        #    为什么一定要设？因为模型有个毛病：写完 Action Input 之后，
        #    它会自己接着编一行"Observation: 8.33"（它根本没算，是猜的）。
        #    设了 stop，它写到 Observation 前面就被掐断，
        #    真正的结果由我们在下面填进去。
        #    （网上把这些叫 stop-token footgun，是 ReAct 最常见的坑）
        raw = llm(msgs, stop=["\nObservation:"])

        # ④ 它说的话，也要记进对话记录，标记成 assistant（模型说的）。
        #    不记的话，下一轮它就"忘了"自己刚才说过什么。
        msgs.append({"role": "assistant", "content": raw})

        # ⑤ 先看它是不是答完了。
        #    在它这段话里找 "Final Answer:" 后面的内容。
        #    re.S 的作用：让点号也能匹配换行，因为答案可能写好几行。
        m = re.search(r"Final Answer:\s*(.*)", raw, re.S)
        if m:
            # 找到了，任务结束。
            # group(1) 就是上面括号 (.*) 圈住的那一段，也就是答案。
            #     r"Final Answer:\s*(.*)"
            #                     ↑ 这一段
            return m.group(1).strip()

        # ⑥ 没答完，那它应该是要用工具。把工具名和参数找出来。
        #    \w+ 表示"一串字母数字下划线"，也就是英文名字。
        #    模型要是写成 "Action: 搜索"，这里就找不到，会走下面的兜底。
        name = re.search(r"Action:\s*(\w+)", raw)
        args = re.search(r"Action Input:\s*(.*)", raw, re.S)

        # ⑦ 分四种情况准备"要回给它的话"。
        #    注意：不管哪种情况，最后都只会得到一句 obs（观察结果），
        #    然后在第 ⑧ 步统一塞回对话里。所以这里不 return。

        if not name:
            # 情况一：它没有按格式写，既没给答案也没说要调工具。
            # 别崩，告诉它该怎么写，让它下一轮重写。
            obs = "格式不对：要么 Action/Action Input，要么 Final Answer"

        elif name.group(1) not in tools:
            # 情况二：它要用的工具不在清单上。
            # 这一步是"权限边界"：没登记的工具，一律不让碰。
            obs = f"没有工具 {name.group(1)}，可用：{list(tools)}"

        else:
            # 情况三、四：工具存在，试着执行。

            try:
                # 它给的参数是一段文字，先变成 Python 字典。
                # 如果根本没写参数，就给个空字典 {}，不然下面会出错。
                kw = json.loads(args.group(1).strip()) if args else {}

            except json.JSONDecodeError:
                # 情况三：参数不是合法的 {"键": "值"}。
                # 把"该怎么写"告诉它，让它自己改。
                obs = 'Action Input 要写成 {"参数名": "值"}'

            else:
                # 走到 else 说明上面解析成功了，可以真的去执行。
                # **kw 的意思是"把字典拆成一个个参数"，
                # 比如 kw = {"expression": "1+1"} 就等价于 calc(expression="1+1")
                try:
                    obs = str(tools[name.group(1)](**kw))
                except Exception as e:
                    # 情况四：工具自己崩了（算错、除零、网络断了……）。
                    # 同样别崩整个程序，把错误原样告诉模型，让它换个办法。
                    obs = f"工具报错：{e}"

        # ⑧ 把结果记进对话记录，当成一条新的"用户消息"。
        #
        #    这里有两个讲究：
        #    a. 用 user 而不是 assistant —— 因为这是"环境给的"，不是模型自己说的。
        #       如果标成 assistant，模型会以为是自己编的，可能顺着幻觉往下走。
        #    b. 是 append 一条新消息，不是塞进模型那段话的末尾 ——
        #       保持"模型说的"和"系统给的"分开，逻辑才干净。
        msgs.append({"role": "user", "content": f"Observation: {obs}"})

    # ⑨ 循环 6 次还没答出来，老实返回"没有"。
    #    注意：真实产品里不能就这么返回空，要把已经查到的信息整理成
    #    一个"部分回答"给用户，并说明哪部分没查完。这点面试常被追问。
    return None


# ============================================================
# 下面是演示用的，面试不用写
# ============================================================

class Mock:
    """假模型：按台词一句句往外吐，不联网、不要钱，用来演示和测试。"""

    def __init__(self, lines):
        self.lines = list(lines)    # 台词本
        self.i = 0                  # 说到第几句了

    def __call__(self, msgs, stop=None):
        # 加了这个方法，这个对象就能像函数一样被调用：mock(msgs, stop=...)
        out = self.lines[self.i] if self.i < len(self.lines) else "Final Answer: 完"
        self.i += 1
        return out


# 台词：先查资料，再算数，最后给答案 —— 正好演示两轮工具调用
DEMO = [
    'Thought: 先查数据\nAction: search\nAction Input: {"query": "答对"}',
    'Thought: 差值是 100-91.67\nAction: calc\nAction Input: {"expression": "100-91.67"}',
    "Thought: 够了\nFinal Answer: 提升 8.33 个百分点",
]

if __name__ == "__main__":
    answer = react("修复后比修复前提升几个百分点？", TOOLS, Mock(DEMO))
    print("答案：", answer)
