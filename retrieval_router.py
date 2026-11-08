# -*- coding: utf-8 -*-
"""检索路由的「规则快速路径」——纯函数，零依赖。

为什么要它
──────────
`is_relation_query()` 在每次检索里都要发一次 DeepSeek 请求。实测
（2026-11-06，backend 日志 08:51:47.369 → 08:51:48.499）这一步单独占 **1130ms**，
而"小智住在哪个城市"这种句子，一条正则就能判，不需要问模型。

但这笔钱不能省错方向：规则判错的代价比省下的 1130ms 更贵——
误判成关系题 → 白跑一次图谱（实测 1723ms）→ 未命中再降级向量。
一误判就等于倒赔 600ms 以上。所以这里的设计原则是：

    **宁可漏判，不可误判。** 漏判只是回到原来"问模型"的路径，零代价。

三条自我约束
────────────
1. 只在**恰好一个实体**时启用（见 `can_use_rules`）。
   关系题必然指向某个主体；没有实体时问"哪个省"本身就不成立，
   而"谁养了猫"这类聚合题恰恰要交给模型去判，规则不该碰。
2. 只认**句式**，不认单词。"宠物"单独出现不算（"有哪些宠物用品"会误判），
   必须是"养了什么宠物/养的猫"这种带谓词的形状。
3. 规则名要能进日志。出现分歧时，看到 `rule:residence` 就知道是哪条规则判的，
   而不是只知道"走了图谱"。

测试在 `tests/test_retrieval_router.py`，其中 `test_no_entity_disables_rules`
锁住约束 1——把 `can_use_rules` 改成恒 True 会立刻变红。
"""

import re
from typing import Sequence, Tuple

# 每条规则都必须能举出至少一个真实问句。加规则前先问自己：
# "有没有正常的非关系问句会被它命中？" 答得出来就别加。
RULES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    # 居住地：住处、省市、籍贯。几乎不存在非关系的"住在"
    ("residence", re.compile(r"住在|住哪|哪个省|哪个城市|哪个国家|哪里人|籍贯|老家在哪")),

    # 归属：组织、团队、阵营
    ("membership", re.compile(r"属于|归属|隶属|哪个组织|哪个团队|哪个部门|哪一派")),

    # 人际/物际关系，必须是"什么关系"这种完整句式
    ("relation", re.compile(
        r"是什么关系|什么关系|什么亲戚|是(?:谁|他|她)的(?:爸爸|妈妈|父亲|母亲|儿子|女儿|妻子|丈夫|兄弟|姐妹)"
    )),

    # 制造者
    ("maker", re.compile(
        r"谁(?:制造|制作|发明|创造|设计|建造|锻造)|是谁(?:做|造|发明|创造)的|由谁(?:制|做|造|发明|创造)"
    )),

    # 材料
    ("material", re.compile(r"由什么(?:制成|做的|构成|材料)|什么材料|用什么(?:做|制)成?的?")),

    # 工作单位。⚠️ 只认"在哪工作"这种**位置**问法。
    # 曾经把 "做什么工作" 也写进来，离线比对（48 题）立刻抓到误判：
    #   C004「小陈是做什么工作的？」问的是**职业（属性）**，不是工作单位（关系），
    #   模型判"否"、规则判"是"，放上线就会白跑一次图谱。删掉后精确率 15/15 = 100%。
    # 这一条就是"宁可漏判不可误判"的具体代价：漏判只是多问一次模型，不花钱不犯错；
    # 误判要赔一次图谱往返 + 一次降级。
    ("workplace", re.compile(r"在哪(?:里)?工作|工作单位|就职于|在哪上班")),

    # 宠物：必须带谓词，"宠物"裸词不算
    ("pet", re.compile(r"养了?什么|养的(?:宠物|猫|狗|鸟|鱼)|养的(?:是)?一只|宠物(?:是|叫|有)什么")),

    # 共同点
    ("common", re.compile(r"共同点|有什么相同|哪里相同|有什么不一样的地方")),
)


def can_use_rules(entities: Sequence[str]) -> bool:
    """规则是否可用：只有恰好一个实体时才可信。

    0 个实体时无从判断主体（"谁养了猫"该交给模型）；
    ≥2 个实体时上游已经走多实体分支，轮不到这里。
    """
    return len(entities) == 1


def rule_relation(query: str) -> Tuple[bool, str]:
    """用规则判断是否关系类查询。

    返回 `(是否命中, 规则名)`。未命中时规则名为空串，调用方应回退到模型判断。
    """
    if not query:
        return False, ""
    for name, pattern in RULES:
        if pattern.search(query):
            return True, name
    return False, ""


def decide(query: str, entities: Sequence[str]) -> Tuple[bool, str, bool]:
    """路由决策：`(是否关系类, 依据, 是否需要问模型)`。

    依据的三种取值，会原样进日志：
      · `rule:<name>` —— 规则命中，省掉一次模型往返
      · `llm`        —— 规则判不了，回退模型（原路径，零行为变化）
      · `no_ent`     —— 没有唯一实体，规则不适用，回退模型
    """
    if not can_use_rules(entities):
        return False, "no_ent", True

    hit, name = rule_relation(query)
    if hit:
        return True, f"rule:{name}", False
    return False, "llm", True
