# -*- coding: utf-8 -*-
"""pytest 共享配置。

唯一职责：把项目根目录放进 sys.path，让 `pytest` 在 tests/ 子目录里也能
`import confidence` / `import obs`。不做别的——这些测试刻意不 import rag_agent，
因为那会加载 chroma + bge-reranker（十几秒、1.2GB），CI 里等于不可用。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
