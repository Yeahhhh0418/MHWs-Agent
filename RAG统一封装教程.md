# RAG `_load_data` 统一封装教程

> 结论：**可以统一，而且收益明显**（4 个方法共 141 行 → 基类约 30 行 + 子类声明约 40 行）。
> 但统一前必须先守住 7 条行为等价性，否则会悄悄改掉 `page_content`，导致已有的向量库作废。
> 本文只给设计和验证方案，**不修改任何现有代码**。
>
> **本文已按当前工作区重测**，对应这轮的三处改动：`load_jsonl` 从 `BaseRAG` 挪到 `mainrag.py` 模块级
> （11–19 行）、`MonsterInfoRAG` 删掉「跳过空值」、武器新增 `Theme` 标签。
> 文中的行号、行数、哈希都是重新实测的，不是上一版遗留。

---

## 一、先看现状：4 份 `_load_data` 的共同骨架

四个类的 `_load_data`（`src/rag/mainrag.py`）去掉各自的特例后，**骨架逐字相同**：

```python
logger.info("加载XX数据", extra={"path": self._data_path})
documents = []
FIELD_LABELS = {...}
data_list = load_jsonl(self._data_path)      # 模块级函数，已不在 BaseRAG 里
for data in data_list:
    parts = []
    for k, v in data.items():
        label = FIELD_LABELS.get(k, k)
        if isinstance(v, list):
            value_str = "、".join(str(item) for item in v)
        else:
            value_str = str(v)
        parts.append(f"{label}：{value_str}")
    content = "，".join(parts)
    documents.append(Document(page_content=content, metadata=data))
self._docs = documents
logger.info("XX数据加载完成", extra={"count": len(documents)})
```

这段骨架的价值在于它不是在「拼字符串」，而是在**决定什么内容进检索索引**——
`page_content` 同时喂给 BM25 和向量模型，字段标签的措辞、字段的排列顺序都会直接影响召回。
所以把它收进基类、让子类只声明「有哪些字段、叫什么、要不要特殊处理」，是对的方向。

### 位置与行数

| 类 | `_load_data` 行号 | 行数 |
|---|---|---|
| `EquipmentCraftingRAG` | 25–72 | 48 |
| `WeaponCraftingRAG` | 77–105 | 29 |
| `MonsterInfoRAG` | 136–171 | 36 |
| `MaterialInfoRAG` | 176–203 | 28 |
| 小计 | | **141** |

（上一版这张表是 14 / 66 / 125 / 169、合计 145 行。`load_jsonl` 挪到文件头后整体下移，
怪物因删掉跳过空值又少了 4 行，所以现在的 141 不是「上一版减几行」能凑出来的，以实测为准。）

外加 `MonsterInfoRAG` 类属性里的三张表（113–134 行，22 行）。

---

## 二、差异只有 4 个正交维度

| 维度 | 装备 | 武器 | 怪物 | 素材 |
|---|---|---|---|---|
| `FIELD_LABELS` 放哪 | 函数内局部变量 | 函数内局部变量 | **类属性** | 函数内局部变量 |
| `isinstance(v, list)` 分支 | 有 | 有 | **没有** | 有 |
| 字段值枚举映射 | `parts_type` × 5 | 无 | `MonsterSize` × 2 | 无 |
| 稀有字段注释 | 无 | 无 | **有**（4 个字段） | 无 |

**四处差异互不干扰**，这是能统一的关键——如果它们互相纠缠（比如「怪物映射了值就不跳空值」），就得靠子类覆写整个方法，统一就失去意义了。

> **刚消失的第五个维度**：怪物原本还有一条「空值字段整条不渲染」的分支，这一轮已经删掉，四个库现在都不跳空值。
> 下面的 ② 讲的就是它——代码虽然没了，但它决定统一后的实现里**不能**把这个分支补回去。

具体差异点：

**① 枚举值映射**——只有两处，都是「字段值 → 中文说明」：

```python
# 装备，50-51 行
if k == "parts_type" and v in PARTS_TYPE_LABELS:
    v = PARTS_TYPE_LABELS[v]

# 怪物，142-145 行
row = dict(data)                      # ← 注意是拷贝
size_text = self.SIZE_LABELS.get(str(row.get("MonsterSize", "")).strip())
if size_text:
    row["MonsterSize"] = size_text
```

**② 跳过空值（已删除，但要记得它在防什么）**——只有怪物有，原注释写明了理由：

```python
# 旧版（HEAD）139-141 行，现已删除
if not str(v).strip():
    # 值为空就不渲染这个字段，免得留下"生态特征："这种悬空标签
    continue
```

删掉它**目前是零影响**：实测四个数据文件的空值字段数都是 **0**（怪物 55 条、11 个字段全非空），
所以 `page_content` 的哈希一字未变（见第五节），**不需要重建任何向量库**。
但它防的是**将来**：哪条怪物的 `EnemyFeatures` 变成空串，正文就会多出一段「生态特征：」这种悬空标签，
和其余 54 条进同一份索引，白白给 BM25 添一个高频词。删是可以删，只是这层保险没了。

**③ 稀有字段注释**——只有怪物有。注意 `label, gloss = ...` 是**元组解包**，
所以 `label` 被赋成了 `"历战王名"`，该字段既进 `parts`（正文）也进 `notes`（注解）：

```python
# 151-156 行
if k in self.RARE_FIELD_GLOSSARY:
    label, gloss = self.RARE_FIELD_GLOSSARY[k]
    notes.append(f"{label} = {gloss}")
else:
    label = self.FIELD_LABELS.get(k, k)
parts.append(f"{label}：{v}")          # ← 在 if/else 之外，两个分支都会执行
```

**④ `FIELD_LABELS` 的位置不一致**——装备/武器/素材是函数内局部变量，怪物是类属性。
这本身没有功能影响，但统一时必须选一种，选类属性。

---

## 三、设计方案：模板方法 + 声明式类属性

思路：把「遍历字段 → 拼标签 → 拼列表 → 装 Document」收进基类，
子类**只声明数据长什么样**，不再写循环。这是标准的 Template Method 模式。

### 3.1 `src/rag/base.py` 的改动

在 `BaseRAG` 里加 3 个可覆写的类属性（`_LOG_NAME` 另算）和 3 个方法，并把 `_load_data` 从抽象方法改成模板方法：

```python
class BaseRAG(ABC):
    _DB_NAME = ""
    RAG_PROMPT_TEMPLATE = ""
    _ID_KEY = "_doc_id"

    # ── 子类用这 3 个属性声明「数据长什么样」──────────────
    FIELD_LABELS: dict = {}            # 字段名 → 中文标签
    VALUE_MAPS: dict = {}              # 字段名 → {原值: 显示值}
    RARE_FIELD_GLOSSARY: dict = {}     # 字段名 → (标签, 何时出现这条字段的说明)
    _LOG_NAME = ""

    def _load_data(self):
        """模板方法：子类不再覆写，只配上表的 3 个属性。"""
        logger.info(f"加载{self._LOG_NAME}数据", extra={"path": self._data_path})
        self._docs = [self._to_document(d) for d in load_jsonl(self._data_path)]
        logger.info(f"{self._LOG_NAME}数据加载完成", extra={"count": len(self._docs)})

    def _to_document(self, data: dict) -> Document:
        parts, notes = [], []
        for k, v in data.items():
            if k in self.VALUE_MAPS:
                v = self.VALUE_MAPS[k].get(str(v).strip(), v)

            if k in self.RARE_FIELD_GLOSSARY:
                label, gloss = self.RARE_FIELD_GLOSSARY[k]
                notes.append(f"{label} = {gloss}")
            else:
                label = self.FIELD_LABELS.get(k, k)
            parts.append(f"{label}：{self._render_value(v)}")

        content = "，".join(parts)
        if notes:
            content += "\n【本条额外字段说明】" + "；".join(notes)
        return Document(page_content=content, metadata=data)   # metadata 永远是原始 data

    @staticmethod
    def _render_value(value) -> str:
        if isinstance(value, list):
            return "、".join(str(item) for item in value)
        return str(value)
```

> ⚠️ **`load_jsonl` 该放哪**：基类的模板方法要调用它，而 `mainrag.py` 是 `import base` 的那一方，
> 所以它**不能只待在 `mainrag.py`**——基类反过来 `from rag.mainrag import load_jsonl` 就是循环导入。
> 两个干净的落点：
> 1. 放在 `rag/base.py` 的**模块级**（不在类里，`from rag.base import BaseRAG, load_jsonl`）；
> 2. 新开一个 `rag/io.py` 之类的工具模块，两边都从它导入。
>
> 现在它在 `mainrag.py` 模块级，只被还没统一的四个子类调用，暂时没问题；等真按本文统一时再挪。

### 3.2 四个子类塌缩成「纯声明」

```python
class EquipmentCraftingRAG(BaseRAG):
    _DB_NAME = "equipment"
    _LOG_NAME = "装备"
    RAG_PROMPT_TEMPLATE = EQUIPMENTRAG_PROMPT
    FIELD_LABELS = {
        "series_id": "装备系列", "parts_type": "装备部位",
        "name": "装备名称", "materials": "制作材料",
    }
    VALUE_MAPS = {"parts_type": {
        "HELM": "头盔", "BODY": "胸甲", "ARM": "腕甲",
        "WAIST": "腰甲", "LEG": "护腿",
    }}


class WeaponCraftingRAG(BaseRAG):
    _DB_NAME = "weapon"
    _LOG_NAME = "武器"
    RAG_PROMPT_TEMPLATE = WEAPONRAG_PROMPT
    FIELD_LABELS = {
        "Name": "武器名称", "WeaponType": "武器类型",
        "materials": "制作材料", "Theme": "武器主题",
    }


class MonsterInfoRAG(BaseRAG):
    _DB_NAME = "monster"
    _LOG_NAME = "怪物"
    RAG_PROMPT_TEMPLATE = MONSTERRAG_PROMPT
    FIELD_LABELS = {
        "EnemyName": "怪物名称", "Species": "种类", "MonsterSize": "体型",
        "EnemyExp": "怪物描述", "EnemyFeatures": "生态特征",
        "EnemyTips": "狩猎提示", "EnemyLegendaryName": "历战个体",
    }
    VALUE_MAPS = {"MonsterSize": {
        "大型": "大型怪物（正式狩猎目标）",
        "小型": "小型怪物（成群出没的杂兵，非正式狩猎目标）",
    }}
    RARE_FIELD_GLOSSARY = {
        "EnemyLegendaryKingName": ("历战王名", "同一只怪物的最强强化版，比历战个体更强，本作只有少数怪物有"),
        "EnemyFrenzyName":        ("狂龙化名", "被狂龙病毒强化后的形态名，攻击更狂暴，不是新怪物"),
        "EnemyExtraName":         ("特殊个体别名", "非标准形态的称呼，如首领个体、克隆体、家畜种"),
        "EnemyBossExp":           ("首领／幼体说明", "对上面这类特殊形态的补充描述，不是另一只怪物的简介"),
    }


class MaterialInfoRAG(BaseRAG):
    _DB_NAME = "material"
    _LOG_NAME = "素材"
    RAG_PROMPT_TEMPLATE = MATERIALRAG_PROMPT
    FIELD_LABELS = {"RawName": "素材名称", "RawExplain": "素材描述", "Rare": "稀有度"}
```

### 3.3 收益

| | 现在 | 统一后 |
|---|---|---|
| 四个 `_load_data` | 141 行 | 0 行（删掉） |
| 子类声明（3.2 那段） | — | ~40 行 |
| 基类新增（3.1 那段） | — | ~30 行 |
| `MonsterInfoRAG` 三张表 | 22 行 | 并入配置，行数持平 |
| **合计** | **163 行** | **~70 行** |

更重要的是：**改一处拼接逻辑，四个库同时生效**。现在想给所有库加个「字段顺序固定」或「超长描述截断」，得改四遍。

### 3.4 备选方案（不推荐）

也可以用自由函数 `build_content(data, field_labels, ...)` 代替模板方法。
但对这个项目不合适——`BaseRAG` 已经是 ABC 子类体系，子类的差异天然是「类级配置」，
再加一层函数参数传递只会让调用处更长。模板方法在这里更贴合已有结构。

---

## 四、行为等价性清单（迁移必须逐条守住）

这 7 条是「统一了但没改行为」的判定标准。**任何一条破了，`page_content` 就变了，已有的 Chroma 库立刻作废**
（`base.py` 的 `_init_vectorstore` docstring 已说明：它只看「目录里有没有文件」，不看数据变没变）。

| # | 规则 | 依据 |
|---|---|---|
| 1 | `metadata=data` 必须是**原始** `data`，不是值映射后的副本 | 怪物 142 行用 `row = dict(data)` 映射、167 行却存 `metadata=data`；所以 metadata 里是 `大型` 而非 `大型怪物（正式狩猎目标）` |
| 2 | 字段顺序 = **文件里的键顺序**，绝不能排序 | 怪物数据有 **8 种**不同键顺序（空字段被删过），排序会改变所有怪物的正文 |
| 3 | `isinstance(v, list)` 分支要对四类**都**生效 | 怪物没有这个分支。当前怪物数据 11 个字段**全是 str**，所以今天加了也不变；这是补潜在缺口，不是行为变更 |
| 4 | 统一实现里**不要**加「跳过空值」分支 | 怪物那条分支刚被删掉，四个库现在都不跳。今天四个文件的空值数都是 0，加了看不出差别；但怪物正文将来会和今天不同 |
| 5 | 稀有字段的 `label` 来自 glossary 元组，且该字段**同时**进正文字段列表和注解 | 怪物 151–156 行，`label, gloss = ...` 是元组解包 |
| 6 | 值映射要 `strip()` 后再查表 | 怪物 143 行有 `.strip()`；装备 50 行没有（但 `parts_type` 值无空格，等价） |
| 7 | 三种拼接符不能混 | 同字段内列表用 `、`；字段之间用 `，`；注解之间用 `；`，且前缀是 `\n【本条额外字段说明】` |

### 另有两个当前状态的事实，供你判断

- **装备的 `materials` 是 list**，所以第 3 条在装备上是**已经在用**的路径，不是新增。
- **素材的 `Rare` 是 int**（不是 str），`_render_value` 的 `str(v)` 分支覆盖它。

---

## 五、迁移步骤

**第 0 步：先备份黄金快照**（改造前跑一次，记下哈希）

> 前置：先给 `mainrag.py` 补上 `import json`（见六·3）。缺它的话下面这个脚本会直接 `NameError`，
> 因为 `load_jsonl` 挪到 `mainrag.py` 后，`base.py` 里那行 `import json` 就不再给它兜底了。

```bash
cd /c/code/MHWI_agent
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
import sys, hashlib; sys.path.insert(0,'src')
from rag.mainrag import EquipmentCraftingRAG, WeaponCraftingRAG, MonsterInfoRAG, MaterialInfoRAG
for cls in [EquipmentCraftingRAG, WeaponCraftingRAG, MonsterInfoRAG, MaterialInfoRAG]:
    r = cls(); r._load_data()
    blob = '\n'.join(d.page_content for d in r._docs)
    print(f'{cls.__name__:24s} docs={len(r._docs):5d} sha={hashlib.sha256(blob.encode()).hexdigest()[:16]}')
"
```

**改造前的基准值**（2026-09-18 按当前工作区重测；与上一版文档给的四个值**逐字相同**）：

| 类 | docs | content sha256 前 16 位 |
|---|---|---|
| `EquipmentCraftingRAG` | 693 | `583c0b9910327ff3` |
| `WeaponCraftingRAG` | 1098 | `9f8f08f3cb53a5d7` |
| `MonsterInfoRAG` | 55 | `29c1065b22a74a3c` |
| `MaterialInfoRAG` | 497 | `e5d1a6a03cf90d1e` |

> **顺带把这轮的改动验了**：
> - 怪物的 `29c1065b22a74a3c` 与「还带跳过空值」的旧版**完全相同**——我照旧逻辑复刻了一遍做对比，逐字节相等。
>   所以删掉跳过空值**不需要重建怪物向量库**（当前数据里根本没有空值可跳，见二·②）。
> - 其余三个值和上一版文档也一致，说明 `load_jsonl` 挪位、武器加 `Theme` 都没碰到正文。
>   换句话说：**这四个数字现在是「改造前」也是「改造后的目标」**，第 3 步拿它们当判据依然有效。

**第 1 步**：按 3.1 改 `base.py`（加属性 + `_load_data` + `_to_document` + `_render_value`）。
注意把 `_load_data` 上的 `@abstractmethod` 去掉——它不再是抽象的。

**第 2 步**：按 3.2 逐个改子类，**删掉各自的 `_load_data`**。

**第 3 步**：重跑第 0 步的脚本，**四个 sha 必须一字不差**。

- 一致 → 迁移成功，**不需要重建任何向量库**（`page_content` 没变）。
- 不一致 → 用下面的逐条 diff 定位是哪条规则破了：

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
import sys, json; sys.path.insert(0,'src')
from rag.mainrag import MonsterInfoRAG
# 把旧实现临时贴成 _load_data_old 再对比，或直接 git stash 对照
r = MonsterInfoRAG(); r._load_data()
for d in r._docs[:2]:
    print(repr(d.page_content)); print('  metadata:', json.dumps(d.metadata, ensure_ascii=False)[:200])
"
```

**第 4 步**：确认 `metadata` 仍是原始值（第 1 条规则最容易破）：

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
import sys; sys.path.insert(0,'src')
from rag.mainrag import MonsterInfoRAG
r = MonsterInfoRAG(); r._load_data()
bad = [(d.metadata.get('MonsterSize'), d.page_content[:40]) for d in r._docs
       if d.metadata.get('MonsterSize') and '怪物（' in str(d.metadata.get('MonsterSize'))]
print('metadata 被污染条数:', len(bad), bad[:2])
"
```

**第 5 步**：跑一次真实检索，确认召回没变：

```bash
.venv/Scripts/python.exe -m pytest tests/ -q     # 注意：tests 目前因旧导入路径跑不起来，见下
```

> ⚠️ `tests/` 整个目录都还在 import 重构前的旧路径：`test_base_rag.py`、`test_tokenizer.py` 用
> `agent.rag.*`，`test_evaluator.py` / `test_nodes.py` / `test_retry.py` 用 `agent.evaluator` /
> `agent.nodes` / `agent.retry`，**当前就是红的**。别把它们的失败当成这次改造的回归。
> 改造后可以顺手把导入路径改成 `rag.*`，然后补一个「四个库 `_load_data` 的 sha 断言」测试，
> 把本文第 0 步固化进 CI。

---

## 六、顺带发现的三个瑕疵

**1. `load_jsonl` 缺 `import json`（这个是真 bug，建议马上补）**

`load_jsonl` 从 `base.py` 挪到 `mainrag.py` 时，`base.py` 顶部那行 `import json` 没跟着走，
而 `mainrag.py` 只 import 了 `Document` / `BaseRAG` / `get_logger` / 四个 prompt。实测：

```
NameError: name 'json' is not defined. Did you forget to import 'json'?
  File "src/rag/mainrag.py", line 18, in load_jsonl
```

后果是**四个库全部加载失败**（`_load_data` 第一句就是它），`setup()` 直接炸。
修法一行——在 `mainrag.py` 顶部加：

```python
import json
```

> 提醒：这正是「把函数挪出类」的典型代价。原来 `BaseRAG.load_jsonl` 能从 `base.py` 的模块命名空间
> 拿到 `json`，现在函数换了家，`json` 也得跟着走。同一个连带问题在统一时会更明显——见 3.1 后面那条
> 关于 `load_jsonl` 该放哪的说明。

**2. `MaterialInfoRAG` 的日志库名（已修）**

原来写的是 `logger.info("武器数据加载完成", ...)`，复制粘贴漏改。当前工作区 203 行已经是
`"素材数据加载完成"`，不用再管；统一后由 `_LOG_NAME` 统一产出，这类错误会彻底绝迹。

**3. 怪物缺 `list` 分支**（潜在）

`MonsterInfoRAG._load_data` 里没有 `isinstance(v, list)` 判断，直接 `f"{label}：{v}"`。
当前怪物数据的 11 个字段全是 str，所以**现在不出问题**；
但哪天加一个列表字段（比如「弱点部位」），正文会渲染出 Python 的 repr：

```
弱点部位：['头', '翅膀', '尾巴']      ← 期望的是 「弱点部位：头、翅膀、尾巴」
```

统一后所有库共用 `_render_value`，这个缺口被自动补上。

---

## 七、什么情况下**不该**这样统一

统一的前提是「差异是正交的声明式配置」。一旦出现下面任一情况，模板方法就会开始拧巴：

1. **某个库的正文需要跨字段计算**——比如「用 A 字段决定 B 字段要不要渲染」。
   现在还能靠 `RARE_FIELD_GLOSSARY` 这类查表糊过去，但如果是「攻击力 > 1000 时加一句说明」，
   就该让那个子类覆写 `_to_document`，别硬塞进配置。
2. **正文要分段落而不是单一字符串**——比如未来给武器加「派生树」多行结构。
   那时 `"".join(parts)` 这个形状就不够了。
3. **不同库要走不同的切分/嵌入策略**——比如怪物描述太长需要 chunking。
   那属于 `_load_data` 之后的环节（`_init_vectorstore`），不该混在这里。
4. **子类数量掉到 2 个以下**——为 2 个类抽象一层基类配置，通常不如直接复制。

现在有 4 个库、差异恰好是 4 个正交维度，**正好落在适合统一的区间**。

---

## 八、附：一张图看清单个文档的组装过程

```
data/Equipment_Crafting_Table.jsonl 一行
  {"series_id":"希望","parts_type":"HELM","name":"希望面具","materials":["初始装备无需素材"]}
        │
        ├─ load_jsonl()  →  dict（mainrag.py 模块级函数；保持文件键顺序）
        │
        ├─ _to_document(data)
        │     ├─ 逐个字段：VALUE_MAPS 映射 → FIELD_LABELS 查标签
        │     ├─ _render_value()：list 用「、」连，其余 str()
        │     ├─ parts 用「，」连成 content
        │     └─ 有稀有字段则追加 "\n【本条额外字段说明】…"（用「；」连）
        │
        └─ Document(page_content=content, metadata=原样的 data)
              │
              ├─ page_content  → BM25（chinese_char_tokenizer）+ 向量模型 → 进检索
              └─ metadata      → 命中后随 Document 返回，供上层过滤/展示
```

关键点：**`page_content` 是检索接口，`metadata` 是数据接口，两者职责不同**——
这也是第 1 条等价性规则（metadata 存原始值）的由来：展示时要 `大型`，
检索时要 `大型怪物（正式狩猎目标）` 这种对模型更友好的措辞，所以两边故意不一样。
