# MHWs Agent

《怪物猎人：荒野》的游戏数据问答 Agent。基于 LangGraph 的多轮链式检索：不预设推理路径，由模型每轮判断"还缺什么"，动态决定下一轮查哪个库、用什么关键词。

回答的是「某装备怎么做」「素材哪来的」「这只怪怎么打」这类需要连着查好几个库的问题。

## 目录

- [功能亮点](#功能亮点)
- [技术栈](#技术栈)
- [环境要求](#环境要求)
- [安装](#安装)
- [配置](#配置)
- [使用](#使用)
- [架构](#架构)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [已知限制](#已知限制)
- [待补充](#待补充)

## 功能亮点

- **多轮链式检索**：analyze 节点每轮读已收集的资料，缺什么就接着查，不缺就出答案。不预先枚举推理路径，靠 `max_iterations` 和整轮时间预算兜底。
- **四个知识库，各自独立调参**：装备、武器、素材、怪物，召回条数 / 相关度阈值 / BM25 与向量的权重都按库单独配。
- **混合检索**：BM25（字符级中文分词）+ 向量（本地 BGE），RRF 融合后按文档编号去重，两路都命中的排更前。
- **怪物库专门处理**：独立走 Markdown 分节（`#` 分怪、`##` 分节），并从问句里**精确**识别怪名，避免「火龙」串到「雌火龙 / 护火龙」。
- **降级留痕**：任意一步失败都会在 state 里记 `degraded` 和原因，评测与接口都能看到，不静默吞掉。
- **成本可见**：每次问答报出耗时、token 用量（含 reasoning）、模型调用次数，以及实际用到了哪几个模型。
- **跨厂商容灾**：主模型调用失败自动切备用模型，用量按模型分开记，能一眼看出是不是备用模型拖慢的。
- **两种入口**：命令行，以及 FastAPI 提供的 HTTP 接口（自带网页页面和 `/docs` 接口文档）。

## 技术栈

| 层 | 用了什么 |
|---|---|
| 语言 | Python ≥ 3.11 |
| 编排 | LangGraph / LangChain |
| 检索 | Chroma（向量）+ rank-bm25（关键词），RRF 融合 |
| 向量模型 | `BAAI/bge-small-zh-v1.5`，本地 CPU 推理 |
| 接口 | FastAPI + uvicorn |
| 模型接入 | OpenAI 兼容协议（主模型 DeepSeek，备用通义千问） |
| 包管理 | uv |
| 质量 | pytest + ruff |

## 环境要求

- Python ≥ 3.11（开发环境用 3.14，见 `.python-version`）
- [uv](https://docs.astral.sh/uv/)
- 一个 OpenAI 兼容的模型 API Key（默认接 DeepSeek）
- 首次运行需要能访问 HuggingFace 下载向量模型（约 90 MB）。下过之后完全离线，不再联网。

## 安装

```bash
git clone <仓库地址>     # 待补充：仓库 URL
cd mhws-agent
uv sync
```

> ⚠️ **别对着已有的 `.venv` 直接跑 `uv sync`** —— 它会按 `uv.lock` 精确同步，把锁文件之外的包（比如你自己装的 jupyter）删掉。要在干净环境验证，用 `UV_PROJECT_ENVIRONMENT=.venv-test uv sync`，验完删掉那个目录。

在项目根目录创建 `.env`（已在 `.gitignore` 中忽略）：

```bash
DEEPSEEK_API_KEY=sk-...     # 主模型，必填
DASHSCOPE_API_KEY=...       # 备用模型，可以留空
```

## 配置

改 `config/config.yaml`：

| 配置项 | 说明 |
|---|---|
| `llm.primary` | 主模型：模型名、`base_url`、从哪个环境变量读密钥、超时与重试 |
| `llm.fallbacks` | 备用模型列表，按优先级排序，建议跨厂商 |
| `embedding` | 向量模型。写**模型名**，不要写本机缓存路径 |
| `retrieval.<库名>` | 每个库的 `bm25_k` / `vector_k` / 权重 / 阈值 / 融合后保留条数 |
| `graph` | `max_iterations` 管"最多转几轮"，`time_budget_seconds` 管"总共花多久" |

`.env` 里的密钥不会被写进配置，`config.yaml` 只记录"从哪个环境变量读"。

## 使用

下面统一用 `.venv/Scripts/python.exe`（Windows）。macOS / Linux 换成 `.venv/bin/python`。

### 命令行

```bash
.venv/Scripts/python.exe scripts/dev.py ask "火龙这个怪要怎么打？"
```

`scripts/dev.py` 是统一入口，共五条命令：

| 命令 | 作用 | 会调模型吗 |
|---|---|---|
| `ask "问题"` | 问一句，打印答案和本次成本 | ✅ |
| `eval [tag,...]` | 跑题库评测，不给 tag 就是全量 31 道 | ✅ |
| `judge` | 判最新一份结果的答案质量，不重跑 Agent | ✅ |
| `test` | 跑单元测试 | ❌ |
| `lint --fix` | ruff 检查 / 格式化 | ❌ |

`dev.py` 只做转发，底层还是那几条命令。它额外保证两件事：工作目录钉在项目根、中文输出不乱码。

也可以用安装好的命令行入口，效果一样：

```bash
.venv/Scripts/mhws-agent.exe "火龙这个怪要怎么打？"
# 或者
.venv/Scripts/python.exe -m app "火龙这个怪要怎么打？"
```

项目以 editable 方式装进 `.venv`，这些命令都**不用加 `PYTHONPATH=src`**。

### HTTP 服务

```bash
.venv/Scripts/python.exe -m uvicorn app.server.main:app --reload
```

浏览器打开 <http://127.0.0.1:8000/> 是用同一个服务提供的问答页面，接口文档在 <http://127.0.0.1:8000/docs>。

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"火龙这个怪要怎么打？"}'
```

启动时会把四个库全部预热（加载向量模型 + 建检索器），所以第一个请求不用等。

### 评测

```bash
.venv/Scripts/python.exe scripts/dev.py eval                 # 全量 31 道
.venv/Scripts/python.exe scripts/dev.py eval 查族名,俗称       # 只跑指定 tag
```

题库在 `src/app/eval/cases.jsonl`，目前 **31 道**，覆盖四个库，其中有几道带 `混淆回归` tag（专防「名字相近的怪串味」）。

脚本自动算**召回率 / MRR / 路由准确率**；答案质量另走 `judge_facts`，拿每题预设的 `key_facts` 去答案里逐条找。

## 架构

```text
START → route → retrieve → analyze ─┬─ continue → retrieve（换库 / 换关键词再查）
                                    └─ finish   → generate → END
```

- **route**：判断问题核心实体，决定第一个查哪个库、用什么关键词
- **retrieve**：混合检索，结果追加进 `collected_info`
- **analyze**：通读已收集的信息，判断还缺什么、下一轮查哪；不缺就 finish
- **generate**：把全部检索结果交给模型组织成人话

### 四个知识库

| 库 | 内容 | 入库条数 |
|---|---|---|
| `equipment` | 防具合成配方（头 / 胸 / 手 / 腰 / 腿） | 693 |
| `weapon` | 武器合成配方、武器类型、主题 | 1098 |
| `material` | 素材名称、说明、稀有度 | 497 |
| `monster` | 怪物种类、体型、生态特征、狩猎提示 | 202 节（55 只怪切出来的） |

前三个库读 `data/*.jsonl`（由 `game_data/` 下的解包 xlsx 整理而来）。

### 怪物库为什么单独走 Markdown

前三个库的字段都短（名字、材料、稀有度），拼成一条就够用。

怪物不一样：一只怪的「物种 / 描述 / 生态 / 打法」回答的是**四个不同的问题**。全拼成一坨之后，问「火龙怎么打」和问「火龙什么生态」会匹配到同一段文字，切不开。

所以怪物数据改存 `data/monster.md`：`#` 分怪、`##` 分节，一节只讲一件事，编号变成两段：

```text
monster|火龙|狩猎提示
monster|火龙|生态特征
```

它是四个库里唯一有长叙述的，也是唯一走 Markdown 的。

### 检索层

四个库共用的部分：

- 字符级中文分词器 —— BM25 默认按空格切，对中文无效
- 两条召回路径按文档编号去重并叠加分数（编号由业务字段拼成，形如 `monster|火龙|狩猎提示`，不是行号）
- `vector_score_threshold` 过滤低相关召回，`ensemble_top_n` 截断后再进模型上下文
- 各库的 `bm25_k` / `vector_k` / 权重都在 `config/config.yaml` 里单独调

怪物库额外做两件事：

- **每节正文自带抬头**（`怪物名称：火龙，小节：狩猎提示。`）—— 有些小节的正文里根本不提自己是谁，按名字查它会沉底
- **名字精确匹配** —— 先从问句里认出怪物名（长名优先，先认「护火龙」再认「火龙」，免得子串误判），命中就只在该怪的小节里检索；认不出来（比如问「天空王者」这种别名）就退回普通检索

## 项目结构

```text
src/app/agent/      LangGraph 图与节点（state / nodes / graph）
src/app/rag/        BaseRAG 基类 + 四个子类 + 中文分词器
src/app/eval/       评测脚本 + 题库（结果写在 eval/results/，不进 git）
src/app/core/       配置、模型实例、日志、用量统计
src/app/server/     HTTP 接口 + 前端页面（FastAPI）
src/app/prompts.py  路由、分析、生成、四个 RAG 的提示词
src/app/cli.py      命令行入口
scripts/            dev.py（统一入口）+ 数据转换脚本
data/               知识数据（三个库的 jsonl + 怪物库的 monster.md）
storage/            Chroma 持久化目录（首次运行自动构建，不进 git）
advise/             设计文档与方案讨论
CLAUDE.md           给 AI 的项目约定（也能当"改哪里会静默坏"的速查）
```

## 常见问题

### 改了数据，要不要手动删向量库？

不用。库目录里存着一份「要入库的内容」的指纹，加载前会重算比对：对不上就自动重建。

⚠️ 但重建必须发生在**新进程**里 —— 同一个进程已经打开过这个库的话，旧文件还占着删不掉，会直接报错（比静默把库搞坏强）。

### 为什么说"编号是数据指纹"？

文档编号由业务字段拼成（如 `monster|火龙|狩猎提示`），`cases.jsonl` 里的 `required_doc_ids` 硬编码了它。

改了编号的拼法、`data/monster.md` 里的 `#` / `##` 标题，题库里的编号就必须同步改 —— **不报错，只会召回率悄悄变 0**。`dev.py test` 里有一条测试专门守这个。

### 命令输出的中文是乱码？

裸敲 Python 命令、输出又被管道或重定向接住时会乱码。前面加 `PYTHONIOENCODING=utf-8`，或者直接用 `dev.py`（它替你设好了）。

### 日志里翻到几个月前的旧错误？

`logs/agent.log` 只追加、从不清空，里面躺着从项目第一天起的所有记录。看它必须带日期过滤，否则会在几个月前的旧错误里找今天的原因。

### 答案里会不会混进《世界》的知识？

会，这是这个项目的已知风险。数据全部解包自《荒野》（MHWS：辟兽、护火龙、锁刃龙、煌雷龙），但**模型自身关于怪猎的知识大多来自《世界》**（MHWI）。两作里同名怪物的机制未必一样 —— 所以答案里凡是资料中没有的内容，都按"可能有版本串味"看待。

## 已知限制

- **没有认证、限流和 CORS 配置**，HTTP 接口只适合本地跑，别直接暴露到公网。
- **只做了单进程、低并发**。接口已确认 3 个并发请求是并行的，但没做过压测。
- **武器主题不全是怪物线**。43 种 `Theme` 里有 10 种是素材线或属性线（「骨素材衍生」「麻痹属性衍生」这类，共 253 条武器），推不出怪名时会如实说明。
- **没有 CI**，测试和 lint 靠本地手动跑。

## 待补充

- **许可证**：仓库里没有 `LICENSE` 文件。
- **仓库地址**：上面的 `git clone` 命令需要填真实 URL。
- **徽章**：项目没有 CI 和发布流程，暂时没有可放的构建状态 / 版本徽章。
- **界面截图**：HTTP 页面还没有截图放进 README。
