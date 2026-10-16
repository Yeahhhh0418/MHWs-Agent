# MHWI Agent

《怪物猎人：荒野》游戏数据问答助手。基于 LangGraph 的多轮链式检索 Agent，回答「某装备怎么做、素材哪来的、这怪怎么打」这类连环问题。

给 AI 的项目约定（怎么跑、改哪里会静默坏、注释规范）在 `CLAUDE.md`。

## 工作方式

```
START → route → retrieve → analyze ─┬─ continue → retrieve（换库/换关键词再查）
                                    └─ finish   → generate → END
```

- **route**：判断问题核心实体，决定第一个查哪个库、用什么关键词
- **retrieve**：混合检索（BM25 关键词 + Chroma 向量，RRF 融合），结果追加进 `collected_info`
- **analyze**：通读已收集信息，判断还缺什么、下一轮查哪；不缺就 finish
- **generate**：把全部检索结果交给 LLM 组织成人话

不预先枚举推理路径，由 analyze 每轮动态决定，靠 `graph.max_iterations` 兜底防死循环。

## 四个知识库

| 库 | 内容 | 入库条数 |
|---|---|---|
| equipment | 防具合成配方（头/胸/手/腰/腿） | 693 |
| weapon | 武器合成配方、武器类型、主题 | 1098 |
| material | 素材名称、说明、稀有度 | 497 |
| monster | 怪物种类、体型、生态特征、狩猎提示 | 202 节（55 只怪切出来的） |

前三个库读 `data/*.jsonl`（由 `game_data/` 下的解包 xlsx 整理而来）；**怪物库读 `data/monster.md`**，原因见下节。

## 怪物库为什么单独走 md

前三个库的字段都短（名字、材料、稀有度），拼成一条就够用。

怪物不一样：一只怪的「物种 / 描述 / 生态 / 打法」回答的是**四个不同的问题**。全拼成一坨之后，问「火龙怎么打」和问「火龙什么生态」会匹配到同一段文字，切不开。

所以怪物数据改存 `data/monster.md`：`#` 分怪、`##` 分节，一节只讲一件事，编号变成两段。

```
monster|火龙|狩猎提示
monster|火龙|生态特征
```

**它是四个库里唯一有长叙述的，也是唯一走 md 的。**

## 检索层

四个库共用的部分：

- 字符级中文分词器 —— BM25 默认按空格切，对中文无效
- 两条召回路径按 `_doc_id` 去重并叠加分数（编号是业务字段拼出来的，形如 `monster|火龙|狩猎提示`，不是行号），两路都命中的排更前
- `vector_score_threshold` 过滤低相关召回，`ensemble_top_n` 截断后再进 LLM 上下文
- 各库的 `bm25_k` / `vector_k` / 权重都在 `config/config.yaml` 里单独调

怪物库额外的两个处理：

- **每节正文自带抬头**（`怪物名称：火龙，小节：狩猎提示。`）—— 有些小节的正文里根本不提自己是谁，按名字查它会沉底。
- **名字精确匹配** —— 先从问句里认出怪物名（长名优先，先认「护火龙」再认「火龙」，免得子串误判），命中就只在该怪的小节里检索；认不出来（比如问「天空王者」这种别名）就退回普通检索。理由：名字是**精确**信息，而「火龙」和「护火龙」在语义上太近，交给相似度去猜必然出错。
- 因为字符级分词同样分不开这几个名字，monster 库的 `bm25_weight` 设成了 **0**；其他三个库仍是 0.5 / 0.5。

## 评测

```bash
.venv/Scripts/python.exe scripts/dev.py eval               # 全量
.venv/Scripts/python.exe scripts/dev.py eval 查族名,俗称     # 只跑指定 tag
.venv/Scripts/python.exe scripts/dev.py judge              # 判答案质量，不重跑 Agent
```

题库 `src/app/eval/cases.jsonl` 现有 **31 道**，覆盖四个库，其中有几道带 `混淆回归` tag（防「名字相近的怪串味」）。脚本自动算**召回率 / MRR / 路由准确率**；答案质量另走 `judge_facts`，拿每题的 `key_facts` 去答案里逐条找。

## 快速开始

```bash
uv sync                      # 按 uv.lock 装依赖，项目本身也以 editable 方式装进 .venv
```

> **别对着已有的 `.venv` 直接跑 `uv sync`** —— 它会按锁文件精确同步，把锁文件之外的包删掉（比如你自己装的 jupyter）。想在干净环境验证，用 `UV_PROJECT_ENVIRONMENT=.venv-test uv sync`，验完删掉那个目录。

在项目根目录建 `.env`（已被 .gitignore 忽略）：

```
DEEPSEEK_API_KEY=...
DASHSCOPE_API_KEY=...        # 备用模型，可留空
```

问一句：

```bash
.venv/Scripts/mhwi-agent.exe "火龙这个怪要怎么打？"
```

`python -m app "…"` 是同一条路。项目是 editable 装的，所以这些命令都**不用加 `PYTHONPATH=src`**。

起 HTTP 服务（页面和接口在同一个地址）：

```bash
.venv/Scripts/python.exe -m uvicorn app.server.main:app --reload
```

然后浏览器打开 <http://127.0.0.1:8000/>（接口文档在 `/docs`）。

只想跑图、不看命令行入口：`.venv/Scripts/python.exe src/app/agent/graph.py`

## 配置

改 `config/config.yaml`：

- `llm.primary` / `llm.fallbacks`：主模型 + 跨厂商备用模型链（`api_key_env` 指定从哪个环境变量读密钥）
- `embedding`：本地 BGE 向量模型，CPU 推理（写**模型名**，不写本机缓存路径）
- `retrieval.<库名>`：各库的召回条数、阈值、BM25/向量权重、RRF 参数
- `graph`：`max_iterations` 管"最多转几轮"，`time_budget_seconds` 管"总共花多久"

日志是结构化 JSON：控制台带色，同时写入 `logs/agent.log`。

## 目录

```
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
CLAUDE.md           给 AI 的项目约定
```

## 注意

- **不用手动删向量库。** 库目录里存着一份「要入库的内容」的指纹，`setup()` 时会重算比对，对不上就自动重建。⚠️ 但重建得在**新进程**里发生 —— 同一进程里已经打开过这个库的话，旧文件还占着删不掉，会直接报错。
- **编号是数据指纹。** 改了 `_ID_FIELDS` 或 `data/monster.md` 里的 `#` / `##` 标题，`src/app/eval/cases.jsonl` 里的编号必须同步改 —— 不会报错，只会召回率悄悄变 0。
- 首次运行要建库 + 加载 BGE 模型，比较慢；本机缓存里没有那个模型时还要联网下载一次（之后就完全离线了）。
