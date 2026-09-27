# MHWI Agent

《怪物猎人：荒野》游戏数据问答助手。基于 LangGraph 的多轮链式检索 Agent，回答「某装备怎么做、素材哪来的、这怪怎么打」这类连环问题。

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

| 库 | 内容 | 条目 |
|---|---|---|
| equipment | 防具合成配方（头/胸/手/腰/腿） | 693 |
| weapon | 武器合成配方、武器类型、主题 | 1098 |
| material | 素材名称、说明、稀有度 | 497 |
| monster | 怪物种类、体型、生态特征、狩猎提示 | 55 |

数据源是 `data/*.jsonl`（由 `game_data/` 下的解包 xlsx 整理而来）。

检索层做了几件事：

- 字符级中文分词器 —— BM25 默认按空格切，对中文无效
- 两条召回路径按 `_doc_id`（行号）去重并叠加分数，两路都命中的排更前
- `vector_score_threshold` 过滤低相关召回，`ensemble_top_n` 截断后再进 LLM 上下文
- 各库的 `bm25_k` / `vector_k` / 权重都在 `config/config.yaml` 里单独调

## 快速开始

```bash
uv sync                      # 或 pip install -e .
```

在项目根目录建 `.env`（已被 .gitignore 忽略）：

```
DEEPSEEK_API_KEY=...
DASHSCOPE_API_KEY=...        # 备用模型，可留空
```

运行：

```bash
PYTHONPATH=src python src/agent/graph.py
```

> `python -m mhwi_agent` 和 `mhwi-agent` 命令尚未接线（`src/mhwi_agent/main.py` 是空的），目前只能走上面的方式。

## 配置

改 `config/config.yaml`：

- `llm.primary` / `llm.fallbacks`：主模型 + 跨厂商备用模型链（`api_key_env` 指定从哪个环境变量读密钥）
- `embedding`：本地 BGE 模型路径，CPU 推理
- `retrieval.<库名>`：各库的召回条数、阈值、BM25/向量权重、RRF 参数
- `graph.max_iterations`：最大检索轮次

日志是结构化 JSON：控制台带色，同时写入 `logs/agent.log`。

## 目录

```
src/agent/          LangGraph 图与节点（state / nodes / graph）
src/rag/            BaseRAG 基类 + 四个子类 + 中文分词器
src/prompts/        路由、分析、生成、四个 RAG 的提示词
src/models/         LLM 实例（主模型 + fallback）
src/mhwi_agent/     配置与日志
data/               JSONL 知识数据
storage/            Chroma 持久化目录（首次运行自动构建）
advise/             设计文档与方案讨论
```

## 注意

- **改了数据或 `_load_data` 之后，必须删掉 `storage/` 下对应的库目录再跑。** 向量库只看「目录里有没有文件」，不会检测数据是否变化；不删就会出现 BM25 用新数据、向量路用旧数据的新旧混检。
- 首次运行要建库 + 加载 BGE 模型，比较慢。
