# LLM 论文问答 Agent

上传论文（arXiv 链接或本地 PDF）→ 自动解析入库 → 用自然语言提问，答案带来源和页码。

## 工作原理

```
论文入口                                        问答入口
POST /api/papers        (arXiv 链接)            POST /api/query/paper
POST /api/papers/upload (本地 PDF)                    │
        │                                             │
        └── 后台任务 ──┐                     agent 自主决定是否检索、检索几轮
                       │                              │
   PyPDFLoader 解析 → 文本清洗 → 切分 → 向量化          │
                       │                              │
                       └──── Milvus（向量）────────────┘
                       └──── SQLite（论文元数据）
```

- **三个工具**：`search_papers` 做语义检索（可换关键词、换语言多轮查，最多 5 轮，能用 `paper_id` 限定到某一篇）；`list_library_papers` 回答"库里有哪些论文"；`view_page_image` 把某一页渲染成图片直接看，用来读文本提取不出来的公式和表格。
- **会话持久化**：不传 `conversation_id` 就新开一个会话，传了就自动带上历史，agent 会把"它""那篇"补全成完整问句再检索。会话存在 SQLite 里，前端刷新后用 `GET /api/conversations/{id}` 恢复。
- **流式输出**：`/api/query/paper/stream` 用 SSE 边检索边推事件（status / sources / token / error / done），不用干等十几秒。
- **引用可定位**：每条来源带 `page`（页码）和 `page_start`/`page_end`（在本页文本里的字符区间），前端据此回原文高亮。
- **来源可信**：返回的 sources 直接取检索工具的真实命中，按 `(paper_id, chunk_id)` 去重，不解析模型写的引用标记。
- **相似度阈值**：低于 `MIN_SCORE` 的命中直接丢弃。向量检索永远会返回"最相似"的 top_k，库里没有相关内容时也会返回，不过滤等于给模型递编答案的素材。

## 环境要求

- Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)
- Milvus（用仓库里的 `docker-compose.yml` 一键起）
- DashScope key（向量化）、DeepSeek key（问答）

## 快速开始

```powershell
# 1. 起向量库（数据落在 ./volumes/ 下，删掉目录等于清库）
docker compose up -d
# 2. 装依赖
uv sync
# 3. 复制 .env.example 为 .env，填 DASHSCOPE_API_KEY 和 DEEPSEEK_API_KEY
Copy-Item .env.example .env
# 4. 起服务
uv run uvicorn main:app --reload
```

打开 http://127.0.0.1:8000/docs 看全部接口。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/papers` | 提交 arXiv 链接，后台下载并入库 |
| POST | `/api/papers/upload` | 上传本地 PDF，后台解析入库 |
| GET | `/api/papers` | 论文列表（含状态、页数、块数） |
| DELETE | `/api/papers/{paper_id}` | 删除论文：向量 + 元数据 + 磁盘文件 |
| POST | `/api/query/paper` | 提问，返回问题、答案、来源；带 `history` 可追问 |
| POST | `/api/query/paper/stream` | 流式问答（SSE），边检索边出字 |
| GET | `/api/conversations` | 会话列表 |
| GET | `/api/conversations/{conversation_id}` | 会话完整记录，刷新页面后用它恢复上下文 |
| DELETE | `/api/conversations/{conversation_id}` | 删除会话 |
| GET | `/api/health` | 健康检查 |

`paper_id` 带来源前缀：`arxiv:2601.00597` 或 `upload:<32位uuid>`。

`status` 流转：`uploaded` → `downloading` / `indexing` → `indexed`，失败进 `failed` 并带 `error`。入库是后台任务，提交后轮询列表看状态，变成 `indexed` 再提问。

### 典型流程

```powershell
# 上传本地论文（立刻返回，解析在后台跑）
curl.exe -X POST "http://127.0.0.1:8000/api/papers/upload" -F "file=@D:\我的论文.pdf"
# 轮询到 status 变成 indexed
curl.exe "http://127.0.0.1:8000/api/papers"
# 提问
curl.exe -X POST "http://127.0.0.1:8000/api/query/paper" -H "Content-Type: application/json" -d "{\"question\": \"论文用了哪些数据集？\", \"top_k\": 3}"
```

追问就是把上一轮的问答放进 `history`：

```json
{
  "question": "那篇的熵是怎么定义的？",
  "history": [
    {"role": "user", "content": "48Ca 那篇论文主要研究了什么？"},
    {"role": "assistant", "content": "它研究了 48Ca 核态从规则到混沌转变过程中的熵与复杂度。"}
  ]
}
```

`role` 只接受 `user` 和 `assistant`（其它值服务端直接丢弃），最多取最近 20 条。

## 经常要调的参数

| 参数 | 位置 / 默认 | 什么时候调 |
| --- | --- | --- |
| `MIN_SCORE` | `.env`，0.28 | 检索老返回无关片段就调高，该查到的查不到就调低。换 embedding 模型或换语料语言后必须重新标定 |
| `EMBEDDING_MODEL` / `EMBED_DIM` | `.env` / `vector_store.py`，v4 / 1024 | 换向量化模型时两者必须一起改，且必须清库重灌——不同模型的向量空间不通用 |
| `MAX_UPLOAD_BYTES` | `.env`，50MB | 要收更大的 PDF 就调 |
| `chunk_size` | `text_splitter.py`，英文 900 / 中文 400 | 块太大检索变模糊，太小上下文断裂。改完要重新解析入库 |
| `MAX_SEARCH_ROUNDS` | `query.py`，5 | 一轮问答 agent 最多检索几次 |
| `MAX_PAGES_PER_QUERY` | `query.py`，2 | 一轮问答最多渲染几页图。看图很贵（实测一次 249k tokens vs 纯文本 6k），这个上限是账单防线 |

## 测试

```powershell
uv run pytest -v
```

`tests/test_query.py` 会 import query 路由（连带连 Milvus），跑测试前确保向量库在运行；其余测试是纯函数，不依赖外部服务。

## 检索评测

```powershell
.venv\Scripts\python.exe eval\run.py
```

`eval/questions.json` 里是「问题 + 期望命中的论文 + 期望关键词」，脚本只走向量检索、不调大模型，整轮只有 embedding 的费用。改 `chunk_size`、`MIN_SCORE`、清洗规则之后先跑它，用数字判断改动是好是坏，再决定要不要重灌。

当前基线：简单题 12/12，难题 9/10；正样本命中@1 17/18、命中@3 18/18，负样本正确过滤 3/4。

注意这只衡量**检索层**。检索层出现假阳不代表答案会错：实测问"论文里用了什么深度学习模型"，检索返回了 5 条物理模型段落（相似度 0.38 > 阈值），但 agent 读完片段后如实回答"没有找到深度学习模型相关内容"，还解释了 `model` 在文中指的是屏蔽势模型。端到端质量要连着问答一起看。

## 已知限制

- **PDF 解析用 PyPDFLoader**：双栏论文可能出现阅读顺序错乱、单词粘连（如 `withlog10`）。彻底解决要换 `pymupdf4llm`；装 `fonttools` 能改善字体编码导致的连字和符号问题。
- **中文提问查英文论文的相似度偏低**（实测 0.29~0.33）。好在 agent 会自己把中文问题改写成英文检索式再查（实测第 2 轮起全英文），实际影响被大幅削弱。
- **评测集偏简单**：`eval/questions.json` 里的问题都是论文主题的直接表述，命中率 100%。细节数字、跨论文对比、模糊指代这些难点还没覆盖，需要继续补题。
- **没有鉴权、CORS、限流**：只适合本地或内网，别直接暴露到公网。
- **标题、作者字段目前是空的**：只填了文件名和 arXiv 编号，要自动抽取得再加一步元数据解析。
- **纯向量检索扛不住词义歧义**：问"论文用了什么深度学习模型"，会把讲物理模型（ZBL/Molière）的段落召回，相似度 0.38 高于阈值，4 个负样本里漏 1 个。目前靠 agent 阅读片段后如实拒答兜住，要根治得上 rerank（cross-encoder 判断整体语义，而不是词面相似）。
- **问答成本**：响应里的 `usage` 是本次问答累计的 token。一次问答有多轮工具调用、每轮一条 AI 消息，成本逐条累加——实测一次 2 轮检索的问答约 6200 tokens。
- **看图很贵**：`view_page_image` 实测一次问答 249k tokens，是纯文本问答的 40 倍。它能读出纯文本拿不到的东西（如 Figure 2 的坐标轴刻度），但只该在公式、表格确实读不出来时用，所以有 `MAX_PAGES_PER_QUERY` 限次。
