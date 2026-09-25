import os

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import DataType, MilvusClient

# 先加载 .env：不调用 load_dotenv() 的话 os.getenv 取不到 DASHSCOPE_API_KEY，
# 向量化时会直接报鉴权失败（api_key 为 None）


load_dotenv()

# 论文标识统一成带前缀的字符串，一眼能看出论文从哪来：
#   arxiv:2601.00597      从 arXiv 下载的
#   upload:3f2a9c...e1    用户本地上传的
# 前缀还有个用处：以后想只在某类论文里检索，可以直接按 "arxiv:%" 前缀过滤。
ARXIV_PREFIX = "arxiv:"
UPLOAD_PREFIX = "upload:"
ID_PATTERN = r"[0-9a-f]{32}"
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
DB_NAME, COLLECTION_NAME = "papers", "docs"
# 向量维度必须和 embedding 模型的实际输出一致，对不上时插入会直接报
# "向量维度不一致"。text-embedding-v4 默认输出 1024 维（也支持 1536/2048），
# 而老的 text-embedding-v2 是固定 1536 维。换模型要同步改这里并重建 collection。
EMBED_DIM = 1024
# 向量化模型，可用 .env 覆盖。不同模型的向量空间不通用，换模型必须重灌数据，
# 不能在同一张表里混用两种模型算出来的向量。
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
# 检索结果的最低相似度（COSINE，越大越相似）。低于这个值的片段和问题基本无关，
# 放行只会给模型提供编答案的素材，所以宁可返回空、让 agent 如实说"没查到"。
#
# 实测（text-embedding-v4，中文提问查英文论文）：
#   相关问题的 top1 落在 0.327~0.615（"论文主要结论"这类泛问偏低）
#   无关问题的 top1 落在 0.156~0.251（"Transformer 注意力机制"这种学术话题最高）
# 取 0.28 能把无关问句整体挡在门外，又不误伤泛问。这是跨语言场景标定出来的经验值，
# 换 embedding 模型、换成中文论文之后都要重新标定。
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.28"))

client = MilvusClient(MILVUS_URI)
if DB_NAME not in client.list_databases():
    client.create_database(db_name=DB_NAME)
client.use_database(db_name=DB_NAME)

# 当前代码期望的字段结构，用于和存量 collection 做比对
EXPECTED_FIELDS = {
    "id": DataType.VARCHAR,          # 字符串主键，如 "2601.00597_0"
    "vector": DataType.FLOAT_VECTOR,  # 文本向量
    "text": DataType.VARCHAR,         # 切分后的原文
    "source": DataType.VARCHAR,       # arxiv_id，方便按论文来源过滤
    "chunk_id": DataType.INT64,
    "page": DataType.INT64,           # 页码
    "page_start": DataType.INT64,     # 块在所属页文本里的起始字符下标
    "page_end": DataType.INT64,       # 结束下标（不含），前端据此高亮
}


def _build_collection():
    """按当前 schema 新建 collection，并给向量字段建索引。"""
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBED_DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("source", DataType.VARCHAR, max_length=128)
    schema.add_field("chunk_id", DataType.INT64)
    schema.add_field("page", DataType.INT64)
    schema.add_field("page_start", DataType.INT64)
    schema.add_field("page_end", DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector",
                           metric_type="COSINE",
                           index_type="AUTOINDEX")

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)


def _existing_fields():
    """读取存量 collection 的字段名 -> 类型，$meta 是动态字段，忽略。"""
    info = client.describe_collection(collection_name=COLLECTION_NAME)
    return {f["name"]: f["type"] for f in info["fields"] if not f["name"].startswith("$")}


def _existing_vector_dim():
    """读取存量 collection 的向量维度。

    dim 藏在字段的 params 里，只比对字段名和类型是发现不了的：换 embedding 模型
    以后字段结构完全没变、维度却对不上，表现就是"表看着没问题，一插入就报
    向量维度不一致"，很难定位到是表建错了。
    """
    info = client.describe_collection(collection_name=COLLECTION_NAME)
    for field in info["fields"]:
        if field["name"] == "vector":
            return (field.get("params") or {}).get("dim")
    return None


def ensure_collection(force_recreate: bool = False):
    """保证 collection 存在，且字段结构、向量维度都和当前代码一致。

    只判断 has_collection 是不够的：collection 一旦存在，schema 就再也改不了。
    早期用自动 schema 建出来的 docs 只有 id(INT64) + vector 两个字段，
    这时再按字符串 id 去 upsert，Milvus 会报：
      The Input data type is inconsistent with defined schema,
      {id} field should be a int64, but got a {<class 'str'>} instead
    所以结构对不上时要重建。维度同理：换了 embedding 模型后维度会变，而 schema
    改不了，只能整表重建。
    """
    if client.has_collection(COLLECTION_NAME):
        # 字段结构和向量维度都要对上；维度对不上时，只要表还是空的就自动重建，
        # 免得换模型后一路报维度错误却不知道该删哪张表
        schema_ok = (
            _existing_fields() == EXPECTED_FIELDS
            and _existing_vector_dim() == EMBED_DIM
        )
        if not force_recreate and schema_ok:
            return

        row_count = client.get_collection_stats(collection_name=COLLECTION_NAME)["row_count"]
        if row_count:
            # 有数据时不做静默删除，避免误删；交给使用者决定改名还是迁移
            raise RuntimeError(
                f"collection `{COLLECTION_NAME}` 的字段结构或向量维度与代码不一致，"
                f"但其中已有 {row_count} 条数据。换过 embedding 模型的话，这些向量已经"
                f"不能用了，请先清理，或把 COLLECTION_NAME 改成新名字。"
            )
        client.drop_collection(collection_name=COLLECTION_NAME)

    _build_collection()


ensure_collection()

embedder = DashScopeEmbeddings(model=EMBEDDING_MODEL, dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))


def _page_of(chunk) -> int:
    """取 chunk 的页码（从 1 开始）。PyPDFLoader 的 page 是 0 开始的下标。"""
    page = chunk.metadata.get("page")
    # 取不到页码时存 -1，不能存 0，否则会被读成"第 0 页"
    return int(page) + 1 if page is not None else -1


def insert_chunks(chunks, paper_id):
    """把一篇论文的切分结果向量化后写入 Milvus。

    写入前先按 source 删掉这篇论文的旧块：只 upsert 的话，如果这次切分出的块数
    比上次少（调整了 chunk_size、arXiv 换了版本），旧的多出来的尾部块会留在库里
    继续被检索命中，答案里就会出现已经不存在的内容。
    """
    if not chunks:
        return 0

    # DashScopeEmbeddings 内部会按模型批量调用（text-embedding-v4 每批 10 条，
    # v2/v1 是 25 条），所以这里不用自己分批，直接传整个列表即可
    vectors = embedder.embed_documents([c.page_content for c in chunks])

    # 模型输出维度和 collection 的 dim 必须一致，否则 upsert 会报维度错误
    if vectors and len(vectors[0]) != EMBED_DIM:
        raise ValueError(
            f"向量维度不一致：模型返回 {len(vectors[0])} 维，collection 定义的是 {EMBED_DIM} 维"
        )

    # paper_id 只有两种来源，都不可能含引号，可以安全拼进 filter 表达式：
    #   arxiv:<编号>   编号过了 parse_arxiv_id 的白名单校验
    #   upload:<uuid>  uuid4().hex，纯十六进制
    # 这一步必须在 upsert 之前执行。
    client.delete(collection_name=COLLECTION_NAME, filter=f'source == "{paper_id}"')

    # 拼接字符串作为唯一ID
    data = [{
        "id": f"{paper_id}_{i}",
        "vector": v,
        "text": chunks[i].page_content,
        "source": paper_id,
        "chunk_id": i,
        "page": _page_of(chunks[i]),
        "page_start": int(chunks[i].metadata.get("page_start", -1)),
        "page_end": int(chunks[i].metadata.get("page_end", -1)),
    } for i, v in enumerate(vectors)]

    client.upsert(collection_name=COLLECTION_NAME, data=data)
    client.flush(collection_name=COLLECTION_NAME)  # 确保数据落盘
    return len(data)


def search_similar_chunks(question: str, top_k: int = 3, paper_id: str | None = None):
    """在已上传的论文里检索与问题最相关的片段。

    相似度低于 MIN_SCORE 的命中会被丢掉。向量检索永远会返回"最相似"的 top_k 条，
    哪怕库里根本没有相关内容（实测无关问题也能拿到 0.25 分），不过滤就直接交给
    模型，等于递给它一堆编答案的素材。过滤后返回空列表是正常结果。

    Args:
        question: 用户的问题原文。
        top_k: 返回的片段数量，默认 3。
        paper_id: 只在指定的某一篇论文里检索（如 "arxiv:2601.00597"），
                  None 表示在整个库里检索。
    """
    # 将用户问题向量化，注意用 embed_query
    query_vector = embedder.embed_query(question)

    # 指定论文时用 Milvus 的过滤表达式把范围缩到那一篇。paper_id 只有
    # arxiv:<编号> 和 upload:<uuid> 两种来源，都不含引号，能安全拼进表达式。
    search_kwargs = {}
    if paper_id:
        search_kwargs["filter"] = f'source == "{paper_id}"'

    # 执行相似度检索，按 COSINE 距离返回最相似的前 top_k 个
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=top_k,
        output_fields=["text", "source", "chunk_id", "page", "page_start", "page_end"],
        **search_kwargs,
    )

    # 整理结果，提取出文本、来源、页码和分数
    hits = [
        {
            "text": hit["entity"]["text"],
            "source": hit["entity"]["source"],
            "chunk_id": hit["entity"]["chunk_id"],
            "page": hit["entity"].get("page", -1),
            "page_start": hit["entity"].get("page_start", -1),
            "page_end": hit["entity"].get("page_end", -1),
            "score": hit["distance"]
        }
        for hit in results[0]
    ]

    # 阈值过滤：返回条数可能少于 top_k，也可能一条都不剩
    return [hit for hit in hits if hit["score"] >= MIN_SCORE]



def delete_paper_vectors(paper_id: str) -> None:
    """删掉一篇论文在向量库里的全部块。

    原来这里的 list_papers 靠 filter="chunk_id == 0" 反推论文列表，
    现在论文清单由元数据表（paper_registry）提供，那个技巧就退休了。
    """
    client.delete(collection_name=COLLECTION_NAME, filter=f'source == "{paper_id}"')
    client.flush(collection_name=COLLECTION_NAME)
