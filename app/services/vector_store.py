import os

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import DataType, MilvusClient

# 先加载 .env：不调用 load_dotenv() 的话 os.getenv 取不到 DASHSCOPE_API_KEY，
# 向量化时会直接报鉴权失败（api_key 为 None）
load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
DB_NAME, COLLECTION_NAME, EMBED_DIM = "papers", "docs", 1536

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
    "chunk_id": DataType.INT64,       # 块序号
}


def _build_collection():
    """按当前 schema 新建 collection，并给向量字段建索引。"""
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBED_DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("source", DataType.VARCHAR, max_length=128)
    schema.add_field("chunk_id", DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", metric_type="COSINE", index_type="AUTOINDEX")

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)


def _existing_fields():
    """读取存量 collection 的字段名 -> 类型，$meta 是动态字段，忽略。"""
    info = client.describe_collection(collection_name=COLLECTION_NAME)
    return {f["name"]: f["type"] for f in info["fields"] if not f["name"].startswith("$")}


def ensure_collection(force_recreate: bool = False):
    """保证 collection 存在，且字段结构和 EXPECTED_FIELDS 一致。

    只判断 has_collection 是不够的：collection 一旦存在，schema 就再也改不了。
    早期用自动 schema 建出来的 docs 只有 id(INT64) + vector 两个字段，
    这时再按字符串 id 去 upsert，Milvus 会报：
      The Input data type is inconsistent with defined schema,
      {id} field should be a int64, but got a {<class 'str'>} instead
    所以结构对不上时要重建。
    """
    if client.has_collection(COLLECTION_NAME):
        if not force_recreate and _existing_fields() == EXPECTED_FIELDS:
            return

        row_count = client.get_collection_stats(collection_name=COLLECTION_NAME)["row_count"]
        if row_count:
            # 有数据时不做静默删除，避免误删；交给使用者决定改名还是迁移
            raise RuntimeError(
                f"collection `{COLLECTION_NAME}` 的字段结构与代码不一致，但其中已有 {row_count} 条数据。"
                f"请先备份/清理，或把 COLLECTION_NAME 改成新名字。"
            )
        client.drop_collection(collection_name=COLLECTION_NAME)

    _build_collection()


ensure_collection()

embedder = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))


def insert_chunks(chunks, arxiv_id):
    if not chunks:
        return 0

    # DashScopeEmbeddings 内部会按模型批量调用（text-embedding-v2 每批 25 条），
    # 所以这里不用自己分批，直接传整个列表即可
    vectors = embedder.embed_documents([c.page_content for c in chunks])

    # 模型输出维度和 collection 的 dim 必须一致，否则 upsert 会报维度错误
    if vectors and len(vectors[0]) != EMBED_DIM:
        raise ValueError(
            f"向量维度不一致：模型返回 {len(vectors[0])} 维，collection 定义的是 {EMBED_DIM} 维"
        )

    # 拼接字符串作为唯一ID
    data = [{"id": f"{arxiv_id}_{i}", "vector": v, "text": chunks[i].page_content, "source": arxiv_id, "chunk_id": i}
            for i, v in enumerate(vectors)]
    client.upsert(collection_name=COLLECTION_NAME, data=data)
    client.flush(collection_name=COLLECTION_NAME)  # 确保数据落盘
    return len(data)


def search_similar_chunks(question: str, top_k: int = 3):
    # 将用户问题向量化，注意用 embed_query
    query_vector = embedder.embed_query(question)

    # 执行相似度检索，按 COSINE 距离返回最相似的前 top_k 个
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=top_k,
        output_fields=["text", "source", "chunk_id"]
    )

    # 整理结果，提取出文本、来源和分数
    return [
        {
            "text": hit["entity"]["text"],
            "source": hit["entity"]["source"],
            "chunk_id": hit["entity"]["chunk_id"],
            "score": hit["distance"]
        }
        for hit in results[0]
    ]



def list_papers():
    res=client.query(
        collection_name=COLLECTION_NAME,
        filter="chunk_id == 0",
        output_fields=["source"]

    )
    # 提取 source 并用 set 去重，最后转回 list
    return list(set([item["source"] for item in res]))
