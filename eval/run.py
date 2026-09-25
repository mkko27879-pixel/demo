"""检索质量评测：跑一批问题，看期望的论文和关键词有没有被召回到前几名。

只走向量检索，不调用大模型，所以整轮只有 embedding 的费用。
改完 chunk_size / MIN_SCORE / 清洗逻辑之后先跑这个，用数字判断改动是好是坏，
而不是靠打开两三个例子肉眼看。

用法：
    .venv\\Scripts\\python.exe eval\\run.py
"""

import json
import sys
from pathlib import Path

# 直接跑脚本时，项目根目录不在 sys.path 里（pytest 有 pythonpath 配置，脚本没有）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.vector_store import MIN_SCORE, search_similar_chunks  # noqa: E402

QUESTIONS = Path(__file__).with_name("questions.json")
TOP_K = 5
# 正样本里"命中的块确实含期望关键词"的比例低于这个值就算不通过
PASS_RATIO = 0.8


def evaluate() -> int:
    cases = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows = []
    hit_at = {1: 0, 3: 0, 5: 0}
    positives = keyword_ok = 0
    negatives = negatives_ok = 0
    # level -> [通过, 总数]，分开统计简单题和难题，才知道难度差在哪
    by_level: dict[str, list[int]] = {}

    for case in cases:
        hits = search_similar_chunks(case["question"], top_k=TOP_K)
        expect = case.get("expect_paper")
        note = case.get("note", "")
        top1 = f"{hits[0]['source'].split(':')[0]} {hits[0]['score']:.3f}" if hits else "-"
        bucket = by_level.setdefault(case.get("level", "easy"), [0, 0])
        bucket[1] += 1

        if expect is None:
            negatives += 1
            ok = not hits
            negatives_ok += ok
            bucket[0] += ok
            result = "PASS 已过滤" if ok else f"FAIL 漏出 {len(hits)} 条"
            rows.append((case["question"], note, result, top1))
            continue

        positives += 1
        rank = next((i + 1 for i, h in enumerate(hits) if h["source"] == expect), None)
        for k in hit_at:
            if rank is not None and rank <= k:
                hit_at[k] += 1

        keywords = [w.lower() for w in case.get("expect_keywords", [])]
        kw_ok = any(
            h["source"] == expect and any(w in h["text"].lower() for w in keywords)
            for h in hits
        ) if keywords else rank is not None
        keyword_ok += kw_ok
        bucket[0] += kw_ok

        rows.append((case["question"], note,
                     f"{'PASS' if kw_ok else 'FAIL'} rank={rank or '-'}", top1))

    print(f"{'问题':<30}{'说明':<20}{'结果':<16}top1")
    print("-" * 88)
    for question, note, result, top1 in rows:
        print(f"{question[:28]:<30}{note[:18]:<20}{result:<16}{top1}")
    print("-" * 88)

    if positives:
        print(f"正样本 {positives}：命中@1 {hit_at[1]} | 命中@3 {hit_at[3]} | "
              f"命中@5 {hit_at[5]} | 关键词命中 {keyword_ok}")
    print(f"负样本 {negatives}：正确过滤 {negatives_ok}（MIN_SCORE={MIN_SCORE}）")
    for level, (ok, total) in sorted(by_level.items()):
        print(f"  {level}: {ok}/{total} = {ok / total:.0%}")

    ratio = keyword_ok / positives if positives else 0
    print(f"\n综合：关键词命中率 {ratio:.0%}（通过线 {PASS_RATIO:.0%}）")
    return 0 if positives and ratio >= PASS_RATIO else 1


if __name__ == "__main__":
    raise SystemExit(evaluate())
