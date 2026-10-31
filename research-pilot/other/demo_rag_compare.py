"""Compare three evidence settings using fictional teaching records."""
import asyncio
import json
from sentence_transformers import SentenceTransformer
from app.agent.paper_search_agent import client, model
from app.rag.retriever import retrieve_papers_semantic

papers = {
    "demo://P1": {"abstract": "In the fictional CEDAR-Z7 radar gesture experiment, random-split accuracy was 94.2%, while subject-independent accuracy was 71.4%."},
    "demo://P2": {"abstract": "In a fictional EEG emotion experiment, graph neural networks were evaluated for recognizing emotional states."},
    "demo://P3": {"abstract": "In a fictional radar hardware experiment, engineers compared battery life and power consumption during continuous sensing."},
}
question = "教学记录 CEDAR-Z7 的跨被试准确率是多少？比随机划分低多少个百分点？请引用资料 ID。"
instructions = "资料均为虚构教学记录。只根据本轮参考资料用中文回答；资料不足时明确说资料不足，不猜测数值。"

async def main():
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    hits = retrieve_papers_semantic(papers, "CEDAR-Z7 subject-independent accuracy versus random split", encoder, top_k=1)
    all_evidence = [{"paper_id": pid, "abstract": item["abstract"]} for pid, item in papers.items()]
    retrieved = [{"paper_id": item["paper_id"], "abstract": item["abstract"]} for item in hits]
    async with client:
        for label, evidence in [("A 无参考资料", []), ("B 全量上下文", all_evidence), ("C RAG", retrieved)]:
            response = await client.responses.create(
                model=model, instructions=instructions, tools=[],
                input=f"问题：{question}\n参考资料：{json.dumps(evidence, ensure_ascii=False)}",
            )
            tokens = response.usage.input_tokens if response.usage else "未提供"
            print(f"\n{label} | 资料数={len(evidence)} | 输入 tokens={tokens}")
            print(response.output_text)

if __name__ == "__main__":
    asyncio.run(main())
