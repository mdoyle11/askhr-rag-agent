from askhr.retrieval.search import search
import asyncio

queries = [
    'What is the attendance policy?',
    'How much PTO do teachers get?',
    'What is the dress code?',
    'What happens if I am late three times?',
]

for q in queries:
    print(f'\n===== {q} =====')
    results = asyncio.run(search(q, top_k=3))
    for i, r in enumerate(results):
        print(f'[{i+1}] score={r["score"]:.3f} section={r.get("section")}')
        print(f'    {r["content"][:150]}...')