TEMPLATE = """Answer the question using only the passages. Reply with a short answer (a few words), no explanation.

{ctx}

Question: {q}
Answer:"""
def build_prompt(q, passages):
    ctx = "\n\n".join(f"[{i+1}] {p['title']}: {p['text']}" for i, p in enumerate(passages))
    return TEMPLATE.format(ctx=ctx, q=q)