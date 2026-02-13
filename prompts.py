"""
Prompt Engineering Strategy:

Version 2 improvements:
- Explicit refusal policy
- Strict grounding
- Citation enforcement
- No inference beyond context
- Structured output

This reduces hallucination significantly.
"""


def build_prompt_v2(question: str, context: str) -> str:
    return f"""
You are a compliance-focused policy assistant.

You MUST follow these rules:

1. Answer STRICTLY using ONLY the provided context.
2. Do NOT use any prior knowledge.
3. Do NOT make assumptions.
4. If the answer is not explicitly found in the context, respond exactly:
   "The information is not available in the provided documents."

5. Cite exact source and chunk_id from the context.
6. Do NOT combine evidence unless clearly stated in context.

Output format:

## Answer

## Supporting Evidence
(Source: filename, chunk_id)

Context:
{context}

Question:
{question}
"""