import tiktoken
from typing import List, Dict

CHUNK_SIZE = 700
CHUNK_OVERLAP = 150

"""
Why 700 tokens?

• Large enough to preserve semantic meaning
• Small enough to maintain retrieval precision
• Reduces context window waste

Why overlap?

• Prevents information loss at chunk boundaries
• Preserves semantic continuity
• Improves answer completeness

Tradeoff:
Large chunks → better coherence but worse precision
Small chunks → better precision but fragmented context
700 with 150 overlap balances both.
"""

def chunk_text(text: str, filename: str) -> List[Dict]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        chunks.append({
            "text": chunk_text,
            "filename": filename,
            "chunk_id": chunk_id
        })

        start += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_id += 1

    return chunks