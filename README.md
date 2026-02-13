📄 Company Policy RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions strictly from company policy documents.

This project was built as part of an AI Engineering internship evaluation to demonstrate:

Data preparation and document processing

Vector search using FAISS

Hallucination-resistant prompting

Confidence scoring

Evaluation framework

Frontend deployment with Streamlit

🚀 Live Demo

https://docsanswering-kszmczvueuwvwbgn5iernm.streamlit.app/

🛠 Tech Stack

Python 3.10+

Streamlit (Frontend)

FAISS (Vector Search)

SentenceTransformers (Local Embeddings – all-MiniLM-L6-v2)

OpenRouter (LLM – mistral-7b-instruct)

PyPDF2 (PDF parsing)

tiktoken (token-aware chunking)

Logging module

python-dotenv

⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/Gurmukh1412/policy-rag-assistant.git
cd policy-rag-assistant
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Mac/Linux
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Add Environment Variable

Create a .env file locally:

OPENROUTER_API_KEY=your_openrouter_api_key

⚠️ Do NOT commit .env to GitHub.

For Streamlit Cloud deployment, add the key under:

Settings → Secrets

5️⃣ Run the Application
streamlit run streamlit_app.py
🏗 Architecture Overview

The system follows a classic Retrieval-Augmented Generation (RAG) architecture.

🔄 RAG Flow

User Question
→ Embed Query
→ Retrieve Top-k Chunks (FAISS + Cosine Similarity)
→ Apply Similarity Threshold Filtering
→ Build Grounded Prompt
→ LLM Generation (OpenRouter)
→ Return Structured Answer + Evidence + Confidence Score

📦 System Components
1️⃣ Data Preparation

Loads .txt, .pdf, .md files

Cleans whitespace

Token-based chunking

Overlap for semantic continuity

Why 700 Token Chunk Size?

Preserves semantic integrity of policy sections

Reduces fragmentation

Efficient use of LLM context window

Balances retrieval precision and completeness

Why 150 Token Overlap?

Prevents boundary information loss

Maintains semantic continuity

Improves retrieval recall

2️⃣ Embedding Layer

Local embeddings using all-MiniLM-L6-v2

384-dimensional vectors

Fully offline embeddings

No API cost

3️⃣ Vector Store

FAISS IndexFlatIP (cosine similarity)

Normalized vectors

Similarity threshold filtering (0.50–0.70 range)

4️⃣ Prompt Engineering

Strict grounding prompt used to prevent hallucination:

You are a compliance-focused policy assistant.

You MUST follow these rules:

1. Answer STRICTLY using ONLY the provided context.
2. Do NOT use prior knowledge.
3. Do NOT make assumptions.
4. If the answer is not explicitly found in the context, respond exactly:
   "The information is not available in the provided documents."

5. Cite exact source and chunk_id from the context.
6. Do NOT infer beyond context.

Output format:

## Answer

## Supporting Evidence
(Source: filename, chunk_id)
🛡 Hallucination Mitigation Strategy

The system reduces hallucinations through:

Similarity threshold gating

Strict grounding prompt

Mandatory citation requirement

Confidence scoring

Explicit fallback message

📊 Evaluation Results

The system was evaluated using 7 structured questions:

Type	Example	Result
Fully Answerable	What is the refund period?	✅ Correct
Fully Answerable	How long does shipping take?	✅ Correct
Partially Answerable	Are international returns allowed?	⚠️ Partial
Unanswerable	What is the CEO's salary?	✅ Correct refusal
Unanswerable	Does company offer gym membership?	✅ Correct refusal
Observations

Strong performance on answerable questions

Correct refusal behavior for out-of-scope queries

No hallucinated answers observed

Confidence score correlates with retrieval similarity

⚖️ Key Trade-offs
Local Embeddings vs API Embeddings

Local → Free and reproducible

API-based → Slightly more accurate but paid

Fixed Threshold

Simple and interpretable

Could be improved with dynamic threshold tuning

Dense Retrieval Only

No keyword-based BM25 search

Hybrid retrieval would improve short-query performance

No Cross-Encoder Reranking

Current approach uses bi-encoder retrieval

Reranking could improve precision

🚀 Improvements With More Time

If extended further, the system could include:

Hybrid Search (BM25 + Dense Retrieval)

Cross-Encoder Reranking

LLM-based automatic evaluation

Query expansion for short queries

Streaming responses

Persistent vector database storage

Section-level metadata filtering

📂 Dynamic File Upload

The system supports:

Uploading policy documents via the frontend

Default fallback to data/ folder if no upload provided

Dynamic knowledge base rebuilding

🏁 Final Notes

This project demonstrates:

End-to-end RAG implementation

Practical hallucination mitigation

Modular and production-ready architecture

Deployment-ready Streamlit frontend

Structured evaluation methodology

👨‍💻 Author

Gurmukh Singh
GitHub: https://github.com/Gurmukh1412
