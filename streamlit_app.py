import streamlit as st
import logging
import re
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline
from handelfile import save_uploaded_files, get_default_data_folder

load_dotenv()
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Company Policy Assistant", layout="wide")

st.title("📄 Company Policy Assistant")

# ---------------------------------------------------------
# File Upload Section
# ---------------------------------------------------------

st.sidebar.header("📂 Data Source")

uploaded_files = st.sidebar.file_uploader(
    "Upload policy files (PDF, TXT, MD)",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True
)

# Determine data source
if uploaded_files:
    st.sidebar.success("Using uploaded files.")
    data_folder = save_uploaded_files(uploaded_files)
    st.info("📂 Using uploaded documents.")
else:
    st.sidebar.info("Using default data folder.")
    data_folder = get_default_data_folder()
    st.info("📁 Using default 'data/' documents.")

# ---------------------------------------------------------
# Initialize RAG (No caching to allow dynamic reload)
# ---------------------------------------------------------

if "rag_instance" not in st.session_state or st.sidebar.button("🔄 Rebuild Knowledge Base"):
    rag = RAGPipeline(
        llm_model="mistralai/mistral-7b-instruct",
        temperature=0.0
    )
    rag.build_knowledge_base(data_folder)
    st.session_state.rag_instance = rag
else:
    rag = st.session_state.rag_instance

# ---------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("question_form"):
    user_question = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Submit")

if submitted and user_question:
    with st.spinner("Generating answer..."):
        answer = rag.answer_question(user_question)
        st.session_state.chat_history.append(
            {"question": user_question, "answer": answer}
        )

for chat in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.markdown(chat["answer"])

        match = re.search(r"## Confidence Score\s*([\d\.]+)", chat["answer"])
        if match:
            st.progress(float(match.group(1)))

st.sidebar.markdown("---")
st.sidebar.markdown("### Model: mistralai/mistral-7b-instruct (Free)")