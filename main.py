import logging
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

load_dotenv()

logging.basicConfig(level=logging.INFO)


def main():
    rag = RAGPipeline(
        llm_model="mistralai/mistral-7b-instruct",  # Free OpenRouter model
        temperature=0.0
    )

    rag.build_knowledge_base("data")

    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        print(rag.answer_question(question))


if __name__ == "__main__":
    main()