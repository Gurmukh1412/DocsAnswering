import os
from typing import List, Tuple
from PyPDF2 import PdfReader


def load_documents(folder_path: str) -> List[Tuple[str, str]]:
    documents = []

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        if file.endswith(".txt") or file.endswith(".md"):
            with open(path, "r", encoding="utf-8") as f:
                documents.append((file, f.read()))

        elif file.endswith(".pdf"):
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            documents.append((file, text))

    return documents