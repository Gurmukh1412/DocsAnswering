import os
import tempfile
from typing import List
from pathlib import Path


def save_uploaded_files(uploaded_files) -> str:
    """
    Saves uploaded Streamlit files to a temporary directory.
    Returns the path of that directory.
    """

    temp_dir = tempfile.mkdtemp()

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    return temp_dir


def get_default_data_folder() -> str:
    """
    Returns default data folder path.
    """
    return str(Path("data").resolve())