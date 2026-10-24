import os
import re
from uuid import uuid4
from typing import List

import pymupdf
from werkzeug.utils import secure_filename


def extract_text_from_pdf(file_path: str) -> str:
    try:
        text_chunks = []
        with pymupdf.open(file_path) as pdf:
            for page in pdf:
                text_chunks.append(page.get_text("text") or "")
        return "\n".join(text_chunks)

    except Exception as e:
        raise RuntimeError(f"Could not extract text from PDF: {e}") from e


def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    cleaned = re.sub(r'\s+', ' ', text).strip()

    if not cleaned:
        return []

    return [
        cleaned[i:i + chunk_size]
        for i in range(0, len(cleaned), chunk_size)
    ]


def save_uploaded_file(uploaded_file, upload_folder: str) -> str:
    original_filename = uploaded_file.filename or ''
    safe_filename = secure_filename(original_filename)
    extension = os.path.splitext(safe_filename)[1].lower()
    if not safe_filename or extension not in {'.pdf', '.txt'}:
        raise ValueError('Please choose a PDF or TXT document.')

    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, f'{uuid4().hex}{extension}')
    uploaded_file.save(file_path)
    return file_path