from pathlib import Path


def read_file(path: str) -> str:
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        return _read_pdf(path)
    elif ext in (".docx", ".doc"):
        return _read_docx(path)
    elif ext in (".txt", ".md", ".csv", ".json"):
        return _read_text(path)
    else:
        return _read_text(path)


def _read_pdf(path: str) -> str:
    import fitz
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def _read_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
