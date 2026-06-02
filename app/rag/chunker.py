def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            split_pos = text.rfind("\n", start, end)
            if split_pos == -1 or split_pos <= start:
                split_pos = text.rfind(". ", start, end)
            if split_pos == -1 or split_pos <= start:
                split_pos = text.rfind(" ", start, end)
            if split_pos > start:
                end = split_pos + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks
