RAG_SEARCH = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": "Search the museum knowledge base for information about artworks, exhibitions, artists, historical periods, techniques, or any museum-related topic. Use this when the user asks a question that requires factual information from the database.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information. Use specific keywords like artwork names, artist names, periods, or topics.",
                }
            },
            "required": ["query"],
        },
    },
}

RAG_IMAGE_SEARCH = {
    "type": "function",
    "function": {
        "name": "rag_image_search",
        "description": "Search the museum knowledge base using an image. Use this when the user provides an image and asks about it. This will find visually similar artworks and related information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A text description or question about the image to combine with visual search.",
                }
            },
            "required": ["query"],
        },
    },
}

ALL_TOOLS = [RAG_SEARCH, RAG_IMAGE_SEARCH]
TEXT_TOOLS = [RAG_SEARCH]
