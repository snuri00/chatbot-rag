import base64
import json

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Chatbot", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main .block-container {padding-top: 1rem; max-width: 900px;}
    .user-bubble {
        background: #2b313e;
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        margin: 8px 0 8px auto;
        max-width: 75%;
        width: fit-content;
        text-align: left;
    }
    .bot-bubble {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 16px 16px 16px 4px;
        padding: 12px 16px;
        margin: 8px auto 8px 0;
        max-width: 85%;
        width: fit-content;
    }
    .source-card {
        background: #262730;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 0.85em;
    }
    .score-high {color: #4CAF50; font-weight: bold;}
    .score-mid {color: #FF9800; font-weight: bold;}
    .score-low {color: #f44336; font-weight: bold;}
</style>
""", unsafe_allow_html=True)


def api_health():
    try:
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except Exception:
        return {"status": "offline", "qdrant": False, "llm": False}


def api_models():
    try:
        return requests.get(f"{API_URL}/models", timeout=5).json()
    except Exception:
        return []


def api_chat(question, image_bytes=None, model=None):
    if image_bytes:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        data = {"question": question}
        if model:
            data["language_model"] = model
        return requests.post(f"{API_URL}/chat/image", files=files, data=data, timeout=300).json()
    payload = {"question": question}
    if model:
        payload["language_model"] = model
    return requests.post(f"{API_URL}/chat", json=payload, timeout=300).json()


def api_chat_stream(question, image_bytes=None, model=None):
    payload = {"question": question}
    if image_bytes:
        payload["image"] = base64.b64encode(image_bytes).decode()
    if model:
        payload["language_model"] = model
    r = requests.post(f"{API_URL}/chat/stream", json=payload, stream=True, timeout=300)
    for line in r.iter_lines():
        if line:
            text = line.decode("utf-8")
            if text.startswith("data: "):
                yield json.loads(text[6:])


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            score = int(src["score"] * 100)
            score_class = "score-high" if score >= 60 else "score-mid" if score >= 40 else "score-low"
            title = src.get("title", "Unknown")
            origin = src.get("source", "")
            snippet = src.get("content", "")[:200]
            if len(src.get("content", "")) > 200:
                snippet += "..."
            st.markdown(f"""<div class="source-card">
<strong>{title}</strong> <span class="{score_class}">{score}%</span>
<br><small>{origin}</small><br>{snippet}
</div>""", unsafe_allow_html=True)


def render_message(msg):
    if msg["role"] == "user":
        html = '<div class="user-bubble">'
        if msg.get("image_bytes"):
            b64 = base64.b64encode(msg["image_bytes"]).decode()
            html += f'<img src="data:image/jpeg;base64,{b64}" style="max-width:200px;border-radius:8px;margin-bottom:8px;display:block;">'
        html += f'{msg["content"]}</div>'
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("sources"):
            render_sources(msg["sources"])


with st.sidebar:
    st.title("RAG Chatbot")
    st.divider()

    health = api_health()
    status_icon = {"healthy": "🟢", "degraded": "🟡"}.get(health["status"], "🔴")
    st.markdown(f"**Status:** {status_icon} {health['status'].upper()}")
    col_q, col_l = st.columns(2)
    col_q.markdown(f"Qdrant: {'✅' if health.get('qdrant') else '❌'}")
    col_l.markdown(f"LLM: {'✅' if health.get('llm') else '❌'}")

    st.divider()

    models = api_models()
    model_map = {m["id"]: m["label"] for m in models if m.get("available")}
    selected_model = None
    if model_map:
        selected_model = st.selectbox("Model", list(model_map.keys()), format_func=lambda x: model_map[x])

    use_streaming = st.toggle("Streaming", value=False)

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_image = None
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None

for msg in st.session_state.messages:
    render_message(msg)

if st.session_state.pending_image:
    cols = st.columns([1, 10])
    with cols[0]:
        st.image(st.session_state.pending_image["bytes"], width=60)
    with cols[1]:
        st.caption(st.session_state.pending_image["name"])
        if st.button("Remove", key="remove_img", type="secondary"):
            st.session_state.pending_image = None
            st.rerun()

prompt = st.chat_input("Ask about an artwork...", accept_file="multiple", file_type=["jpg", "jpeg", "png", "webp"])

if prompt:
    text = prompt.text if hasattr(prompt, "text") else str(prompt)
    files = prompt.files if hasattr(prompt, "files") else []

    image_bytes = None
    image_name = None

    if files:
        image_bytes = files[0].read()
        image_name = files[0].name
    elif st.session_state.pending_image:
        image_bytes = st.session_state.pending_image["bytes"]
        image_name = st.session_state.pending_image["name"]

    st.session_state.pending_image = None

    if not text.strip():
        text = "Describe this image"

    st.session_state.messages.append({
        "role": "user",
        "content": text,
        "image_bytes": image_bytes,
    })

    render_message(st.session_state.messages[-1])

    if use_streaming and not image_bytes:
        sources = []
        full_text = ""
        placeholder = st.empty()
        for event in api_chat_stream(text, model=selected_model):
            if event["type"] == "sources":
                sources = event.get("sources", [])
            elif event["type"] == "content":
                full_text += event["content"]
                placeholder.markdown(f'<div class="bot-bubble">{full_text}▌</div>', unsafe_allow_html=True)
            elif event["type"] == "done":
                placeholder.markdown(f'<div class="bot-bubble">{full_text}</div>', unsafe_allow_html=True)
        render_sources(sources)
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_text,
            "sources": sources,
        })
    else:
        with st.spinner(""):
            response = api_chat(text, image_bytes=image_bytes, model=selected_model)
        if "answer" in response:
            assistant_msg = {
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("sources", []),
            }
            st.session_state.messages.append(assistant_msg)
            render_message(assistant_msg)
        else:
            st.error(response.get("detail", "Error"))
