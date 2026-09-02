from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(
    page_title="星座运势助手",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

WELCOME_MESSAGE = "hello，我是专业的星座运势大师，你有什么关于星座运势的问题吗？"
SUGGESTED_PROMPTS = [
    "我的今日运势怎么样？",
    "帮我看看感情运势",
    "我最近事业运如何？",
    "我是谁？",
]


def post_json(
    base_url: str,
    path: str,
    *,
    params: dict | None = None,
    data=None,
    headers: dict | None = None,
    timeout: int = 300,
):
    response = requests.post(
        f"{base_url.rstrip('/')}{path}",
        params=params,
        data=data,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def render_status_card(title: str, result: dict | None):
    if not result:
        return

    status = result.get("status", "unknown")
    message = result.get("message", "")
    chunk_count = result.get("chunk_count", 0)

    st.markdown(
        f"""
        <div style="padding: 12px 14px; border-radius: 12px; border: 1px solid #ddd; background: #fafafa; margin-bottom: 12px;">
            <div style="font-size: 0.95rem; font-weight: 700; margin-bottom: 6px;">{title}</div>
            <div style="margin-bottom: 6px;">状态: <b>{status}</b>，分块数: <b>{chunk_count}</b></div>
            <div>{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def conversation_title(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and message.get("content"):
            title = str(message["content"]).strip()
            return title[:18] if len(title) <= 18 else f"{title[:18]}..."
    return "新对话"


def ensure_chat_state():
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "active_history_id" not in st.session_state:
        st.session_state.active_history_id = None
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "processing_query" not in st.session_state:
        st.session_state.processing_query = None
    if "processing_phase" not in st.session_state:
        st.session_state.processing_phase = None


def build_auth_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    dashscope_api_key = st.session_state.get("dashscope_api_key_input", "").strip()
    yuanfenju_api_key = st.session_state.get("yuanfenju_api_key_input", "").strip()
    tavily_api_key = st.session_state.get("tavily_api_key_input", "").strip()

    if dashscope_api_key:
        headers["X-DASHSCOPE-API-KEY"] = dashscope_api_key
    if yuanfenju_api_key:
        headers["X-YUANFENJU-API-KEY"] = yuanfenju_api_key
    if tavily_api_key:
        headers["X-TAVILY-API-KEY"] = tavily_api_key
    return headers


def archive_current_chat() -> None:
    messages = st.session_state.chat_messages
    meaningful_messages = [message for message in messages if message.get("content") and message.get("content") != "星座大师思考中..."]
    if not meaningful_messages:
        return
    if meaningful_messages == [{"role": "assistant", "content": WELCOME_MESSAGE}]:
        return

    st.session_state.conversation_history.insert(
        0,
        {
            "id": str(uuid.uuid4()),
            "title": conversation_title(meaningful_messages),
            "messages": [{"role": message["role"], "content": message["content"]} for message in meaningful_messages],
        },
    )


def start_new_chat() -> None:
    archive_current_chat()
    st.session_state.chat_messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.active_history_id = None
    st.session_state.pending_query = None
    st.session_state.processing_query = None
    st.session_state.processing_phase = None


def clear_current_chat() -> None:
    st.session_state.chat_messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.active_history_id = None
    st.session_state.pending_query = None
    st.session_state.processing_query = None
    st.session_state.processing_phase = None


def load_history_chat(history_id: str) -> None:
    for history_item in st.session_state.conversation_history:
        if history_item["id"] == history_id:
            st.session_state.chat_messages = [
                {"role": message["role"], "content": message["content"]} for message in history_item["messages"]
            ]
            st.session_state.active_history_id = history_id
            st.session_state.pending_query = None
            st.session_state.processing_query = None
            st.session_state.processing_phase = None
            return


def delete_history_chat(history_id: str) -> None:
    st.session_state.conversation_history = [
        history_item for history_item in st.session_state.conversation_history if history_item["id"] != history_id
    ]
    if st.session_state.active_history_id == history_id:
        st.session_state.active_history_id = None
        st.session_state.chat_messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
        st.session_state.pending_query = None
        st.session_state.processing_query = None
        st.session_state.processing_phase = None


def sync_active_history() -> None:
    active_history_id = st.session_state.active_history_id
    if not active_history_id:
        return

    for history_item in st.session_state.conversation_history:
        if history_item["id"] == active_history_id:
            history_item["messages"] = [
                {"role": message["role"], "content": message["content"]} for message in st.session_state.chat_messages
            ]
            history_item["title"] = conversation_title(st.session_state.chat_messages)
            return


def reset_chat():
    clear_current_chat()


def fetch_chat_answer(base_url: str, user_id: str, query: str) -> str:
    result = post_json(
        base_url,
        "/chat",
        params={"query": query, "user_id": user_id},
        headers=build_auth_headers(),
    )
    return result.get("message", "")


def handle_chat_query(base_url: str, user_id: str, query: str):
    try:
        answer = fetch_chat_answer(base_url, user_id, query)
        st.session_state.chat_messages[-1]["content"] = answer or "暂无回答"
    except Exception as exc:
        st.session_state.chat_messages[-1]["content"] = f"聊天请求失败: {exc}"


def stream_response_text(text: str):
    """逐字流式输出，同时保留完整文本。"""
    full_response = []
    for character in text:
        full_response.append(character)
        yield character
    return "".join(full_response)


ensure_chat_state()

with st.sidebar:
    st.header("控制台")
    backend_url = st.text_input("后端地址", value="https://staroracle-agent.onrender.com")
    st.subheader("API Keys")
    dashscope_api_key = st.text_input(
        "DASHSCOPE_API_KEY",
        key="dashscope_api_key_input",
        type="password",
        placeholder="请输入 DashScope API Key",
    )
    yuanfenju_api_key = st.text_input(
        "YUANFENJU_API_KEY",
        key="yuanfenju_api_key_input",
        type="password",
        placeholder="请输入元亨聚 API Key",
    )
    tavily_api_key = st.text_input(
        "TAVILY_API_KEY",
        key="tavily_api_key_input",
        type="password",
        placeholder="请输入 Tavily API Key",
    )

    st.caption("这些 key 只保存在当前浏览器会话，不会写入服务器环境变量。")

    user_id = st.text_input("用户 ID", value="default")

    col_new, col_clear = st.columns(2)
    with col_new:
        if st.button("新对话", use_container_width=True):
            start_new_chat()
            st.rerun()
    with col_clear:
        if st.button("清空", use_container_width=True):
            clear_current_chat()
            st.rerun()

    st.divider()
    st.subheader("历史对话")

    if st.session_state.conversation_history:
        for index, conversation in enumerate(st.session_state.conversation_history, start=1):
            row_cols = st.columns([8, 1])
            with row_cols[0]:
                button_label = f"{index}. {conversation['title']}"
                is_active = conversation["id"] == st.session_state.active_history_id
                if st.button(
                    button_label,
                    use_container_width=True,
                    key=f"history_{conversation['id']}",
                    type="primary" if is_active else "secondary",
                ):
                    sync_active_history()
                    load_history_chat(conversation["id"])
                    st.rerun()
            with row_cols[1]:
                if st.button("×", key=f"delete_{conversation['id']}", help="删除历史对话"):
                    delete_history_chat(conversation["id"])
                    st.rerun()
    else:
        st.caption("暂无历史对话")

    # st.divider()
    # st.write("当前能力")
    # st.write("- 聊天")
    # st.write("- URL / PDF / 文本入库")
    # st.write("- 知识库 RAG 检索")

    st.divider()
    st.subheader("知识入库")

    with st.expander("URL 入库", expanded=False):
        with st.form("ingest_url_form"):
            url_value = st.text_input("待入库 URL", placeholder="https://example.com/article")
            submit_url = st.form_submit_button("入库 URL", use_container_width=True)
            if submit_url:
                if not url_value.strip():
                    st.warning("请输入 URL。")
                else:
                    with st.spinner("正在处理 URL..."):
                        try:
                            result = post_json(
                                backend_url,
                                "/add_urls",
                                params={"URL": url_value},
                                headers=build_auth_headers(),
                            )
                            render_status_card("URL 入库结果", result)
                        except Exception as exc:
                            st.error(f"URL 入库失败: {exc}")

    with st.expander("PDF 入库", expanded=False):
        uploaded_pdf = st.file_uploader("选择 PDF 文件", type=["pdf"])
        if st.button("入库 PDF", use_container_width=True):
            if uploaded_pdf is None:
                st.warning("请先上传 PDF 文件。")
            else:
                with st.spinner("正在处理 PDF..."):
                    try:
                        temp_dir = Path(tempfile.gettempdir()) / "p6_streamlit_uploads"
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        target_path = temp_dir / uploaded_pdf.name
                        target_path.write_bytes(uploaded_pdf.getbuffer())
                        result = post_json(
                            backend_url,
                            "/add_pdfs",
                            params={"pdf_path": str(target_path)},
                            headers=build_auth_headers(),
                        )
                        render_status_card("PDF 入库结果", result)
                    except Exception as exc:
                        st.error(f"PDF 入库失败: {exc}")

    with st.expander("文本入库", expanded=False):
        with st.form("ingest_text_form"):
            source_name = st.text_input("文本来源名", value="manual_text")
            text_value = st.text_area("待入库文本", height=180, placeholder="输入一段要保存到知识库的文本")
            submit_text = st.form_submit_button("入库文本", use_container_width=True)
            if submit_text:
                if not text_value.strip():
                    st.warning("请输入文本内容。")
                else:
                    with st.spinner("正在处理文本..."):
                        try:
                            result = post_json(
                                backend_url,
                                "/add_texts",
                                params={"source_name": source_name},
                                data=text_value.encode("utf-8"),
                                headers={**build_auth_headers(), "Content-Type": "text/plain; charset=utf-8"},
                            )
                            render_status_card("文本入库结果", result)
                        except Exception as exc:
                            st.error(f"文本入库失败: {exc}")

chat_col, _ = st.columns([7, 3])
with chat_col:
    st.markdown("## 星座运势助手-对话")

    display_messages = list(st.session_state.chat_messages)

    chat_history = st.container(height=690, border=True)
    with chat_history:
        for message in display_messages:
            st.chat_message(message["role"]).write(message["content"])

    prompt_cols = st.columns(len(SUGGESTED_PROMPTS))
    for index, prompt in enumerate(SUGGESTED_PROMPTS):
        with prompt_cols[index]:
            if st.button(prompt, use_container_width=True, key=f"prompt_{index}"):
                st.session_state.pending_query = prompt
                st.rerun()

    processing_phase = st.session_state.processing_phase
    if processing_phase == "fetching":
        handle_chat_query(backend_url, user_id, st.session_state.processing_query)
        sync_active_history()
        st.session_state.processing_query = None
        st.session_state.processing_phase = None
        st.rerun()

    pending_query = st.session_state.pending_query
    query = st.chat_input("输入问题，开始聊天")
    active_query = pending_query or query
    if active_query:
        active_query = active_query.strip()
        if active_query:
            st.session_state.pending_query = None
            st.session_state.chat_messages.append({"role": "user", "content": active_query})
            st.session_state.chat_messages.append({"role": "assistant", "content": "星座大师思考中..."})
            st.session_state.processing_query = active_query
            st.session_state.processing_phase = "fetching"
            st.rerun()
