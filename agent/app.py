import streamlit as st
import os
import base64
from pathlib import Path

# Import hàm khởi tạo từ file retrieval.py của bạn
from retrieval import build_answer_style_instruction, get_rag_components

# ================= 1. CẤU HÌNH TRANG =================
st.set_page_config(page_title="RAG Chatbot", layout="wide", initial_sidebar_state="expanded")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = True

ENV_FILE = Path(__file__).resolve().parent / ".env"


def _to_bool(raw_value, default=True):
    if raw_value is None:
        return default
    value = str(raw_value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def read_env_setting(key, default_value=None):
    # Allow process env to override local .env value.
    env_raw = os.getenv(key)
    if env_raw is not None:
        return env_raw

    if not ENV_FILE.exists():
        return default_value

    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        k, v = row.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return default_value


def save_env_setting(key, value):
    lines = []
    found = False

    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()

    updated_lines = []
    for line in lines:
        row = line.strip()
        if row and not row.startswith("#") and "=" in line:
            k, _ = line.split("=", 1)
            if k.strip() == key:
                updated_lines.append(f"{key}={value}")
                found = True
                continue
        updated_lines.append(line)

    if not found:
        if updated_lines and updated_lines[-1].strip() != "":
            updated_lines.append("")
        updated_lines.append(f"{key}={value}")

    ENV_FILE.write_text("\n".join(updated_lines).rstrip() + "\n", encoding="utf-8")


if "ocr_fallback_toggle" not in st.session_state:
    current_ocr = _to_bool(read_env_setting("OCR_FALLBACK_ENABLED", "true"), default=True)
    st.session_state.ocr_fallback_toggle = current_ocr

# ================= 2. KHỞI TẠO AI TỪ FILE RETRIEVAL.PY =================
@st.cache_resource(show_spinner="Đang khởi tạo hệ thống AI...")
def load_components():
    # Gọi hàm từ file retrieval.py
    return get_rag_components()

retriever, prompt_template, llm, format_docs = load_components()


def render_skeleton(container):
    skeleton_html = """
    <style>
    @keyframes pulseBox {
        0% {opacity: 0.78; transform: scale(1);}
        50% {opacity: 1; transform: scale(1.01);}
        100% {opacity: 0.78; transform: scale(1);}
    }
    @keyframes sheenMove {
        0% {transform: translateX(-160%);}
        100% {transform: translateX(160%);}
    }
    @keyframes blinkDot {
        0%, 20% {opacity: 0.25;}
        50% {opacity: 1;}
        100% {opacity: 0.25;}
    }
    .sk-panel {
        position: relative;
        overflow: hidden;
        margin-top: 8px;
        max-width: 720px;
        min-height: 116px;
        border-radius: 18px;
        background: linear-gradient(135deg, #0b2a3b 0%, #123e57 48%, #1a5670 100%);
        border: 1px solid #2e6f8d;
        box-shadow: 0 10px 24px rgba(10, 40, 58, 0.42);
        animation: pulseBox 1.7s ease-in-out infinite;
    }
    .sk-panel::after {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 38%;
        height: 100%;
        background: linear-gradient(90deg, rgba(67, 147, 179, 0) 0%, rgba(67, 147, 179, 0.38) 50%, rgba(67, 147, 179, 0) 100%);
        animation: sheenMove 1.55s linear infinite;
    }
    .sk-title {
        position: absolute;
        top: 14px;
        left: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        color: #cde9f6;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.2px;
    }
    .sk-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #7dd3f6;
        animation: blinkDot 1.1s infinite;
    }
    .sk-dot.d2 {animation-delay: 0.15s;}
    .sk-dot.d3 {animation-delay: 0.3s;}
    .sk-hint {
        position: absolute;
        bottom: 14px;
        left: 14px;
        color: #9cd3e9;
        font-size: 12px;
        opacity: 0.9;
    }
    </style>
    <div class="sk-panel">
        <div class="sk-title">
            <span>Assistant đang soạn phản hồi</span>
            <span class="sk-dot"></span>
            <span class="sk-dot d2"></span>
            <span class="sk-dot d3"></span>
        </div>
        <div class="sk-hint">Đang xử lý ngữ cảnh và tổng hợp câu trả lời...</div>
    </div>
    """
    container.markdown(skeleton_html, unsafe_allow_html=True)


def chunk_to_text(chunk):
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(item.get("text", ""))
        return "".join(out)
    return str(content)

# ================= 3. HÀM XỬ LÝ POPUP TÀI LIỆU =================
@st.dialog("📄 Chi tiết tài liệu", width="large")
def show_pdf_popup(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi mở file: {e}")

# ================= 4. GIAO DIỆN NAVBAR BÊN TRÁI =================
with st.sidebar:
    st.title("⚙️ Điều khiển")

    st.markdown("### OCR Fallback")
    st.toggle(
        "Bật OCR khi trang PDF rỗng",
        key="ocr_fallback_toggle",
        help="Áp dụng OCR cho các trang không lấy được text layer.",
    )
    if st.button("💾 Lưu cài đặt OCR", use_container_width=True):
        save_env_setting(
            "OCR_FALLBACK_ENABLED",
            "true" if st.session_state.ocr_fallback_toggle else "false",
        )
        st.success("Đã lưu OCR_FALLBACK_ENABLED vào .env")
    st.markdown("---")
    
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("### 🕒 Lịch sử Chat")
    st.button("💬 Tổng quan dự án", use_container_width=True)
    st.button("💬 Hỏi đáp quy trình", use_container_width=True)

    st.markdown("<br>" * 8, unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🎧 Hỗ trợ")
    st.markdown("📧 support@yourdomain.com")
    st.markdown("---")

    st.markdown("### 📂 Nguồn tài liệu")
    paper_dir = "./papers/ai_thucchien"
    
    if os.path.exists(paper_dir):
        files = sorted([f for f in os.listdir(paper_dir) if f.endswith('.pdf')])
        if files:
            st.caption(f"Tổng tài liệu: {len(files)}")
            keyword = st.text_input(
                "Tìm tài liệu",
                placeholder="Nhập tên file...",
                key="doc_filter_keyword",
            ).strip().lower()

            filtered_files = [f for f in files if keyword in f.lower()] if keyword else files

            if not filtered_files:
                st.info("Không có tài liệu khớp từ khóa.")
            else:
                selected_file = st.selectbox(
                    "Chọn tài liệu",
                    options=filtered_files,
                    key="selected_source_file",
                )
                if st.button("📄 Mở tài liệu", use_container_width=True):
                    file_path = os.path.join(paper_dir, selected_file)
                    show_pdf_popup(file_path)
        else:
            st.warning("Thư mục trống.")
    else:
        st.error("Chưa tạo thư mục ./papers")

# ================= 5. GIAO DIỆN CHATBOT CHÍNH =================
st.title("🤖 RAG Knowledge Assistant")

st.toggle("Hiện tiến trình AI suy nghĩ", key="show_thinking")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hỏi tôi bất cứ điều gì về tài liệu..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            thought_holder = st.empty()

            def show_step(step_text):
                if st.session_state.show_thinking:
                    thought_holder.info(step_text)

            show_step("B1: Đang truy xuất hybrid (vector + keyword) và rerank ngữ cảnh...")
            docs = retriever.invoke(prompt)

            sources = []
            for d in docs[:5]:
                src = d.metadata.get("source", "N/A")
                page = d.metadata.get("page")
                if page is not None:
                    sources.append(f"- {src} (page {page + 1})")
                else:
                    sources.append(f"- {src}")

            show_step(f"B2: Đang chuẩn bị prompt từ {len(docs)} đoạn ngữ cảnh...")
            context = format_docs(docs)
            answer_style_instruction = build_answer_style_instruction(prompt)
            prompt_value = prompt_template.invoke(
                {
                    "context": context,
                    "question": prompt,
                    "answer_style_instruction": answer_style_instruction,
                }
            )

            show_step("B3: Đang tạo câu trả lời...")

            skeleton_holder = st.empty()
            render_skeleton(skeleton_holder)
            response_holder = st.empty()

            full_response = ""
            first_token_arrived = False
            for chunk in llm.stream(prompt_value):
                text = chunk_to_text(chunk)
                if not text:
                    continue
                if not first_token_arrived:
                    skeleton_holder.empty()
                    first_token_arrived = True
                full_response += text
                response_holder.markdown(full_response + "▌")

            if not first_token_arrived:
                skeleton_holder.empty()

            if sources:
                full_response += "\n\nNguồn tham khảo:\n" + "\n".join(sources)

            thought_holder.empty()
            response_holder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            if "thought_holder" in locals():
                thought_holder.empty()
            st.error(f"Lỗi khi gọi mô hình: {e}")