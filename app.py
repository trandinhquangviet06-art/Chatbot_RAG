"""
app.py - Giao dien chatbot RAG viet bang Streamlit.
Phong cach Apple Design System, tuong duong voi app/ui.py (Gradio).

Deploy: Streamlit Cloud → Main file path: app.py
"""
import streamlit as st
import sys
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.generation.generator import answer_query

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinanceBot - Trợ lý Phân tích Tài chính",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Apple Design System CSS
# ---------------------------------------------------------------------------
APPLE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --apple-white:     #FFFFFF;
    --apple-soft:      #FBFBFD;
    --apple-gray-050:  #F5F5F7;
    --apple-ink:       #1D1D1F;
    --apple-muted:     #86868B;
    --apple-blue:      #0071E3;
    --apple-blue-deep: #0066CC;
    --apple-border:    #D2D2D7;
    --shadow-soft:     0 6px 20px rgba(0,0,0,.08);
    --shadow-card:     0 12px 32px rgba(0,0,0,.10);
    --shadow-focus:    0 0 0 4px rgba(0,113,227,.14);
    --font-sans: "Inter", -apple-system, "Helvetica Neue", Arial, sans-serif;
    --font-mono: "SF Mono", Menlo, Monaco, monospace;
}

html, body, [class*="css"] { font-family: var(--font-sans) !important; }
.stApp { background: var(--apple-soft) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* NAV */
.nav-bar {
    position: sticky; top: 0; z-index: 100;
    background: rgba(255,255,255,.82);
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    border-bottom: 1px solid var(--apple-border);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 48px; height: 52px;
}
.nav-logo { font-size: 20px; font-weight: 600; letter-spacing: -.02em; color: var(--apple-ink); display: flex; align-items: center; gap: 10px; }
.nav-dot  { width: 10px; height: 10px; border-radius: 50%; background: var(--apple-blue); display: inline-block; }
.nav-status { font-size: 13px; color: var(--apple-muted); }

/* HERO */
.hero-section { text-align: center; padding: 64px 24px 36px; max-width: 760px; margin: 0 auto; }
.hero-eyebrow { font-size: 12px; font-weight: 600; letter-spacing: .06em; color: var(--apple-blue); text-transform: uppercase; margin-bottom: 12px; }
.hero-title   { font-size: clamp(32px, 5vw, 52px); font-weight: 700; line-height: 1.06; letter-spacing: -.03em; color: var(--apple-ink); margin-bottom: 14px; }
.hero-sub     { font-size: 18px; line-height: 1.5; letter-spacing: -.01em; color: var(--apple-muted); }

/* CHIP BUTTONS */
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
    background: var(--apple-white) !important;
    border: 1px solid var(--apple-border) !important;
    border-radius: 999px !important;
    color: var(--apple-ink) !important;
    font-size: 13px !important;
    font-family: var(--font-sans) !important;
    padding: 8px 18px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.06) !important;
    transition: all .15s !important;
    width: auto !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover {
    border-color: var(--apple-blue) !important;
    color: var(--apple-blue) !important;
    box-shadow: 0 2px 8px rgba(0,113,227,.12) !important;
}

/* CHAT CARD */
.chat-card-wrap { max-width: 860px; margin: 0 auto 24px; padding: 0 16px; }
.chat-card      { background: var(--apple-white); border-radius: 24px; box-shadow: var(--shadow-card); overflow: hidden; }
.chat-card-header { background: var(--apple-gray-050); border-bottom: 1px solid var(--apple-border); padding: 14px 24px; display: flex; align-items: center; gap: 8px; }
.hd   { width: 13px; height: 13px; border-radius: 50%; display: inline-block; }
.hd-r { background: #FF5F57; }
.hd-a { background: #FFBD2E; }
.hd-g { background: #28C840; }
.chat-header-title { font-size: 13px; font-weight: 500; color: var(--apple-muted); margin-left: 4px; }

/* MESSAGES */
.chat-messages {
    padding: 24px 28px; min-height: 360px; max-height: 500px;
    overflow-y: auto; scroll-behavior: smooth;
    display: flex; flex-direction: column; gap: 14px;
}
.chat-messages::-webkit-scrollbar { width: 5px; }
.chat-messages::-webkit-scrollbar-thumb { background: var(--apple-border); border-radius: 99px; }
.msg-row      { display: flex; align-items: flex-end; gap: 10px; }
.msg-row.user { flex-direction: row-reverse; }
.msg-row.bot  { flex-direction: row; }
.msg-bubble   { max-width: 72%; font-size: 15px; line-height: 1.6; padding: 12px 18px; border-radius: 20px; word-break: break-word; }
.msg-row.user .msg-bubble { background: var(--apple-blue); color: var(--apple-white); border-radius: 20px 20px 6px 20px; box-shadow: 0 2px 8px rgba(0,113,227,.22); }
.msg-row.bot  .msg-bubble { background: var(--apple-gray-050); color: var(--apple-ink); border-radius: 20px 20px 20px 6px; border: 1px solid var(--apple-border); max-width: 82%; }

/* STREAMLIT CHAT INPUT override */
div[data-testid="stChatInput"] {
    max-width: 860px !important;
    margin: 0 auto !important;
    padding: 0 16px !important;
}
div[data-testid="stChatInput"] textarea {
    border-radius: 14px !important;
    font-family: var(--font-sans) !important;
    font-size: 15px !important;
    border: 1px solid var(--apple-border) !important;
    background: var(--apple-white) !important;
    box-shadow: var(--shadow-soft) !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: var(--apple-blue) !important;
    box-shadow: var(--shadow-focus) !important;
}

/* SOURCE CARD */
.source-card-wrap { max-width: 860px; margin: 16px auto 32px; padding: 0 16px; }
.source-card  { background: var(--apple-white); border-radius: 16px; border: 1px solid var(--apple-border); padding: 16px 22px; box-shadow: var(--shadow-soft); }
.source-label { font-size: 11px; font-weight: 600; letter-spacing: .05em; color: var(--apple-muted); text-transform: uppercase; margin-bottom: 8px; }
.source-text  { font-family: var(--font-mono); font-size: 13px; line-height: 1.6; color: var(--apple-ink); background: var(--apple-gray-050); border: 1px solid var(--apple-border); border-radius: 10px; padding: 10px 14px; }

/* FOOTER */
.footer { text-align: center; padding: 24px 24px 40px; font-size: 13px; color: var(--apple-muted); border-top: 1px solid var(--apple-border); margin-top: 8px; }
</style>
"""

# ---------------------------------------------------------------------------
# State init — chỉ chạy 1 lần
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Xin chào! Tôi là FinanceBot. Hãy đặt câu hỏi về báo cáo tài chính và tôi sẽ tìm kiếm câu trả lời chính xác từ tài liệu."}
    ]
if "status" not in st.session_state:
    st.session_state.status = "Sẵn sàng nhận câu hỏi của bạn."
# pending_query: chứa câu hỏi chờ xử lý (None = không có gì cần xử lý)
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ---------------------------------------------------------------------------
# XỬ LÝ QUERY — thực hiện TRƯỚC KHI render bất cứ thứ gì
# Đây là điểm mấu chốt: xử lý 1 lần rồi set pending_query = None → không loop
# ---------------------------------------------------------------------------
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None          # ← xóa ngay, tránh loop

    # Thêm câu hỏi user
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.status = "Đang tìm kiếm trong tài liệu..."

    # Gọi RAG pipeline
    try:
        accumulated = ""
        for chunk in answer_query(query):
            accumulated += chunk
        st.session_state.messages.append({"role": "bot", "content": accumulated})
        st.session_state.status = "Hoàn thành. Sẵn sàng cho câu hỏi tiếp theo."
    except Exception as e:
        st.session_state.messages.append({"role": "bot", "content": f"⚠️ Lỗi: {e}"})
        st.session_state.status = f"Lỗi: {e}"

    st.rerun()  # rerun để hiển thị messages mới; lần này pending_query = None nên dừng

# ---------------------------------------------------------------------------
# RENDER UI
# ---------------------------------------------------------------------------
st.markdown(APPLE_CSS, unsafe_allow_html=True)

# NAV
st.markdown("""
<nav class="nav-bar">
    <span class="nav-logo"><span class="nav-dot"></span>FinanceBot</span>
    <span class="nav-status">Trợ lý Phân tích Tài chính &middot; Gemini Flash</span>
</nav>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero-section">
    <p class="hero-eyebrow">RAG &middot; Retrieval-Augmented Generation</p>
    <h1 class="hero-title">Phân tích báo cáo tài chính<br>một cách thông minh.</h1>
    <p class="hero-sub">Đặt câu hỏi bằng Tiếng Việt — FinanceBot sẽ tra cứu tài liệu,
    trích dẫn nguồn chính xác và trả lời tức thì.</p>
</div>
""", unsafe_allow_html=True)

# SUGGESTION CHIPS
SUGGESTIONS = [
    "Chi phí R&D của 3M năm 2015?",
    "Doanh thu thuần năm gần nhất?",
    "Lợi nhuận gộp và biên lợi nhuận?",
]
cols = st.columns([2, 3, 3, 3, 2])
for i, s in enumerate(SUGGESTIONS):
    with cols[i + 1]:
        if st.button(s, key=f"chip_{i}"):
            st.session_state.pending_query = s
            st.rerun()

# CHAT CARD
def render_messages(messages):
    html = '<div class="chat-messages" id="chat-messages">'
    for m in messages:
        role = "user" if m["role"] == "user" else "bot"
        content = m["content"].replace("\n", "<br>")
        html += f'<div class="msg-row {role}"><div class="msg-bubble">{content}</div></div>'
    html += "</div>"
    return html

st.markdown('<div class="chat-card-wrap"><div class="chat-card">', unsafe_allow_html=True)
st.markdown("""
<div class="chat-card-header">
    <span class="hd hd-r"></span><span class="hd hd-a"></span><span class="hd hd-g"></span>
    <span class="chat-header-title">FinanceBot — Cuộc trò chuyện</span>
</div>
""", unsafe_allow_html=True)
st.markdown(render_messages(st.session_state.messages), unsafe_allow_html=True)
st.markdown("</div></div>", unsafe_allow_html=True)

# CHAT INPUT — st.chat_input tự xóa sau mỗi lần submit, không gây loop
if prompt := st.chat_input("Hỏi về báo cáo tài chính..."):
    st.session_state.pending_query = prompt
    st.rerun()

# STATUS CARD
st.markdown(f"""
<div class="source-card-wrap">
  <div class="source-card">
    <p class="source-label">Trạng thái</p>
    <div class="source-text">{st.session_state.status}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# XÓA CHAT
col_clear, _ = st.columns([1, 5])
with col_clear:
    if st.button("🗑️ Xóa cuộc trò chuyện", key="clear_btn"):
        st.session_state.messages = [
            {"role": "bot", "content": "Cuộc trò chuyện đã được xóa. Hãy đặt câu hỏi mới!"}
        ]
        st.session_state.status = "Sẵn sàng nhận câu hỏi của bạn."
        st.rerun()

# FOOTER
st.markdown("""
<footer class="footer">
    Copyright &copy; 2025 FinanceBot &nbsp;&middot;&nbsp;
    Powered by Gemini Flash &amp; LangChain &nbsp;&middot;&nbsp;
    Dữ liệu chỉ từ tài liệu đã nạp vào hệ thống
</footer>
""", unsafe_allow_html=True)
