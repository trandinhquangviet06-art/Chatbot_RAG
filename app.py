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

/* === Reset Streamlit chrome === */
html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
}
.stApp {
    background: var(--apple-soft) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none !important; }

/* === NAV BAR === */
.nav-bar {
    position: sticky; top: 0; z-index: 100;
    background: rgba(255,255,255,.82);
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    border-bottom: 1px solid var(--apple-border);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 48px; height: 52px;
}
.nav-logo {
    font-size: 20px; font-weight: 600; letter-spacing: -.02em;
    color: var(--apple-ink); display: flex; align-items: center; gap: 10px;
}
.nav-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--apple-blue); display: inline-block; }
.nav-status { font-size: 13px; color: var(--apple-muted); }

/* === HERO === */
.hero-section {
    text-align: center; padding: 64px 24px 36px;
    max-width: 760px; margin: 0 auto;
}
.hero-eyebrow {
    font-size: 12px; font-weight: 600; letter-spacing: .06em;
    color: var(--apple-blue); text-transform: uppercase; margin-bottom: 12px;
}
.hero-title {
    font-size: clamp(32px, 5vw, 52px); font-weight: 700;
    line-height: 1.06; letter-spacing: -.03em;
    color: var(--apple-ink); margin-bottom: 14px;
}
.hero-sub {
    font-size: 18px; line-height: 1.5; letter-spacing: -.01em;
    color: var(--apple-muted);
}

/* === SUGGESTION CHIPS === */
.suggestions {
    max-width: 860px; margin: 0 auto 24px;
    display: flex; gap: 10px; flex-wrap: wrap; padding: 0 16px;
    justify-content: center;
}
.chip {
    background: var(--apple-white);
    border: 1px solid var(--apple-border);
    border-radius: 999px; padding: 8px 18px;
    font-size: 13px; color: var(--apple-ink);
    cursor: pointer; font-family: var(--font-sans);
    transition: background .15s, border-color .15s, box-shadow .15s;
    text-decoration: none; display: inline-block;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.chip:hover {
    background: var(--apple-gray-050);
    border-color: var(--apple-blue);
    box-shadow: 0 2px 8px rgba(0,113,227,.12);
    color: var(--apple-blue);
}

/* === CHAT CARD === */
.chat-card-wrap {
    max-width: 860px; margin: 0 auto 24px; padding: 0 16px;
}
.chat-card {
    background: var(--apple-white); border-radius: 24px;
    box-shadow: var(--shadow-card); overflow: hidden;
}
.chat-card-header {
    background: var(--apple-gray-050);
    border-bottom: 1px solid var(--apple-border);
    padding: 14px 24px; display: flex; align-items: center; gap: 8px;
}
.hd { width: 13px; height: 13px; border-radius: 50%; display: inline-block; }
.hd-r { background: #FF5F57; }
.hd-a { background: #FFBD2E; }
.hd-g { background: #28C840; }
.chat-header-title { font-size: 13px; font-weight: 500; color: var(--apple-muted); margin-left: 4px; }

/* === MESSAGES === */
.chat-messages {
    padding: 24px 28px;
    min-height: 360px; max-height: 500px;
    overflow-y: auto; scroll-behavior: smooth;
    display: flex; flex-direction: column; gap: 14px;
}
.chat-messages::-webkit-scrollbar { width: 5px; }
.chat-messages::-webkit-scrollbar-track { background: transparent; }
.chat-messages::-webkit-scrollbar-thumb { background: var(--apple-border); border-radius: 99px; }

.msg-row { display: flex; align-items: flex-end; gap: 10px; }
.msg-row.user { flex-direction: row-reverse; }
.msg-row.bot  { flex-direction: row; }

.msg-bubble {
    max-width: 72%; font-size: 15px; line-height: 1.6;
    padding: 12px 18px; border-radius: 20px;
    word-break: break-word;
}
.msg-row.user .msg-bubble {
    background: var(--apple-blue);
    color: var(--apple-white);
    border-radius: 20px 20px 6px 20px;
    box-shadow: 0 2px 8px rgba(0,113,227,.22);
}
.msg-row.bot .msg-bubble {
    background: var(--apple-gray-050);
    color: var(--apple-ink);
    border-radius: 20px 20px 20px 6px;
    border: 1px solid var(--apple-border);
    max-width: 82%;
}

/* === INPUT BAR === */
.input-bar {
    border-top: 1px solid var(--apple-border);
    padding: 14px 20px; display: flex; align-items: flex-end; gap: 10px;
    background: var(--apple-white);
}

/* === SOURCE / STATUS CARD === */
.source-card-wrap { max-width: 860px; margin: 0 auto 32px; padding: 0 16px; }
.source-card {
    background: var(--apple-white); border-radius: 16px;
    border: 1px solid var(--apple-border); padding: 16px 22px;
    box-shadow: var(--shadow-soft);
}
.source-label {
    font-size: 11px; font-weight: 600; letter-spacing: .05em;
    color: var(--apple-muted); text-transform: uppercase; margin-bottom: 8px;
}
.source-text {
    font-family: var(--font-mono); font-size: 13px;
    line-height: 1.6; color: var(--apple-ink);
    background: var(--apple-gray-050);
    border: 1px solid var(--apple-border); border-radius: 10px;
    padding: 10px 14px;
}

/* === FOOTER === */
.footer {
    text-align: center; padding: 24px 24px 40px;
    font-size: 13px; color: var(--apple-muted);
    border-top: 1px solid var(--apple-border);
    margin-top: 8px;
}

/* === Streamlit widget cleanup === */
div[data-testid="stChatMessage"] { display: none; }
div[data-testid="stChatInput"]   { display: none; }
div[data-testid="stButton"] button {
    border-radius: 999px !important;
    font-family: var(--font-sans) !important;
    font-size: 15px !important; font-weight: 500 !important;
    transition: all .18s ease !important;
}
div[data-testid="stTextInput"] > div > div > input {
    border-radius: 14px !important;
    font-family: var(--font-sans) !important;
    font-size: 16px !important;
    border: 1px solid var(--apple-border) !important;
    background: var(--apple-gray-050) !important;
    color: var(--apple-ink) !important;
    padding: 12px 16px !important;
}
div[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--apple-blue) !important;
    background: var(--apple-white) !important;
    box-shadow: var(--shadow-focus) !important;
}
/* Hide label for text input */
div[data-testid="stTextInput"] label { display: none !important; }
</style>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def render_messages(messages):
    """Render chat messages as styled HTML bubbles."""
    html = '<div class="chat-messages" id="chat-messages">'
    for m in messages:
        role_class = "user" if m["role"] == "user" else "bot"
        content = m["content"].replace("\n", "<br>")
        html += f"""
        <div class="msg-row {role_class}">
            <div class="msg-bubble">{content}</div>
        </div>"""
    html += "</div>"
    return html

# ---------------------------------------------------------------------------
# State init
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Xin chào! Tôi là FinanceBot. Hãy đặt câu hỏi về báo cáo tài chính và tôi sẽ tìm kiếm câu trả lời chính xác từ tài liệu."}
    ]
if "status" not in st.session_state:
    st.session_state.status = "Sẵn sàng nhận câu hỏi của bạn."
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ---------------------------------------------------------------------------
# Inject CSS
# ---------------------------------------------------------------------------
st.markdown(APPLE_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# NAV BAR
# ---------------------------------------------------------------------------
st.markdown("""
<nav class="nav-bar">
    <span class="nav-logo">
        <span class="nav-dot"></span>FinanceBot
    </span>
    <span class="nav-status">Trợ lý Phân tích Tài chính &middot; Gemini Flash</span>
</nav>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-section">
    <p class="hero-eyebrow">RAG &middot; Retrieval-Augmented Generation</p>
    <h1 class="hero-title">Phân tích báo cáo tài chính<br>một cách thông minh.</h1>
    <p class="hero-sub">
        Đặt câu hỏi bằng Tiếng Việt — FinanceBot sẽ tra cứu tài liệu,
        trích dẫn nguồn chính xác và trả lời tức thì.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SUGGESTION CHIPS — set pending_query via st.button
# ---------------------------------------------------------------------------
SUGGESTIONS = [
    "Chi phí R&D của 3M năm 2015?",
    "Doanh thu thuần năm gần nhất?",
    "Lợi nhuận gộp và biên lợi nhuận gộp?",
]

cols = st.columns([1, 3, 3, 3, 1])
for i, s in enumerate(SUGGESTIONS):
    with cols[i + 1]:
        if st.button(s, key=f"chip_{i}", use_container_width=True):
            st.session_state.pending_query = s

# ---------------------------------------------------------------------------
# CHAT CARD
# ---------------------------------------------------------------------------
st.markdown("""
<div class="chat-card-wrap">
  <div class="chat-card">
    <div class="chat-card-header">
      <span class="hd hd-r"></span>
      <span class="hd hd-a"></span>
      <span class="hd hd-g"></span>
      <span class="chat-header-title">FinanceBot — Cuộc trò chuyện</span>
    </div>
""", unsafe_allow_html=True)

# Render existing messages
st.markdown(render_messages(st.session_state.messages), unsafe_allow_html=True)

# Auto-scroll JS
st.markdown("""
<script>
    const el = document.getElementById('chat-messages');
    if(el) el.scrollTop = el.scrollHeight;
</script>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# INPUT BAR (inside chat card)
# ---------------------------------------------------------------------------
st.markdown('<div class="input-bar">', unsafe_allow_html=True)
in_col1, in_col2, in_col3 = st.columns([8, 1, 1])

with in_col1:
    user_input = st.text_input(
        label="query",
        placeholder="Hỏi về báo cáo tài chính... (Enter để gửi)",
        key="query_input",
        label_visibility="collapsed",
    )
with in_col2:
    send = st.button("Gửi", key="send_btn", use_container_width=True, type="primary")
with in_col3:
    clear = st.button("Xóa", key="clear_btn", use_container_width=True)

st.markdown("</div></div></div>", unsafe_allow_html=True)  # close input-bar + chat-card + chat-card-wrap

# ---------------------------------------------------------------------------
# STATUS / SOURCE CARD
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="source-card-wrap">
  <div class="source-card">
    <p class="source-label">Trạng thái</p>
    <div class="source-text" id="status-box">{st.session_state.status}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("""
<footer class="footer">
    Copyright &copy; 2025 FinanceBot &nbsp;&middot;&nbsp;
    Powered by Gemini Flash &amp; LangChain &nbsp;&middot;&nbsp;
    Dữ liệu chỉ từ tài liệu đã nạp vào hệ thống
</footer>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# LOGIC — xử lý query
# ---------------------------------------------------------------------------
query = None

# Từ suggestion chip
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

# Từ input bar
elif send and user_input and user_input.strip():
    query = user_input.strip()

elif user_input and user_input.strip() and not send:
    # Enter pressed (st.text_input submits on Enter)
    query = user_input.strip()

if query:
    # Thêm câu hỏi user vào history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.status = "Đang tìm kiếm trong tài liệu..."
    st.rerun()

# Nếu message cuối là user → cần sinh câu trả lời
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_query = st.session_state.messages[-1]["content"]
    accumulated = ""
    st.session_state.status = "Đang trả lời..."

    try:
        for chunk in answer_query(last_query):
            accumulated += chunk
        st.session_state.messages.append({"role": "bot", "content": accumulated})
        st.session_state.status = "Hoàn thành. Sẵn sàng cho câu hỏi tiếp theo."
    except Exception as e:
        st.session_state.messages.append({"role": "bot", "content": f"⚠️ Lỗi: {e}"})
        st.session_state.status = f"Lỗi: {e}"

    st.rerun()

# Xóa chat
if clear:
    st.session_state.messages = [
        {"role": "bot", "content": "Cuộc trò chuyện đã được xóa. Hãy đặt câu hỏi mới!"}
    ]
    st.session_state.status = "Sẵn sàng nhận câu hỏi của bạn."
    st.rerun()
