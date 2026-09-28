"""
app.py - FinanceBot RAG Chatbot voi Streaming thuc su.
- Tung token duoc stream ra ngay khi nhan duoc tu LLM.
- Hieu ung "dang suy nghi" voi animated dots.
- Dark-mode premium UI voi glassmorphism.

Deploy: Streamlit Cloud -> Main file path: app.py
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
    page_title="FinanceBot - Tro ly Phan tich Tai chinh",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Premium Dark-mode CSS
# ---------------------------------------------------------------------------
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-base:       #0A0A0F;
    --bg-surface:    #13131A;
    --bg-card:       #1A1A24;
    --bg-input:      #1E1E2A;
    --accent:        #6366F1;
    --accent-glow:   rgba(99,102,241,.35);
    --accent-light:  #818CF8;
    --text-primary:  #F1F1F3;
    --text-secondary:#A0A0B0;
    --text-muted:    #60607A;
    --border:        rgba(255,255,255,.08);
    --shadow-card:   0 20px 60px rgba(0,0,0,.5);
    --font-sans: "Inter", -apple-system, "Helvetica Neue", Arial, sans-serif;
    --font-mono: "SF Mono", Menlo, Monaco, "Courier New", monospace;
}

html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
}
.stApp { background: var(--bg-base) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

.fin-nav {
    position: sticky; top: 0; z-index: 200;
    background: rgba(10,10,15,.85);
    backdrop-filter: saturate(180%) blur(24px);
    -webkit-backdrop-filter: saturate(180%) blur(24px);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 48px; height: 56px;
}
.fin-nav-logo {
    font-size: 18px; font-weight: 700; letter-spacing: -.02em;
    color: var(--text-primary); display: flex; align-items: center; gap: 10px;
}
.fin-nav-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 10px var(--accent-glow);
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%,100% { box-shadow: 0 0 6px var(--accent-glow); }
    50%      { box-shadow: 0 0 18px var(--accent-glow), 0 0 30px var(--accent-glow); }
}
.fin-nav-badge {
    font-size: 11px; font-weight: 600; letter-spacing: .04em;
    color: var(--accent-light);
    background: rgba(99,102,241,.12);
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 999px; padding: 3px 10px;
}

.fin-hero {
    text-align: center; padding: 60px 24px 32px;
    max-width: 800px; margin: 0 auto;
}
.fin-hero-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; font-weight: 700; letter-spacing: .08em;
    color: var(--accent-light); text-transform: uppercase;
    background: rgba(99,102,241,.1);
    border: 1px solid rgba(99,102,241,.2);
    border-radius: 999px; padding: 5px 14px; margin-bottom: 20px;
}
.fin-hero-title {
    font-size: clamp(28px,4.5vw,52px);
    font-weight: 700; line-height: 1.08; letter-spacing: -.03em;
    color: var(--text-primary); margin-bottom: 14px;
}
.fin-hero-sub {
    font-size: 17px; line-height: 1.6; color: var(--text-secondary);
    max-width: 560px; margin: 0 auto;
}

div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-family: var(--font-sans) !important;
    padding: 8px 18px !important;
    transition: all .2s !important;
    width: auto !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover {
    border-color: rgba(99,102,241,.5) !important;
    color: var(--accent-light) !important;
    background: rgba(99,102,241,.08) !important;
}

.fin-chat-wrap { max-width: 880px; margin: 0 auto 20px; padding: 0 16px; }
.fin-chat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 24px;
    box-shadow: var(--shadow-card);
    overflow: hidden;
}
.fin-chat-header {
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex; align-items: center; gap: 8px;
}
.fin-dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
.fin-dot-r { background: #FF5F57; }
.fin-dot-a { background: #FFBD2E; }
.fin-dot-g { background: #28C840; }
.fin-chat-title { font-size: 13px; font-weight: 500; color: var(--text-muted); margin-left: 6px; }

.fin-messages {
    padding: 24px 28px;
    min-height: 380px; max-height: 520px;
    overflow-y: auto; scroll-behavior: smooth;
    display: flex; flex-direction: column; gap: 16px;
}
.fin-messages::-webkit-scrollbar { width: 4px; }
.fin-messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

.fin-msg-row { display: flex; align-items: flex-end; gap: 10px; }
.fin-msg-row.user { flex-direction: row-reverse; }
.fin-msg-row.bot  { flex-direction: row; }

.fin-bubble {
    font-size: 15px; line-height: 1.65;
    padding: 12px 18px; word-break: break-word;
}
.fin-msg-row.user .fin-bubble {
    max-width: 75%;
    background: linear-gradient(135deg, #6366F1, #4F46E5);
    color: #FFFFFF;
    border-radius: 20px 20px 5px 20px;
    box-shadow: 0 4px 20px rgba(99,102,241,.35);
}
.fin-msg-row.bot .fin-bubble {
    max-width: 82%;
    background: var(--bg-surface);
    color: var(--text-primary);
    border-radius: 20px 20px 20px 5px;
    border: 1px solid var(--border);
}

.fin-thinking {
    display: flex; align-items: center; gap: 6px;
    padding: 12px 18px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 20px 20px 20px 5px;
}
.fin-thinking-label {
    font-size: 13px; color: var(--text-muted); font-style: italic;
    margin-right: 4px;
}
.fin-thinking-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent);
    animation: think-bounce 1.3s ease-in-out infinite;
}
.fin-thinking-dot:nth-child(2) { animation-delay: .15s; }
.fin-thinking-dot:nth-child(3) { animation-delay: .30s; }
.fin-thinking-dot:nth-child(4) { animation-delay: .45s; }
@keyframes think-bounce {
    0%,60%,100% { transform: translateY(0); opacity: .4; }
    30%          { transform: translateY(-6px); opacity: 1; }
}

.fin-cursor {
    display: inline-block; width: 2px; height: 16px;
    background: var(--accent-light); border-radius: 1px;
    margin-left: 2px; vertical-align: middle;
    animation: blink-cur .7s ease-in-out infinite;
}
@keyframes blink-cur {
    0%,100% { opacity: 1; } 50% { opacity: 0; }
}

div[data-testid="stChatInput"] {
    max-width: 880px !important;
    margin: 0 auto !important;
    padding: 0 16px !important;
}
div[data-testid="stChatInput"] textarea {
    border-radius: 16px !important;
    font-family: var(--font-sans) !important;
    font-size: 15px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-input) !important;
    color: var(--text-primary) !important;
    box-shadow: none !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.2) !important;
}
div[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}

.fin-status-wrap { max-width: 880px; margin: 12px auto 28px; padding: 0 16px; }
.fin-status {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 12px 18px;
    display: flex; align-items: center; gap: 10px;
    box-shadow: 0 4px 20px rgba(0,0,0,.2);
}
.fin-status-icon { font-size: 14px; }
.fin-status-text { font-size: 13px; color: var(--text-secondary); font-family: var(--font-mono); }

.fin-footer {
    text-align: center; padding: 24px 24px 40px;
    font-size: 12px; color: var(--text-muted);
    border-top: 1px solid var(--border); margin-top: 12px;
    letter-spacing: .02em;
}
</style>
"""

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Xin chao! Toi la **FinanceBot**. Hay dat cau hoi ve bao cao tai chinh."}
    ]
if "status" not in st.session_state:
    st.session_state.status = ("✅", "San sang nhan cau hoi cua ban.")
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ---------------------------------------------------------------------------
# Render CSS + NAV
# ---------------------------------------------------------------------------
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

st.markdown("""
<nav class="fin-nav">
    <span class="fin-nav-logo">
        <span class="fin-nav-dot"></span>FinanceBot
    </span>
    <span class="fin-nav-badge">RAG · Gemini Flash</span>
</nav>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="fin-hero">
    <div class="fin-hero-pill">📊 Retrieval-Augmented Generation</div>
    <h1 class="fin-hero-title">Phan tich bao cao tai chinh<br>mot cach thong minh.</h1>
    <p class="fin-hero-sub">Dat cau hoi bang Tieng Viet — FinanceBot tra cuu tai lieu,
    trich dan nguon chinh xac va tra loi ngay theo thoi gian thuc.</p>
</div>
""", unsafe_allow_html=True)

# SUGGESTION CHIPS
SUGGESTIONS = [
    "Chi phi R&D cua 3M nam 2015?",
    "Doanh thu thuan nam gan nhat?",
    "Loi nhuan gop va bien loi nhuan?",
]
cols = st.columns([1, 3, 3, 3, 1])
for i, s in enumerate(SUGGESTIONS):
    with cols[i + 1]:
        if st.button(s, key=f"chip_{i}"):
            st.session_state.pending_query = s
            st.rerun()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# CHAT CARD header
st.markdown("""
<div class="fin-chat-wrap">
  <div class="fin-chat-card">
    <div class="fin-chat-header">
      <span class="fin-dot fin-dot-r"></span>
      <span class="fin-dot fin-dot-a"></span>
      <span class="fin-dot fin-dot-g"></span>
      <span class="fin-chat-title">FinanceBot — Cuoc tro chuyen</span>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MESSAGES + REAL STREAMING
# ---------------------------------------------------------------------------
def render_bubble(role: str, content: str) -> str:
    content_html = content.replace("\n", "<br>")
    return (
        f'<div class="fin-msg-row {role}">'
        f'<div class="fin-bubble">{content_html}</div>'
        f'</div>'
    )

messages_placeholder = st.empty()

def render_all_messages(extra_html: str = ""):
    html = '<div class="fin-messages" id="fin-messages">'
    for m in st.session_state.messages:
        html += render_bubble(m["role"], m["content"])
    html += extra_html
    html += "</div>"
    messages_placeholder.markdown(html, unsafe_allow_html=True)

THINKING_HTML = """
<div class="fin-msg-row bot">
  <div class="fin-thinking">
    <span class="fin-thinking-label">Dang suy nghi</span>
    <span class="fin-thinking-dot"></span>
    <span class="fin-thinking-dot"></span>
    <span class="fin-thinking-dot"></span>
    <span class="fin-thinking-dot"></span>
  </div>
</div>"""

if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

    # Them user message va hien thi thinking
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.status = ("🔍", "Dang tim kiem trong tai lieu...")
    render_all_messages(extra_html=THINKING_HTML)

    # STREAMING: moi chunk tu LLM hien thi ngay lap tuc
    accumulated = ""
    try:
        st.session_state.status = ("⚡", "Dang tra loi theo thoi gian thuc...")
        for chunk in answer_query(query):
            accumulated += chunk
            # Hien thi noi dung dang stream + cursor nhay
            streaming_html = (
                '<div class="fin-msg-row bot">'
                '<div class="fin-bubble">'
                + accumulated.replace("\n", "<br>")
                + '<span class="fin-cursor"></span>'
                '</div></div>'
            )
            render_all_messages(extra_html=streaming_html)

        # Luu vao messages, bo cursor
        st.session_state.messages.append({"role": "bot", "content": accumulated})
        st.session_state.status = ("✅", "Hoan thanh. San sang cho cau hoi tiep theo.")

    except Exception as e:
        err_msg = f"Loi: {e}"
        st.session_state.messages.append({"role": "bot", "content": err_msg})
        st.session_state.status = ("❌", f"Loi: {e}")

    render_all_messages()

else:
    render_all_messages()

st.markdown("</div></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Hoi ve bao cao tai chinh...", key="main_input"):
    st.session_state.pending_query = prompt
    st.rerun()

# ---------------------------------------------------------------------------
# STATUS BAR
# ---------------------------------------------------------------------------
icon, text = st.session_state.status
st.markdown(f"""
<div class="fin-status-wrap">
  <div class="fin-status">
    <span class="fin-status-icon">{icon}</span>
    <span class="fin-status-text">{text}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# XOA CHAT
# ---------------------------------------------------------------------------
col_clear, _ = st.columns([1, 7])
with col_clear:
    if st.button("🗑️ Xoa chat", key="clear_btn"):
        st.session_state.messages = [
            {"role": "bot", "content": "Cuoc tro chuyen da duoc xoa. Hay dat cau hoi moi!"}
        ]
        st.session_state.status = ("✅", "San sang nhan cau hoi cua ban.")
        st.rerun()

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("""
<footer class="fin-footer">
    Copyright &copy; 2025 FinanceBot &nbsp;&middot;&nbsp;
    Powered by Gemini Flash &amp; LangChain &nbsp;&middot;&nbsp;
    Du lieu chi tu tai lieu da nap vao he thong
</footer>
""", unsafe_allow_html=True)