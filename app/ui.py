"""
ui.py - Giao dien chatbot RAG theo phong cach Apple Design System.

Chay:
    python app/ui.py
"""
import sys, os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr
from src.generation.generator import answer_query

# ---------------------------------------------------------------------------
# Apple Design System CSS
# ---------------------------------------------------------------------------
APPLE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

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
    --font-sans: "SF Pro Display","SF Pro Text",Inter,-apple-system,"Helvetica Neue",Arial,sans-serif;
    --font-mono: "SF Mono",Menlo,Monaco,monospace;
}

body, .gradio-container {
    background: var(--apple-soft) !important;
    font-family: var(--font-sans) !important;
    color: var(--apple-ink) !important;
    margin: 0 !important; padding: 0 !important;
}
.gradio-container { max-width: 100% !important; }

#nav-bar {
    position: sticky; top: 0; z-index: 100;
    background: rgba(255,255,255,.82);
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    border-bottom: 1px solid var(--apple-border);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 48px; height: 52px;
}
#nav-logo {
    font-size: 22px; font-weight: 600; letter-spacing: -.02em;
    color: var(--apple-ink); display: flex; align-items: center; gap: 10px;
}
#nav-logo .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--apple-blue); display: inline-block; }
#nav-status { font-size: 13px; color: var(--apple-muted); }

#hero-section { text-align: center; padding: 72px 24px 40px; max-width: 760px; margin: 0 auto; }
#hero-eyebrow { font-size: 12px; font-weight: 600; letter-spacing: .06em; color: var(--apple-blue); text-transform: uppercase; margin-bottom: 16px; }
#hero-title { font-size: clamp(34px,5vw,56px); font-weight: 600; line-height: 1.06; letter-spacing: -.03em; color: var(--apple-ink); margin-bottom: 16px; }
#hero-sub { font-size: 19px; line-height: 1.47; letter-spacing: -.01em; color: var(--apple-muted); }

#chat-card {
    background: var(--apple-white); border-radius: 28px;
    box-shadow: var(--shadow-card); overflow: hidden;
    max-width: 860px; margin: 0 auto 32px;
}
#chat-header {
    background: var(--apple-gray-050); border-bottom: 1px solid var(--apple-border);
    padding: 18px 28px; display: flex; align-items: center; gap: 8px;
}
.hd { width: 13px; height: 13px; border-radius: 50%; display: inline-block; }
.hd-r { background: #FF5F57; }
.hd-a { background: #FFBD2E; }
.hd-g { background: #28C840; }
#chat-header-title { font-size: 14px; font-weight: 500; color: var(--apple-muted); margin-left: 6px; }

#chatbot-window {
    background: var(--apple-white) !important;
    border: none !important;
    min-height: 420px; max-height: 520px;
    padding: 24px 28px !important;
    overflow-y: auto; scroll-behavior: smooth;
}
#chatbot-window .message.user {
    background: var(--apple-blue) !important;
    color: var(--apple-white) !important;
    border-radius: 20px 20px 6px 20px !important;
    padding: 12px 18px !important; max-width: 72% !important;
    font-size: 15px !important; line-height: 1.5 !important;
    box-shadow: 0 2px 8px rgba(0,113,227,.22) !important;
    margin-left: auto !important;
}
#chatbot-window .message.bot {
    background: var(--apple-gray-050) !important;
    color: var(--apple-ink) !important;
    border-radius: 20px 20px 20px 6px !important;
    padding: 14px 18px !important; max-width: 82% !important;
    font-size: 15px !important; line-height: 1.6 !important;
    border: 1px solid var(--apple-border) !important;
}
#chatbot-window .message-wrap .avatar-container { display: none !important; }

#input-bar {
    background: var(--apple-white); border-top: 1px solid var(--apple-border);
    padding: 16px 24px; display: flex; align-items: flex-end; gap: 12px;
}
#query-input textarea {
    font-family: var(--font-sans) !important;
    font-size: 16px !important; line-height: 1.5 !important;
    color: var(--apple-ink) !important; background: var(--apple-gray-050) !important;
    border: 1px solid var(--apple-border) !important; border-radius: 14px !important;
    padding: 12px 16px !important; resize: none !important;
    min-height: 48px !important; max-height: 140px !important;
    transition: border-color .18s ease, box-shadow .18s ease;
    box-shadow: none !important;
}
#query-input textarea:focus {
    border-color: var(--apple-blue) !important;
    box-shadow: var(--shadow-focus) !important;
    background: var(--apple-white) !important; outline: none !important;
}
#query-input textarea::placeholder { color: var(--apple-muted) !important; }
#query-input label, #query-input .label-wrap { display: none !important; }
#query-input { flex: 1; }

#send-btn, #clear-btn {
    border-radius: 999px !important; min-height: 44px !important;
    padding: 0 22px !important; font-size: 15px !important;
    font-weight: 500 !important; font-family: var(--font-sans) !important;
    letter-spacing: -.01em !important; cursor: pointer !important;
    transition: background .18s ease, transform .12s ease, box-shadow .18s ease !important;
    border: none !important;
}
#send-btn {
    background: var(--apple-blue) !important; color: var(--apple-white) !important;
    box-shadow: 0 2px 10px rgba(0,113,227,.30) !important;
}
#send-btn:hover {
    background: var(--apple-blue-deep) !important; transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0,113,227,.36) !important;
}
#send-btn:active { transform: translateY(0) !important; }
#clear-btn {
    background: transparent !important; color: var(--apple-blue) !important;
    border: 1.5px solid var(--apple-border) !important;
}
#clear-btn:hover {
    border-color: var(--apple-blue) !important;
    background: rgba(0,113,227,.05) !important;
}

#source-card {
    background: var(--apple-white); border-radius: 20px;
    border: 1px solid var(--apple-border); padding: 20px 24px;
    max-width: 860px; margin: 0 auto 40px; box-shadow: var(--shadow-soft);
}
#source-output {
    font-family: var(--font-mono) !important; font-size: 13px !important;
    line-height: 1.6 !important; color: var(--apple-ink) !important;
    background: var(--apple-gray-050) !important;
    border: 1px solid var(--apple-border) !important; border-radius: 10px !important;
    padding: 12px 14px !important; min-height: 56px;
}
#source-card > .wrap > .label-wrap { display: none !important; }

#footer {
    text-align: center; padding: 28px 24px 48px; font-size: 13px;
    color: var(--apple-muted); border-top: 1px solid var(--apple-border);
}

#chatbot-window::-webkit-scrollbar { width: 5px; }
#chatbot-window::-webkit-scrollbar-track { background: transparent; }
#chatbot-window::-webkit-scrollbar-thumb { background: var(--apple-border); border-radius: 99px; }
#chatbot-window::-webkit-scrollbar-thumb:hover { background: var(--apple-muted); }

@media (max-width: 734px) {
    #nav-bar { padding: 0 20px; }
    #hero-section { padding: 48px 20px 28px; }
    #chat-card, #source-card { border-radius: 18px; margin-left: 12px; margin-right: 12px; }
    #chatbot-window { padding: 16px !important; }
    #input-bar { padding: 12px 14px; gap: 8px; }
}
"""

NAV_HTML = """
<nav id="nav-bar">
    <span id="nav-logo"><span class="dot"></span>FinanceBot</span>
    <span id="nav-status">Tro ly Phan tich Tai chinh &middot; Gemini 1.5 Flash</span>
</nav>"""

HERO_HTML = """
<div id="hero-section">
    <p id="hero-eyebrow">RAG &middot; Retrieval-Augmented Generation</p>
    <h1 id="hero-title">Phan tich bao cao tai chinh<br>mot cach thong minh.</h1>
    <p id="hero-sub">
        Dat cau hoi bang Tieng Viet &mdash; FinanceBot se tra cuu tai lieu,
        trich dan nguon chinh xac va tra loi tuc thi.
    </p>
</div>"""

HEADER_HTML = """
<div id="chat-header">
    <span class="hd hd-r"></span>
    <span class="hd hd-a"></span>
    <span class="hd hd-g"></span>
    <span id="chat-header-title">FinanceBot &mdash; Cuoc tro chuyen moi</span>
</div>"""

SUGGEST_HTML = """
<div style="max-width:860px;margin:0 auto 32px;display:flex;gap:10px;flex-wrap:wrap;padding:0 4px;">
    <button onclick="(function(){var t=document.querySelector('#query-input textarea');if(t){t.value='Chi phi nghien cuu va phat trien cua 3M nam 2015 la bao nhieu?';t.dispatchEvent(new Event('input',{bubbles:true}));}})()"
        style="background:#F5F5F7;border:1px solid #D2D2D7;border-radius:999px;padding:8px 18px;font-size:14px;color:#1D1D1F;cursor:pointer;font-family:inherit;transition:background .15s;">
        Chi phi R&amp;D 3M 2015
    </button>
    <button onclick="(function(){var t=document.querySelector('#query-input textarea');if(t){t.value='Doanh thu thuan cua cong ty trong nam gan nhat?';t.dispatchEvent(new Event('input',{bubbles:true}));}})()"
        style="background:#F5F5F7;border:1px solid #D2D2D7;border-radius:999px;padding:8px 18px;font-size:14px;color:#1D1D1F;cursor:pointer;font-family:inherit;transition:background .15s;">
        Doanh thu thuan
    </button>
    <button onclick="(function(){var t=document.querySelector('#query-input textarea');if(t){t.value='Loi nhuan gop va bien loi nhuan gop la bao nhieu?';t.dispatchEvent(new Event('input',{bubbles:true}));}})()"
        style="background:#F5F5F7;border:1px solid #D2D2D7;border-radius:999px;padding:8px 18px;font-size:14px;color:#1D1D1F;cursor:pointer;font-family:inherit;transition:background .15s;">
        Loi nhuan gop
    </button>
</div>"""

FOOTER_HTML = """
<footer id="footer">
    Copyright &copy; 2025 FinanceBot &nbsp;&middot;&nbsp;
    Powered by Gemini 1.5 Flash &amp; LangChain &nbsp;&middot;&nbsp;
    Du lieu chi tu tai lieu da nap vao he thong
</footer>"""


def build_ui():
    with gr.Blocks(
        css=APPLE_CSS,
        title="FinanceBot - Tro ly Phan tich Tai chinh",
        theme=gr.themes.Base(font=["Inter", "Helvetica Neue", "Arial"]),
    ) as demo:

        gr.HTML(NAV_HTML)
        gr.HTML(HERO_HTML)

        with gr.Column(elem_id="chat-card"):
            gr.HTML(HEADER_HTML)
            chatbot = gr.Chatbot(
                label="",
                elem_id="chatbot-window",
                bubble_full_width=False,
                show_label=False,
                height=480,
                avatar_images=(None, None),
            )
            with gr.Row(elem_id="input-bar"):
                query_input = gr.Textbox(
                    placeholder="Hoi ve bao cao tai chinh... (Enter de gui)",
                    lines=1, max_lines=5,
                    show_label=False, elem_id="query-input", scale=7,
                )
                send_btn = gr.Button(
                    "Gui", elem_id="send-btn", scale=1, variant="primary"
                )
                clear_btn = gr.Button(
                    "Xoa", elem_id="clear-btn", scale=1, variant="secondary"
                )

        with gr.Column(elem_id="source-card"):
            gr.Markdown("**TRANG THAI**")
            source_output = gr.Textbox(
                value="San sang nhan cau hoi cua ban.",
                label="", interactive=False,
                elem_id="source-output", lines=2, max_lines=6,
            )

        gr.HTML(SUGGEST_HTML)
        gr.HTML(FOOTER_HTML)

        # ---- Event handlers ----
        def on_submit(query, history):
            if not query or not query.strip():
                yield history or [], "", "San sang."
                return
            full_history = (history or []) + [[query.strip(), ""]]
            yield full_history, "", "Dang tim kiem trong tai lieu..."
            accumulated = ""
            try:
                for chunk in answer_query(query.strip()):
                    accumulated += chunk
                    full_history[-1][1] = accumulated
                    yield full_history, "", "Dang tra loi..."
            except Exception as exc:
                full_history[-1][1] = f"Loi: {exc}"
                yield full_history, "", f"Loi: {exc}"
                return
            yield full_history, "", "Hoan thanh. San sang cho cau hoi tiep theo."

        send_btn.click(
            fn=on_submit,
            inputs=[query_input, chatbot],
            outputs=[chatbot, query_input, source_output],
        )
        query_input.submit(
            fn=on_submit,
            inputs=[query_input, chatbot],
            outputs=[chatbot, query_input, source_output],
        )
        clear_btn.click(
            fn=lambda: ([], "", "San sang nhan cau hoi cua ban."),
            inputs=None,
            outputs=[chatbot, query_input, source_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
    )
