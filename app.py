import os
import sys
import html
import re
import logging
import tempfile
import streamlit as st

# Console logs contain characters such as "→" / "❌". On Windows, when output is redirected
# (log file, non-UTF-8 console) Python uses cp1252 and print() raises UnicodeEncodeError,
# which would abort the pipeline. Force UTF-8 so logging can never break a run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import time
from dotenv import load_dotenv

# Load configuration BEFORE importing core/ — core/transcriber.py reads
# SARVAM_API_KEY / WHISPER_MODEL / SARVAM_STT_MODEL at import time.
# Local: .env   ·   Streamlit Community Cloud: st.secrets (copied into os.environ)
load_dotenv()
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str) and _key not in os.environ:
            os.environ[_key] = _value
except Exception:
    pass  # no secrets.toml (normal for local runs) — .env is used instead

from utils.audio_processor import process_input, cleanup_files, DOWNLOAD_DIR
from utils.exporter import build_txt, build_pdf
from utils.errors import UserFacingError
from utils.supadata import get_transcript_for_source, is_url
from core.llm import describe_llm_error
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

from core.vector_store import warm_up_embeddings

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _start_model_warm_up():
    # Runs once per server process: every analysis needs the embedding model, so load it in the
    # background instead of making the first user wait for it after transcription
    warm_up_embeddings()
    return True

_start_model_warm_up()


# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VAANI: Your AI Video Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
# Light "Vaani" design system. Base colours also live in .streamlit/config.toml (keep in sync).
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #FAF8F5;          /* warm off-white canvas */
    --surface: #FFFFFF;
    --surface-2: #F6F4F0;
    --border: #E6E1D9;
    --text: #1F2330;        /* charcoal */
    --muted: #6B6F7B;
    --brand: #5B4BDB;       /* Vaani indigo-purple: primary actions, branding, chat */
    --brand-dark: #4A3BC4;
    --brand-soft: #EEEBFC;
    --coral: #E8673C;       /* summary */
    --coral-soft: #FDEEE7;
    --amber: #C98A0B;       /* action items */
    --amber-soft: #FDF4DD;
    --green: #1E9A5A;       /* decisions / success */
    --green-soft: #E5F5EC;
    --blue: #2F72C9;        /* open questions */
    --blue-soft: #E8F0FB;
    --teal: #0E8F80;        /* transcript */
    --teal-soft: #E1F4F1;
    --red: #CF3F3F;
    --red-soft: #FBEAEA;
    --radius: 14px;
    --shadow: 0 1px 2px rgba(31,35,48,.04), 0 4px 14px rgba(31,35,48,.05);
}

/* Text font comes from .streamlit/config.toml (theme.font = Inter). No global !important font
   override here: it would also replace the Material icon font and show icon names as text. */
html, body { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stMainBlockContainer"], .block-container { padding-top: 2.5rem; max-width: 1180px; }
h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.01em; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 0.5rem; }
.v-side-brand { display: flex; align-items: center; gap: .65rem; margin: 0 0 .25rem; }
.v-side-tag { color: var(--muted); font-size: .8rem; margin: 0 0 1.1rem 2.9rem; }
.v-section-label {
    font-size: .7rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
    color: var(--muted); margin: .4rem 0 .5rem;
}
.v-or { display: flex; align-items: center; gap: .6rem; color: var(--muted); font-size: .72rem;
        font-weight: 600; text-transform: uppercase; letter-spacing: .08em; margin: .35rem 0 .6rem; }
.v-or::before, .v-or::after { content: ""; flex: 1; height: 1px; background: var(--border); }
.v-side-note { color: var(--muted); font-size: .75rem; line-height: 1.5; margin-top: .35rem; }
.v-side-footer { color: var(--muted); font-size: .72rem; margin-top: 1.4rem; padding-top: .9rem;
                 border-top: 1px solid var(--border); line-height: 1.6; }

/* Logo mark + wordmark */
.v-logo { width: 34px; height: 34px; border-radius: 10px; flex-shrink: 0; display: grid; place-items: center;
          background: linear-gradient(135deg, var(--brand) 0%, #8B5CF6 55%, var(--coral) 100%);
          color: #fff; font-weight: 800; font-size: 1.05rem; box-shadow: 0 3px 10px rgba(91,75,219,.25); }
.v-wordmark { font-weight: 800; font-size: 1.15rem; letter-spacing: .06em; color: var(--text); line-height: 1; }
.v-wordmark .v-colon { color: var(--brand); }
.v-wordmark .v-sub { font-weight: 500; letter-spacing: 0; color: var(--muted); font-size: .98rem; }

/* ── Text input helper ("Press Enter to apply" / "…submit form"): Streamlit positions it absolutely
      INSIDE the input, where it overlapped the URL. Reserve a permanent strip BELOW every text input
      and place the hint there: it never overlaps the field, and because the space is always reserved
      the layout does not jump when the hint appears/disappears (a jump on blur moved the Send button
      between mouse-down and mouse-up, so clicks could be lost). ── */
[data-testid="stTextInput"] { padding-bottom: 18px; }
[data-testid="InputInstructions"] {
    position: absolute !important; top: auto !important; bottom: 0 !important; right: 2px !important; left: auto !important;
    font-size: .72rem !important; line-height: 16px; color: var(--muted) !important;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%;
}

/* ── Inputs ── */
[data-testid="stTextInputRootElement"], [data-baseweb="select"] > div { border-radius: 10px !important; }
[data-testid="stWidgetLabel"] p { font-weight: 600; font-size: .85rem; color: var(--text); }

/* ── File uploader: compact, clearly a drop zone ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--brand-soft); border: 1.5px dashed #C9C1F4; border-radius: 12px; padding: .8rem;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--brand); }
[data-testid="stFileUploaderDropzoneInstructions"] span { color: var(--text); font-weight: 600; }
[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--muted); }

/* ── Buttons ── */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    border-radius: 10px; font-weight: 600; transition: background .15s, box-shadow .15s, transform .15s;
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] {
    background: var(--brand); border: 1px solid var(--brand); color: #fff;
    box-shadow: 0 2px 8px rgba(91,75,219,.28);
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {
    background: var(--brand-dark); border-color: var(--brand-dark); transform: translateY(-1px);
}
.stButton > button[kind="secondary"], .stDownloadButton > button {
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
}
.stDownloadButton > button p, .stFormSubmitButton > button p { white-space: nowrap; }
.stButton > button[kind="secondary"]:hover, .stDownloadButton > button:hover {
    border-color: var(--brand); color: var(--brand); background: var(--surface);
}

/* ── Top bar ── */
.v-topbar { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
            padding-bottom: 1rem; margin-bottom: 1.25rem; border-bottom: 1px solid var(--border); }
.v-brand { display: flex; align-items: center; gap: .7rem; min-width: 0; }
.v-pill { display: inline-flex; align-items: center; gap: .4rem; padding: .3rem .7rem; border-radius: 999px;
          font-size: .78rem; font-weight: 600; white-space: nowrap; }
.v-pill::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.v-pill-idle { background: var(--surface-2); color: var(--muted); border: 1px solid var(--border); }
.v-pill-ready { background: var(--green-soft); color: var(--green); }
.v-pill-busy { background: var(--brand-soft); color: var(--brand); }

/* ── Generic card ── */
.v-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
          box-shadow: var(--shadow); padding: 1.1rem 1.25rem 1rem; margin-bottom: 1rem; position: relative;
          overflow: hidden; }
.v-card::before { content: ""; position: absolute; left: 0; top: 0; right: 0; height: 3px; background: var(--accent); }
.v-card-head { display: flex; align-items: center; gap: .6rem; margin-bottom: .55rem; }
.v-icon { width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center; font-size: .95rem;
          background: var(--accent-soft); flex-shrink: 0; }
.v-card-title { font-weight: 700; font-size: .95rem; color: var(--text); flex: 1; }
.v-count { font-size: .72rem; font-weight: 700; padding: .15rem .55rem; border-radius: 999px;
           background: var(--accent-soft); color: var(--accent); }
.v-card-body { font-size: .9rem; line-height: 1.6; color: var(--text); overflow-wrap: anywhere; }
.v-card-body p { margin: 0 0 .4rem; }
.v-card-body ul, .v-card-body ol { margin: .1rem 0 .3rem; padding-left: 1.2rem; }
.v-card-body li { margin: .18rem 0; }
.v-card-body li::marker { color: var(--accent); font-weight: 700; }
.v-summary   { --accent: var(--coral); --accent-soft: var(--coral-soft); }
/* long summaries scroll inside the card instead of pushing the dashboard down (nothing is cut) */
.v-summary .v-card-body { max-height: 440px; overflow-y: auto; padding-right: .3rem; }
.v-actions   { --accent: var(--amber); --accent-soft: var(--amber-soft); }
.v-decisions { --accent: var(--green); --accent-soft: var(--green-soft); }
.v-questions { --accent: var(--blue);  --accent-soft: var(--blue-soft); }
.v-export    { --accent: var(--brand); --accent-soft: var(--brand-soft); }
.v-chat      { --accent: var(--brand); --accent-soft: var(--brand-soft); }

/* ── Result title card ── */
.v-title-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
                box-shadow: var(--shadow); padding: 1.15rem 1.35rem; margin-bottom: 1rem;
                border-left: 4px solid var(--brand); }
.v-eyebrow { font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--brand); }
.v-title { font-size: 1.45rem; font-weight: 700; line-height: 1.3; margin: .2rem 0 .6rem; color: var(--text);
           overflow-wrap: anywhere; }
.v-meta { display: flex; gap: .45rem; flex-wrap: wrap; }
.v-chip { font-size: .76rem; font-weight: 600; padding: .25rem .6rem; border-radius: 999px;
          background: var(--surface-2); color: var(--muted); border: 1px solid var(--border); }
.v-chip-ok { background: var(--green-soft); color: var(--green); border-color: transparent; }

/* ── Section heading (export / chat) ── */
.v-section-head { display: flex; align-items: center; gap: .65rem; margin: 1.4rem 0 .7rem; }
.v-section-title { font-weight: 700; font-size: 1.05rem; color: var(--text); }
.v-section-sub { font-size: .8rem; color: var(--muted); }

/* ── Transcript expander ── */
[data-testid="stExpander"] details { background: var(--surface); border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; box-shadow: var(--shadow); border-left: 4px solid var(--teal) !important; }
[data-testid="stExpander"] summary { font-weight: 600; }
[data-testid="stExpander"] summary:hover { color: var(--teal); }
.v-transcript { background: var(--teal-soft); border-radius: 10px; padding: 1rem 1.1rem; font-size: .88rem;
                line-height: 1.75; color: #24443F; max-height: 340px; overflow-y: auto; white-space: pre-wrap;
                overflow-wrap: anywhere; }

/* ── Progress stepper ── */
.v-progress { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
              box-shadow: var(--shadow); padding: 1.1rem 1.25rem; margin-bottom: 1rem; }
.v-progress-title { font-weight: 700; font-size: .95rem; margin-bottom: .9rem; }
.v-steps { display: flex; gap: .35rem; flex-wrap: wrap; }
.v-step { flex: 1 1 110px; display: flex; flex-direction: column; align-items: center; text-align: center; gap: .4rem;
          position: relative; min-width: 96px; }
.v-step:not(:last-child)::after { content: ""; position: absolute; top: 15px; left: calc(50% + 20px); right: calc(-50% + 20px);
          height: 2px; background: var(--border); }
.v-step.done:not(:last-child)::after { background: var(--green); }
.v-dot { width: 32px; height: 32px; border-radius: 50%; display: grid; place-items: center; font-size: .8rem; font-weight: 700;
         background: var(--surface-2); color: var(--muted); border: 2px solid var(--border); z-index: 1; }
.v-step.done .v-dot { background: var(--green); border-color: var(--green); color: #fff; }
.v-step.active .v-dot { background: var(--brand); border-color: var(--brand); color: #fff;
                        box-shadow: 0 0 0 5px var(--brand-soft); animation: v-pulse 1.4s ease-in-out infinite; }
.v-step.failed .v-dot { background: var(--red); border-color: var(--red); color: #fff; }
.v-step-label { font-size: .78rem; font-weight: 600; color: var(--muted); line-height: 1.3; }
.v-step.done .v-step-label { color: var(--green); }
.v-step.active .v-step-label { color: var(--brand); }
.v-step.failed .v-step-label { color: var(--red); }
.v-progress-detail { margin-top: .9rem; font-size: .84rem; color: var(--muted); }
@keyframes v-pulse { 0%,100% { box-shadow: 0 0 0 4px var(--brand-soft); } 50% { box-shadow: 0 0 0 8px var(--brand-soft); } }

/* ── Chat ── */
.v-chat-box { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
              box-shadow: var(--shadow); padding: 1rem; margin-bottom: .75rem; max-height: 460px; overflow-y: auto;
              /* column-reverse + a single inner wrapper keeps the scroll anchored at the NEWEST message
                 (otherwise new answers appeared below the visible area once the history was long) */
              display: flex; flex-direction: column-reverse; }
.v-msg { display: flex; gap: .6rem; margin: .55rem 0; align-items: flex-end; }
.v-msg.v-user { justify-content: flex-end; }
.v-bubble { max-width: min(78%, 680px); padding: .65rem .9rem; border-radius: 14px; font-size: .9rem; line-height: 1.55;
            overflow-wrap: anywhere; }
.v-user .v-bubble { background: var(--brand); color: #fff; border-bottom-right-radius: 4px; }
.v-bot .v-bubble { background: var(--surface-2); color: var(--text); border: 1px solid var(--border);
                   border-bottom-left-radius: 4px; }
.v-avatar { width: 28px; height: 28px; border-radius: 9px; flex-shrink: 0; display: grid; place-items: center;
            background: linear-gradient(135deg, var(--brand), #8B5CF6); color: #fff; font-size: .8rem; font-weight: 800; }
.v-role { font-size: .7rem; font-weight: 700; color: var(--muted); margin-bottom: .2rem; }
.v-chat-empty { text-align: center; padding: 1.4rem 1rem; color: var(--muted); font-size: .88rem; }
.v-chat-empty b { color: var(--text); }
.v-hint { display: inline-block; margin: .5rem .2rem 0; padding: .3rem .7rem; border-radius: 999px; font-size: .78rem;
          background: var(--brand-soft); color: var(--brand); font-weight: 600; }

/* ── Empty state ── */
.v-empty { background: var(--surface); border: 1px solid var(--border); border-radius: 18px; box-shadow: var(--shadow);
           padding: 1.8rem 2rem; position: relative; overflow: hidden; }
.v-empty::after { content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px; border-radius: 50%;
                  background: radial-gradient(circle, rgba(91,75,219,.10), rgba(232,103,60,.06) 60%, transparent 70%); }
.v-empty h2 { font-size: 1.35rem; font-weight: 700; margin: .5rem 0 .35rem; }
.v-empty p { color: var(--muted); font-size: .92rem; max-width: 560px; margin: 0 0 1.1rem; line-height: 1.6; }
.v-howto { display: flex; gap: .75rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.v-howto div { flex: 1 1 170px; background: var(--surface-2); border: 1px solid var(--border); border-radius: 12px;
               padding: .7rem .85rem; font-size: .84rem; color: var(--text); }
.v-howto b { display: inline-grid; place-items: center; width: 22px; height: 22px; border-radius: 50%; margin-right: .4rem;
             background: var(--brand); color: #fff; font-size: .72rem; }
.v-features { display: flex; gap: .45rem; flex-wrap: wrap; }
.v-feature { font-size: .76rem; font-weight: 600; padding: .28rem .65rem; border-radius: 999px; }

@media (max-width: 760px) {
    .v-title { font-size: 1.2rem; }
    .v-bubble { max-width: 88%; }
    .v-empty { padding: 1.3rem 1.2rem; }
    .v-step:not(:last-child)::after { display: none; }
}
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
LANGUAGE_LABELS = {"english": "English", "hinglish": "Hinglish (Hindi + English)"}
LOGO = '<div class="v-logo">V</div>'

def count_items(text: str) -> int:
    """Number of list items in an LLM list (0 for 'No … found.')."""
    return len(re.findall(r"^\s*(?:\d+[.)]|[-*•])\s+", text or "", flags=re.MULTILINE))

def render_card(kind: str, icon: str, title: str, content: str, show_count: bool = True):
    # Unindented markup with blank lines around the content: Markdown inside an HTML block is only
    # parsed after a blank line (otherwise lists/bold show as raw text), and indented lines would
    # become code blocks. Content is HTML-escaped; Markdown syntax is unaffected by escaping.
    n = count_items(content) if show_count else None
    badge = f'<span class="v-count">{n if n else "None"}</span>' if show_count else ""
    st.markdown(
        f'<div class="v-card v-{kind}"><div class="v-card-head"><span class="v-icon">{icon}</span>'
        f'<span class="v-card-title">{title}</span>{badge}</div><div class="v-card-body">\n\n'
        f'{html.escape(content or "")}\n\n</div></div>',
        unsafe_allow_html=True,
    )

# Five user-facing stages mapped onto the pipeline's internal steps (logic unchanged)
STAGES = [
    ("Extract audio",        ["audio"]),
    ("Transcribe",           ["transcript"]),
    ("Generate insights",    ["title", "summary", "extract"]),
    ("Build knowledge base", ["rag"]),
    ("Ready to chat",        None),
]

def render_progress(placeholder, steps: dict, detail: str = "", finished: bool = False, error: str = None):
    parts = []
    for i, (label, keys) in enumerate(STAGES, start=1):
        if keys is None:
            state = "done" if finished else "pending"
        else:
            states = [steps.get(k) for k in keys]
            if "failed" in states:
                state = "failed"
            elif all(s == "done" for s in states):
                state = "done"
            elif any(s in ("active", "done") for s in states):
                state = "active"
            else:
                state = "pending"
        mark = {"done": "✓", "failed": "✕"}.get(state, str(i))
        parts.append(f'<div class="v-step {state}"><div class="v-dot">{mark}</div>'
                     f'<div class="v-step-label">{label}</div></div>')
    heading = "Analysis complete" if finished else ("Analysis stopped" if error else "Analysing your video…")
    with placeholder.container():
        st.markdown(
            f'<div class="v-progress"><div class="v-progress-title">{heading}</div>'
            f'<div class="v-steps">{"".join(parts)}</div>'
            + (f'<div class="v-progress-detail">{html.escape(detail)}</div>' if detail else "")
            + '</div>',
            unsafe_allow_html=True,
        )
        if error:
            st.error(f"❌ {error}")

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div class="v-side-brand">{LOGO}<div class="v-wordmark">VAANI</div></div>'
                '<div class="v-side-tag">Video intelligence, simplified.</div>', unsafe_allow_html=True)

    st.markdown('<div class="v-section-label">Input</div>', unsafe_allow_html=True)
    source = st.text_input(
        "YouTube URL",
        placeholder="Paste a YouTube link",
        help="Paste a YouTube video link. When running Vaani on your own computer you can also paste a local file path.",
    )
    st.markdown('<div class="v-or">or</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a video or audio file", help="If a file is uploaded it is used instead of the YouTube link.")

    language = st.selectbox("Spoken language", ["english", "hinglish"], index=0,
                            format_func=lambda v: LANGUAGE_LABELS[v])

    run_btn = st.button("Analyse video", type="primary", icon=":material/auto_awesome:", use_container_width=True)
    st.markdown('<div class="v-side-note">English is transcribed with Whisper; Hinglish with Sarvam AI '
                '(translated to English).</div>', unsafe_allow_html=True)

    st.markdown('<div class="v-side-footer">Powered by Whisper · Sarvam AI · Gemini<br>'
                'ChromaDB · HuggingFace embeddings</div>', unsafe_allow_html=True)

# ─── Top bar ────────────────────────────────────────────────────────────────────
if run_btn and (source.strip() or uploaded_file is not None):
    _status = '<span class="v-pill v-pill-busy">Analysing…</span>'
elif st.session_state.result:
    _status = '<span class="v-pill v-pill-ready">Analysis ready</span>'
else:
    _status = '<span class="v-pill v-pill-idle">Ready</span>'
st.markdown(
    f'<div class="v-topbar"><div class="v-brand">{LOGO}<div class="v-wordmark">VAANI<span class="v-colon">:</span> '
    f'<span class="v-sub">Your AI Video Assistant</span></div></div>{_status}</div>',
    unsafe_allow_html=True,
)

# ── Run Pipeline ────────────────────────────────────────────────────────────────
pipeline_started = False
if run_btn:
    if not source.strip() and uploaded_file is None:
        st.error("Please enter a YouTube URL or file path, or upload a file.")
    elif not (os.getenv("GEMINI_API_KEY") or "").strip():
        st.error("Gemini API key is not configured. Add GEMINI_API_KEY to your .env file (local) or to the app's Secrets on Streamlit Cloud.")
    elif language == "hinglish" and not os.getenv("SARVAM_API_KEY"):
        st.error("Sarvam API key is required for Hinglish transcription. Add SARVAM_API_KEY to your .env file (local) or Streamlit secrets (cloud).")
    else:
        pipeline_started = True
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        progress_placeholder = st.empty()

        step_labels = {
            "audio": "Extracting and preparing audio",
            "transcript": "Transcribing audio with " + ("Sarvam AI" if language == "hinglish" else "Whisper") + " (this is the longest step)",
            "title": "Generating title",
            "summary": "Summarising transcript",
            "extract": "Extracting action items, decisions and questions",
            "rag": "Building the chat index",
        }
        if uploaded_file is None and is_url(source.strip()):
            step_labels["audio"] = "Fetching the video's transcript (audio is downloaded only if needed)"

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state
            if state == "active":
                render_progress(progress_placeholder, st.session_state.pipeline_steps, f"{step_labels[key]}…")

        upload_path = None
        chunks = []
        transcript_source = "audio"
        try:
            update_step("audio", "active")
            if uploaded_file is not None:
                # Save the upload to a uniquely named temp file (only the extension is kept from the user's filename)
                suffix = os.path.splitext(uploaded_file.name)[1]
                fd, upload_path = tempfile.mkstemp(suffix=suffix, dir=DOWNLOAD_DIR)
                with os.fdopen(fd, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                chunks = process_input(upload_path)
                update_step("audio", "done")

                update_step("transcript", "active")
                transcript = transcribe_all(chunks, language)
                update_step("transcript", "done")
            else:
                # YouTube URL: Supadata transcript first, the existing audio pipeline as fallback.
                # A local file path goes straight to the existing audio pipeline.
                transcript, transcript_source = get_transcript_for_source(source.strip(), language, on_step=update_step)

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            action_items  = extract_action_items(transcript)
            decisions     = extract_key_decisions(transcript)
            questions     = extract_questions(transcript)
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
                "language": language,
                "transcript_source": transcript_source,
            }
            st.session_state.pipeline_done = True
            render_progress(progress_placeholder, st.session_state.pipeline_steps, "Opening your results…", finished=True)
            time.sleep(0.5)
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            logger.exception("Pipeline failed")  # full traceback goes to the server log only
            failed_step = "processing"
            for k in ["audio","transcript","title","summary","extract","rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "failed"
                    failed_step = k
            if isinstance(e, (UserFacingError, FileNotFoundError)):
                message = str(e)  # our own readable messages (FFmpeg, YouTube, missing file, Sarvam, no speech)
            elif failed_step == "audio":
                message = "Could not read or convert this audio/video file. Check that it is a valid media file."
            elif failed_step == "transcript":
                message = f"Transcription failed ({type(e).__name__}). Check the server logs for details."
            else:
                message = describe_llm_error(e)  # Gemini quota / rate limit / timeout / key problems, no secrets
            render_progress(progress_placeholder, st.session_state.pipeline_steps, error=f"Error: {message}")
        finally:
            cleanup_files(chunks + [upload_path])

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result
    words = len(r["transcript"].split())
    lang = r.get("language", "english")
    engine = ("YouTube captions (Supadata)" if r.get("transcript_source") == "supadata"
              else "YouTube captions (Supadata), translated to English by Gemini" if r.get("transcript_source") == "supadata_translated"
              else "Sarvam AI" if lang == "hinglish" else "Whisper")

    # Title card
    st.markdown(
        f'<div class="v-title-card"><div class="v-eyebrow">Video analysis</div>'
        f'<div class="v-title">{html.escape(r["title"])}</div><div class="v-meta">'
        f'<span class="v-chip">{html.escape(LANGUAGE_LABELS.get(lang, lang))} · {engine}</span>'
        f'<span class="v-chip">{words:,} words transcribed</span>'
        f'<span class="v-chip v-chip-ok">Ready to chat</span></div></div>',
        unsafe_allow_html=True,
    )

    # Insights: summary + action items, then decisions + open questions
    col1, col2 = st.columns([3, 2], gap="medium")
    with col1:
        render_card("summary", "📝", "Summary", r["summary"], show_count=False)
    with col2:
        render_card("actions", "✅", "Action Items", r["action_items"])

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        render_card("decisions", "🤝", "Key Decisions", r["key_decisions"])
    with c2:
        render_card("questions", "❓", "Open Questions", r["open_questions"])

    # Transcript — collapsed so long transcripts don't take over the page (full text, never truncated)
    with st.expander(f"Transcript · {words:,} words", icon=":material/subject:", expanded=False):
        st.markdown(f'<div class="v-transcript">{html.escape(r["transcript"])}</div>', unsafe_allow_html=True)

    # Export
    st.markdown('<div class="v-section-head v-export"><span class="v-icon">⬇️</span><div>'
                '<div class="v-section-title">Export analysis</div>'
                '<div class="v-section-sub">Title, summary, insights and the full transcript.</div></div></div>',
                unsafe_allow_html=True)
    if "export_txt" not in r:  # built once per analysis, reused on every rerun (chat reruns the page)
        r["export_txt"] = build_txt(r)
        try:
            r["export_pdf"] = build_pdf(r)
        except Exception:
            logger.exception("PDF export failed")
            r["export_pdf"] = None
    exp1, exp2, _ = st.columns([1, 1, 1.4], gap="small")
    with exp1:
        st.download_button("Download TXT", data=r["export_txt"], file_name="vaani_analysis.txt",
                           mime="text/plain", icon=":material/description:", use_container_width=True)
    with exp2:
        if r["export_pdf"] is not None:
            st.download_button("Download PDF", data=r["export_pdf"], file_name="vaani_analysis.pdf",
                               mime="application/pdf", icon=":material/picture_as_pdf:", use_container_width=True)
        else:
            st.error("PDF export failed — TXT export is still available.")

    # ── RAG Chat ──────────────────────────────────────────────────────────────
    st.markdown('<div class="v-section-head v-chat"><span class="v-icon">💬</span><div>'
                '<div class="v-section-title">Chat with Vaani</div>'
                '<div class="v-section-sub">Answers come only from this video\'s transcript.</div></div></div>',
                unsafe_allow_html=True)

    # Chat history display (every message is HTML-escaped before it is placed in the markup)
    if st.session_state.chat_history:
        chat_html = '<div class="v-chat-box"><div>'
        for msg in st.session_state.chat_history:
            text = html.escape(msg["content"]).replace(chr(10), "<br>")
            if msg["role"] == "user":
                chat_html += f'<div class="v-msg v-user"><div class="v-bubble">{text}</div></div>'
            else:
                chat_html += (f'<div class="v-msg v-bot"><div class="v-avatar">V</div><div class="v-bubble">'
                              f'<div class="v-role">Vaani</div>{text}</div></div>')
        chat_html += '</div></div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown('<div class="v-chat-box"><div class="v-chat-empty"><b>Ask Vaani about this video.</b><br>'
                    'For example:<br><span class="v-hint">What were the key decisions?</span>'
                    '<span class="v-hint">Who is responsible for what?</span>'
                    '<span class="v-hint">What is still undecided?</span></div></div>', unsafe_allow_html=True)

    # Chat input — a form so that pressing Enter sends the question (with separate widgets, Enter only
    # reran the page and the question was silently dropped) and the box is cleared after sending
    with st.form("chat_form", clear_on_submit=True, border=False):
        chat_col1, chat_col2 = st.columns([5, 1], gap="small", vertical_alignment="top")
        with chat_col1:
            user_input = st.text_input("Your question", placeholder="Ask Vaani anything about this video…", label_visibility="collapsed")
        with chat_col2:
            send_btn = st.form_submit_button("Send", type="primary", icon=":material/send:", use_container_width=True)

    if send_btn and user_input.strip():
        try:
            with st.spinner("Vaani is thinking…"):
                # This analysis's chat so far (reset on every new analysis, so nothing leaks across videos)
                answer = ask_question(r["rag_chain"], user_input.strip(), st.session_state.chat_history)
        except Exception as e:
            logger.exception("RAG question failed")
            st.error(f"❌ Could not get an answer: {describe_llm_error(e)}")
        else:
            st.session_state.chat_history.append({"role": "user",      "content": user_input.strip()})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear chat", type="secondary", icon=":material/delete_sweep:"):
            st.session_state.chat_history = []
            st.rerun()

elif not pipeline_started:
    # Empty state
    st.markdown("""<div class="v-empty">
<h2>Vaani is ready to analyse a video</h2>
<p>Turn any video or recording into a summary, action items, decisions and open questions, then chat with it.</p>
<div class="v-howto">
<div><b>1</b>Paste a YouTube link or upload a file in the sidebar</div>
<div><b>2</b>Choose the spoken language</div>
<div><b>3</b>Click <strong>Analyse video</strong></div>
</div>
<div class="v-features">
<span class="v-feature" style="background:var(--coral-soft);color:var(--coral)">Summary</span>
<span class="v-feature" style="background:var(--amber-soft);color:var(--amber)">Action items</span>
<span class="v-feature" style="background:var(--green-soft);color:var(--green)">Decisions</span>
<span class="v-feature" style="background:var(--blue-soft);color:var(--blue)">Open questions</span>
<span class="v-feature" style="background:var(--teal-soft);color:var(--teal)">Full transcript</span>
<span class="v-feature" style="background:var(--brand-soft);color:var(--brand)">Chat with Vaani</span>
<span class="v-feature" style="background:var(--surface-2);color:var(--muted)">TXT &amp; PDF export</span>
</div>
</div>""", unsafe_allow_html=True)
