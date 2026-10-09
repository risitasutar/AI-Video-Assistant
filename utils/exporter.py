import re
from fpdf import FPDF

SECTIONS = [
    ("Summary", "summary"),
    ("Action Items", "action_items"),
    ("Key Decisions", "key_decisions"),
    ("Open Questions", "open_questions"),
    ("Full Transcript", "transcript"),
]

# The built-in PDF fonts only support Latin-1, so map common LLM punctuation first
_PDF_REPLACEMENTS = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "•": "-", "…": "...",
    " ": " ", "​": "",
})


def _plain(text) -> str:
    """Markdown from the LLM -> plain text for exports; empty sections are labelled."""
    text = str(text or "").strip()
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # '### Heading' -> 'Heading'
    return text or "Not available"


def build_txt(result: dict) -> str:
    parts = [f"VAANI: Your AI Video Assistant — {_plain(result.get('title'))}", "=" * 60]
    for heading, key in SECTIONS:
        parts += ["", heading.upper(), "-" * 60, _plain(result.get(key))]
    return "\n".join(parts) + "\n"


def _pdf_safe(text: str) -> str:
    return str(text).translate(_PDF_REPLACEMENTS).encode("latin-1", "replace").decode("latin-1")


def build_pdf(result: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 9, _pdf_safe(_plain(result.get("title"))), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for heading, key in SECTIONS:
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5.5, _pdf_safe(_plain(result.get(key))), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    return bytes(pdf.output())
