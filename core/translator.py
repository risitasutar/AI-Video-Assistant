"""Translate a non-English transcript (e.g. Hindi YouTube captions) to English with Gemini.

VAANI's downstream pipeline works on English text: the audio paths already produce English
(Whisper English, Sarvam translate mode) and the embedding model all-MiniLM-L6-v2 is English-only,
so retrieval over untranslated Devanagari text would be unreliable.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import get_llm

# Large sections keep the number of Gemini calls low; no overlap, so nothing is translated twice
SECTION_CHARS = 12000

TRANSLATE_PROMPT = (
    "Translate this video transcript into natural English. It may be in Hindi or another Indian "
    "language, or mixed with English (Hinglish), in Devanagari or Latin script. Translate faithfully "
    "and completely: keep names, numbers and technical terms, do not summarise, do not add or omit "
    "anything, and do not add notes. Output only the English translation."
)


def translate_to_english(text: str) -> str:
    """English translation of the whole transcript (raises on Gemini errors; never returns partial text)."""
    sections = RecursiveCharacterTextSplitter(chunk_size=SECTION_CHARS, chunk_overlap=0).split_text(text)
    chain = (ChatPromptTemplate.from_messages([("system", TRANSLATE_PROMPT), ("human", "{text}")])
             | get_llm(temperature=0) | StrOutputParser())
    return " ".join(chain.invoke({"text": section}).strip() for section in sections).strip()
