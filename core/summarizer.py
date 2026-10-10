from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os
from core.llm import get_llm as _gemini_llm
from utils.errors import UserFacingError

# Shared grounding rules: the transcript may be a meeting OR any other video (lecture, talk, vlog)
GROUNDING_RULES = (
    "The transcript is the ONLY source of truth. Never add facts, names, attendees, dates, years, "
    "deadlines, numbers, project names or technical details that are not stated in it, and do not "
    "use outside knowledge (no extra terminology, explanations or formulas the speakers did not say). "
    "Do not use placeholders such as [Project Name] or [Insert Date]. "
    "Report proposals, opinions and possibilities as such - never as decisions. "
    "Keep any uncertainty expressed in the transcript."
)

# Transcripts up to this size are summarised in a single call (no lossy map step).
# 16k characters ≈ 4k tokens ≈ 18 minutes of speech. Longer inputs are processed in sections:
# a single call over a ~41k-character transcript was measured to miss items buried in it.
SINGLE_PASS_CHARS = 16000


def get_llm():
    return _gemini_llm(temperature=0.2)


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 12000,
        chunk_overlap = 500
    )

    return splitter.split_text(transcript)

SUMMARY_INSTRUCTIONS = (
    "Write a faithful summary of this transcript (it may be a meeting or any other kind of video) "
    "as concise bullet points grouped by topic, in the order the topics appear. "
    "Include any decisions, assignments, numbers and open issues that are stated. "
    "Start directly with the content: no introduction, no title line, no closing remarks, "
    "no 'Next Steps' or 'Key Takeaways' unless the speakers state them. "
)

def summarize(transcript : str) -> str:
    if not (transcript or "").strip():
        raise UserFacingError("The transcript is empty, so there is nothing to analyse.")
    llm = get_llm()

    final_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", SUMMARY_INSTRUCTIONS + GROUNDING_RULES),
        ("human", "{text}"),
    ]
    )
    final_chain = final_prompt | llm | StrOutputParser()

    if len(transcript) <= SINGLE_PASS_CHARS:
        return final_chain.invoke({"text": transcript})

    # Long transcripts: map (faithful notes per section) → reduce (merge notes)
    map_prompt = ChatPromptTemplate.from_messages(
        [
        ("system",
         "Write faithful, compact notes on this section of a longer transcript. Keep every decision, "
         "assignment (who / what / when), number, name and open question that is stated, using the "
         "speakers' wording where possible. No introduction. " + GROUNDING_RULES),
        ("human", "{text}"),
    ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [map_chain.invoke({"text" : chunk}) for chunk in chunks]

    combined = "\n\n".join(f"Section {i + 1} notes:\n{s}" for i, s in enumerate(chunk_summaries))

    combined_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are given notes from consecutive sections of one transcript. Merge them into one "
            "summary, removing duplicates. Use only information present in the notes. "
            + SUMMARY_INSTRUCTIONS + GROUNDING_RULES,
        ),
        ("human", "{text}"),
    ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def _clean_title(title: str) -> str:
    # The model sometimes wraps the title in markdown/quotes; the UI shows it as plain text
    title = title.strip().splitlines()[0] if title.strip() else "Untitled"
    return title.replace("**", "").replace("#", "").strip().strip('"').strip("'").strip()


def generate_title(transcipt : str) -> str:
    if not (transcipt or "").strip():
        raise UserFacingError("The transcript is empty, so there is nothing to analyse.")
    llm = get_llm()



    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) |
        ChatPromptTemplate.from_messages([
             (
                "system",
                "Generate a short, specific title (max 8 words) describing what this transcript "
                "is about. It may be a meeting or any other kind of video. Use only topics that "
                "are clearly discussed; do not add dates, quarters, names or topics that are not "
                "in the transcript. Return only the title as plain text: no quotes, no markdown.",
            ),
            ("human", "{text}"),
        ])
        | llm
        |StrOutputParser()
        | RunnableLambda(_clean_title)
    )

    # Beginning-only (2000 chars) titles described just the intro of long videos
    return title_chain.invoke(transcipt[:12000])




