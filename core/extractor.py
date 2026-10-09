#Actionableitems , decision , questions

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.summarizer import GROUNDING_RULES, SINGLE_PASS_CHARS, split_transcript
import httpx
import os


def get_llm():
    llm = ChatMistralAI(model = os.getenv("MISTRAL_MODEL", "mistral-small-latest"), mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.1)
    # retry HTTP errors such as 429 (free-tier rate limits), ~30s total backoff
    return llm.with_retry(retry_if_exception_type=(httpx.HTTPStatusError,), stop_after_attempt=6)


OUTPUT_RULES = (
    "Return only the numbered list - no introduction, no notes, no closing remarks. "
)

ACTION_ITEMS_PROMPT = (
    "Extract the action items from this transcript. An action item is a concrete task that the "
    "transcript explicitly assigns to someone or that someone explicitly commits to doing. "
    "Do NOT include suggestions ('we could', 'someone should probably'), opinions, possibilities, "
    "open questions, undecided topics, or things that were already done. "
    "Format each item as: **Task** - Owner: <name, or 'Not specified'> - Deadline: <as stated, "
    "or 'Not specified'>. Never guess an owner or deadline. "
)
NO_ACTION_ITEMS = "No action items found."

DECISIONS_PROMPT = (
    "Extract the decisions from this transcript. A decision is something the speakers explicitly "
    "agreed, confirmed or settled. Do NOT include proposals, suggestions, opinions, possibilities, "
    "topics that were postponed or left open, or task assignments (those are action items). "
    "Write each decision as one short sentence. "
)
NO_DECISIONS = "No key decisions found."

QUESTIONS_PROMPT = (
    "List the open questions and unresolved issues from this transcript: questions that were "
    "asked but not answered, topics explicitly postponed, and proposals explicitly left undecided. "
    "Only include items the speakers themselves raise as open - general discussion or chit-chat is not an open question. "
    "Do NOT include questions that were answered later in the transcript, rhetorical questions, "
    "and do not invent follow-up questions of your own. "
    "Write each as one short line. "
)
NO_QUESTIONS = "No open questions found."


def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt + " " + OUTPUT_RULES + GROUNDING_RULES),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )


def _extract(transcript: str, task_prompt: str, none_text: str) -> str:
    none_rule = f"If there are none, reply exactly: {none_text}"
    if len(transcript) <= SINGLE_PASS_CHARS:
        return build_chain(task_prompt + none_rule).invoke(transcript)

    # Long transcript: a single call misses items buried in the middle, so first collect
    # candidates section by section (recall), then check them against the full transcript.
    section_chain = build_chain(task_prompt + none_rule)
    candidates = [section_chain.invoke(section) for section in split_transcript(transcript)]
    candidates = [c.strip() for c in candidates if none_text.lower() not in c.lower()]
    if not candidates:
        return none_text

    verify_prompt = ChatPromptTemplate.from_messages([
        ("system",
         task_prompt +
         "You are given candidate items that were extracted from consecutive sections of the full "
         "transcript below. Keep only candidates that meet the definition above when the FULL "
         "transcript is considered (drop any that are resolved, answered or contradicted elsewhere "
         "in it), merge duplicates, and keep transcript order. Do not add new items. "
         + none_rule + " " + OUTPUT_RULES + GROUNDING_RULES),
        ("human", "Candidates:\n{candidates}\n\nFull transcript:\n{text}"),
    ])
    verify_chain = verify_prompt | get_llm() | StrOutputParser()
    return verify_chain.invoke({"candidates": "\n".join(candidates), "text": transcript})


def extract_action_items(transcript:str)->str:
    return _extract(transcript, ACTION_ITEMS_PROMPT, NO_ACTION_ITEMS)


def extract_key_decisions(transcript: str) -> str:
    return _extract(transcript, DECISIONS_PROMPT, NO_DECISIONS)


def extract_questions(transcript: str) -> str:
    return _extract(transcript, QUESTIONS_PROMPT, NO_QUESTIONS)
