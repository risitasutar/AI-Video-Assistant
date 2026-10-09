import os
import re
import httpx
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

NOT_FOUND = "I could not find this information in the transcript."

# Recent turns sent with each message (enough for follow-ups, small enough to stay cheap)
HISTORY_MESSAGES = 6          # = last 3 question/answer pairs
HISTORY_CHARS = 600           # per message
MAX_CONTEXT_CHUNKS = 8

# One call per message: the model routes the message itself using these rules
RAG_SYSTEM_PROMPT = f"""You are Vaani, a friendly assistant in a video-analysis app. The user has analysed a video and is chatting with you about it. You see the recent conversation and transcript excerpts retrieved for the latest message.

First decide what the latest message is, then respond accordingly:

1. Small talk (greetings, thanks, acknowledgements such as "great", "okay", "cool"): reply briefly and naturally. Do not mention the transcript or searching.

2. A question about the video (its speakers, what was said, events, decisions, "in this video/meeting/talk"), including any question that refers to a specific person, team, product or event without saying who or what it is ("the CEO", "the team", "she", "the launch" - these mean the ones in the video), and follow-ups that refer to earlier messages ("she", "that", "what next", "explain more"): use the conversation to work out what is meant, then answer ONLY from the transcript excerpts. Explain naturally in your own words. Never invent names, quotes, dates, numbers or events. The transcript has no speaker labels: say who is speaking only if the transcript states it (e.g. "my name is ..."), and do not attribute words to a person unless the transcript does. If the excerpts do not contain the answer, say: "{NOT_FOUND}" If they answer only part of it, answer that part and say which part is not in the transcript. Distinguish what was decided from what was only proposed or left open.

3. A general-knowledge question that is not about this video (science, history, definitions, advice, jokes, news and current events...): answer helpfully from your general knowledge, starting with "This isn't from the video, but". Use that opening only when you actually give a general-knowledge answer - if you are saying the transcript does not contain something, just use the not-found sentence from rule 2. Never present general knowledge as something said in the video. For time-sensitive facts (current office-holders, recent events, prices, statistics) add that your knowledge may be out of date and that you cannot check live sources.

4. A mixed question: answer each part, clearly saying which part comes from the video and which from general knowledge.

Be concise. Reply in plain text without markdown formatting.

Transcript excerpts (in transcript order; may be empty or unrelated for small talk and general questions):
{{context}}"""

# Pure courtesy/acknowledgement messages: answered naturally, without a transcript search
_SMALL_TALK_WORDS = {
    "hi", "hii", "hello", "hey", "hiya", "yo", "good", "morning", "afternoon", "evening", "night",
    "thanks", "thank", "thankyou", "thx", "ty", "you", "u", "so", "much", "a", "lot", "very",
    "great", "nice", "cool", "awesome", "perfect", "amazing", "excellent", "wonderful", "brilliant",
    "ok", "okay", "okk", "k", "alright", "sure", "yes", "yeah", "yep", "no", "nope", "got", "it",
    "understood", "makes", "sense", "fine", "wow", "bye", "goodbye", "see", "later", "cheers",
    "vaani", "that's", "thats", "helpful", "helped", "that", "is", "was", "really",
}


def _is_small_talk(message: str) -> bool:
    words = re.findall(r"[a-z']+", message.lower())
    return 0 < len(words) <= 6 and all(w in _SMALL_TALK_WORDS for w in words) and "?" not in message


def _history_messages(history):
    msgs = []
    for m in (history or [])[-HISTORY_MESSAGES:]:
        text = m["content"][:HISTORY_CHARS]
        msgs.append(HumanMessage(text) if m["role"] == "user" else AIMessage(text))
    return msgs


def get_llm():
    llm = ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.1,
    )
    # retry HTTP errors such as 429 (free-tier rate limits), ~30s total backoff
    return llm.with_retry(retry_if_exception_type=(httpx.HTTPStatusError,), stop_after_attempt=6)

def format_docs(docs):
    # Put retrieved chunks back in transcript order so cross-section answers read coherently
    docs = sorted(docs, key=lambda d: d.metadata.get("chunk_index", 0))
    return "\n\n".join(f"[Excerpt {doc.metadata.get('chunk_index', 0) + 1}] {doc.page_content}" for doc in docs)


def _build_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])

    def retrieve(inputs):
        question, history = inputs["question"], inputs.get("history") or []
        if _is_small_talk(question):
            return ""  # "Great!" / "Thanks" are not transcript searches
        docs = list(retriever.invoke(question))
        # History-aware retrieval for follow-ups ("what did she say next?", "explain that"):
        # also search with the previous exchange attached, so pronouns resolve to the right part
        # of the transcript - a local similarity search, no extra LLM call.
        if history:
            last_q = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
            last_a = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
            docs += retriever.invoke(f"{last_q} {last_a[:300]} {question}")
        unique = {d.metadata.get("chunk_index", i): d for i, d in enumerate(docs)}
        return format_docs(list(unique.values())[:MAX_CONTEXT_CHUNKS])

    #full LCEL Rag pipeline
    return (
        RunnableLambda(lambda inp: {
            "question": inp["question"],
            "history": _history_messages(inp.get("history")),
            "context": retrieve(inp),
        })
        | prompt | get_llm() | StrOutputParser()
        | RunnableLambda(lambda a: a.replace("**", ""))  # chat bubbles show plain text
    )


def build_rag_chain(transcript:str):

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k = 6)

    return _build_chain(retriever)


def load_rag_chain():
    vector_store = load_vector_store()
    return _build_chain(get_retriever(vector_store, k = 6))


def ask_question(rag_chain, question:str, history: list = None) -> str:
    """history: this analysis's previous chat messages [{"role": "user"|"assistant", "content": str}]."""
    print(f"Question : {question}")
    answer = rag_chain.invoke({"question": question, "history": history or []})
    print(f"answer :{answer}")
    return answer
