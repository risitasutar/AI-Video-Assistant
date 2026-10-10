"""Offline tests for the Gemini integration (core/llm.py and the chains that use it).

The Gemini model is replaced by local fakes: no network, no API key, no quota used.
The summary/extraction/RAG chains, prompts, splitting and retrieval are the real project code.
Run from the project root:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.api_core import exceptions as gexc
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

import core.llm as llm_mod
from core.llm import describe_llm_error, LLMResponseError
from core.summarizer import summarize, generate_title, SINGLE_PASS_CHARS
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question
from utils.errors import UserFacingError

FAKE_KEY = "AIzaFAKE-test-key-not-real-000000000"
TRANSCRIPT = ("Good morning everyone. We decided to launch the mobile app on 15 November. "
              "Priya will prepare the marketing plan by next Friday. Pricing is still undecided.")


class FakeGemini:
    """Replaces ChatGoogleGenerativeAI: records constructor kwargs and every prompt it receives."""
    instances, prompts = [], []

    def __init__(self, replies=None, error=None):
        self.replies, self.error = list(replies or []), error

    def __call__(self, **kwargs):  # used as the patched class
        FakeGemini.instances.append(kwargs)
        return RunnableLambda(self._respond)

    def _respond(self, prompt_value):
        FakeGemini.prompts.append(prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value))
        if self.error is not None:
            raise self.error
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply if isinstance(reply, AIMessage) else AIMessage(content=reply)


class GeminiTestCase(unittest.TestCase):
    def setUp(self):
        FakeGemini.instances, FakeGemini.prompts = [], []
        self.env = mock.patch.dict(os.environ, {"GEMINI_API_KEY": FAKE_KEY}, clear=False)
        self.env.start()
        os.environ.pop("GEMINI_MODEL", None)

    def tearDown(self):
        self.env.stop()

    def fake(self, replies=None, error=None):
        return mock.patch.object(llm_mod, "ChatGoogleGenerativeAI", FakeGemini(replies, error))


class ConfigurationTests(GeminiTestCase):
    def test_real_client_configuration_without_exposing_key(self):
        model = llm_mod.ChatGoogleGenerativeAI(  # real class, constructed only (no request is sent)
            model=llm_mod.gemini_model_name(), google_api_key=llm_mod.get_gemini_api_key(),
            temperature=0.1, max_retries=llm_mod.MAX_ATTEMPTS, timeout=llm_mod.REQUEST_TIMEOUT)
        self.assertEqual(model.model.split("/")[-1], "gemini-3.5-flash-lite")
        self.assertEqual(model.max_retries, 2)
        self.assertEqual(model.timeout, 60)
        for text in (str(model), repr(model), str(model.google_api_key)):
            self.assertNotIn(FAKE_KEY, text, "the key must not appear in string representations")
        self.assertEqual(model.google_api_key.get_secret_value(), FAKE_KEY)

    def test_factory_passes_settings(self):
        with self.fake(["ok"]):
            llm_mod.get_llm(temperature=0.2)
        kw = FakeGemini.instances[-1]
        self.assertEqual((kw["model"], kw["temperature"], kw["max_retries"], kw["timeout"]),
                         ("gemini-3.5-flash-lite", 0.2, 2, 60))
        self.assertEqual(kw["google_api_key"], FAKE_KEY)

    def test_model_override(self):
        with mock.patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.5-flash-lite"}), self.fake(["ok"]):
            llm_mod.get_llm(temperature=0.1)
        self.assertEqual(FakeGemini.instances[-1]["model"], "gemini-2.5-flash-lite")

    def test_missing_key(self):
        for value in ("", "   "):
            with mock.patch.dict(os.environ, {"GEMINI_API_KEY": value}), self.fake(["ok"]):
                with self.assertRaises(UserFacingError) as ctx:
                    summarize(TRANSCRIPT)
            self.assertIn("GEMINI_API_KEY", str(ctx.exception))
            self.assertIn("Streamlit Cloud", str(ctx.exception))
        self.assertEqual(FakeGemini.prompts, [], "no request may be attempted without a key")


class GenerationTests(GeminiTestCase):
    def test_summary_and_title(self):
        with self.fake(["- Launch on 15 November\n- Priya: marketing plan"]):
            self.assertEqual(summarize(TRANSCRIPT), "- Launch on 15 November\n- Priya: marketing plan")
        self.assertIn("15 November", FakeGemini.prompts[0], "the transcript is sent to the model")
        with self.fake(["**Mobile App Launch Plan**\nextra line"]):
            self.assertEqual(generate_title(TRANSCRIPT), "Mobile App Launch Plan")  # existing title clean-up kept

    def test_long_transcript_uses_existing_map_reduce(self):
        long_text = (TRANSCRIPT + " ") * (SINGLE_PASS_CHARS // len(TRANSCRIPT) + 5)
        with self.fake(["section notes"]):
            summarize(long_text)
        self.assertGreaterEqual(len(FakeGemini.prompts), 3, "map step per section + final reduce step")
        self.assertIn("Merge them into one summary", FakeGemini.prompts[-1])

    def test_extraction(self):
        with self.fake(["1. **Prepare the marketing plan** - Owner: Priya - Deadline: next Friday"]):
            self.assertIn("Priya", extract_action_items(TRANSCRIPT))
        with self.fake(["1. The app launches on 15 November."]):
            self.assertIn("15 November", extract_key_decisions(TRANSCRIPT))
        with self.fake(["1. What should the pricing be?"]):
            self.assertIn("pricing", extract_questions(TRANSCRIPT))
        self.assertIn("action items", FakeGemini.prompts[0].lower())
        self.assertIn("decisions", FakeGemini.prompts[1].lower())
        self.assertIn("open questions", FakeGemini.prompts[2].lower())

    def test_rag_answer_uses_retrieved_transcript(self):
        with self.fake(["Priya will prepare the marketing plan by next Friday."]):
            chain = build_rag_chain(TRANSCRIPT)             # real MiniLM embeddings + ChromaDB
            answer = ask_question(chain, "Who prepares the marketing plan?", [])
        self.assertEqual(answer, "Priya will prepare the marketing plan by next Friday.")
        prompt = FakeGemini.prompts[-1]
        self.assertIn("Priya will prepare the marketing plan", prompt, "retrieved transcript excerpt is in the prompt")
        self.assertIn("Who prepares the marketing plan?", prompt)

    def test_list_content_is_normalised(self):
        reply = AIMessage(content=[{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}])
        with self.fake([reply]):
            self.assertEqual(summarize(TRANSCRIPT), "Part one. Part two.")


class TranslatorTests(GeminiTestCase):
    def test_hindi_is_translated_with_gemini(self):
        from core.translator import translate_to_english
        with self.fake(["Hello team. The launch will be on 15 November."]):
            out = translate_to_english("नमस्ते टीम। लॉन्च 15 नवंबर को होगा।")
        self.assertEqual(out, "Hello team. The launch will be on 15 November.")
        self.assertIn("नमस्ते टीम", FakeGemini.prompts[0], "the Hindi text is sent for translation")
        self.assertIn("Translate this video transcript into natural English", FakeGemini.prompts[0])
        self.assertEqual(FakeGemini.instances[-1]["temperature"], 0)

    def test_long_transcript_is_translated_in_sections_without_overlap(self):
        from core.translator import translate_to_english, SECTION_CHARS
        hindi = ("यह एक लंबा वाक्य है जो बार बार दोहराया जाता है। " * 1200)
        with self.fake(["Section."]):
            out = translate_to_english(hindi)
        sections = len(FakeGemini.prompts)
        self.assertGreaterEqual(sections, len(hindi) // SECTION_CHARS)
        self.assertEqual(out, " ".join(["Section."] * sections))
        sent = "".join(prompt.split("Human: ", 1)[1] for prompt in FakeGemini.prompts)
        self.assertLessEqual(len(sent.replace(" ", "")), len(hindi.replace(" ", "")), "no text is translated twice")

    def test_translation_errors_propagate(self):
        from core.translator import translate_to_english
        with self.fake(error=gexc.ResourceExhausted("429 quota")):
            with self.assertRaises(gexc.ResourceExhausted):
                translate_to_english("नमस्ते")


class ErrorTests(GeminiTestCase):
    def test_empty_input_is_rejected_without_calling_gemini(self):
        with self.fake(["should not be used"]):
            for fn in (summarize, generate_title, extract_action_items, extract_key_decisions, extract_questions, build_rag_chain):
                with self.assertRaises(UserFacingError) as ctx:
                    fn("   ")
                self.assertIn("transcript is empty", str(ctx.exception))
        self.assertEqual(FakeGemini.prompts, [])

    def test_empty_or_blocked_response_is_an_error_not_a_fake_summary(self):
        for bad in ("", "   ", AIMessage(content=[])):
            with self.fake([bad]):
                with self.assertRaises(LLMResponseError):
                    summarize(TRANSCRIPT)

    def test_provider_errors_propagate_and_are_described(self):
        cases = [
            (gexc.ResourceExhausted("429 You exceeded your current quota"), "free-tier quota or rate limit"),
            (gexc.TooManyRequests("429 rate limit"), "free-tier quota or rate limit"),
            (gexc.DeadlineExceeded("504 Deadline Exceeded"), "timed out"),
            (gexc.ServiceUnavailable("503 overloaded"), "temporarily unavailable"),
            (gexc.InternalServerError("500 internal"), "temporarily unavailable"),
            (ChatGoogleGenerativeAIError("Invalid argument provided to Gemini: 400 API key not valid. Please pass a valid API key."), "key was rejected"),
            (gexc.PermissionDenied("403 permission denied"), "key was rejected"),
            (gexc.NotFound("404 models/gemini-x is not found"), "model is not available"),
            (RuntimeError("something unexpected"), "request failed (RuntimeError)"),
        ]
        for error, expected in cases:
            with self.fake(error=error):
                with self.assertRaises(Exception) as ctx:      # a failed call is never turned into a summary
                    summarize(TRANSCRIPT)
            self.assertIs(ctx.exception, error)
            message = describe_llm_error(ctx.exception)
            self.assertIn(expected, message, type(error).__name__)
            self.assertNotIn(FAKE_KEY, message)

    def test_rag_errors_propagate(self):
        with self.fake(error=gexc.ResourceExhausted("429 quota")):
            chain = build_rag_chain(TRANSCRIPT)
            with self.assertRaises(gexc.ResourceExhausted):
                ask_question(chain, "When is the launch?", [])


if __name__ == "__main__":
    unittest.main()
