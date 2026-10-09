"""Offline tests for utils/supadata.py (Supadata YouTube transcripts + audio-pipeline fallback).

HTTP calls and the audio pipeline are mocked: no network access, no API keys, no Whisper.
Run from the project root:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import utils.supadata as sd
from utils.errors import UserFacingError

YT = "https://www.youtube.com/watch?v=abc123XYZ00"
FAKE_KEY = "test-key-not-real-0123456789"


class _Resp:
    def __init__(self, status=200, body=None, json_error=False):
        self.status_code, self._body, self._json_error = status, body, json_error

    def json(self):
        if self._json_error:
            raise ValueError("not json")
        return self._body


def _ok(content, lang="en"):
    return _Resp(200, {"content": content, "lang": lang, "availableLangs": [lang]})


class SupadataAdapterTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"SUPADATA_API_KEY": FAKE_KEY})
        self.env.start()
        self.sleep = mock.patch.object(sd.time, "sleep")
        self.sleep.start()

    def tearDown(self):
        self.env.stop()
        self.sleep.stop()

    def _fetch(self, *responses):
        with mock.patch.object(sd.requests, "get", side_effect=list(responses)) as get:
            return sd.fetch_youtube_transcript(YT), get

    def _fetch_error(self, *responses):
        with self.assertRaises(sd.SupadataError) as ctx:
            self._fetch(*responses)
        msg = str(ctx.exception)
        self.assertNotIn(FAKE_KEY, msg, "the API key must never appear in error messages")
        return msg

    # ── success ──
    def test_success_normalizes_segments(self):
        segments = [{"text": "Hello  everyone,", "offset": 0, "duration": 1500, "lang": "en"},
                    {"text": "it&#39;s launch\nday &amp; we&#39;re ready.", "offset": 1500, "duration": 2000, "lang": "en"},
                    {"text": "  Thanks! ", "offset": 3500, "duration": 800, "lang": "en"}]
        transcript, get = self._fetch(_ok(segments))
        self.assertEqual(transcript, "Hello everyone, it's launch day & we're ready. Thanks!")
        url, = get.call_args.args
        self.assertEqual(url, "https://api.supadata.ai/v1/transcript")
        self.assertEqual(get.call_args.kwargs["params"], {"url": YT, "lang": "en"})
        self.assertEqual(get.call_args.kwargs["headers"], {"x-api-key": FAKE_KEY})
        self.assertIsNotNone(get.call_args.kwargs["timeout"])

    def test_success_plain_text_content(self):
        transcript, _ = self._fetch(_ok("  Plain   text transcript. "))
        self.assertEqual(transcript, "Plain text transcript.")

    def test_async_job_is_polled_until_completed(self):
        transcript, get = self._fetch(
            _Resp(202, {"jobId": "job-1"}),
            _Resp(200, {"status": "active"}),
            _Resp(200, {"status": "completed", "lang": "en", "content": [{"text": "Done.", "offset": 0, "duration": 1}]}))
        self.assertEqual(transcript, "Done.")
        self.assertEqual(get.call_args_list[1].args[0], "https://api.supadata.ai/v1/transcript/job-1")

    # ── errors ──
    def test_missing_key(self):
        with mock.patch.dict(os.environ, {"SUPADATA_API_KEY": ""}):
            with mock.patch.object(sd.requests, "get") as get:
                with self.assertRaises(sd.SupadataError) as ctx:
                    sd.fetch_youtube_transcript(YT)
        self.assertIn("SUPADATA_API_KEY is not configured", str(ctx.exception))
        get.assert_not_called()

    def test_api_errors_have_readable_messages(self):
        cases = {401: "rejected", 402: "plan limit", 403: "not allowed", 404: "could not find this video",
                 206: "no transcript is available", 429: "rate limit", 500: "internal error", 418: "HTTP 418"}
        for status, expected in cases.items():
            body = {"error": "x", "message": "server says no", "details": "..."}
            self.assertIn(expected, self._fetch_error(_Resp(status, body)), f"HTTP {status}")

    def test_timeout(self):
        self.assertIn("timed out", self._fetch_error(requests.Timeout("read timed out")))

    def test_connection_error(self):
        self.assertIn("could not connect", self._fetch_error(requests.ConnectionError("dns")))

    def test_malformed_responses(self):
        for resp in (_Resp(200, json_error=True), _Resp(200, ["not", "a", "dict"]),
                     _Resp(200, {"lang": "en"}), _Resp(200, {"content": [{"no_text": 1}], "lang": "en"}),
                     _Resp(200, {"content": 42, "lang": "en"}), _Resp(202, {"no_job_id": True})):
            self.assertIn("unexpected", self._fetch_error(resp))

    def test_empty_transcript(self):
        for content in ([], [{"text": "  ", "offset": 0, "duration": 1}], ""):
            self.assertIn("empty transcript", self._fetch_error(_ok(content)))

    def test_failed_and_slow_jobs(self):
        self.assertIn("could not generate", self._fetch_error(_Resp(202, {"jobId": "j"}), _Resp(200, {"status": "failed"})))
        with mock.patch.object(sd, "POLL_MAX_WAIT", -1):
            self.assertIn("too long", self._fetch_error(_Resp(202, {"jobId": "j"}), _Resp(200, {"status": "queued"})))

    def test_non_english_transcript_is_not_used(self):
        self.assertIn("'hi' transcript", self._fetch_error(_ok([{"text": "नमस्ते", "offset": 0, "duration": 1}], lang="hi")))


class TranscriptRoutingTests(unittest.TestCase):
    """get_transcript_for_source: Supadata first for URLs, existing audio pipeline as fallback."""

    def setUp(self):
        self.patches = [
            mock.patch.object(sd, "process_input", return_value=["chunk_0.wav"]),
            mock.patch.object(sd, "transcribe_all", return_value="Audio transcript."),
            mock.patch.object(sd, "cleanup_files"),
        ]
        self.process_input, self.transcribe_all, self.cleanup = (p.start() for p in self.patches)
        self.steps = []

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def _get(self, source, language="english"):
        return sd.get_transcript_for_source(source, language, on_step=lambda k, s: self.steps.append((k, s)))

    def test_supadata_success_bypasses_audio_pipeline(self):
        with mock.patch.object(sd, "fetch_youtube_transcript", return_value="Caption transcript.") as fetch:
            transcript, method = self._get(YT)
        self.assertEqual((transcript, method), ("Caption transcript.", "supadata"))
        fetch.assert_called_once_with(YT)
        self.process_input.assert_not_called()
        self.transcribe_all.assert_not_called()
        self.assertEqual(self.steps, [("audio", "done"), ("transcript", "done")])

    def test_supadata_failure_falls_back_to_audio_pipeline(self):
        with mock.patch.object(sd, "fetch_youtube_transcript", side_effect=sd.SupadataError("no transcript is available")):
            transcript, method = self._get(YT, "hinglish")
        self.assertEqual((transcript, method), ("Audio transcript.", "audio"))
        self.process_input.assert_called_once_with(YT)
        self.transcribe_all.assert_called_once_with(["chunk_0.wav"], "hinglish")
        self.cleanup.assert_called_once_with(["chunk_0.wav"])
        self.assertEqual(self.steps, [("audio", "done"), ("transcript", "active"), ("transcript", "done")])

    def test_both_methods_failing_gives_readable_error(self):
        self.process_input.side_effect = UserFacingError(
            "Unable to download audio from this YouTube URL. unable to download video data: HTTP Error 403: Forbidden")
        with mock.patch.dict(os.environ, {"SUPADATA_API_KEY": FAKE_KEY}), \
                mock.patch.object(sd.requests, "get", return_value=_Resp(429, {"error": "limit-exceeded"})):
            with self.assertRaises(UserFacingError) as ctx:
                self._get(YT)
        msg = str(ctx.exception)
        self.assertIn("Could not get a transcript for this YouTube video", msg)
        self.assertIn("rate limit", msg)                       # transcript-service reason
        self.assertIn("HTTP Error 403", msg)                    # audio-download reason
        self.assertIn("Upload a video or audio file", msg)
        self.assertNotIn(FAKE_KEY, msg)
        self.transcribe_all.assert_not_called()

    def test_unexpected_audio_failure_after_supadata_failure_is_readable(self):
        self.process_input.side_effect = RuntimeError("internal pydub detail")
        with mock.patch.object(sd, "fetch_youtube_transcript", side_effect=sd.SupadataError("could not connect to Supadata.")):
            with self.assertRaises(UserFacingError) as ctx:
                self._get(YT)
        self.assertNotIn("internal pydub detail", str(ctx.exception))
        self.assertIn("Upload a video or audio file", str(ctx.exception))

    def test_local_file_path_uses_existing_pipeline_only(self):
        with mock.patch.object(sd, "fetch_youtube_transcript") as fetch:
            transcript, method = self._get(r"C:\videos\meeting.mp4")
        fetch.assert_not_called()
        self.assertEqual((transcript, method), ("Audio transcript.", "audio"))
        self.process_input.assert_called_once_with(r"C:\videos\meeting.mp4")

    def test_local_file_errors_are_unchanged(self):
        self.process_input.side_effect = FileNotFoundError("File not found: x.mp4")
        with mock.patch.object(sd, "fetch_youtube_transcript") as fetch:
            with self.assertRaises(FileNotFoundError):
                self._get("x.mp4")
        fetch.assert_not_called()


class UploadPathUnchangedTests(unittest.TestCase):
    """The Streamlit upload branch must keep calling the existing pipeline directly (never Supadata)."""

    def test_app_upload_branch_uses_existing_pipeline(self):
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"), encoding="utf-8") as f:
            src = f.read()
        branch = src.split("if uploaded_file is not None:", 1)[1].split("else:", 1)[0]
        self.assertIn("chunks = process_input(upload_path)", branch)
        self.assertIn("transcript = transcribe_all(chunks, language)", branch)
        self.assertNotIn("supadata", branch.lower())
        self.assertNotIn("get_transcript_for_source", branch)


if __name__ == "__main__":
    unittest.main()
