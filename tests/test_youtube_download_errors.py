"""Offline tests for YouTube download retries, the web_embedded fallback and error messages
in utils/audio_processor.py.

yt-dlp is replaced by a scripted stub, so no network access or API keys are needed.
Run from the project root:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yt_dlp
import utils.audio_processor as ap
from utils.errors import UserFacingError

URL = "https://www.youtube.com/watch?v=test1234567"
CLOUD_HINT = "YouTube may block downloads from cloud-hosted servers"
UPLOAD_HINT = "Upload a video or audio file"
E403 = "ERROR: unable to download video data: HTTP Error 403: Forbidden"
OK = "ok"


class _ScriptedYDL:
    """Stand-in for yt_dlp.YoutubeDL. Each extract_info() call consumes the next scripted outcome
    (OK or an error message) and records which player client the attempt used."""
    script = []
    clients = []

    def __init__(self, opts):
        args = (opts.get("extractor_args") or {}).get("youtube", {})
        self.client = (args.get("player_client") or ["default"])[0]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        type(self).clients.append(self.client)
        outcome = type(self).script.pop(0)
        if outcome != OK:
            raise yt_dlp.utils.DownloadError(outcome)
        return {"id": "test1234567", "ext": "webm"}

    def prepare_filename(self, info):
        return os.path.join(ap.DOWNLOAD_DIR, "stub_test1234567.webm")


class YouTubeDownloadTests(unittest.TestCase):
    def _download(self, script, file_exists=True):
        _ScriptedYDL.script, _ScriptedYDL.clients = list(script), []
        with mock.patch.object(ap.yt_dlp, "YoutubeDL", _ScriptedYDL), \
                mock.patch.object(ap.time, "sleep"), \
                mock.patch.object(ap.os.path, "exists", return_value=file_exists):
            return ap.download_youtube_audio(URL)

    def _download_error(self, script):
        with self.assertRaises(UserFacingError) as ctx:
            self._download(script)
        return str(ctx.exception)

    # ── success paths ──
    def test_default_client_success_does_not_use_fallback(self):
        path = self._download([OK])
        self.assertTrue(path.endswith("stub_test1234567.wav"))
        self.assertEqual(_ScriptedYDL.clients, ["default"])

    def test_default_403_then_retry_success(self):
        path = self._download([E403, OK])
        self.assertTrue(path.endswith(".wav"))
        self.assertEqual(_ScriptedYDL.clients, ["default", "default"])

    def test_fallback_success_after_default_403s(self):
        path = self._download([E403, E403, E403, OK])
        self.assertTrue(path.endswith(".wav"))
        self.assertEqual(_ScriptedYDL.clients, ["default"] * 3 + ["web_embedded"])

    # ── failure paths ──
    def test_both_default_and_fallback_403_suggest_upload(self):
        msg = self._download_error([E403] * 5)
        self.assertEqual(_ScriptedYDL.clients, ["default"] * 3 + ["web_embedded"] * 2,
                         "3 default attempts, then 2 bounded fallback attempts")
        self.assertTrue(msg.startswith("Unable to download audio from this YouTube URL."))
        self.assertIn("HTTP Error 403: Forbidden", msg)
        self.assertIn(CLOUD_HINT, msg)
        self.assertIn(UPLOAD_HINT, msg)

    def test_fallback_unsupported_for_video_suggests_upload(self):
        for unsupported in ("ERROR: [youtube] test1234567: Requested format is not available",
                            "ERROR: [youtube] test1234567: Playback on other websites has been disabled by the video owner"):
            msg = self._download_error([E403, E403, E403, unsupported])
            self.assertEqual(_ScriptedYDL.clients, ["default"] * 3 + ["web_embedded"],
                             "an unsupported fallback is not retried")
            self.assertIn(CLOUD_HINT, msg)
            self.assertIn(UPLOAD_HINT, msg)

    def test_non_403_error_fails_immediately_without_fallback_or_hint(self):
        msg = self._download_error(["ERROR: [youtube] test1234567: Video unavailable"])
        self.assertEqual(_ScriptedYDL.clients, ["default"], "no retry and no fallback for non-403 errors")
        self.assertIn("Video unavailable", msg)
        self.assertNotIn(CLOUD_HINT, msg)

    def test_missing_output_file_is_reported(self):
        with self.assertRaises(UserFacingError) as ctx:
            self._download([OK], file_exists=False)
        self.assertIn("no audio file was produced", str(ctx.exception))
        self.assertEqual(_ScriptedYDL.clients, ["default"])


if __name__ == "__main__":
    unittest.main()
