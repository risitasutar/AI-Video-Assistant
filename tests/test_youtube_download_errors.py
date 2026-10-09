"""Offline tests for the YouTube download error messages in utils/audio_processor.py.

yt-dlp is replaced by a stub, so no network access or API keys are needed.
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


class _FailingYDL:
    """Stand-in for yt_dlp.YoutubeDL whose download always raises the given error."""
    calls = 0
    error = ""

    def __init__(self, opts):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        type(self).calls += 1
        raise yt_dlp.utils.DownloadError(self.error)


class YouTubeDownloadErrorTests(unittest.TestCase):
    def _run(self, error):
        _FailingYDL.calls, _FailingYDL.error = 0, error
        with mock.patch.object(ap.yt_dlp, "YoutubeDL", _FailingYDL), mock.patch.object(ap.time, "sleep"):
            with self.assertRaises(UserFacingError) as ctx:
                ap.download_youtube_audio(URL)
        return str(ctx.exception)

    def test_403_after_retries_suggests_upload(self):
        msg = self._run("ERROR: unable to download video data: HTTP Error 403: Forbidden")
        self.assertEqual(_FailingYDL.calls, 3, "403 should be retried 3 times in total")
        self.assertTrue(msg.startswith("Unable to download audio from this YouTube URL."))
        self.assertIn("HTTP Error 403: Forbidden", msg)
        self.assertIn(CLOUD_HINT, msg)
        self.assertIn("Upload a video or audio file", msg)

    def test_other_errors_fail_immediately_without_cloud_hint(self):
        msg = self._run("ERROR: [youtube] test1234567: Video unavailable")
        self.assertEqual(_FailingYDL.calls, 1, "non-403 errors must not be retried")
        self.assertIn("Video unavailable", msg)
        self.assertNotIn(CLOUD_HINT, msg)

    def test_403_then_success_returns_file(self):
        attempts = {"n": 0}

        class _FlakyYDL(_FailingYDL):
            def extract_info(self, url, download=True):
                attempts["n"] += 1
                if attempts["n"] == 1:
                    raise yt_dlp.utils.DownloadError("ERROR: unable to download video data: HTTP Error 403: Forbidden")
                return {"id": "test1234567", "ext": "webm"}

            def prepare_filename(self, info):
                return os.path.join(ap.DOWNLOAD_DIR, "stub_test1234567.webm")

        with mock.patch.object(ap.yt_dlp, "YoutubeDL", _FlakyYDL), mock.patch.object(ap.time, "sleep"), \
                mock.patch.object(ap.os.path, "exists", return_value=True):
            path = ap.download_youtube_audio(URL)
        self.assertEqual(attempts["n"], 2)
        self.assertTrue(path.endswith("stub_test1234567.wav"))


if __name__ == "__main__":
    unittest.main()
